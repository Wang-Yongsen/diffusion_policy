"""Open the hand after insertion, lift the empty flange, then return to ready."""

from copy import deepcopy
from numbers import Real
from pathlib import Path
import threading
import time

import numpy as np

from rail_common.runtime import execution_summary
from rail_common.core import difference, matrix, transform, vector, write_json
from rail_common.planner import Planner
from .hardware import initial_receiving_pose
from .logic import hand_targets

DEFAULT_RELEASE_RETURN = dict(
    lift_height_m=0.15,
    lift_step_m=0.005,
    max_joint_step_rad=0.05,
    path_tolerance_m=0.005,
    velocity_ratio=1,
    acceleration_ratio=1,
    total_timeout_s=600.0,
)


def validate_settings(settings):
    for name in DEFAULT_RELEASE_RETURN:
        value = settings.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid release_return.{name}: require a finite positive number")
    for name in ("velocity_ratio", "acceleration_ratio"):
        if type(settings[name]) is not int or not 1 <= settings[name] <= 100:
            raise ValueError(f"release_return.{name} must be an integer from 1 to 100")
    if settings["lift_step_m"] > 0.01 or settings["max_joint_step_rad"] > 0.1:
        raise ValueError("Release lift steps must be at most 10 mm / 0.1 rad")
    if settings["lift_height_m"] / settings["lift_step_m"] > 1024:
        raise ValueError("Release lift exceeds the 1024-step planning budget")


def plan_release_return(kinematics, config, joints):
    """Validate the entire empty-hand route before releasing the rail."""
    settings = config["release_return"]
    validate_settings(settings)
    q = vector(joints, 7).copy()
    start = matrix(kinematics.fk("right", tuple(q)))
    planner = Planner(config["move_above"], kinematics, side="right")
    home = initial_receiving_pose(config, kinematics)
    count = int(np.ceil(settings["lift_height_m"] / settings["lift_step_m"]))
    points, flanges = [q.tolist()], [start.tolist()]
    began = time.monotonic()
    for index in range(1, count + 2):
        target = home.copy() if index == count + 1 else start.copy()
        if index <= count:
            target[1, 3] += settings["lift_height_m"] * index / count
        endpoint = planner.plan(q, target).points[-1]
        if time.monotonic() - began > config["move_above"]["planning_timeout_s"]:
            raise TimeoutError("Release/return IK planning timed out")
        if index <= count and np.max(np.abs(endpoint - q)) > settings["max_joint_step_rad"]:
            raise ValueError("Release lift IK changed branch or exceeded the joint step limit")
        points.append(endpoint.tolist())
        flanges.append(target.tolist())
        q = endpoint
    return dict(
        side="right",
        control_mode="position",
        lift_axis="right_base_+Y",
        lift_steps=count,
        settings=dict(settings),
        joint_waypoints_rad=points,
        flange_waypoints=flanges,
        T_R_F_initial=start.tolist(),
        T_R_F_home=home.tolist(),
        collision_checked=False,
    )


class ReleaseReturn:
    """One-shot lifecycle: submit the opening trajectory, wait, lift and return."""

    def __init__(self, config, kinematics, directory, state, insertion):
        if (
            insertion.get("status") != "insertion_reached"
            or insertion.get("insertion_executed") is not True
        ):
            raise ValueError("Release/return requires completed insertion")
        self.config, self.settings = config, dict(config["release_return"])
        validate_settings(self.settings)
        self.arrival = config["move_above"]
        self.directory = Path(directory)
        # Freeze insertion evidence: the old grasp is invalid once the hand opens.
        self.insertion = deepcopy(insertion)
        self.open_target = hand_targets(config)["open"]
        self.start = transform(state["T_B_F"]).copy()
        self.initial_joints = vector(state["joints_rad"], 7).copy()
        self.status, self.previous_status = "planning_return", None
        self.started = time.monotonic()
        self.done = threading.Event()
        self.report = self.error = None
        self.release_attempted = self.release_command_sent = self.release_sequence_completed = False
        self.hand_feedback = None
        self.index, self.settled_since = 0, None
        self.evidence = []
        self.save()

        def plan():
            try:
                self.report = plan_release_return(
                    kinematics, dict(config, release_return=self.settings), self.initial_joints
                )
            except Exception as error:
                self.error = error
            finally:
                self.done.set()

        threading.Thread(target=plan, daemon=True, name="right-release-return-planner").start()

    def save(self, reason=None):
        action = (
            "keep_open"
            if self.release_sequence_completed
            else "opening" if self.release_attempted else "keep_grasp"
        )
        result = dict(
            status=self.status,
            previous_status=self.previous_status,
            reason=reason,
            side="right",
            control_mode="position",
            hand_action=action,
            automatic_release=True,
            release_attempted=self.release_attempted,
            release_command_sent=self.release_command_sent,
            release_sequence_completed=self.release_sequence_completed,
            release_completion_basis="open_command_then_wait",
            hand_open_confirmed=False,
            hand_feedback=deepcopy(self.hand_feedback),
            lift_height_m=self.settings["lift_height_m"],
            returned_to_initial="returned_initial" in (self.status, self.previous_status),
            insertion_executed=self.insertion["insertion_executed"],
            insertion=self.insertion,
            execution_summary=execution_summary(self.evidence),
            elapsed_s=max(0.0, time.monotonic() - self.started),
            assembly_verified=False,
            force_sensor_used=False,
            collision_checked=False,
        )
        if self.status == "paused":
            result["hand_action"] = "retain_last_target"
        write_json(self.directory / "result.json", result)
        return result

    def paused_summary(self, reason):
        if self.status != "paused":
            self.previous_status = self.status
        self.status = "paused"
        return self.save(str(reason))

    def _check_hold(self, hardware, state):
        distance, angle = difference(state["T_B_F"], self.start)
        if (
            not hardware.position_move_finished()
            or distance > self.arrival["position_tolerance_m"]
            or angle > self.arrival["rotation_tolerance_rad"]
        ):
            raise RuntimeError("Right arm moved while planning return or opening the hand")
        if (
            np.max(np.abs(vector(state["joints_rad"], 7) - self.initial_joints))
            > self.config["arm_tracking_error_rad"]
        ):
            raise RuntimeError("Right-arm joint branch changed before release lift")

    def _record_hand_feedback(self, hardware, state, now):
        measured = vector(state["hand_position_rad"], 20)
        current = state.get("hand_current_a")
        self.hand_feedback = dict(
            position_rad=measured.tolist(),
            target_rad=self.open_target.tolist(),
            target_error_rad=(measured - self.open_target).tolist(),
            current_a=None if current is None else vector(current, 20).tolist(),
            elapsed_s=now - self.started,
        )

    def _dispatch(self, hardware, state, now):
        points = self.report["joint_waypoints_rad"]
        joints = vector(state["joints_rad"], 7)
        if (
            np.max(np.abs(joints - np.asarray(points[self.index])))
            > self.config["arm_tracking_error_rad"]
        ):
            raise RuntimeError("Right-arm joints changed from the planned return path")
        error = difference(state["T_B_F"], self.report["flange_waypoints"][self.index])
        if (
            error[0] > self.arrival["position_tolerance_m"]
            or error[1] > self.arrival["rotation_tolerance_rad"]
        ):
            raise RuntimeError("Right flange left the planned return segment start")
        self.status = (
            "lifting_after_release"
            if self.index < self.report["lift_steps"]
            else "returning_initial"
        )
        self.evidence.append(
            dict(
                stage=self.status,
                step=self.index + 1,
                command_queued=False,
                endpoint_verified=False,
            )
        )
        hardware.move_to_position(
            joints,
            np.asarray(points[self.index + 1]),
            timeout_s=self.arrival["position_move_timeout_s"],
            velocity_ratio=self.settings["velocity_ratio"],
            acceleration_ratio=self.settings["acceleration_ratio"],
        )
        self.evidence[-1]["command_queued"] = True
        self.step_started, self.settled_since = now, None
        self.save()

    def advance(self, hardware, state):
        if self.status in ("returned_initial", "paused"):
            return self.status
        hardware.raise_if_fault()
        if hardware.paused:
            raise RuntimeError("Release/return is paused")
        now = time.monotonic()
        if now - self.started > self.settings["total_timeout_s"]:
            raise TimeoutError("Release/return total timeout exceeded")
        if self.release_command_sent:
            self._record_hand_feedback(hardware, state, now)
        if self.status == "planning_return":
            self._check_hold(hardware, state)
            if now - self.started > self.arrival["planning_timeout_s"]:
                raise TimeoutError("Release/return IK planning timed out")
            if self.done.is_set():
                if self.error is not None:
                    raise self.error
                self.release_attempted = True
                hardware.begin_hand(self.open_target)
                self.release_command_sent = True
                self.status = "releasing_hand"
                self.save()
        elif self.status == "releasing_hand":
            self._check_hold(hardware, state)
            if hardware.hand_command_finished(now):
                self.release_sequence_completed = True
                self._dispatch(hardware, state, now)
        else:
            actual = transform(state["T_B_F"])
            target = np.asarray(self.report["flange_waypoints"][self.index + 1])
            if self.status == "lifting_after_release":
                start = np.asarray(self.report["flange_waypoints"][self.index])
                lateral = np.linalg.norm((actual[:3, 3] - start[:3, 3])[[0, 2]])
                margin = self.arrival["position_tolerance_m"]
                if (
                    lateral > self.settings["path_tolerance_m"]
                    or difference(actual, start)[1] > self.arrival["rotation_tolerance_rad"]
                    or not start[1, 3] - margin <= actual[1, 3] <= target[1, 3] + margin
                ):
                    raise RuntimeError("Release lift deviated from its vertical segment")
            if now - self.step_started > self.arrival["position_move_timeout_s"]:
                raise TimeoutError("Release/return segment did not reach its endpoint")
            distance, angle = difference(actual, target)
            sdk_completed = bool(hardware.position_move_finished())
            self.evidence[-1].update(
                sdk_completed=sdk_completed, position_error_m=distance, rotation_error_rad=angle
            )
            if (
                sdk_completed
                and distance <= self.arrival["position_tolerance_m"]
                and angle <= self.arrival["rotation_tolerance_rad"]
            ):
                self.settled_since = now if self.settled_since is None else self.settled_since
                if now - self.settled_since >= self.arrival["settle_s"]:
                    self.evidence[-1]["endpoint_verified"] = True
                    self.index += 1
                    if self.index == len(self.report["joint_waypoints_rad"]) - 1:
                        self.status = "returned_initial"
                        self.save()
                    else:
                        self._dispatch(hardware, state, now)
            else:
                self.settled_since = None
        return self.status
