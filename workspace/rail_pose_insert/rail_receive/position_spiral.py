"""Feedback-gated position spiral; no impedance controller or force-sensor input.

The rail keeps its orientation, spirals in right-base XZ, and descends along -Y.
The SDK interpolates joints between the fully preplanned small endpoints; the
measured rail pose is checked against each straight Cartesian segment.
"""

from numbers import Real
from pathlib import Path
import threading
import time

import numpy as np

from rail_common.runtime import execution_summary
from rail_common.core import difference, matrix, transform, vector, write_json
from rail_common.planner import Planner

DEFAULT_POSITION_SPIRAL = dict(
    radius_m=0.003,
    turns=3,
    max_step_m=0.0005,
    velocity_ratio=1,
    acceleration_ratio=1,
    position_tolerance_m=0.0002,
    rotation_tolerance_rad=0.01745329252,
    path_tolerance_m=0.003,
    overshoot_tolerance_m=0.001,
    settle_s=0.1,
    step_timeout_s=20.0,
    total_timeout_s=600.0,
    max_joint_step_rad=0.05,
    max_descent_m=0.06,
)
MAX_STEPS = 4096


def validate_settings(settings):
    for name in DEFAULT_POSITION_SPIRAL:
        value = settings.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"Invalid position_spiral setting: {name}; require a finite positive number"
            )
    for name in ("velocity_ratio", "acceleration_ratio"):
        if type(settings[name]) is not int or not 1 <= settings[name] <= 100:
            raise ValueError(f"position_spiral.{name} must be an integer from 1 to 100")
    limits = dict(
        radius_m=0.005,
        turns=100,
        max_step_m=0.001,
        rotation_tolerance_rad=np.radians(5),
        path_tolerance_m=0.005,
        overshoot_tolerance_m=0.002,
        max_joint_step_rad=0.1,
        max_descent_m=0.1,
        settle_s=5.0,
        step_timeout_s=120.0,
        total_timeout_s=3600.0,
    )
    for name, limit in limits.items():
        if settings[name] > limit:
            raise ValueError(f"position_spiral.{name} must be at most {limit:g}")
    if settings["position_tolerance_m"] >= settings["max_step_m"] / 2:
        raise ValueError("Position spiral arrival tolerance must be smaller than half max_step_m")
    if settings["path_tolerance_m"] < settings["position_tolerance_m"]:
        raise ValueError("Position spiral path tolerance must cover arrival tolerance")
    if not settings["settle_s"] < settings["step_timeout_s"] <= settings["total_timeout_s"]:
        raise ValueError("Position spiral requires settle_s < step_timeout_s <= total_timeout_s")
    # Even a zero-height spiral cannot exceed this allocation/IK budget.
    minimum_length_bound = np.hypot(
        np.pi * settings["radius_m"], 2 * np.pi * settings["turns"] * settings["radius_m"]
    )
    if minimum_length_bound / settings["max_step_m"] > MAX_STEPS:
        raise ValueError(f"Position spiral exceeds the {MAX_STEPS}-step planning budget")


def plan_position_spiral(kinematics, config, initial_joints, grasp, goal, above):
    """Preplan every IK endpoint before any motion is dispatched.

    r(u)=radius*sin(pi*u) expands then closes smoothly at the exact goal.
    Bounding |d position / du| bounds every Cartesian chord by max_step_m.
    """
    settings = config["position_spiral"]
    validate_settings(settings)
    grasp, goal, above = map(transform, (grasp, goal, above))
    q = vector(initial_joints, 7)
    actual = matrix(kinematics.fk("right", tuple(q))) @ grasp
    distance, angle = difference(actual, above)
    if (
        distance > config["move_above"]["position_tolerance_m"]
        or angle > settings["rotation_tolerance_rad"]
    ):
        raise ValueError(
            "Right arm is no longer at the above-goal pose; position spiral not started"
        )
    height = float(actual[1, 3] - goal[1, 3])
    if not 0 < height <= settings["max_descent_m"] or above[1, 3] <= goal[1, 3]:
        raise ValueError("Position spiral descent must be positive and within max_descent_m")
    if not np.allclose(above[[0, 2], 3], goal[[0, 2], 3], atol=1e-9, rtol=0):
        raise ValueError("Position spiral requires a vertical above-goal start")
    if difference(above, goal)[1] > 1e-8:
        raise ValueError("Position spiral must preserve orientation")
    centerline = goal[:3, 3] - actual[:3, 3]
    # The triangle inequality also covers the small horizontal correction from
    # the measured above pose to the goal; the first move is bounded identically.
    spiral_speed_bound = float(
        np.hypot(np.pi * settings["radius_m"], 2 * np.pi * settings["turns"] * settings["radius_m"])
    )
    speed_bound = float(np.hypot(height, np.linalg.norm(centerline[[0, 2]]) + spiral_speed_bound))
    requested_count = max(speed_bound / settings["max_step_m"], 16 * settings["turns"])
    if not np.isfinite(requested_count) or requested_count > MAX_STEPS:
        raise ValueError(f"Position spiral exceeds the {MAX_STEPS}-step planning budget")
    count = int(np.ceil(requested_count))
    if count * settings["settle_s"] >= settings["total_timeout_s"]:
        raise ValueError("Position spiral settling alone would exceed total_timeout_s")
    planner = Planner(config["move_above"], kinematics, side="right")
    inverse_grasp = np.linalg.inv(grasp)
    points, rails, flanges = [q.copy()], [], []
    previous = actual
    started = time.monotonic()
    deadline = min(config["move_above"]["planning_timeout_s"], settings["total_timeout_s"])
    for index in range(1, count + 1):
        if time.monotonic() - started > deadline:
            raise TimeoutError("Position spiral IK planning timed out")
        u = index / count
        rail = goal.copy()
        rail[:3, 3] = (1 - u) * actual[:3, 3] + u * goal[:3, 3]
        radius = settings["radius_m"] * np.sin(np.pi * u)
        theta = 2 * np.pi * settings["turns"] * u
        rail[[0, 2], 3] += radius * np.array([np.cos(theta), np.sin(theta)])
        if index == count:
            rail = goal.copy()  # Exact terminal pose, including zero lateral offset.
        if np.linalg.norm(rail[:3, 3] - previous[:3, 3]) > settings["max_step_m"] + 1e-12:
            raise ValueError("Position spiral Cartesian segment exceeds max_step_m")
        flange = rail @ inverse_grasp
        endpoint = planner.plan(q, flange).points[-1]
        if np.max(np.abs(endpoint - q)) > settings["max_joint_step_rad"]:
            raise ValueError(
                "Position spiral IK changed joint branch or exceeded the per-step joint limit"
            )
        points.append(endpoint.copy())
        rails.append(rail.tolist())
        flanges.append(flange.tolist())
        q, previous = endpoint, rail
    return dict(
        schema_version=1,
        side="right",
        control_mode="position",
        mode="position_spiral",
        path_type="feedback_gated_position_spiral",
        spiral_plane="right_base_XZ",
        descent_axis="-Y",
        radius_profile="radius_m*sin(pi*u)",
        height_m=height,
        steps=count,
        settings=dict(settings),
        centerline="measured_start_to_goal",
        T_F_O=grasp.tolist(),
        T_R_O_goal=goal.tolist(),
        T_R_O_above=above.tolist(),
        T_R_O_initial_from_fk=actual.tolist(),
        joint_waypoints_rad=np.asarray(points).tolist(),
        rail_targets=rails,
        flange_targets=flanges,
        interpolation_owner="controller",
        collision_checked=False,
        force_controlled=False,
        force_sensor_used=False,
        visual_verification_required=False,
        visual_verified=False,
        contact_verified=False,
        assembly_verified=False,
        automatic_release=True,
        hand_action="keep_grasp",
    )


class PositionSpiral:
    """Asynchronous planner and position-only executor through insertion arrival."""

    def __init__(self, config, kinematics, directory, joints, grasp, goal, above):
        self.config, self.settings = config, dict(config["position_spiral"])
        self.directory = Path(directory)
        self.grasp, self.goal, self.above = [
            transform(value).copy() for value in (grasp, goal, above)
        ]
        self.status = "planning_position_spiral"
        self.started = time.monotonic()
        self.done = threading.Event()
        self.report = self.error = None
        self.index, self.settled_since = 0, None
        self.evidence = []
        self.last_actual = self.last_joints = None
        self.reason = None
        self.previous_status = None
        initial_joints = vector(joints, 7).copy()

        def plan():
            try:
                self.report = plan_position_spiral(
                    kinematics,
                    dict(config, position_spiral=self.settings),
                    initial_joints,
                    self.grasp,
                    self.goal,
                    self.above,
                )
            except Exception as error:
                self.error = error
            finally:
                self.done.set()

        threading.Thread(target=plan, daemon=True, name="right-position-spiral-planner").start()

    def _observe(self, state):
        actual = transform(state["T_B_F"]) @ self.grasp
        joints = vector(state["joints_rad"], 7)
        self.last_actual, self.last_joints = actual.copy(), joints.copy()
        return actual

    def _dispatch(self, hardware, state, now):
        joints = vector(state["joints_rad"], 7)
        planned_start = np.asarray(self.report["joint_waypoints_rad"][self.index])
        if np.max(np.abs(joints - planned_start)) > self.config["arm_tracking_error_rad"]:
            raise RuntimeError("Right-arm joints changed from the planned position spiral branch")
        target = np.asarray(self.report["joint_waypoints_rad"][self.index + 1])
        actual = transform(state["T_B_F"]) @ self.grasp
        rail_target = np.asarray(self.report["rail_targets"][self.index])
        # Limit measured drift to the nominal step plus its arrival tolerance.
        if (
            difference(actual, rail_target)[0]
            > self.settings["max_step_m"] + self.settings["position_tolerance_m"]
        ):
            raise RuntimeError("Measured start is too far from the next position spiral target")
        self.evidence.append(
            dict(
                step=self.index + 1,
                queue_attempt_elapsed_s=now - self.started,
                measured_start_joints_rad=joints.tolist(),
                measured_start_rail_pose=actual.tolist(),
                dispatch_attempted=True,
                command_queued=False,
                sdk_completed=False,
                endpoint_verified=False,
            )
        )
        hardware.move_to_position(
            joints,
            target,
            timeout_s=self.settings["step_timeout_s"],
            velocity_ratio=self.settings["velocity_ratio"],
            acceleration_ratio=self.settings["acceleration_ratio"],
        )
        self.evidence[-1]["command_queued"] = True
        self.step_started, self.settled_since = now, None

    def _check_corridor(self, actual):
        start = np.asarray(
            self.report["rail_targets"][self.index - 1]
            if self.index
            else self.report["T_R_O_initial_from_fk"]
        )
        target = np.asarray(self.report["rail_targets"][self.index])
        segment = target[:3, 3] - start[:3, 3]
        fraction = np.clip(
            np.dot(actual[:3, 3] - start[:3, 3], segment) / np.dot(segment, segment), 0.0, 1.0
        )
        path_error = float(np.linalg.norm(actual[:3, 3] - (start[:3, 3] + fraction * segment)))
        angle = difference(actual, target)[1]
        if (
            path_error > self.settings["path_tolerance_m"]
            or angle > self.settings["rotation_tolerance_rad"]
        ):
            raise RuntimeError(
                f"Position spiral deviated from segment corridor: {path_error*1000:.2f} mm / "
                f"{np.degrees(angle):.2f} deg"
            )
        tolerance = self.settings["overshoot_tolerance_m"]
        if actual[1, 3] < target[1, 3] - tolerance or actual[1, 3] > start[1, 3] + tolerance:
            raise RuntimeError("Position spiral exceeded the commanded vertical segment")

    def result(self, state=None, reason=None):
        """Return execution evidence; a missing/invalid snapshot keeps last valid FK."""
        if state is not None:
            try:
                self._observe(state)
            except (ValueError, KeyError, TypeError):
                pass
        result = dict(
            schema_version=1,
            status=self.status,
            side="right",
            control_mode="position",
            mode="position_spiral",
            hand_action="keep_grasp",
            automatic_release=True,
            force_controlled=False,
            force_sensor_used=False,
            contact_verified=False,
            visual_verification_required=False,
            visual_verified=False,
            preplace_alignment=getattr(self, "preplace_alignment", None),
            assembly_verified=False,
            collision_checked=False,
            insertion_executed=True if self.index else (None if self.evidence else False),
            execution_attempted=bool(self.evidence),
            previous_status=self.previous_status,
            completed_steps=self.index,
            queued_steps=sum(record["command_queued"] for record in self.evidence),
            planned_steps=None if self.report is None else self.report["steps"],
            elapsed_s=max(0.0, time.monotonic() - self.started),
            execution_summary=execution_summary(self.evidence),
            T_R_O_goal=self.goal.tolist(),
            reason=self.reason if reason is None else str(reason),
        )
        if self.last_actual is not None:
            distance, angle = difference(self.last_actual, self.goal)
            result.update(
                T_R_O_from_fk=self.last_actual.tolist(),
                measured_joints_rad=self.last_joints.tolist(),
                position_error_m=distance,
                rotation_error_rad=angle,
            )
        return result

    def _save_result(self, result):
        write_json(self.directory / "result.json", result)
        return result

    def paused_summary(self, state=None, reason=None):
        """Save a stopped run; the caller owns hardware.pause and any manual recovery."""
        if self.status != "paused":
            self.previous_status = self.status
        self.status = "paused"
        self.reason = str(reason) if reason is not None else self.reason
        result = self.result(state)
        result["hand_action"] = "retain_last_target"
        return self._save_result(result)

    def advance(self, hardware, state):
        if self.status in ("insertion_reached", "paused"):
            return self.status
        hardware.raise_if_fault()
        now = time.monotonic()
        if now - self.started > self.settings["total_timeout_s"]:
            raise TimeoutError("Position spiral total timeout exceeded")
        actual = self._observe(state)
        if self.status == "planning_position_spiral":
            if now - self.started > self.config["move_above"]["planning_timeout_s"]:
                raise TimeoutError("Position spiral IK planning timed out")
            if self.done.is_set():
                if self.error is not None:
                    raise self.error
                distance, angle = difference(actual, self.report["T_R_O_initial_from_fk"])
                if (
                    distance > self.settings["position_tolerance_m"]
                    or angle > self.settings["rotation_tolerance_rad"]
                ):
                    raise RuntimeError("Right arm moved during position spiral planning")
                self._dispatch(hardware, state, now)
                self.status = "position_spiral"
        elif self.status == "position_spiral":
            self._check_corridor(actual)
            if now - self.step_started > self.settings["step_timeout_s"]:
                raise TimeoutError(
                    f"Position spiral step {self.index+1} did not reach its measured endpoint"
                )
            distance, angle = difference(actual, self.report["rail_targets"][self.index])
            sdk_completed = bool(hardware.position_move_finished())
            reached = (
                sdk_completed
                and distance <= self.settings["position_tolerance_m"]
                and angle <= self.settings["rotation_tolerance_rad"]
            )
            self.evidence[-1].update(
                sdk_completed=sdk_completed,
                position_error_m=distance,
                rotation_error_rad=angle,
                last_feedback_elapsed_s=now - self.started,
            )
            if reached:
                self.settled_since = now if self.settled_since is None else self.settled_since
                if now - self.settled_since >= self.settings["settle_s"]:
                    self.evidence[-1].update(
                        endpoint_verified=True,
                        reached_elapsed_s=now - self.started,
                        measured_rail_pose=actual.tolist(),
                    )
                    self.index += 1
                    if self.index == self.report["steps"]:
                        self.status = "insertion_reached"
                        self._save_result(self.result())
                    else:
                        self._dispatch(hardware, state, now)
            else:
                self.settled_since = None
        return self.status
