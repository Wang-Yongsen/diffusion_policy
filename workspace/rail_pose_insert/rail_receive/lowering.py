"""Operator-triggered, feedback-gated lowering in SDK position mode."""
from pathlib import Path
import threading
import time

import numpy as np

from rail_preinsert.core import difference, matrix, transform, vector, write_json
from rail_preinsert.planner import Planner


DEFAULT_LOWERING = dict(step_m=.005, velocity_ratio=1, acceleration_ratio=1,
    position_tolerance_m=.001, rotation_tolerance_rad=.01745329252,
    lateral_tolerance_m=.003, overshoot_tolerance_m=.001, settle_s=.2,
    step_timeout_s=20., max_joint_step_rad=.05)


def validate_settings(settings):
    for name in DEFAULT_LOWERING:
        value = settings[name]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid lowering setting: {name}")
    for name in ("velocity_ratio", "acceleration_ratio"):
        if type(settings[name]) is not int or not 1 <= settings[name] <= 100:
            raise ValueError(f"lowering.{name} must be an integer from 1 to 100")
    if settings["step_m"] > .005:
        raise ValueError("Lowering steps must be at most 5 mm")
    if settings["position_tolerance_m"] >= settings["step_m"]:
        raise ValueError("Lowering arrival tolerance must be smaller than a step")


def plan_lowering(kinematics, config, initial_joints, grasp, goal, above):
    """Plan a few vertical endpoints before sending any motion commands."""
    s = config["lowering"]
    validate_settings(s)
    grasp, goal, above = transform(grasp), transform(goal), transform(above)
    q = vector(initial_joints, 7)
    actual = matrix(kinematics.fk("right", tuple(q))) @ grasp
    distance, angle = difference(actual, above)
    if distance > config["move_above"]["position_tolerance_m"] or angle > s["rotation_tolerance_rad"]:
        raise ValueError("Right arm is no longer at the above-goal pose; lowering not started")
    height = above[1, 3]-goal[1, 3]
    if height <= 0 or not np.allclose(above[[0, 2], 3], goal[[0, 2], 3], atol=1e-9):
        raise ValueError("Lowering requires a vertical above-goal start")
    if difference(above, goal)[1] > 1e-8:
        raise ValueError("Lowering must preserve orientation")
    count = int(np.ceil(height/s["step_m"]-1e-9))
    planner = Planner(config["move_above"], kinematics, side="right")
    points, rails, flanges = [q.copy()], [], []
    for index in range(1, count+1):
        rail = goal.copy()
        rail[1, 3] += height*(1-index/count)
        flange = rail @ np.linalg.inv(grasp)
        endpoint = planner.plan(q, flange).points[-1]
        if np.max(np.abs(endpoint-q)) > s["max_joint_step_rad"]:
            raise ValueError("Lowering IK changed joint branch or exceeded the per-step joint limit")
        points.append(endpoint.copy())
        rails.append(rail.tolist())
        flanges.append(flange.tolist())
        q = endpoint
    return {"schema_version": 1, "side": "right", "control_mode": "position",
        "path_type": "feedback_gated_vertical_steps", "height_m": height,
        "T_F_O": grasp.tolist(), "T_R_O_goal": goal.tolist(), "T_R_O_above": above.tolist(),
        "joint_waypoints_rad": np.asarray(points).tolist(), "rail_targets": rails,
        "flange_targets": flanges, "steps": count, "settings": dict(s),
        "collision_checked": False, "force_controlled": False, "automatic_release": False}


class Lowering:
    def __init__(self, config, kinematics, directory, joints, grasp, goal, above):
        self.config, self.settings = config, config["lowering"]
        self.directory = Path(directory)
        self.grasp, self.goal, self.above = transform(grasp), transform(goal), transform(above)
        self.status = "planning_lowering"
        self.started = time.monotonic()
        self.done = threading.Event()
        self.report = self.error = None
        self.index, self.settled_since = 0, None
        def plan():
            try:
                self.report = plan_lowering(kinematics, config, joints, self.grasp, self.goal, self.above)
                write_json(self.directory/"lowering_plan.json", self.report)
            except Exception as error:
                self.error = error
            finally:
                self.done.set()
        threading.Thread(target=plan, daemon=True, name="right-lowering-planner").start()

    def _dispatch(self, hardware, state, now):
        planned_start = np.asarray(self.report["joint_waypoints_rad"][self.index])
        if np.max(np.abs(state["joints_rad"]-planned_start)) > self.config["arm_tracking_error_rad"]:
            raise RuntimeError("Right-arm joints changed from the planned lowering branch")
        target = np.asarray(self.report["joint_waypoints_rad"][self.index+1])
        hardware.move_to_position(state["joints_rad"], target,
            timeout_s=self.settings["step_timeout_s"],
            velocity_ratio=self.settings["velocity_ratio"],
            acceleration_ratio=self.settings["acceleration_ratio"])
        self.step_started, self.settled_since = now, None
        print(f"缓慢下放：第 {self.index+1}/{self.report['steps']} 段，"
              f"距装配目标高度 {(self.report['rail_targets'][self.index][1][3]-self.goal[1, 3])*1000:.1f} mm。",
              flush=True)

    def _check_corridor(self, state):
        actual = transform(state["T_B_F"]) @ self.grasp
        lateral = np.linalg.norm((actual-self.goal)[[0, 2], 3])
        angle = difference(actual, self.goal)[1]
        if lateral > self.settings["lateral_tolerance_m"] or angle > self.settings["rotation_tolerance_rad"]:
            raise RuntimeError(f"Lowering deviated from vertical corridor: {lateral*1000:.2f} mm / "
                               f"{np.degrees(angle):.2f} deg")
        floor = self.report["rail_targets"][self.index][1][3]
        if (actual[1, 3] < floor-self.settings["overshoot_tolerance_m"]
                or actual[1, 3] > self.above[1, 3]+self.config["move_above"]["position_tolerance_m"]):
            raise RuntimeError("Lowering exceeded the commanded vertical segment")
        return actual

    def advance(self, hardware, state):
        hardware.raise_if_fault()
        now = time.monotonic()
        if self.status == "planning_lowering":
            if now-self.started > self.config["move_above"]["planning_timeout_s"]:
                raise TimeoutError("Lowering IK planning timed out")
            if self.done.is_set():
                if self.error is not None:
                    raise self.error
                # Planning is asynchronous: check measured start again at dispatch.
                error = difference(transform(state["T_B_F"]) @ self.grasp, self.above)
                if (error[0] > self.config["move_above"]["position_tolerance_m"]
                        or error[1] > self.settings["rotation_tolerance_rad"]):
                    raise RuntimeError("Right arm moved during lowering planning")
                self._dispatch(hardware, state, now)
                self.status = "lowering"
        elif self.status == "lowering":
            actual = self._check_corridor(state)
            error = difference(state["T_B_F"], self.report["flange_targets"][self.index])
            reached = (hardware.position_move_finished()
                       and error[0] <= self.settings["position_tolerance_m"]
                       and error[1] <= self.settings["rotation_tolerance_rad"])
            if reached:
                self.settled_since = now if self.settled_since is None else self.settled_since
                if now-self.settled_since >= self.settings["settle_s"]:
                    self.index += 1
                    if self.index == self.report["steps"]:
                        self.status = "placement_reached"
                        distance, angle = difference(actual, self.goal)
                        write_json(self.directory/"result.json", {"status": self.status,
                            "side": "right", "control_mode": "position", "hand_action": "keep_grasp",
                            "lowering_executed": True, "automatic_release": False,
                            "contact_verified": False, "assembly_verified": False,
                            "collision_checked": False, "force_controlled": False,
                            "T_R_O_from_fk": actual.tolist(), "T_R_O_goal": self.goal.tolist(),
                            "position_error_m": distance, "rotation_error_rad": angle})
                    else:
                        self._dispatch(hardware, state, now)
            else:
                self.settled_since = None
            if self.status == "lowering" and now-self.step_started > self.settings["step_timeout_s"]:
                raise TimeoutError(f"Lowering step {self.index+1} did not reach its measured endpoint")
        return self.status
