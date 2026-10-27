"""Estimate the held rail-to-flange transform and move the right arm above its goal."""

from __future__ import annotations

from pathlib import Path
import threading
import time

import numpy as np

from rail_common.core import (
    PoseStabilityError,
    difference,
    digest,
    interpolate_joints,
    load_target,
    matrix,
    mean_pose,
    transform,
    vector,
    write_json,
)
from rail_common.planner import Planner
from rail_common.pose_alignment import RailPoseAlignment


def above_targets(T_R_C, T_C_goal, height):
    if isinstance(height, bool) or not np.isfinite(height) or height <= 0:
        raise ValueError("Above-goal height must be finite and positive")
    goal = transform(T_R_C) @ transform(T_C_goal)
    above = goal.copy()
    above[1, 3] += height  # Right SDK base: +Y is up, independent of CAD orientation.
    return goal, above


def plan_above(kinematics, config, initial_joints, T_F_O, T_R_above):
    """Solve only the final flange target; the SDK owns position interpolation."""
    q = vector(initial_joints, 7)
    T_F_O = transform(T_F_O)
    initial = matrix(kinematics.fk("right", tuple(q))) @ T_F_O
    alignment = RailPoseAlignment.from_config(config)
    if alignment is not None:
        alignment.check_motion(initial, T_R_above)
    from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import load_config

    profile = load_config(config["arm_config"])["profile"]["parameters"]["arms"]["right"]
    target = transform(T_R_above) @ np.linalg.inv(T_F_O)
    # Reuse bounded endpoint IK. Its two-point JointPath is only a container;
    # neither .at() nor its nominal duration is used to execute this movement.
    path = Planner(config["move_above"], kinematics, side="right").plan(q, target)
    return path, {
        "schema_version": 2,
        "side": "right",
        "end_stage": "above_target",
        "parent_frame": "tianji_right_base",
        "unit": "m",
        "angle_unit": "rad",
        "T_R_O_initial": initial.tolist(),
        "T_F_O": T_F_O.tolist(),
        "T_R_O_above": transform(T_R_above).tolist(),
        "T_R_F_above": target.tolist(),
        "joint_waypoints_rad": path.points.tolist(),
        "velocity_ratio": min(config["velocity_ratio"], profile["velocity_ratio"]),
        "acceleration_ratio": min(config["acceleration_ratio"], profile["acceleration_ratio"]),
        "control_mode": "position",
        "interpolation_owner": "controller",
        "path_type": "sdk_position_point_to_point",
        "intermediate_targets": 0,
        "collision_checked": False,
        "insertion_executed": False,
    }


class AboveTransfer:
    def __init__(self, config, geometry, kinematics, directory, *, draw_bboxes=False):
        self.config, self.geometry, self.kinematics = config, geometry, kinematics
        self.directory = Path(directory)
        self.settings = config["move_above"]
        self.alignment = RailPoseAlignment.from_config(config)
        for key, value in self.settings.items():
            if key not in ("target",) and (
                isinstance(value, bool) or not np.isfinite(value) or value <= 0
            ):
                raise ValueError(f"Invalid move_above setting: {key}")
        if type(self.settings["samples"]) is not int or self.settings["samples"] < 2:
            raise ValueError("move_above.samples must be an integer >= 2")
        target_path = Path(self.settings["target"])
        self.target_path = (Path(config["config_path"]).parent / target_path).resolve()
        target, camera_goal = load_target(self.target_path, geometry.calibration, config["mesh"])
        taught_calibration = target.get("calibration_sha256_at_teach")
        if taught_calibration and taught_calibration != digest(config["calibration"]):
            raise ValueError(
                "Target was taught with a different camera calibration; teach it again"
            )
        self.goal, self.above = above_targets(
            geometry.T_B_C, camera_goal, self.settings["height_m"]
        )
        self.input_digests = {
            "target": digest(self.target_path),
            **{k: digest(config[k]) for k in ("mesh", "calibration", "urdf", "manifest")},
        }
        self.preview_poses = {
            "T_C_O_goal": camera_goal.tolist(),
            "T_C_O_above": (np.linalg.inv(geometry.T_B_C) @ self.above).tolist(),
        }
        self.status, self.needs_reinitialize = "idle", False
        self.records, self.grasps = [], []
        self.path = self.report = self.planning_error = None
        self.planning_done = threading.Event()
        self.settled_since = None
        self.pose_sampling = None
        self.manual_move_message = None
        self.cached_grasp = None
        self.spiral = None
        self.release_return = None
        self.manual_lower_message = None
        self.preplace = None
        self.draw_bboxes = draw_bboxes
        self.accepted_prompts = set()
        self.waiting_prompt_stage = None
        self.prompt_frame_not_before_ns = None
        self.prompt_capture_not_before_ns = None
        self.prompt_hold_flange = None

    def begin(self):
        self.not_before_ns = time.monotonic_ns()
        self.status, self.needs_reinitialize = "estimating_held_pose", True
        self.records.clear()
        self.grasps.clear()
        self.path = self.report = self.planning_error = None
        self.planning_done.clear()
        self.settled_since = None
        self.pose_sampling = None
        self.manual_move_message = None
        self.cached_grasp = None
        self.spiral = None
        self.release_return = None
        self.manual_lower_message = None
        self.preplace = None
        self.accepted_prompts.clear()
        self.waiting_prompt_stage = None
        self.prompt_hold_flange = None
        if self.prompt_required("initial"):
            self._wait_for_prompt("initial")

    def prompt_required(self, stage):
        if self.draw_bboxes and stage not in self.accepted_prompts:
            return True
        try:
            self._load_prompt(stage)
        except (ValueError, OSError, KeyError):
            return True
        return False

    def _load_prompt(self, stage):
        from .sam_prompt import load_prompt, prompt_path

        return load_prompt(
            prompt_path(self.config, stage),
            self.config["receive_image_size"],
            self.geometry.calibration["camera_serial"],
        )

    def _wait_for_prompt(self, stage, state=None):
        from .sam_prompt import PROMPT_NAMES, prompt_path

        self.waiting_prompt_stage = stage
        self.status = f"waiting_{stage}_bbox"
        self.needs_reinitialize = False
        self.prompt_frame_not_before_ns = time.time_ns()
        self.prompt_capture_not_before_ns = time.monotonic_ns()
        self.prompt_hold_flange = None if state is None else transform(state["T_B_F"]).copy()
        self.records.clear()
        self.grasps.clear()
        write_json(
            self.directory / "result.json",
            dict(
                status=self.status,
                side="right",
                prompt_stage=stage,
                prompt_path=str(prompt_path(self.config, stage)),
                hand_action="keep_grasp",
                insertion_executed=False,
                assembly_verified=False,
            ),
        )
        print(
            f"等待绘制{PROMPT_NAMES[stage]}：窗口按 E，鼠标拖框，Enter 保存并继续；"
            f"也可直接 Enter 复用已保存的本阶段框。保存到 {prompt_path(self.config, stage)}。"
            "等待期间保持位置，不计入估姿超时，M/N 不执行。",
            flush=True,
        )

    def _check_prompt_hold(self, hardware, state):
        hardware.raise_if_fault()
        if hardware.paused:
            raise RuntimeError("Cannot continue bbox capture while the arm is paused")
        actual = transform(state["T_B_F"])
        if self.prompt_hold_flange is None:
            self.prompt_hold_flange = actual.copy()
        distance, angle = difference(actual, self.prompt_hold_flange)
        if (
            distance > self.settings["position_tolerance_m"]
            or angle > self.settings["rotation_tolerance_rad"]
            or (self.waiting_prompt_stage == "preplace" and not hardware.position_move_finished())
        ):
            raise RuntimeError("Right arm moved while waiting for bbox capture")

    def accept_prompt(self, stage, hardware, state, *, reused=False):
        """Consume only the current slot; saving cannot also trigger M or N."""
        if self.waiting_prompt_stage != stage:
            raise ValueError("Saved bbox does not match the current waiting stage")
        self._check_prompt_hold(hardware, state)
        self._load_prompt(stage)

        self.accepted_prompts.add(stage)
        self.waiting_prompt_stage = None
        self.prompt_frame_not_before_ns = None
        self.prompt_capture_not_before_ns = None
        self.records.clear()
        self.grasps.clear()
        if stage == "initial":
            self.not_before_ns = time.monotonic_ns()
            self.status, self.needs_reinitialize = "estimating_held_pose", True
        else:
            self._start_preplace(state)
        return self.status

    def _start_preplace(self, state):
        from .preplace_transfer import PreplaceRefinement

        self.preplace = PreplaceRefinement(self, state)
        self.status = self.preplace.status
        self.preplace.save()

    @property
    def reestimate_predicted_pose(self):
        return None if self.preplace is None else self.preplace.predicted_pose.copy()

    def perception_initialized(self, generation):
        if self.preplace is not None:
            self.preplace.initialized(generation)
        self.needs_reinitialize = False

    def _finish_position_arrival(self, state, error):
        """Finish coarse transfer and optionally start fresh visual correction."""
        self.status = "above_target"
        result = {
            "status": self.status,
            "side": "right",
            "control_mode": "position",
            "arrival_check": "sdk_and_fk",
            "visual_verification_required": False,
            "visual_verified": False,
            "position_error_m": error[0],
            "rotation_error_rad": error[1],
            "T_R_F_from_fk": transform(state["T_B_F"]).tolist(),
            "T_R_F_above": transform(self.report["T_R_F_above"]).tolist(),
            "T_R_O_above": self.above.tolist(),
            "pose_sampling": self.pose_sampling,
            "height_m": self.settings["height_m"],
            "hand_action": "keep_grasp",
            "insertion_executed": False,
            "assembly_verified": False,
            "collision_checked": False,
        }
        write_json(self.directory / "result.json", result)
        if self.prompt_required("preplace"):
            self._wait_for_prompt("preplace", state)
        else:
            self._start_preplace(state)

    def _observation(self, record, hardware):
        if self.needs_reinitialize or not record or record.get("valid") is not True:
            return None
        if (
            record.get("replay_only")
            or record.get("test_data")
            or record.get("camera_serial") != self.geometry.calibration["camera_serial"]
            or record.get("mesh_sha256") != self.geometry.mesh_digest
            or record.get("timestamp_source") != "sdk_global_timestamp_mapped_to_host_monotonic"
        ):
            raise ValueError("Held-rail observation is not a matching live RGB-D measurement")
        stamp = int(record["timestamp_ns"])
        if (
            stamp <= self.not_before_ns
            or not 0 <= time.monotonic_ns() - stamp <= self.config["observation_max_age_s"] * 1e9
        ):
            return None
        if self.records and stamp <= self.records[-1]["timestamp_ns"]:
            return None
        with hardware.lock:
            history = list(hardware.history)
        try:
            q = interpolate_joints(history, stamp)
        except ValueError:
            return None
        if self.alignment is not None:
            record = self.alignment.align_record(record, self.preview_poses["T_C_O_goal"])
        T_R_O = self.geometry.T_B_C @ transform(record["T_C_CAD"])
        grasp = np.linalg.inv(matrix(self.kinematics.fk("right", tuple(q)))) @ T_R_O
        return T_R_O, grasp, record

    def _plan(self, joints, grasp):
        try:
            self.path, self.report = plan_above(
                self.kinematics, self.config, joints, grasp, self.above
            )
            self.report.update(
                T_R_O_goal=self.goal.tolist(),
                height_m=self.settings["height_m"],
                visual_verification_required=False,
                input_sha256=self.input_digests,
                pose_sampling=self.pose_sampling,
                pose_alignment_policy=self.config.get("pose_alignment"),
                pose_alignment=[r.get("pose_alignment") for r in self.records],
            )
        except Exception as error:
            self.planning_error = error
        finally:
            self.planning_done.set()

    def _prepare_mean(self):
        """Freeze the accepted observations and mean once, before M is pressed."""
        position_limit = self.settings.get("pose_consistency_position_tolerance_m", 0.005)
        stable = True
        try:
            grasp = mean_pose(self.grasps, max_translation_m=position_limit)
        except PoseStabilityError:
            stable = False
            # Still validate every rigid transform and use the same SO(3) mean.
            grasp = mean_pose(self.grasps, max_translation_m=np.inf, max_rotation_rad=np.inf)
        self.cached_grasp = grasp.copy()
        errors = [difference(pose, grasp) for pose in self.grasps]
        self.pose_sampling = {
            "stable": stable,
            "operator_requested": False,
            "stability_overridden": False,
            "request_key": None,
            "max_position_deviation_m": max(error[0] for error in errors),
            "max_rotation_deviation_rad": max(error[1] for error in errors),
            "position_tolerance_m": position_limit,
            "rotation_tolerance_rad": 0.035,
            "samples": len(self.records),
            "mean_saved_monotonic_ns": time.monotonic_ns(),
        }

    def _start_planning(self, state, *, force_move=False):
        if self.cached_grasp is None:
            self.manual_move_message = "M 未执行：还没有保存的位姿均值，请等待估姿完成后再按 M。"
            return
        stamp = time.monotonic_ns()
        self.pose_sampling.update(
            operator_requested=force_move,
            request_key="M" if force_move else None,
            stability_overridden=bool(force_move and not self.pose_sampling["stable"]),
            decision_monotonic_ns=stamp,
            saved_mean_age_s=(stamp - self.pose_sampling["mean_saved_monotonic_ns"]) / 1e9,
            pose_source="saved_mean",
        )
        self.status, self.planning_started = "planning_above", time.monotonic()
        if force_move:
            self.manual_move_message = (
                "M 已收到：直接使用已保存的位姿均值规划移动，不重新采样或检查位姿稳定性。"
            )
        threading.Thread(
            target=self._plan,
            args=(state["joints_rad"].copy(), self.cached_grasp.copy()),
            daemon=True,
            name="right-above-planner",
        ).start()

    def advance(self, hardware, state, record, *, force_move=False, lower_requested=False):
        now = time.monotonic()
        s = self.settings
        self.manual_move_message = None
        self.manual_lower_message = None
        if self.release_return is not None:
            if lower_requested:
                self.manual_lower_message = "N 未执行：插入已结束，正在松手回位或已回到初始位置。"
            self.status = self.release_return.advance(hardware, state)
            return self.status
        if self.waiting_prompt_stage is not None:
            self._check_prompt_hold(hardware, state)
            if force_move:
                self.manual_move_message = "M 未执行：请先在窗口完成本阶段 bbox 绘制或复用确认。"
            if lower_requested:
                self.manual_lower_message = "N 未执行：请先完成 bbox 和位姿对准；完成后再次按 N。"
            return self.status
        if self.preplace is not None and not self.preplace.completed:
            if lower_requested:
                self.manual_lower_message = (
                    "N 未执行：正在重新估姿并纠偏；纠偏到位后需再次按 N 下放。"
                )
            self.status = self.preplace.advance(hardware, state, record)
            return self.status
        if lower_requested:
            if hardware.paused:
                self.manual_lower_message = "N 未执行：机械臂已暂停，本次不能恢复。"
                return self.status
            if self.status != "above_target":
                reason = {
                    "planning_position_spiral": "正在规划位置螺旋插入",
                    "position_spiral": "正在执行位置螺旋插入",
                    "insertion_reached": "本次插入已结束，准备松手回位，不能重复启动",
                }.get(self.status, "尚未到达上方位置")
                self.manual_lower_message = f"N 未执行：{reason}。"
            elif not hardware.position_move_finished():
                self.manual_lower_message = "N 未执行：请等待当前点动到位。"
            else:
                error = (
                    difference(transform(state["T_B_F"]) @ self.cached_grasp, self.above)
                    if self.preplace is not None
                    else difference(state["T_B_F"], self.report["T_R_F_above"])
                )
                limits = self.preplace.settings if self.preplace is not None else s
                if (
                    error[0] > limits["position_tolerance_m"]
                    or error[1] > limits["rotation_tolerance_rad"]
                ):
                    self.manual_lower_message = (
                        "N 未执行：当前末端已偏离上方位置，请先回到上方目标。"
                    )
                else:
                    from .position_spiral import PositionSpiral

                    self._check_inputs()
                    self.spiral = PositionSpiral(
                        self.config,
                        self.kinematics,
                        self.directory,
                        state["joints_rad"].copy(),
                        self.cached_grasp.copy(),
                        self.goal,
                        self.above,
                    )
                    if self.preplace is not None:
                        self.spiral.preplace_alignment = self.preplace.summary()
                    self.status = self.spiral.status
                    self.manual_lower_message = (
                        "N 已收到：规划位置控制螺旋下降；到位后自动松手、抬升并回初始位置。"
                    )
                    return self.status
            if self.spiral is None:
                return self.status
        if self.spiral is not None:
            if self.spiral.status == "planning_position_spiral" and self.spiral.done.is_set():
                self._check_inputs()
            self.status = self.spiral.advance(hardware, state)
            if self.status == "insertion_reached":
                from .release_return import ReleaseReturn

                self._check_inputs()
                self.release_return = ReleaseReturn(
                    self.config, self.kinematics, self.directory, state, self.spiral.result()
                )
                self.status = self.release_return.status
            return self.status
        if force_move and hardware.paused:
            self.manual_move_message = "M 未执行：机械臂已暂停。"
            return self.status
        if self.status in ("waiting_move_key",):
            if force_move:
                self._start_planning(state, force_move=True)
        elif self.status == "estimating_held_pose":
            observation = self._observation(record, hardware)
            if observation is not None:
                _, grasp, record = observation
                self.records.append(record)
                self.grasps.append(grasp)
            if len(self.records) >= s["samples"]:
                self._prepare_mean()
                self.status = "waiting_move_key"
                if force_move:
                    self._start_planning(state, force_move=True)
            elif force_move:
                self.manual_move_message = (
                    "M 未执行：还没有保存的位姿均值，请等待估姿完成后再按 M。"
                )
            elif (
                not force_move
                and time.monotonic_ns() - self.not_before_ns
                > self.config["perception_timeout_s"] * 1e9
            ):
                raise TimeoutError("No fresh held-rail pose; arm remains at its current position")
        elif self.status == "planning_above":
            if now - self.planning_started > s["planning_timeout_s"]:
                raise TimeoutError("Right-arm above-goal planning timed out")
            if self.planning_done.is_set():
                if self.planning_error is not None:
                    raise self.planning_error
                self._check_inputs()
                hardware.move_to_position(
                    self.path.points[0],
                    self.path.points[-1],
                    timeout_s=s["position_move_timeout_s"],
                )
                self.motion_started, self.status = now, "moving_above"
                self.controller_completed_at = None
        elif self.status == "moving_above":
            error = difference(state["T_B_F"], self.report["T_R_F_above"])
            reached = (
                error[0] <= s["position_tolerance_m"] and error[1] <= s["rotation_tolerance_rad"]
            )
            completed = hardware.position_move_finished()
            if completed and self.controller_completed_at is None:
                self.controller_completed_at = now
            if completed and reached:
                self.settled_since = self.settled_since or now
                if now - self.settled_since >= s["settle_s"]:
                    self._finish_position_arrival(state, error)
            else:
                self.settled_since = None
            if not completed and now - self.motion_started > s["position_move_timeout_s"]:
                raise TimeoutError("Right-arm SDK position move timed out")
            if completed and now - self.controller_completed_at > s["reach_timeout_s"]:
                raise TimeoutError("Right flange did not reach the planned above-goal pose")
        return self.status

    def _check_inputs(self):
        for key, expected in self.input_digests.items():
            if digest(self.target_path if key == "target" else self.config[key]) != expected:
                raise ValueError(f"{key} changed during right-arm planning")
