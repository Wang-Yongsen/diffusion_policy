"""Estimate the held rail-to-flange transform and move the right arm above its goal."""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json
import threading
import time

import numpy as np

from rail_preinsert.core import (PoseStabilityError, difference, digest, interpolate_joints, load_target, matrix,
                                mean_pose, transform, vector, write_json)
from rail_preinsert.planner import Planner
from rail_preinsert.pose_alignment import RailPoseAlignment


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
    return path, {"schema_version": 2, "side": "right", "end_stage": "above_target",
        "parent_frame": "tianji_right_base", "unit": "m", "angle_unit": "rad",
        "T_R_O_initial": initial.tolist(), "T_F_O": T_F_O.tolist(), "T_R_O_above": transform(T_R_above).tolist(),
        "T_R_F_above": target.tolist(),
        "joint_waypoints_rad": path.points.tolist(),
        "velocity_ratio": min(config["velocity_ratio"], profile["velocity_ratio"]),
        "acceleration_ratio": min(config["acceleration_ratio"], profile["acceleration_ratio"]),
        "control_mode": "position", "interpolation_owner": "controller",
        "path_type": "sdk_position_point_to_point", "intermediate_targets": 0,
        "collision_checked": False, "insertion_executed": False}


class AboveTransfer:
    def __init__(self, config, geometry, kinematics, directory):
        self.config, self.geometry, self.kinematics = config, geometry, kinematics
        self.directory = Path(directory)
        self.settings = config["move_above"]
        self.alignment = RailPoseAlignment.from_config(config)
        for key, value in self.settings.items():
            if key not in ("target", "start_mode") and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(f"Invalid move_above setting: {key}")
        if self.settings.get("start_mode", "manual") not in ("manual", "auto"):
            raise ValueError("move_above.start_mode must be manual or auto")
        if type(self.settings["samples"]) is not int or self.settings["samples"] < 2:
            raise ValueError("move_above.samples must be an integer >= 2")
        target_path = Path(self.settings["target"])
        self.target_path = (Path(config["config_path"]).parent / target_path).resolve()
        target, camera_goal = load_target(self.target_path, geometry.calibration, config["mesh"])
        taught_calibration = target.get("calibration_sha256_at_teach")
        if taught_calibration and taught_calibration != digest(config["calibration"]):
            raise ValueError("Target was taught with a different camera calibration; teach it again")
        self.goal, self.above = above_targets(geometry.T_B_C, camera_goal, self.settings["height_m"])
        self.input_digests = {"target": digest(self.target_path),
                             **{k: digest(config[k]) for k in ("mesh", "calibration", "urdf", "manifest")}}
        self.preview_poses = {"T_C_O_goal": camera_goal.tolist(),
            "T_C_O_above": (np.linalg.inv(geometry.T_B_C) @ self.above).tolist()}
        self.status, self.needs_reinitialize = "idle", False
        self.records, self.grasps = [], []
        self.path = self.report = self.planning_error = None
        self.planning_done = threading.Event()
        self.settled_since = None
        self.pose_sampling = None
        self.manual_move_message = None
        self.cached_grasp = None
        self.lowering = None
        self.manual_lower_message = None
        self.insertion_started = False
        self.insertion_summary = None
        self.insertion_trace_seq = 0
        self.insertion_saved_state = None
        self.visual_verification = None
        self.preplace = None

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
        self.lowering = None
        self.manual_lower_message = None
        self.insertion_started = False
        self.insertion_summary = None
        self.insertion_trace_seq = 0
        self.insertion_saved_state = None
        self.visual_verification = None
        self.preplace = None

    def visual_verification_message(self):
        result = self.visual_verification
        if result is None:
            return "尚无上方视觉复核结果"
        delta = np.asarray(result["translation_error_base_m"])*1000
        return (f"视觉位置误差 {result['position_error_m']*1000:.2f} mm / "
                f"限值 {result['position_tolerance_m']*1000:.2f} mm，"
                f"姿态误差 {np.degrees(result['rotation_error_rad']):.2f}° / "
                f"限值 {np.degrees(result['rotation_tolerance_rad']):.2f}°；"
                f"右基座 XYZ 偏差 [{delta[0]:+.2f}, {delta[1]:+.2f}, {delta[2]:+.2f}] mm")

    @property
    def requires_visual_verification(self):
        return self.config["lowering"].get("mode", "force_spiral") != "position_spiral"

    @property
    def requires_preplace_alignment(self):
        return (self.config["lowering"].get("mode") == "position_spiral"
                and self.config.get("preplace_alignment", {}).get("enabled", True))

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
        self.visual_verification = None
        result = {"status": self.status, "side": "right", "control_mode": "position",
            "arrival_check": "sdk_and_fk", "visual_verification_required": False,
            "visual_verified": False, "position_error_m": error[0], "rotation_error_rad": error[1],
            "T_R_F_from_fk": transform(state["T_B_F"]).tolist(),
            "T_R_F_above": transform(self.report["T_R_F_above"]).tolist(), "T_R_O_above": self.above.tolist(),
            "pose_sampling": self.pose_sampling, "height_m": self.settings["height_m"],
            "hand_action": "keep_grasp", "insertion_executed": False,
            "assembly_verified": False, "collision_checked": False}
        write_json(self.directory / "above_arrival.json", result)
        write_json(self.directory / "result.json", result)
        if self.requires_preplace_alignment:
            from .preplace_transfer import PreplaceRefinement
            self.preplace = PreplaceRefinement(self, state)
            self.status = self.preplace.status
            self.preplace.save()

    def _start_insertion(self, hardware):
        from .insertion import validate_settings
        settings = self.config.get("insertion", {})
        try:
            missing = validate_settings(settings, require_ready=False)
        except (ValueError, TypeError, KeyError) as error:
            self.manual_lower_message = f"N 未执行：插入配置无效：{error}。"
            return
        if missing:
            self.manual_lower_message = "N 未执行：力控插入缺少现场参数：" + ", ".join(missing) + "。"
            return
        self._check_inputs()
        request = {"settings": deepcopy(settings), "grasp": self.cached_grasp.tolist(),
                   "goal": self.goal.tolist(), "above": self.above.tolist()}
        write_json(self.directory / "insertion_plan.json", {
            "schema_version": 1, "side": "right", "control_mode": "cartesian_impedance",
            "parent_frame": "tianji_right_base", "search_plane": "XZ", "insertion_axis": "-Y",
            "unit": "m", "angle_unit": "rad", "settings": request["settings"],
            "T_F_O": request["grasp"], "T_R_O_goal": request["goal"], "T_R_O_above": request["above"],
            "input_sha256": self.input_digests, "automatic_release": False,
            "assembly_verified": False, "collision_checked": False})
        try:
            hardware.queue_insertion(request)
        except ValueError as error:
            # A rejected request has not handed control to the insertion owner.
            self.manual_lower_message = f"N 未执行：插入请求检查未通过：{error}。"
            return
        self.insertion_started = True
        self.status = "switching_to_insertion"
        self.manual_lower_message = "N 已收到：切换至阻抗控制，轻触后螺旋找孔；保持抓握，完成后等待人工确认。"

    def poll_insertion(self, hardware):
        """Drain copied controller telemetry on the UI thread, never in the servo."""
        if not self.insertion_started:
            return None
        snapshot = hardware.insertion_snapshot(after_seq=self.insertion_trace_seq)
        if not isinstance(snapshot, dict) or not snapshot:
            return self.insertion_summary
        snapshot = deepcopy(snapshot)
        trace = snapshot.pop("trace", [])
        records = [sample for sample in trace if sample["trace_seq"] > self.insertion_trace_seq]
        if records:
            with (self.directory / "insertion_trace.jsonl").open("a", encoding="utf-8") as stream:
                first_available = snapshot.get("trace_first_seq", records[0]["trace_seq"])
                if first_available > self.insertion_trace_seq + 1:
                    stream.write(json.dumps({"event": "telemetry_gap",
                        "expected_trace_seq": self.insertion_trace_seq + 1,
                        "first_available_trace_seq": first_available}, ensure_ascii=False) + "\n")
                for sample in records:
                    stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
            self.insertion_trace_seq = max(sample["trace_seq"] for sample in records)
        self.insertion_summary = snapshot
        self.status = snapshot.get("status", self.status)
        result = dict(snapshot, side="right", control_mode="cartesian_impedance", force_controlled=True,
                      hand_action=snapshot.get("hand_action", "keep_grasp"), automatic_release=False,
                      assembly_verified=False, collision_checked=False)
        # Persist status transitions immediately; telemetry carries cycle detail.
        saved_state = (self.status, snapshot.get("reason"), snapshot.get("insertion_executed"))
        if saved_state != self.insertion_saved_state:
            write_json(self.directory / "insertion_result.json", result)
            write_json(self.directory / "result.json", result)
            self.insertion_saved_state = saved_state
        return snapshot

    def _observation(self, record, hardware):
        if self.needs_reinitialize or not record or record.get("valid") is not True:
            return None
        if (record.get("replay_only") or record.get("test_data")
                or record.get("camera_serial") != self.geometry.calibration["camera_serial"]
                or record.get("mesh_sha256") != self.geometry.mesh_digest
                or record.get("timestamp_source") != "sdk_global_timestamp_mapped_to_host_monotonic"):
            raise ValueError("Held-rail observation is not a matching live RGB-D measurement")
        stamp = int(record["timestamp_ns"])
        if (stamp <= self.not_before_ns or not 0 <= time.monotonic_ns()-stamp
                <= self.config["observation_max_age_s"]*1e9):
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
            self.path, self.report = plan_above(self.kinematics, self.config, joints, grasp, self.above)
            self.report.update(T_R_O_goal=self.goal.tolist(), height_m=self.settings["height_m"],
                               visual_verification_required=self.requires_visual_verification,
                               input_sha256=self.input_digests,
                               pose_sampling=self.pose_sampling,
                               pose_alignment_policy=self.config.get("pose_alignment"),
                               pose_alignment=[r.get("pose_alignment") for r in self.records])
            write_json(self.directory / "above_plan.json", self.report)
        except Exception as error:
            self.planning_error = error
        finally:
            self.planning_done.set()

    def _prepare_mean(self):
        """Freeze the accepted observations and mean once, before M is pressed."""
        position_limit = self.settings.get("pose_consistency_position_tolerance_m", .005)
        stable = True
        try:
            grasp = mean_pose(self.grasps, max_translation_m=position_limit)
        except PoseStabilityError:
            stable = False
            # Still validate every rigid transform and use the same SO(3) mean.
            grasp = mean_pose(self.grasps, max_translation_m=np.inf, max_rotation_rad=np.inf)
        self.cached_grasp = grasp.copy()
        errors = [difference(pose, grasp) for pose in self.grasps]
        self.pose_sampling = {"stable": stable, "operator_requested": False,
            "stability_overridden": False, "request_key": None,
            "max_position_deviation_m": max(error[0] for error in errors),
            "max_rotation_deviation_rad": max(error[1] for error in errors),
            "position_tolerance_m": position_limit, "rotation_tolerance_rad": .035,
            "samples": len(self.records), "mean_saved_monotonic_ns": time.monotonic_ns()}
        self._save_mean()
        write_json(self.directory / "pose_sampling_wait.json", self.pose_sampling)

    def _save_mean(self):
        write_json(self.directory / "held_observation.json", {"frame": "T_C_CAD", "unit": "m",
            "records": self.records, "T_F_O": self.cached_grasp.tolist(), "replay_only": False,
            "pose_sampling": self.pose_sampling})

    def _start_planning(self, state, *, force_move=False):
        if self.cached_grasp is None:
            self.manual_move_message = "M 未执行：还没有保存的位姿均值，请等待估姿完成后再按 M。"
            return
        stamp = time.monotonic_ns()
        self.pose_sampling.update(operator_requested=force_move, request_key="M" if force_move else None,
            stability_overridden=bool(force_move and not self.pose_sampling["stable"]),
            decision_monotonic_ns=stamp,
            saved_mean_age_s=(stamp-self.pose_sampling["mean_saved_monotonic_ns"])/1e9,
            pose_source="saved_mean")
        self._save_mean()
        self.status, self.planning_started = "planning_above", time.monotonic()
        if force_move:
            self.manual_move_message = "M 已收到：直接使用已保存的位姿均值规划移动，不重新采样或检查位姿稳定性。"
        threading.Thread(target=self._plan, args=(state["joints_rad"].copy(), self.cached_grasp.copy()),
                         daemon=True, name="right-above-planner").start()

    def advance(self, hardware, state, record, *, force_move=False, lower_requested=False):
        now = time.monotonic()
        s = self.settings
        self.manual_move_message = None
        self.manual_lower_message = None
        if self.insertion_started:
            self.poll_insertion(hardware)
            if lower_requested:
                self.manual_lower_message = "N 未执行：本次力控插入已启动或已结束，不能重复启动。"
            return self.status
        if self.preplace is not None and not self.preplace.converged:
            if lower_requested:
                self.manual_lower_message = "N 未执行：正在重新估姿并纠偏；对准完成后需再次按 N 下放。"
            self.status = self.preplace.advance(hardware, state, record)
            return self.status
        if lower_requested:
            if hardware.paused:
                self.manual_lower_message = "N 未执行：机械臂已暂停，本次不能恢复。"
                return self.status
            elif self.status == "above_target_unverified":
                # A retry only samples new vision while stationary. It never
                # consumes the same N press to start insertion after success.
                if not hardware.position_move_finished():
                    self.manual_lower_message = "N 未执行：请等待当前点动到位，再重新视觉复核。"
                    return self.status
                error = difference(state["T_B_F"], self.report["T_R_F_above"])
                if error[0] > s["position_tolerance_m"] or error[1] > s["rotation_tolerance_rad"]:
                    self.manual_lower_message = "N 未执行：当前末端已偏离上方位置，请先回到上方目标，再重新视觉复核。"
                    return self.status
                self._check_inputs()
                self.status, self.not_before_ns = "verifying_above", time.monotonic_ns()
                self.records.clear()
                self.manual_lower_message = ("N 仅重新进行上方视觉复核，本次不下放；"
                    "复核通过后需再次按 N 才启动插入。上次" + self.visual_verification_message() + "。")
                return self.status
            elif self.status != "above_target":
                reason = {"above_target_unverified": "上方视觉复核未通过",
                          "verifying_above": "正在等待新的上方视觉复核结果，请稍后再按 N",
                          "planning_lowering": "正在规划下放", "lowering": "正在下放",
                          "planning_position_spiral": "正在规划位置螺旋插入",
                          "position_spiral": "正在执行位置螺旋插入",
                          "awaiting_confirmation": "本次插入轨迹已结束，等待人工确认，不能重复启动",
                          "placement_reached": "已到装配目标，不重复下放"}.get(self.status,
                              "尚未到达上方位置" if not self.requires_visual_verification
                              else "尚未到达并复核上方位置")
                self.manual_lower_message = f"N 未执行：{reason}。"
            elif not hardware.position_move_finished():
                self.manual_lower_message = "N 未执行：请等待当前点动到位。"
            else:
                # The fresh grasp is the insertion reference. A small angular
                # residual at a long rail's origin can amplify flange distance.
                error = (difference(transform(state["T_B_F"]) @ self.cached_grasp, self.above)
                         if self.preplace is not None and self.preplace.converged
                         else difference(state["T_B_F"], self.report["T_R_F_above"]))
                arrival_limits = (self.preplace.settings if self.preplace is not None and self.preplace.converged else s)
                if error[0] > arrival_limits["position_tolerance_m"] or error[1] > arrival_limits["rotation_tolerance_rad"]:
                    self.manual_lower_message = "N 未执行：当前末端已偏离上方位置，请先回到上方目标。"
                else:
                    if self.config["lowering"].get("mode", "force_spiral") == "force_spiral":
                        self._start_insertion(hardware)
                        return self.status
                    self._check_inputs()
                    if self.config["lowering"]["mode"] == "position_spiral":
                        from .position_spiral import PositionSpiral
                        controller = PositionSpiral
                        message = "N 已收到：规划位置控制螺旋下降，不使用六维力；到目标后保持抓握，等待人工确认。"
                    else:
                        from .lowering import Lowering
                        controller = Lowering
                        message = "N 已收到：规划缓慢竖直下放，到装配目标后保持抓握。"
                    self.lowering = controller(self.config, self.kinematics, self.directory,
                        state["joints_rad"].copy(), self.cached_grasp.copy(), self.goal, self.above)
                    if self.preplace is not None:
                        self.lowering.preplace_alignment = self.preplace.summary()
                    self.status = self.lowering.status
                    self.manual_lower_message = message
                    return self.status
            if self.lowering is None and self.status != "verifying_above":
                return self.status
            # Repeated N must not suppress verification or motion checks.
        if self.lowering is not None:
            if self.lowering.status in ("planning_lowering", "planning_position_spiral") and self.lowering.done.is_set():
                self._check_inputs()
            self.status = self.lowering.advance(hardware, state)
            return self.status
        if force_move and hardware.paused:
            self.manual_move_message = "M 未执行：机械臂已暂停。"
            return self.status
        if self.status in ("waiting_move_key", "waiting_pose_override"):
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
                self.status = "waiting_move_key" if s.get("start_mode", "manual") == "manual" else "waiting_pose_override"
                if force_move or (s.get("start_mode") == "auto" and self.pose_sampling["stable"]):
                    self._start_planning(state, force_move=force_move)
            elif force_move:
                self.manual_move_message = "M 未执行：还没有保存的位姿均值，请等待估姿完成后再按 M。"
            elif (not force_move
                  and time.monotonic_ns()-self.not_before_ns > self.config["perception_timeout_s"]*1e9):
                raise TimeoutError("No fresh held-rail pose; arm remains at its current position")
        elif self.status == "planning_above":
            if now-self.planning_started > s["planning_timeout_s"]:
                raise TimeoutError("Right-arm above-goal planning timed out")
            if self.planning_done.is_set():
                if self.planning_error is not None:
                    raise self.planning_error
                self._check_inputs()
                hardware.move_to_position(self.path.points[0], self.path.points[-1],
                                          timeout_s=s["position_move_timeout_s"])
                self.motion_started, self.status = now, "moving_above"
                self.controller_completed_at = None
                self.last_motion_report = now-2.
        elif self.status == "moving_above":
            error = difference(state["T_B_F"], self.report["T_R_F_above"])
            reached = error[0] <= s["position_tolerance_m"] and error[1] <= s["rotation_tolerance_rad"]
            completed = hardware.position_move_finished()
            if completed and self.controller_completed_at is None:
                self.controller_completed_at = now
            if now-self.last_motion_report >= 2.:
                print(f"位置控制搬运：已等待 {now-self.motion_started:.1f} 秒，"
                      f"末端距目标 {error[0]*1000:.2f} mm / {np.degrees(error[1]):.2f}°，"
                      f"控制器{'已到位' if completed else '运动中'}。", flush=True)
                self.last_motion_report = now
            if completed and reached:
                self.settled_since = self.settled_since or now
                if now-self.settled_since >= s["settle_s"]:
                    if self.requires_visual_verification:
                        self.status, self.not_before_ns = "verifying_above", time.monotonic_ns()
                        self.records.clear()
                    else:
                        self._finish_position_arrival(state, error)
            else:
                self.settled_since = None
            if not completed and now-self.motion_started > s["position_move_timeout_s"]:
                raise TimeoutError("Right-arm SDK position move timed out")
            if (completed and now-self.controller_completed_at > s["reach_timeout_s"]):
                raise TimeoutError("Right flange did not reach the planned above-goal pose")
        elif self.status == "verifying_above":
            hardware.raise_if_fault()
            error = difference(state["T_B_F"], self.report["T_R_F_above"])
            if (not hardware.position_move_finished() or error[0] > s["position_tolerance_m"]
                    or error[1] > s["rotation_tolerance_rad"]):
                self.records.clear()
                raise RuntimeError("Right arm did not remain at the above-goal pose during visual verification")
            if time.monotonic_ns()-self.not_before_ns > self.config["perception_timeout_s"]*1e9:
                raise TimeoutError("Above-goal arm endpoint reached; visual verification timed out")
            observation = self._observation(record, hardware)
            if observation is not None:
                actual, _, record = observation
                self.records.append(record)
                if len(self.records) >= s["samples"]:
                    actual = self.geometry.T_B_C @ mean_pose(
                        [transform(r["T_C_CAD"]) for r in self.records], max_translation_m=
                        s.get("pose_consistency_position_tolerance_m", .005))
                    distance, angle = difference(actual, self.above)
                    passed = distance <= s["visual_position_tolerance_m"] and angle <= s["visual_rotation_tolerance_rad"]
                    self.status = "above_target" if passed else "above_target_unverified"
                    self.visual_verification = {
                        "passed": bool(passed), "position_error_m": distance, "rotation_error_rad": angle,
                        "position_tolerance_m": s["visual_position_tolerance_m"],
                        "rotation_tolerance_rad": s["visual_rotation_tolerance_rad"],
                        "translation_error_base_m": (actual[:3, 3]-self.above[:3, 3]).tolist(),
                        "frame": "tianji_right_base", "samples": len(self.records),
                        "first_timestamp_ns": self.records[0]["timestamp_ns"],
                        "last_timestamp_ns": self.records[-1]["timestamp_ns"]}
                    print(self.visual_verification_message() + f"；复核{'通过' if passed else '未通过'}。", flush=True)
                    result = {"status": self.status, "side": "right",
                        "position_error_m": distance, "rotation_error_rad": angle,
                        "visual_verification": deepcopy(self.visual_verification),
                        "T_R_O_observed": actual.tolist(), "T_R_O_above": self.above.tolist(),
                        "pose_alignment": [r.get("pose_alignment") for r in self.records],
                        "pose_sampling": self.pose_sampling,
                        "height_m": s["height_m"], "hand_action": "keep_grasp",
                        "insertion_executed": False, "collision_checked": False}
                    write_json(self.directory / "above_verification.json", result)
                    write_json(self.directory / "result.json", result)
        return self.status

    def _check_inputs(self):
        for key, expected in self.input_digests.items():
            if digest(self.target_path if key == "target" else self.config[key]) != expected:
                raise ValueError(f"{key} changed during right-arm planning")
