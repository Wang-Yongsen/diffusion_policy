"""Fresh visual observations and bounded position corrections before insertion."""
from copy import deepcopy
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from rail_preinsert.core import PoseStabilityError, difference, mean_pose, transform, vector, write_json
from .preplace_alignment import plan_preplace_alignment, validate_settings


class PreplaceRefinement:
    def __init__(self, transfer, state):
        self.transfer = transfer
        self.settings = transfer.config["preplace_alignment"]
        validate_settings(self.settings)
        self.started = time.monotonic()
        self.corrections = 0
        self.plans = 0
        self.finalizing = False
        self.measurements, self.evidence = [], []
        self.report = None
        self.converged = False
        self.expected_generation = None
        self._sample_again(state)

    def _sample_again(self, state):
        t = self.transfer
        self.status = "reestimating_preplace"
        self.finalizing = False
        self.sample_started = time.monotonic()
        self.stationary_flange = transform(state["T_B_F"]).copy()
        self.predicted_pose = (np.linalg.inv(t.geometry.T_B_C) @ self.stationary_flange @ t.cached_grasp)
        self.expected_generation = None
        t.not_before_ns = time.monotonic_ns()
        t.records.clear()
        t.grasps.clear()
        t.needs_reinitialize = True

    def initialized(self, generation):
        if not isinstance(generation, str) or not generation:
            raise ValueError("Preplacement re-estimation requires a new perception generation")
        self.expected_generation = generation

    def summary(self, reason=None):
        return dict(status=self.status, converged=self.converged,
            previous_status=getattr(self, "previous_status", None),
            corrections=self.corrections, planned_moves=self.plans, final_pose_lock=self.finalizing,
            measurements=deepcopy(self.measurements),
            execution_evidence=deepcopy(self.evidence), reason=None if reason is None else str(reason),
            elapsed_s=max(0., time.monotonic()-self.started),
            force_sensor_used=False, automatic_release=False, assembly_verified=False,
            collision_checked=False)

    def save(self, reason=None):
        result = self.summary(reason)
        write_json(self.transfer.directory / "preplace_alignment_result.json", result)
        write_json(self.transfer.directory / "result.json", dict(status=self.status, side="right",
            preplace_alignment=result, hand_action="keep_grasp", insertion_executed=False,
            force_sensor_used=False, assembly_verified=False, collision_checked=False))
        return result

    def stopped(self, reason):
        self.previous_status = self.status
        self.status = "paused"
        return self.save(reason)

    def _complete(self, state):
        t = self.transfer
        self.status, self.converged = "above_target", True
        t.report.update(T_R_F_measured=transform(state["T_B_F"]).tolist(),
            measured_joints_rad=vector(state["joints_rad"], 7).tolist(),
            T_R_O_from_fk=(transform(state["T_B_F"]) @ t.cached_grasp).tolist(),
            last_correction_plan=(f"preplace_alignment_plan_{self.plans}.json" if self.plans else None),
            preplace_alignment=self.save())
        write_json(t.directory / "above_aligned_plan.json", t.report)

    def _check_stationary(self, hardware, state):
        distance, angle = difference(state["T_B_F"], self.stationary_flange)
        if (not hardware.position_move_finished() or distance > self.settings["position_tolerance_m"]
                or angle > self.settings["rotation_tolerance_rad"]):
            raise RuntimeError("Right arm moved during preplacement re-estimation/planning")

    def _measure(self, hardware, state, record):
        t, s = self.transfer, self.settings
        self._check_stationary(hardware, state)
        if time.monotonic()-self.sample_started > s["perception_timeout_s"]:
            raise TimeoutError("Preplacement FoundationPose re-estimation timed out; insertion not started")
        if self.expected_generation is None or not record or record.get("generation") != self.expected_generation:
            return
        if record.get("reason") == "held_pose_unavailable":
            raise RuntimeError("Preplacement FoundationPose failed: " + str(record.get("detail", "unknown")))
        observation = t._observation(record, hardware)
        if observation is None:
            return
        _, grasp, record = observation
        t.records.append(record)
        t.grasps.append(grasp)
        if len(t.records) < s["samples"]:
            return
        try:
            grasp = mean_pose(t.grasps, max_translation_m=s["consistency_position_m"],
                              max_rotation_rad=s["consistency_rotation_rad"])
        except PoseStabilityError:
            # Wait for a stable rolling window, within the same finite deadline.
            t.records.pop(0)
            t.grasps.pop(0)
            return
        actual = transform(state["T_B_F"]) @ grasp
        distance, angle = difference(actual, t.above)
        delta = actual[:3, 3]-t.above[:3, 3]
        midpoint = np.r_[vector(t.config["rail_midpoint_cad_m"], 3), 1.]
        measurement = dict(round=len(self.measurements)+1, generation=self.expected_generation,
            position_error_m=distance, rotation_error_rad=angle,
            translation_error_base_m=delta.tolist(),
            rotation_error_base_rad=Rotation.from_matrix(actual[:3, :3] @ t.above[:3, :3].T).as_rotvec().tolist(),
            midpoint_error_base_m=(actual @ midpoint-t.above @ midpoint)[:3].tolist(),
            T_R_O_observed=actual.tolist(), T_R_O_above=t.above.tolist(), T_F_O=grasp.tolist(),
            first_timestamp_ns=t.records[0]["timestamp_ns"], last_timestamp_ns=t.records[-1]["timestamp_ns"],
            samples=len(t.records), position_tolerance_m=s["visual_position_tolerance_m"],
            rotation_tolerance_rad=s["visual_rotation_tolerance_rad"])
        self.measurements.append(measurement)
        self.save()
        print(f"预放置重新估姿：位置偏差 {distance*1000:.2f} mm，姿态偏差 {np.degrees(angle):.2f}°；"
              f"右基座 XYZ [{delta[0]*1000:+.2f}, {delta[1]*1000:+.2f}, {delta[2]*1000:+.2f}] mm。", flush=True)
        if distance > s["max_correction_m"] or angle > s["max_correction_rad"]:
            raise ValueError("Preplacement visual deviation exceeds the configured correction limits")
        t.cached_grasp = grasp.copy()
        t.pose_sampling = dict(stable=True, source="preplace_reestimation", generation=self.expected_generation,
            samples=len(t.records), mean_saved_monotonic_ns=time.monotonic_ns(),
            max_position_deviation_m=max(difference(g, grasp)[0] for g in t.grasps),
            max_rotation_deviation_rad=max(difference(g, grasp)[1] for g in t.grasps))
        # Keep the original coarse path in above_plan.json; its joint targets
        # cannot be relabelled with a newly measured grasp.
        t.path = None
        t.report = dict(schema_version=3, side="right", control_mode="position",
            coarse_plan_file="above_plan.json", T_F_O=grasp.tolist(),
            T_R_F_above=(t.above @ np.linalg.inv(grasp)).tolist(),
            T_R_O_goal=t.goal.tolist(), T_R_O_above=t.above.tolist(),
            height_m=t.settings["height_m"], input_sha256=t.input_digests,
            preplace_alignment_required=True, pose_sampling=deepcopy(t.pose_sampling),
            collision_checked=False, assembly_verified=False)
        write_json(t.directory / "aligned_observation.json", dict(records=t.records,
            T_F_O=grasp.tolist(), pose_sampling=t.pose_sampling, measurement=measurement))
        accepted = distance <= s["visual_position_tolerance_m"] and angle <= s["visual_rotation_tolerance_rad"]
        measurement["within_visual_tolerance"] = bool(accepted)
        if accepted and distance <= s["position_tolerance_m"] and angle <= s["rotation_tolerance_rad"]:
            self._complete(state)
            return
        if not accepted and self.corrections >= s["max_corrections"]:
            raise RuntimeError("Preplacement alignment did not converge within max_corrections; insertion not started")
        # Once vision is within tolerance, freeze this measured grasp and finish
        # its bounded pose correction before entering the fixed-attitude spiral.
        # This final small move does not start another visual correction loop.
        self.finalizing = bool(accepted)
        if not self.finalizing:
            self.corrections += 1
        self.plans += 1
        self.status, self.planning_started = "planning_preplace", time.monotonic()
        self.done, self.error, self.report = threading.Event(), None, None
        joints = vector(state["joints_rad"], 7).copy()

        def plan():
            try:
                self.report = plan_preplace_alignment(t.kinematics, t.config, joints, grasp, t.goal, t.above)
                self.report.update(round=self.corrections, plan_number=self.plans,
                                   final_pose_lock=self.finalizing, observation=measurement,
                                   input_sha256=t.input_digests)
                write_json(t.directory / f"preplace_alignment_plan_{self.plans}.json", self.report)
            except Exception as error:
                self.error = error
            finally:
                self.done.set()
        threading.Thread(target=plan, daemon=True, name="right-preplace-planner").start()
        self.save()

    def _dispatch(self, hardware, state, now):
        t, s = self.transfer, self.settings
        t._check_inputs()
        joints = vector(state["joints_rad"], 7)
        planned_start = np.asarray(self.report["joint_waypoints_rad"][self.index])
        if np.max(np.abs(joints-planned_start)) > t.config["arm_tracking_error_rad"]:
            raise RuntimeError("Right-arm joints changed from the preplacement IK branch")
        actual = transform(state["T_B_F"]) @ t.cached_grasp
        start_error = difference(actual, self.report["T_R_O_waypoints"][self.index])
        if start_error[0] > s["position_tolerance_m"] or start_error[1] > s["rotation_tolerance_rad"]:
            raise RuntimeError("Measured start differs from planned preplacement segment")
        target = np.asarray(self.report["joint_waypoints_rad"][self.index+1])
        self.evidence.append(dict(round=self.corrections, step=self.index+1, command_queued=False,
            plan_number=self.plans, final_pose_lock=self.finalizing,
            endpoint_verified=False, measured_start_joints_rad=joints.tolist()))
        hardware.move_to_position(joints, target, timeout_s=s["step_timeout_s"],
            velocity_ratio=s["velocity_ratio"], acceleration_ratio=s["acceleration_ratio"])
        self.evidence[-1]["command_queued"] = True
        self.step_started, self.settled_since = now, None

    def _move(self, hardware, state, now):
        t, s = self.transfer, self.settings
        actual = transform(state["T_B_F"]) @ t.cached_grasp
        start = np.asarray(self.report["T_R_O_waypoints"][self.index])
        target = np.asarray(self.report["T_R_O_waypoints"][self.index+1])
        segment = target[:3, 3]-start[:3, 3]
        length2 = float(segment @ segment)
        fraction = 0. if length2 < 1e-18 else np.clip((actual[:3, 3]-start[:3, 3]) @ segment/length2, 0., 1.)
        path_error = float(np.linalg.norm(actual[:3, 3]-start[:3, 3]-fraction*segment))
        angular_bound = difference(start, target)[1]+s["path_rotation_tolerance_rad"]
        if (path_error > s["path_tolerance_m"] or difference(actual, start)[1] > angular_bound
                or difference(actual, target)[1] > angular_bound):
            raise RuntimeError("Preplacement correction deviated from its segment corridor")
        corners = np.asarray(self.report["mesh_bound_corners_cad_m"])
        goal_y = (corners @ t.goal[:3, :3].T+t.goal[:3, 3])[:, 1]
        def clearance(pose):
            return float(np.min((corners @ pose[:3, :3].T+pose[:3, 3])[:, 1]-goal_y))
        # SDK joint interpolation is not the planned SO(3) arc. Check measured
        # geometry as well; only endpoint measurement tolerance is allowed here.
        allowance = s["position_tolerance_m"]+2*np.linalg.norm(corners, axis=1).max()*np.sin(s["rotation_tolerance_rad"]/2)
        floor = min(clearance(start), clearance(target))-self.report["rotation_chord_margin_m"]-allowance
        if clearance(actual) < floor:
            raise RuntimeError("Measured rail geometry fell below the preplacement segment clearance")
        if now-self.step_started > s["step_timeout_s"]:
            raise TimeoutError("Preplacement correction step did not reach its measured endpoint")
        distance, angle = difference(actual, target)
        completed = bool(hardware.position_move_finished())
        self.evidence[-1].update(sdk_completed=completed, position_error_m=distance, rotation_error_rad=angle)
        if completed and distance <= s["position_tolerance_m"] and angle <= s["rotation_tolerance_rad"]:
            self.settled_since = now if self.settled_since is None else self.settled_since
            if now-self.settled_since >= s["settle_s"]:
                self.evidence[-1]["endpoint_verified"] = True
                self.index += 1
                if self.index == len(self.report["joint_waypoints_rad"])-1:
                    if self.finalizing:
                        self._complete(state)
                    else:
                        self._sample_again(state)
                        self.save()
                else:
                    self._dispatch(hardware, state, now)
        else:
            self.settled_since = None

    def advance(self, hardware, state, record):
        if self.converged:
            return self.status
        hardware.raise_if_fault()
        if hardware.paused:
            raise RuntimeError("Preplacement alignment is paused")
        now = time.monotonic()
        if now-self.started > self.settings["total_timeout_s"]:
            raise TimeoutError("Preplacement alignment total timeout exceeded")
        if self.status == "reestimating_preplace":
            self._measure(hardware, state, record)
        elif self.status == "planning_preplace":
            self._check_stationary(hardware, state)
            if now-self.planning_started > self.settings["planning_timeout_s"]:
                raise TimeoutError("Preplacement correction IK planning timed out")
            if self.done.is_set():
                if self.error is not None:
                    raise self.error
                self.index = 0
                self._dispatch(hardware, state, now)
                self.status = "aligning_preplace"
                self.save()
                label = "收敛后姿态精调" if self.finalizing else f"第 {self.corrections}/{self.settings['max_corrections']} 轮"
                print(f"预放置纠偏：{label}，"
                      f"{len(self.report['joint_waypoints_rad'])-1} 段，抬升 {self.report['lift_m']*1000:.1f} mm。", flush=True)
        elif self.status == "aligning_preplace":
            self._move(hardware, state, now)
        return self.status
