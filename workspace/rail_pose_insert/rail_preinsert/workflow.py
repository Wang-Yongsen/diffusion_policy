from __future__ import annotations

from copy import deepcopy
import numpy as np

from .core import (digest, difference, interpolate_joints, load_calibration, load_target,
                   matrix, mean_pose, read_json, targets, transform, vector)
from .geometry import RobotGeometry
from .planner import Planner
from .pose_alignment import RailPoseAlignment


def observation_poses(observation, calibration, mesh, *, allow_test_data=False,
                      alignment=None, reference=None):
    if observation.get("test_data", False) and not allow_test_data:
        raise ValueError("Synthetic observation cannot be used on hardware")
    if observation.get("replay_only", False) and not allow_test_data:
        raise ValueError("Offline replay observation cannot stand in for a fresh live measurement")
    if (observation.get("camera_serial") != calibration["camera_serial"]
            or observation.get("unit") != "m" or observation.get("frame") != "T_C_CAD"
            or observation.get("mesh_sha256") != digest(mesh)):
        raise ValueError("Observation camera/CAD/coordinate convention mismatch")
    records = observation.get("records", [])
    if not records or any(not r.get("valid") for r in records):
        raise ValueError("No valid rail pose observations")
    times = [int(r["timestamp_ns"]) for r in records]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Pose times must increase")
    if not allow_test_data and any(r.get("timestamp_source") != "sdk_global_timestamp_mapped_to_host_monotonic"
                                   for r in records):
        raise ValueError("Live pose timestamps must be mapped SDK GLOBAL_TIME")
    if alignment is not None:
        reference = records[0]["T_C_CAD"] if reference is None else reference
        records = [alignment.align_record(r, reference) for r in records]
    matrices = [transform(r["T_C_CAD"]) for r in records]
    mean_pose(matrices)  # Align first; still reject genuine stationary drift.
    return records, matrices


def hand_references(config):
    from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_LIMITS_RAD, JOINT_NAMES
    data = read_json(config["hand_reference"])
    candidates = {v["label"]: v for v in data["candidates"]}
    result = {}
    for kind in ("open", "grasp"):
        item = candidates[config[f"{kind}_label"]]
        if item["joint_order"] != list(JOINT_NAMES):
            raise ValueError("Candidate joint order differs from Hand2 driver")
        q = vector(item["command_position_rad"], 20, "recorded hand target")
        if any(not low <= value <= high for value, (low, high) in zip(q, JOINT_LIMITS_RAD)):
            raise ValueError("Candidate hand command exceeds limits")
        result[kind] = q
    return result


def prepare_plan(config, kinematics, state, observation, *, allow_test_data=False):
    input_digests = {key: digest(config[key]) for key in ("calibration", "target", "mesh", "urdf", "manifest")}
    calibration, X = load_calibration(config["calibration"], allow_test_data=allow_test_data)
    _, T_C_goal = load_target(config["target"], calibration, config["mesh"], allow_test_data=allow_test_data)
    if state.get("test_data", False) and not allow_test_data:
        raise ValueError("Synthetic state cannot be used for live execution")
    if state.get("model_sha256", kinematics.model.digest) != kinematics.model.digest:
        raise ValueError("Measured state uses another kinematic model")
    alignment = RailPoseAlignment.from_config(config)
    records, poses = observation_poses(observation, calibration, config["mesh"],
        allow_test_data=allow_test_data, alignment=alignment, reference=T_C_goal)
    full = vector(state["full_joints"], 54, "full measured state")
    grasps = []
    for record, T_C_O in zip(records, poses):
        joints = interpolate_joints(state["left_joint_history"], record["timestamp_ns"])
        T_B_F = matrix(kinematics.fk("left", tuple(joints)))
        grasps.append(np.linalg.inv(T_B_F) @ X @ T_C_O)
    grasp = mean_pose(grasps)
    values = targets(X, T_C_goal, poses[0], matrix(kinematics.fk("left", tuple(full[:7]))),
                     config["outward_axis_cad"], config["preinsert_offset_m"])
    values["T_F_O"] = grasp
    values["T_B_F_preinsert"] = values["T_B_O_preinsert"] @ np.linalg.inv(grasp)
    if alignment is not None:
        alignment.check_motion(matrix(kinematics.fk("left", tuple(full[:7]))) @ grasp,
                               values["T_B_O_preinsert"])
    frames = RobotGeometry(config, kinematics)
    effective = deepcopy(config)
    from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import load_config
    profile = load_config(config["arm_config"])["profile"]["parameters"]["arms"]["left"]
    arm = kinematics.model.arm("left")
    effective["joint_speed_rad_s"] = min(config["joint_speed_rad_s"],
        min(arm.max_joint_velocity_rad_s) * profile["velocity_ratio"] / 100)
    effective["joint_acceleration_rad_s2"] = min(config["joint_acceleration_rad_s2"],
        min(np.radians(row[3]) for row in arm.limits_native) * profile["acceleration_ratio"] / 100)
    planner = Planner(effective, kinematics)
    trajectory = planner.plan(full[:7], values["T_B_F_preinsert"])
    if any(digest(config[key]) != value for key, value in input_digests.items()):
        raise ValueError("Planning input changed during computation; repeat planning")
    report = {
        "schema_version": 2, "end_stage": "preinsert", "status": "planned",
        "test_data": bool(allow_test_data), "unit": "m", "angle_unit": "rad",
        "parent_frame": "tianji_left_base", "model_sha256": kinematics.model.digest,
        "mesh_sha256": digest(config["mesh"]),
        "calibration_sha256": digest(config["calibration"]), "target_sha256": digest(config["target"]),
        "urdf_sha256": digest(config["urdf"]),
        "manifest_sha256": digest(config["manifest"]),
        "initial_full_joints": full.tolist(), "targets": {k: v.tolist() for k, v in values.items()},
        "joint_waypoints_rad": trajectory.points.tolist(), "segment_durations_s": trajectory.durations.tolist(),
        "duration_s": trajectory.duration, "joint_speed_rad_s": effective["joint_speed_rad_s"],
        "joint_acceleration_rad_s2": effective["joint_acceleration_rad_s2"],
        "path_type": "direct_joint_interpolation", "collision_checked": False,
        "urdf_sdk_registration_error_m": frames.registration_error_m,
        "pose_alignment": [r.get("pose_alignment") for r in records],
        "pose_alignment_policy": config.get("pose_alignment"),
    }
    return trajectory, report, frames


def check_plan_inputs(config, report):
    if report.get("schema_version") != 2 or report.get("collision_checked") is not False:
        raise ValueError("Regenerate the plan using the current direct interpolation workflow")
    if report.get("test_data") or report.get("end_stage") != "preinsert":
        raise ValueError("Only live preinsert plans can execute")
    if report.get("pose_alignment_policy") != config.get("pose_alignment"):
        raise ValueError("Pose alignment policy changed; regenerate the plan")
    for key in ("calibration", "target", "mesh", "urdf", "manifest"):
        if digest(config[key]) != report[f"{key}_sha256"]:
            raise ValueError(f"{key} changed after planning")
    if (report["joint_speed_rad_s"] > config["joint_speed_rad_s"]
            or report["joint_acceleration_rad_s2"] > config["joint_acceleration_rad_s2"]):
        raise ValueError("Plan dynamic limits exceed experiment configuration")


def verify_preinsert(config, report, observation, *, allow_test_data=False):
    calibration, X = load_calibration(config["calibration"], allow_test_data=allow_test_data)
    expected = transform(report["targets"]["T_B_O_preinsert"])
    records, poses = observation_poses(observation, calibration, config["mesh"],
        allow_test_data=allow_test_data, alignment=RailPoseAlignment.from_config(config),
        reference=np.linalg.inv(X) @ expected)
    actual = X @ mean_pose(poses)
    distance, angle = difference(actual, expected)
    passed = (distance <= config["visual_position_tolerance_m"]
              and angle <= config["visual_rotation_tolerance_rad"])
    return {"status": "preinsert_reached" if passed else "preinsert_unverified",
            "position_error_m": distance, "rotation_error_rad": angle,
            "T_B_O_observed": actual.tolist(), "end_stage": "preinsert",
            "pose_alignment": [r.get("pose_alignment") for r in records],
            "hand_action": "keep_grasp", "insertion_executed": False,
            "collision_checked": False,
            "verification": "stationary visual pose error only"}
