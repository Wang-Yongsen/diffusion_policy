from __future__ import annotations

from copy import deepcopy
import numpy as np
import trimesh

from .core import (digest, difference, interpolate_joints, load_calibration, load_target,
                   matrix, mean_pose, read_json, targets, transform, vector)
from .geometry import CollisionWorld
from .planner import Planner


def observation_poses(observation, calibration, mesh, *, allow_test_data=False):
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
    matrices = [transform(r["T_C_CAD"]) for r in records]
    mean_pose(matrices)  # Reject inconsistent stationary estimates / direction flips.
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
    input_digests = {key: digest(config[key]) for key in ("calibration", "target", "scene", "mesh", "urdf", "manifest")}
    calibration, X = load_calibration(config["calibration"], allow_test_data=allow_test_data)
    target, T_C_goal = load_target(config["target"], calibration, config["mesh"], allow_test_data=allow_test_data)
    if state.get("test_data", False) and not allow_test_data:
        raise ValueError("Synthetic state cannot be used for live execution")
    if state.get("model_sha256", kinematics.model.digest) != kinematics.model.digest:
        raise ValueError("Measured state uses another kinematic model")
    records, poses = observation_poses(observation, calibration, config["mesh"], allow_test_data=allow_test_data)
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
    world = CollisionWorld(config, kinematics, full, grasp, values["T_B_O_goal"], allow_test_data=allow_test_data)
    effective = deepcopy(config)
    from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import load_config
    profile = load_config(config["arm_config"])["profile"]["parameters"]["arms"]["left"]
    arm = kinematics.model.arm("left")
    effective["joint_speed_rad_s"] = min(config["joint_speed_rad_s"],
        min(arm.max_joint_velocity_rad_s) * profile["velocity_ratio"] / 100)
    effective["joint_acceleration_rad_s2"] = min(config["joint_acceleration_rad_s2"],
        min(np.radians(row[3]) for row in arm.limits_native) * profile["acceleration_ratio"] / 100)
    planner = Planner(effective, kinematics, world)
    trajectory = planner.plan(full[:7], values["T_B_F_preinsert"])
    if any(digest(config[key]) != value for key, value in input_digests.items()):
        raise ValueError("Planning input changed during computation; repeat planning")
    report = {
        "schema_version": 1, "end_stage": "preinsert", "status": "planned",
        "test_data": bool(allow_test_data), "unit": "m", "angle_unit": "rad",
        "parent_frame": "tianji_left_base", "model_sha256": kinematics.model.digest,
        "mesh_sha256": digest(config["mesh"]), "scene_sha256": digest(config["scene"]),
        "calibration_sha256": digest(config["calibration"]), "target_sha256": digest(config["target"]),
        "urdf_sha256": digest(config["urdf"]),
        "manifest_sha256": digest(config["manifest"]),
        "initial_full_joints": full.tolist(), "targets": {k: v.tolist() for k, v in values.items()},
        "joint_waypoints_rad": trajectory.points.tolist(), "segment_durations_s": trajectory.durations.tolist(),
        "duration_s": trajectory.duration, "joint_speed_rad_s": effective["joint_speed_rad_s"],
        "joint_acceleration_rad_s2": effective["joint_acceleration_rad_s2"],
        "collision_check": {"geometry": "conservative convex hulls; measured fixed obstacles + target-anchored fixture",
                            "edge_resolution_rad": config["edge_resolution_rad"],
                            "margin_m": config["collision_margin_m"], "configurations_checked": world.check_count,
                            "urdf_sdk_registration_error_m": world.model.registration_error_m,
                            "allowed_internal_contacts": "fixed measured left hand + held rail; adjacent serial/mount links",
                            "limitation": "Discrete joint-path validation, not a continuous swept-volume certificate"},
    }
    return trajectory, report, world


def check_plan_inputs(config, report):
    if report.get("test_data") or report.get("end_stage") != "preinsert":
        raise ValueError("Only live preinsert plans can execute")
    for key in ("calibration", "target", "scene", "mesh", "urdf", "manifest"):
        if digest(config[key]) != report[f"{key}_sha256"]:
            raise ValueError(f"{key} changed after planning")
    if (report["joint_speed_rad_s"] > config["joint_speed_rad_s"]
            or report["joint_acceleration_rad_s2"] > config["joint_acceleration_rad_s2"]):
        raise ValueError("Plan dynamic limits exceed experiment configuration")


def verify_preinsert(config, report, observation, *, allow_test_data=False):
    calibration, X = load_calibration(config["calibration"], allow_test_data=allow_test_data)
    _, poses = observation_poses(observation, calibration, config["mesh"], allow_test_data=allow_test_data)
    actual = X @ mean_pose(poses)
    expected = transform(report["targets"]["T_B_O_preinsert"])
    goal = transform(report["targets"]["T_B_O_goal"])
    distance, angle = difference(actual, expected)
    vertices = np.asarray(trimesh.load(config["mesh"], force="mesh").vertices)
    delta = vertices @ (actual[:3, :3] - goal[:3, :3]).T + actual[:3, 3] - goal[:3, 3]
    n = vector(report["targets"]["outward_axis_B"], 3)
    minimum_offset = float(np.min(delta @ n))
    manifest = read_json(config["manifest"])
    fixture_file = __import__("pathlib").Path(config["manifest"]).parent / manifest["cad"]["models"]["fixture"]["mesh"]
    fixture = trimesh.load(fixture_file, force="mesh")
    pin_height = float(fixture.bounds[1, 2] - manifest["fixture_geometry"]["support_height_m"])
    passed = (distance <= config["visual_position_tolerance_m"] and angle <= config["visual_rotation_tolerance_rad"]
              and minimum_offset > pin_height + config["collision_margin_m"])
    return {"status": "preinsert_reached" if passed else "preinsert_unverified",
            "position_error_m": distance, "rotation_error_rad": angle,
            "minimum_rail_offset_from_seated_m": minimum_offset, "tallest_pin_m": pin_height,
            "T_B_O_observed": actual.tolist(), "end_stage": "preinsert",
            "hand_action": "keep_grasp", "insertion_executed": False,
            "verification": "stationary visual pose and nominal CAD clearance; physical precision still requires hardware validation"}
