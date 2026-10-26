from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
DP_ROOT = HERE.parents[1]


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(value, size, name="vector"):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected {size} finite values")
    return result


def transform(value, name="transform"):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{name}: expected finite 4x4 matrix")
    if (not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            or abs(np.linalg.det(T[:3, :3]) - 1) > 1e-5):
        raise ValueError(f"{name}: invalid rigid transform")
    return T


def matrix(pose):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(pose.orientation_xyzw).as_matrix()
    T[:3, 3] = pose.position_m
    return T


def pose(T, side="left"):
    from diffusion_policy.third_party.bimanual_teleop.types import Pose
    T = transform(T)
    return Pose(f"tianji_{side}_base", f"tianji_{side}_flange", tuple(T[:3, 3]),
                tuple(Rotation.from_matrix(T[:3, :3]).as_quat()))


def difference(a, b):
    a, b = transform(a), transform(b)
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()))


class PoseStabilityError(ValueError):
    """Valid poses disagree beyond the requested multi-frame tolerance."""


def mean_pose(poses, max_translation_m=.005, max_rotation_rad=.035):
    poses = np.array([transform(p) for p in poses])
    if not len(poses):
        raise ValueError("No pose observations")
    result = np.eye(4)
    result[:3, 3] = np.mean(poses[:, :3, 3], axis=0)
    result[:3, :3] = Rotation.from_matrix(poses[:, :3, :3]).mean().as_matrix()
    errors = [difference(p, result) for p in poses]
    if max(e[0] for e in errors) > max_translation_m or max(e[1] for e in errors) > max_rotation_rad:
        raise PoseStabilityError("Stationary pose observations disagree; reinitialize and inspect CAD direction")
    return result


def load_calibration(path, *, allow_test_data=False):
    data = read_json(path)
    if data.get("test_data", False) and not allow_test_data:
        raise ValueError("Synthetic calibration is forbidden for live use")
    if (data.get("parent_frame") != "tianji_left_base"
            or data.get("child_frame") != "head_color_optical" or data.get("unit") != "m"):
        raise ValueError("Calibration must map head_color_optical into tianji_left_base in metres")
    if not data.get("camera_serial"):
        raise ValueError("Calibration camera_serial is required")
    return data, transform(data["T_B_C"], "T_B_C")


def load_target(path, calibration, mesh_path, *, allow_test_data=False):
    data = read_json(path)
    if data.get("test_data", False) and not allow_test_data:
        raise ValueError("Synthetic target is forbidden for live use")
    if data.get("camera_serial") != calibration["camera_serial"] or data.get("unit") != "m":
        raise ValueError("Target camera/units do not match calibration")
    if data.get("mesh_sha256") != digest(mesh_path):
        raise ValueError("Target CAD mesh differs from current inference mesh")
    return data, transform(data["T_C_O_goal"], "T_C_O_goal")


def targets(T_B_C, T_C_O_goal, T_C_O_initial, T_B_F_initial, outward_axis_cad, offset_m=.02):
    X, goal, initial, flange = map(transform, (T_B_C, T_C_O_goal, T_C_O_initial, T_B_F_initial))
    axis = vector(outward_axis_cad, 3, "outward_axis_cad")
    if abs(np.linalg.norm(axis) - 1) > 1e-5 or not 0 < offset_m <= .1:
        raise ValueError("Require unit outward CAD axis and a positive SI preinsert offset")
    goal_B = X @ goal
    pre = goal_B.copy()
    pre[:3, 3] += offset_m * goal_B[:3, :3] @ axis
    grasp = np.linalg.inv(flange) @ X @ initial
    return {"T_B_O_goal": goal_B, "T_B_O_preinsert": pre, "T_F_O": grasp,
            "T_B_F_preinsert": pre @ np.linalg.inv(grasp), "outward_axis_B": goal_B[:3, :3] @ axis}


def interpolate_joints(history, stamp_ns, max_gap_s=.05):
    times = np.array([v[0] for v in history], dtype=np.int64)
    index = int(np.searchsorted(times, stamp_ns))
    if index < len(times) and times[index] == stamp_ns:
        return np.asarray(history[index][1], dtype=float).copy()
    if index == 0 or index == len(times):
        raise ValueError("Camera time is outside measured-state coverage; no extrapolation")
    if times[index] - times[index - 1] > max_gap_s * 1e9:
        raise ValueError("Measured-state gap exceeds synchronization budget")
    fraction = (stamp_ns - times[index - 1]) / (times[index] - times[index - 1])
    return (1 - fraction) * np.asarray(history[index - 1][1]) + fraction * np.asarray(history[index][1])


def load_config(path):
    import yaml
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    for key in ("arm_config", "hand_config", "mesh", "urdf", "manifest", "foundationpose_root",
                "calibration", "target", "hand_reference"):
        value = Path(data[key])
        data[key] = str((path.parent / value).resolve() if not value.is_absolute() else value)
    data["config_path"] = str(path)
    if data["end_stage"] != "preinsert":
        raise ValueError("This experiment ends at preinsert; insertion is not implemented")
    if not np.isclose(data["preinsert_offset_m"], .02):
        raise ValueError("This experiment uses a 20 mm preinsert offset")
    for key in ("joint_speed_rad_s", "joint_acceleration_rad_s2",
                "planning_timeout_s", "arm_tracking_error_rad", "hand_transition_s", "state_max_age_s",
                "preinsert_position_tolerance_m", "preinsert_rotation_tolerance_rad", "settle_s",
                "right_arm_drift_rad", "hand_shape_drift_rad", "visual_position_tolerance_m",
                "visual_rotation_tolerance_rad", "perception_timeout_s"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    axis = vector(data["outward_axis_cad"], 3, "outward_axis_cad")
    if abs(np.linalg.norm(axis) - 1) > 1e-5:
        raise ValueError("outward_axis_cad must be a unit vector")
    if not isinstance(data["perception_samples"], int) or not 1 <= data["perception_samples"] <= 10:
        raise ValueError("perception_samples must be an integer from 1 to 10")
    return data
