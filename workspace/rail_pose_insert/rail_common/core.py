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
    if (
        not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
        or abs(np.linalg.det(T[:3, :3]) - 1) > 1e-5
    ):
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
    return Pose(
        f"tianji_{side}_base",
        f"tianji_{side}_flange",
        tuple(T[:3, 3]),
        tuple(Rotation.from_matrix(T[:3, :3]).as_quat()),
    )


def difference(a, b):
    a, b = transform(a), transform(b)
    return (
        float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()),
    )


class PoseStabilityError(ValueError):
    """Valid poses disagree beyond the requested multi-frame tolerance."""


def mean_pose(poses, max_translation_m=0.005, max_rotation_rad=0.035):
    poses = np.array([transform(p) for p in poses])
    if not len(poses):
        raise ValueError("No pose observations")
    result = np.eye(4)
    result[:3, 3] = np.mean(poses[:, :3, 3], axis=0)
    result[:3, :3] = Rotation.from_matrix(poses[:, :3, :3]).mean().as_matrix()
    errors = [difference(p, result) for p in poses]
    if (
        max(e[0] for e in errors) > max_translation_m
        or max(e[1] for e in errors) > max_rotation_rad
    ):
        raise PoseStabilityError(
            "Stationary pose observations disagree; reinitialize and inspect CAD direction"
        )
    return result


def load_calibration(path, *, allow_test_data=False):
    data = read_json(path)
    if data.get("test_data", False) and not allow_test_data:
        raise ValueError("Synthetic calibration is forbidden for live use")
    if (
        data.get("parent_frame") != "tianji_left_base"
        or data.get("child_frame") != "head_color_optical"
        or data.get("unit") != "m"
    ):
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


def interpolate_joints(history, stamp_ns, max_gap_s=0.05):
    times = np.array([v[0] for v in history], dtype=np.int64)
    index = int(np.searchsorted(times, stamp_ns))
    if index < len(times) and times[index] == stamp_ns:
        return np.asarray(history[index][1], dtype=float).copy()
    if index == 0 or index == len(times):
        raise ValueError("Camera time is outside measured-state coverage; no extrapolation")
    if times[index] - times[index - 1] > max_gap_s * 1e9:
        raise ValueError("Measured-state gap exceeds synchronization budget")
    fraction = (stamp_ns - times[index - 1]) / (times[index] - times[index - 1])
    return (1 - fraction) * np.asarray(history[index - 1][1]) + fraction * np.asarray(
        history[index][1]
    )


def observation_poses(
    observation, calibration, mesh, *, allow_test_data=False, alignment=None, reference=None
):
    if observation.get("test_data", False) and not allow_test_data:
        raise ValueError("Synthetic observation cannot be used on hardware")
    if observation.get("replay_only", False) and not allow_test_data:
        raise ValueError("Offline replay observation cannot stand in for a fresh live measurement")
    if (
        observation.get("camera_serial") != calibration["camera_serial"]
        or observation.get("unit") != "m"
        or observation.get("frame") != "T_C_CAD"
        or observation.get("mesh_sha256") != digest(mesh)
    ):
        raise ValueError("Observation camera/CAD/coordinate convention mismatch")
    records = observation.get("records", [])
    if not records or any(not r.get("valid") for r in records):
        raise ValueError("No valid rail pose observations")
    times = [int(r["timestamp_ns"]) for r in records]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Pose times must increase")
    if not allow_test_data and any(
        r.get("timestamp_source") != "sdk_global_timestamp_mapped_to_host_monotonic"
        for r in records
    ):
        raise ValueError("Live pose timestamps must be mapped SDK GLOBAL_TIME")
    if alignment is not None:
        reference = records[0]["T_C_CAD"] if reference is None else reference
        records = [alignment.align_record(r, reference) for r in records]
    matrices = [transform(r["T_C_CAD"]) for r in records]
    mean_pose(matrices)  # Align first; still reject genuine stationary drift.
    return records, matrices
