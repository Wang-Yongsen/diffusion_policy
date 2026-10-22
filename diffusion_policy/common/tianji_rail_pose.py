"""Rail policy representations and the rotvec recording/controller boundary.

Policy observations concatenate two xyz + xyzw quaternion poses (14-D).
Policy actions concatenate two xyz + rotation-6D poses, then two 20-D hands
(58-D). Rotation-6D flattens the first two matrix rows, matching DP's
PyTorch3D RotationTransformer. Stored/controller EEF actions remain 52-D.
"""

import numpy as np
from scipy.spatial.transform import Rotation

ACTION_DIMS = {"eef": 58, "joint": 54}
RAW_ACTION_DIMS = {"eef": 52, "joint": 54}
EEF_OBS_ROTATION_INDICES = (3, 4, 5, 6, 10, 11, 12, 13)
EEF_ACTION_ROTATION_INDICES = (3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17)


def _vector(value, dim):
    value = np.asarray(value)
    if value.ndim < 1 or value.shape[-1] != dim or not np.isfinite(value).all():
        raise ValueError(f"Require finite (..., {dim}) values")
    if not np.issubdtype(value.dtype, np.floating):
        value = value.astype(np.float64)
    return value


def eef_observation_to_quaternion(value):
    """Convert two recorded xyz+rotvec poses to xyz+quaternion (xyzw)."""
    value = _vector(value, 12)
    poses = value.reshape(value.shape[:-1] + (2, 6))
    quaternion = Rotation.from_rotvec(poses[..., 3:6].reshape(-1, 3)).as_quat(canonical=True)
    quaternion = quaternion.reshape(poses.shape[:-1] + (4,))
    return np.concatenate((poses[..., :3], quaternion), axis=-1).reshape(
        value.shape[:-1] + (14,)
    ).astype(value.dtype)


def eef_action_to_6d(value):
    """Encode recorded/controller absolute targets for diffusion training."""
    value = _vector(value, 52)
    poses = value[..., :12].reshape(value.shape[:-1] + (2, 6))
    matrix = Rotation.from_rotvec(poses[..., 3:6].reshape(-1, 3)).as_matrix()
    rotation = matrix[:, :2, :].reshape(poses.shape[:-1] + (6,))
    arms = np.concatenate((poses[..., :3], rotation), axis=-1).reshape(
        value.shape[:-1] + (18,)
    )
    return np.concatenate((arms, value[..., 12:]), axis=-1).astype(value.dtype)


def rotation_6d_to_matrix(value):
    """Project predicted row vectors onto SO(3) with Gram-Schmidt."""
    value = _vector(value, 6).astype(np.float64)
    first, second = value[..., :3], value[..., 3:]
    norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate 6D rotation: first row has zero length")
    first = first / norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate 6D rotation: rows are collinear")
    second = second / norm
    return np.stack((first, second, np.cross(first, second)), axis=-2)


def eef_action_to_rotvec(value):
    """Decode 58-D policy targets to the 52-D controller/trace contract."""
    value = _vector(value, 58)
    poses = value[..., :18].reshape(value.shape[:-1] + (2, 9))
    matrix = rotation_6d_to_matrix(poses[..., 3:9]).reshape(-1, 3, 3)
    rotation = Rotation.from_matrix(matrix).as_rotvec().reshape(poses.shape[:-1] + (3,))
    arms = np.concatenate((poses[..., :3], rotation), axis=-1).reshape(
        value.shape[:-1] + (12,)
    )
    return np.concatenate((arms, value[..., 18:]), axis=-1)


def policy_actions_to_controller(value, action_space="eef"):
    """Decode model output before controller validation, scheduling and tracing."""
    if action_space == "eef":
        return eef_action_to_rotvec(value)
    if action_space == "joint":
        return _vector(value, ACTION_DIMS["joint"]).astype(np.float64)
    raise ValueError("action_space must be eef or joint")
