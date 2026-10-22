"""Deterministic image conversion shared by datasets and deployment."""

import cv2
import numpy as np

from diffusion_policy.common.tianji_rail_pose import eef_observation_to_quaternion

INTERPOLATIONS = {
    "area": cv2.INTER_AREA,
    "bilinear": cv2.INTER_LINEAR,
    "nearest": cv2.INTER_NEAREST,
}


def preprocess_rgb(images, shape, *, resize_enabled=True, interpolation="area"):
    """Convert uint8 NHWC RGB to contiguous float32 NCHW in [0, 1]."""
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("Require RGB uint8 NHWC")
    if resize_enabled:
        images = np.stack(
            [
                cv2.resize(image, (shape[2], shape[1]), interpolation=INTERPOLATIONS[interpolation])
                for image in images
            ]
        )
    return np.ascontiguousarray(np.moveaxis(images, -1, 1), dtype=np.float32) / 255.0


def preprocess_observation(obs, shape_meta, dataset_config):
    """Apply the checkpoint's deterministic input contract without augmentation."""
    if not set(shape_meta["obs"]) <= set(obs):
        raise ValueError("Checkpoint observation keys are not provided by the real rig")
    result = {}
    for key, spec in shape_meta["obs"].items():
        values = np.asarray(obs[key])
        shape = tuple(spec["shape"])
        if spec.get("type", "low_dim") == "rgb":
            result[key] = preprocess_rgb(
                values,
                shape,
                resize_enabled=dataset_config.get("resize_enabled", True),
                interpolation=dataset_config.get("resize_interpolation", "area"),
            )
        else:
            if key == "robot_eef_pose" and shape == (14,) and values.shape[-1] == 12:
                values = eef_observation_to_quaternion(values.astype(np.float32))
            result[key] = values.astype(np.float32)
        if result[key].shape[1:] != shape or not np.isfinite(result[key]).all():
            raise ValueError(f"{key}: shape/finite check failed against checkpoint")
    return result
