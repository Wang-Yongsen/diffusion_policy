"""Small rail data and configuration fixtures shared by offline regressions."""

from pathlib import Path

import numpy as np
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.dataset.tianji_rail_image_dataset import TianjiRailImageDataset

CONFIG_DIR = Path(__file__).resolve().parents[1] / "diffusion_policy/config"


def write_dataset(path, *, legacy=False, action_space="eef"):
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(schema_version=1 if legacy else 2, action_space=action_space)
    data, meta = root.create_group("data"), root.create_group("meta")
    # First demonstration has a 20-frame segment and a 5-frame short segment.
    meta.array("episode_ends", np.asarray([20, 25, 45] if legacy else [25, 45], np.int64))
    if not legacy:
        meta.array("segment_ends", np.asarray([20, 25, 45], np.int64))
    meta.attrs["segments"] = [{"source_episode": s} for s in ("a", "a", "b")]
    timeline = np.arange(45, dtype=np.float32) / 100
    data.array("action", np.repeat(timeline[:, None], 52 if action_space == "eef" else 54, axis=1))
    data.array("robot_eef_pose", np.repeat(timeline[:, None], 12, axis=1))
    data.array("robot_joint", np.repeat(timeline[:, None], 14, axis=1))
    data.array("hand_joint", np.repeat(timeline[:, None], 40, axis=1))
    # Distinct channels and a spatial pattern expose transpose/resize errors.
    images = np.zeros((45, 12, 16, 3), np.uint8)
    images[..., 0], images[..., 1], images[..., 2] = 255, np.arange(16)[None, None, :], 50
    for key in ("camera_0", "camera_1", "camera_2"):
        data.array(key, images, chunks=(1, 12, 16, 3))
    return root


def make_dataset(path, **options):
    shape = options.pop("image_shape", (3, 6, 8))
    obs = {key: {"shape": shape, "type": "rgb"} for key in ("camera_0", "camera_1", "camera_2")}
    obs.update(robot_eef_pose={"shape": [14]}, hand_joint={"shape": [40]})
    return TianjiRailImageDataset(
        path, {"obs": obs, "action": {"shape": [58]}}, image_shape=shape, **options
    )


def rail_config(name, overrides=()):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=name, overrides=list(overrides))
        OmegaConf.resolve(cfg)
        return cfg
