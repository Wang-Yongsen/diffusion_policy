"""Lazy RGB loading, source-level splits and gap-safe windows for Tianji rail."""
import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import zarr

from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

# eef: arm flange xyz+rotvec 6+6, hands 20+20; joint: arms 7+7, hands 20+20.
ACTION_DIMS = {"eef": 52, "joint": 54}


class TianjiRailImageDataset(BaseImageDataset):
    def __init__(self, dataset_path, shape_meta, horizon=16, n_obs_steps=2,
                 seed=42, val_ratio=0., image_shape=(3, 240, 320),
                 resize_enabled=True, resize_interpolation="area", sample_hz=None,
                 action_space="eef"):
        if action_space not in ACTION_DIMS:
            raise ValueError("action_space must be eef or joint")
        if type(horizon) is not int or type(n_obs_steps) is not int or not 1 <= n_obs_steps <= horizon:
            raise ValueError("Require 1 <= n_obs_steps <= horizon, both integers")
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0,1)")
        if type(resize_enabled) is not bool:
            raise ValueError("resize_enabled must be boolean")
        interpolation = {"area": cv2.INTER_AREA, "bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}
        if resize_interpolation not in interpolation:
            raise ValueError("resize_interpolation must be area, bilinear or nearest")
        if (len(image_shape) != 3 or image_shape[0] != 3
                or any(type(v) is not int or v <= 0 for v in image_shape)):
            raise ValueError("image_shape must be positive [3,H,W] integers")
        self.root = zarr.open_group(str(Path(dataset_path).expanduser()), mode="r")
        self.data = self.root["data"]
        self.shape_meta = shape_meta
        self.horizon, self.n_obs_steps = horizon, n_obs_steps
        self.resize_enabled, self.interpolation = resize_enabled, interpolation[resize_interpolation]
        self.image_shape = tuple(image_shape)
        self.rgb_keys, self.lowdim_keys = [], []
        count = len(self.data["action"])
        dim = ACTION_DIMS[action_space]
        stored_space = self.root.attrs.get("action_space", "eef")
        if stored_space != action_space:
            raise ValueError(f"Dataset action_space={stored_space} does not match configured action_space={action_space}")
        if tuple(shape_meta["action"]["shape"]) != (dim,) or self.data["action"].shape != (count, dim):
            raise ValueError(f"Tianji rail action_space={action_space} requires {dim}-D actions")
        self.action_space = action_space
        for key, spec in shape_meta["obs"].items():
            if key == "action" or key not in self.data:
                raise ValueError(f"Missing or invalid observation: {key}")
            array = self.data[key]
            shape = tuple(spec["shape"])
            if len(array) != count:
                raise ValueError(f"{key}: inconsistent frame count")
            if spec.get("type", "low_dim") == "rgb":
                if (shape != self.image_shape or array.ndim != 4 or array.shape[-1] != 3
                        or array.dtype != np.dtype("uint8")):
                    raise ValueError(f"{key}: RGB shape_meta must match image_shape; source must be uint8 NHWC")
                if not resize_enabled and array.shape[1:] != shape[1:] + (3,):
                    raise ValueError(f"{key}: resize disabled but source resolution differs from shape_meta")
                self.rgb_keys.append(key)
            elif spec.get("type", "low_dim") == "low_dim" and array.shape[1:] == shape:
                self.lowdim_keys.append(key)
            else:
                raise ValueError(f"{key}: unsupported observation type or shape")
        if not self.rgb_keys or not self.lowdim_keys:
            raise ValueError("Select RGB and low-dimensional observations in shape_meta")

        meta = self.root["meta"]
        episode_ends = np.asarray(meta["episode_ends"][:])
        if "segment_ends" not in meta and self.root.attrs.get("schema_version", 1) >= 2:
            raise ValueError("schema v2 requires segment_ends")
        self.segment_ends = np.asarray(meta.get("segment_ends", meta["episode_ends"])[:])
        for name, ends in (("episode_ends", episode_ends), ("segment_ends", self.segment_ends)):
            if (ends.ndim != 1 or ends.dtype != np.dtype("int64") or not len(ends)
                    or ends[0] <= 0 or np.any(np.diff(ends) <= 0) or ends[-1] != count):
                raise ValueError(f"Invalid {name}: must partition all frames")
        if not np.isin(episode_ends, self.segment_ends).all():
            raise ValueError("segment_ends must preserve episode boundaries")
        # Older checkpoint configs omit sample_hz. New experiments explicitly
        # require both declared and measured sampling intervals to agree.
        self.sample_hz = sample_hz
        if sample_hz is not None:
            if (isinstance(sample_hz, bool) or not isinstance(sample_hz, (int, float))
                    or not np.isfinite(sample_hz) or sample_hz <= 0):
                raise ValueError("sample_hz must be finite and positive")
            declared_hz = self.root.attrs.get("target_fps")
            if (isinstance(declared_hz, bool) or not isinstance(declared_hz, (int, float))
                    or not np.isfinite(declared_hz) or not np.isclose(declared_hz, sample_hz, rtol=0, atol=1e-8)):
                raise ValueError(f"Dataset target_fps={declared_hz} does not match sample_hz={sample_hz}; re-export the data")
            if "timestamp" not in self.data or self.data["timestamp"].shape != (count,):
                raise ValueError("Sampling-frequency validation requires data/timestamp with one value per frame")
            times = np.asarray(self.data["timestamp"][:], dtype=np.float64)
            if not np.isfinite(times).all():
                raise ValueError("Nonfinite data/timestamp")
            intervals = np.diff(times)
            # Source time may reset at episode/segment boundaries.
            within_segment = np.ones(len(intervals), dtype=bool)
            within_segment[self.segment_ends[:-1] - 1] = False
            if not np.allclose(intervals[within_segment], 1 / sample_hz, rtol=0, atol=1e-6):
                raise ValueError(f"Within-segment timestamp intervals do not match sample_hz={sample_hz}")
        segments = meta.attrs.get("segments", [])
        if segments and len(segments) != len(self.segment_ends):
            raise ValueError("Segment metadata count differs from segment_ends")
        episode_ids = np.searchsorted(episode_ends, self.segment_ends)
        self.sources = [s["source_episode"] for s in segments] if segments else [str(i) for i in episode_ids]
        for episode in np.unique(episode_ids):
            if len({self.sources[i] for i in np.flatnonzero(episode_ids == episode)}) != 1:
                raise ValueError("Segments within a source episode must share source_episode")
        unique = list(dict.fromkeys(self.sources))
        self.n_source_episodes = len(unique)
        # Reject an impossible requested holdout rather than silently train on it.
        if val_ratio > 0 and len(unique) < 2:
            raise ValueError("val_ratio > 0 requires at least two independent source episodes")
        source_val = get_val_mask(len(unique), val_ratio, seed)
        self.val_mask = np.asarray([source_val[unique.index(s)] for s in self.sources])
        self.train_mask = ~self.val_mask
        self.indices = self._indices(self.train_mask)
        if not len(self.indices):
            raise ValueError("No training windows: need a continuous training segment of at least horizon frames")

    def _indices(self, mask):
        starts = np.r_[0, self.segment_ends[:-1]]
        windows = [(start, segment) for segment, (low, high) in enumerate(zip(starts, self.segment_ends))
                   if mask[segment] for start in range(int(low), int(high) - self.horizon + 1)]
        return np.asarray(windows, dtype=np.int64).reshape(-1, 2)

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.indices = self._indices(self.val_mask)
        return result

    def _training_values(self, key):
        starts = np.r_[0, self.segment_ends[:-1]]
        values = np.concatenate([self.data[key][low:high] for low, high, keep
                                 in zip(starts, self.segment_ends, self.train_mask) if keep])
        if not np.isfinite(values).all():
            raise ValueError(f"{key}: nonfinite training values")
        return values.astype(np.float32)

    def get_normalizer(self, **kwargs):
        result = LinearNormalizer()
        result.fit({key: self._training_values(key) for key in self.lowdim_keys + ["action"]},
                   last_n_dims=1, **kwargs)
        for key in self.rgb_keys:
            result[key] = get_image_range_normalizer()
        return result

    def get_all_actions(self):
        return torch.from_numpy(self._training_values("action"))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start = int(self.indices[index, 0])
        obs = {}
        for key in self.rgb_keys:
            images = self.data[key][start:start+self.n_obs_steps]
            if self.resize_enabled:
                images = np.stack([cv2.resize(image, (self.image_shape[2], self.image_shape[1]), interpolation=self.interpolation)
                                   for image in images])
            obs[key] = torch.from_numpy(np.ascontiguousarray(np.moveaxis(images, -1, 1), dtype=np.float32) / 255.)
        for key in self.lowdim_keys:
            obs[key] = torch.from_numpy(self.data[key][start:start+self.n_obs_steps].astype(np.float32))
        action = torch.from_numpy(self.data["action"][start:start+self.horizon].astype(np.float32))
        if not torch.isfinite(action).all() or any(not torch.isfinite(x).all() for x in obs.values()):
            raise ValueError(f"Nonfinite sample at frame {start}")
        return {"obs": obs, "action": action}
