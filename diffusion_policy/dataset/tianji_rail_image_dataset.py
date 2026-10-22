"""Lazy RGB windows, source-level splits and gap-safe rail sampling."""

import copy
from pathlib import Path

import numpy as np
import torch
import zarr

from diffusion_policy.common.image_preprocessing import INTERPOLATIONS, preprocess_rgb
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.common.tianji_rail_pose import (
    ACTION_DIMS,
    RAW_ACTION_DIMS,
    EEF_ACTION_ROTATION_INDICES,
    EEF_OBS_ROTATION_INDICES,
    eef_action_to_6d,
    eef_observation_to_quaternion,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

class TianjiRailImageDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_path,
        shape_meta,
        horizon=16,
        n_obs_steps=2,
        seed=42,
        val_ratio=0.0,
        image_shape=(3, 240, 320),
        resize_enabled=True,
        resize_interpolation="area",
        sample_hz=None,
        action_space="eef",
        image_augmentation=None,
    ):
        self._validate_options(
            horizon,
            n_obs_steps,
            val_ratio,
            image_shape,
            resize_enabled,
            resize_interpolation,
            action_space,
            image_augmentation,
        )
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.image_shape = tuple(image_shape)
        self.resize_enabled = resize_enabled
        self.resize_interpolation = resize_interpolation
        self.interpolation = INTERPOLATIONS[resize_interpolation]
        self.image_augmentation = image_augmentation
        self.action_space = action_space
        self.sample_hz = sample_hz
        self.root = zarr.open_group(str(Path(dataset_path).expanduser()), mode="r")
        self.data = self.root["data"]
        self._validate_arrays()
        self._load_segments()
        self._validate_frequency()
        self._split_sources(seed, val_ratio)
        self.indices = self._indices(self.train_mask)
        if not len(self.indices):
            raise ValueError(
                "No training windows: need a continuous training segment of at least horizon frames"
            )

    @staticmethod
    def _validate_options(
        horizon,
        n_obs_steps,
        val_ratio,
        image_shape,
        resize_enabled,
        resize_interpolation,
        action_space,
        image_augmentation,
    ):
        if action_space not in ACTION_DIMS:
            raise ValueError("action_space must be eef or joint")
        if (
            type(horizon) is not int
            or type(n_obs_steps) is not int
            or not 1 <= n_obs_steps <= horizon
        ):
            raise ValueError("Require 1 <= n_obs_steps <= horizon, both integers")
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0,1)")
        if type(resize_enabled) is not bool:
            raise ValueError("resize_enabled must be boolean")
        if image_augmentation is not None and not callable(image_augmentation):
            raise ValueError("image_augmentation must be a callable transform or None")
        if resize_interpolation not in INTERPOLATIONS:
            raise ValueError("resize_interpolation must be area, bilinear or nearest")
        if (
            len(image_shape) != 3
            or image_shape[0] != 3
            or any(type(value) is not int or value <= 0 for value in image_shape)
        ):
            raise ValueError("image_shape must be positive [3,H,W] integers")

    def _validate_arrays(self):
        count = len(self.data["action"])
        dim = ACTION_DIMS[self.action_space]
        raw_dim = RAW_ACTION_DIMS[self.action_space]
        stored_space = self.root.attrs.get("action_space", "eef")
        if stored_space != self.action_space:
            raise ValueError(
                f"Dataset action_space={stored_space} does not match configured action_space={self.action_space}"
            )
        if tuple(self.shape_meta["action"]["shape"]) != (dim,) or self.data["action"].shape != (
            count,
            raw_dim,
        ):
            raise ValueError(
                f"Tianji rail action_space={self.action_space} requires {dim}-D policy actions "
                f"and {raw_dim}-D stored actions"
            )
        self.rgb_keys = []
        self.lowdim_keys = []
        for key, spec in self.shape_meta["obs"].items():
            if key == "action" or key not in self.data:
                raise ValueError(f"Missing or invalid observation: {key}")
            array = self.data[key]
            shape = tuple(spec["shape"])
            if len(array) != count:
                raise ValueError(f"{key}: inconsistent frame count")
            if spec.get("type", "low_dim") == "rgb":
                if (
                    shape != self.image_shape
                    or array.ndim != 4
                    or array.shape[-1] != 3
                    or array.dtype != np.dtype("uint8")
                ):
                    raise ValueError(
                        f"{key}: RGB shape_meta must match image_shape; source must be uint8 NHWC"
                    )
                if not self.resize_enabled and array.shape[1:] != shape[1:] + (3,):
                    raise ValueError(
                        f"{key}: resize disabled but source resolution differs from shape_meta"
                    )
                self.rgb_keys.append(key)
            elif spec.get("type", "low_dim") == "low_dim":
                stored_shape = shape
                if self.action_space == "eef" and key == "robot_eef_pose":
                    if shape != (14,):
                        raise ValueError("EEF robot_eef_pose must be 14-D xyz+quaternion (xyzw)")
                    stored_shape = (12,)
                if array.shape[1:] != stored_shape:
                    raise ValueError(f"{key}: stored observation shape mismatch")
                self.lowdim_keys.append(key)
            else:
                raise ValueError(f"{key}: unsupported observation type or shape")
        if not self.rgb_keys or not self.lowdim_keys:
            raise ValueError("Select RGB and low-dimensional observations in shape_meta")

    def _load_segments(self):
        meta = self.root["meta"]
        count = len(self.data["action"])
        episode_ends = np.asarray(meta["episode_ends"][:])
        if "segment_ends" not in meta and self.root.attrs.get("schema_version", 1) >= 2:
            raise ValueError("schema v2 requires segment_ends")
        self.segment_ends = np.asarray(meta.get("segment_ends", meta["episode_ends"])[:])
        for name, ends in (("episode_ends", episode_ends), ("segment_ends", self.segment_ends)):
            if (
                ends.ndim != 1
                or ends.dtype != np.dtype("int64")
                or not len(ends)
                or ends[0] <= 0
                or np.any(np.diff(ends) <= 0)
                or ends[-1] != count
            ):
                raise ValueError(f"Invalid {name}: must partition all frames")
        if not np.isin(episode_ends, self.segment_ends).all():
            raise ValueError("segment_ends must preserve episode boundaries")
        segments = meta.attrs.get("segments", [])
        if segments and len(segments) != len(self.segment_ends):
            raise ValueError("Segment metadata count differs from segment_ends")
        episode_ids = np.searchsorted(episode_ends, self.segment_ends)
        self.sources = (
            [item["source_episode"] for item in segments]
            if segments
            else [str(i) for i in episode_ids]
        )
        for episode in np.unique(episode_ids):
            if len({self.sources[i] for i in np.flatnonzero(episode_ids == episode)}) != 1:
                raise ValueError("Segments within a source episode must share source_episode")

    def _validate_frequency(self):
        # Historical checkpoint configs omit sample_hz; new runs validate both metadata and time.
        if self.sample_hz is None:
            return
        hz = self.sample_hz
        if (
            isinstance(hz, bool)
            or not isinstance(hz, (int, float))
            or not np.isfinite(hz)
            or hz <= 0
        ):
            raise ValueError("sample_hz must be finite and positive")
        declared_hz = self.root.attrs.get("target_fps")
        if (
            isinstance(declared_hz, bool)
            or not isinstance(declared_hz, (int, float))
            or not np.isfinite(declared_hz)
            or not np.isclose(declared_hz, hz, rtol=0, atol=1e-8)
        ):
            raise ValueError(
                f"Dataset target_fps={declared_hz} does not match sample_hz={hz}; re-export the data"
            )
        count = len(self.data["action"])
        if "timestamp" not in self.data or self.data["timestamp"].shape != (count,):
            raise ValueError(
                "Sampling-frequency validation requires data/timestamp with one value per frame"
            )
        times = np.asarray(self.data["timestamp"][:], dtype=np.float64)
        if not np.isfinite(times).all():
            raise ValueError("Nonfinite data/timestamp")
        within_segment = np.ones(count - 1, dtype=bool)
        within_segment[self.segment_ends[:-1] - 1] = False
        if not np.allclose(np.diff(times)[within_segment], 1 / hz, rtol=0, atol=1e-6):
            raise ValueError(f"Within-segment timestamp intervals do not match sample_hz={hz}")

    def _split_sources(self, seed, val_ratio):
        unique = list(dict.fromkeys(self.sources))
        self.n_source_episodes = len(unique)
        if val_ratio > 0 and len(unique) < 2:
            raise ValueError("val_ratio > 0 requires at least two independent source episodes")
        source_val = get_val_mask(len(unique), val_ratio, seed)
        by_source = dict(zip(unique, source_val))
        self.val_mask = np.asarray([by_source[source] for source in self.sources])
        self.train_mask = ~self.val_mask

    def _indices(self, mask):
        starts = np.r_[0, self.segment_ends[:-1]]
        windows = [
            (start, segment)
            for segment, (low, high) in enumerate(zip(starts, self.segment_ends))
            if mask[segment]
            for start in range(int(low), int(high) - self.horizon + 1)
        ]
        return np.asarray(windows, dtype=np.int64).reshape(-1, 2)

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.indices = self._indices(self.val_mask)
        result.image_augmentation = None
        return result

    def _training_values(self, key):
        starts = np.r_[0, self.segment_ends[:-1]]
        values = np.concatenate(
            [
                self.data[key][low:high]
                for low, high, keep in zip(starts, self.segment_ends, self.train_mask)
                if keep
            ]
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{key}: nonfinite training values")
        return self._policy_values(key, values)

    def _policy_values(self, key, values):
        values = np.asarray(values, dtype=np.float32)
        if self.action_space == "eef":
            if key == "action":
                return eef_action_to_6d(values)
            if key == "robot_eef_pose":
                return eef_observation_to_quaternion(values)
        return values

    def get_normalizer(self, **kwargs):
        result = LinearNormalizer()
        result.fit(
            {key: self._training_values(key) for key in self.lowdim_keys + ["action"]},
            last_n_dims=1,
            **kwargs,
        )
        if self.action_space == "eef":
            # Quaternions and rotation-6D already lie in [-1, 1], as in DP.
            for key, indices in (("robot_eef_pose", EEF_OBS_ROTATION_INDICES),
                                 ("action", EEF_ACTION_ROTATION_INDICES)):
                if key in self.lowdim_keys or key == "action":
                    params = result[key].params_dict
                    with torch.no_grad():
                        params["scale"][list(indices)] = 1
                        params["offset"][list(indices)] = 0
        for key in self.rgb_keys:
            result[key] = get_image_range_normalizer()
        return result

    def get_all_actions(self):
        return torch.from_numpy(self._training_values("action"))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start = int(self.indices[index, 0])
        stop = start + self.n_obs_steps
        obs = {}
        for key in self.rgb_keys:
            images = preprocess_rgb(
                self.data[key][start:stop],
                self.image_shape,
                resize_enabled=self.resize_enabled,
                interpolation=self.resize_interpolation,
            )
            obs[key] = torch.from_numpy(images)
            if self.image_augmentation is not None:
                # One draw per camera history preserves temporal consistency.
                obs[key] = self.image_augmentation(obs[key])
        for key in self.lowdim_keys:
            obs[key] = torch.from_numpy(self._policy_values(key, self.data[key][start:stop]))
        action = torch.from_numpy(
            self._policy_values("action", self.data["action"][start : start + self.horizon])
        )
        if not torch.isfinite(action).all() or any(
            not torch.isfinite(value).all() for value in obs.values()
        ):
            raise ValueError(f"Nonfinite sample at frame {start}")
        return {"obs": obs, "action": action}
