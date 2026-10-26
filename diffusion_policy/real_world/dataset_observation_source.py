"""Raw training/validation observations on a step-locked dataset clock."""

from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import zarr

from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.tianji_rail_image_dataset import (
    ACTION_DIMS,
    TianjiRailImageDataset,
)


class DatasetObservationSource:
    """One continuous segment; advance only after a completed action chunk.

    Full datasets use the checkpoint's source-level split; episode caches recover
    membership from original source provenance. Both retain the full training
    horizon. No measured state is replaced in the device env.
    """

    def __init__(
        self,
        cfg,
        path,
        *,
        split=None,
        mode="auto",
        episode=None,
        segment=None,
        max_windows=None,
        n_obs_steps,
        n_action_steps,
    ):
        if split not in (None, "train", "val"):
            raise ValueError("Dataset split must be train or val")
        if mode not in ("auto", "full", "episode"):
            raise ValueError("Dataset mode must be auto, full or episode")
        if max_windows is not None and (type(max_windows) is not int or max_windows < 1):
            raise ValueError("dataset-max-windows must be a positive integer")
        dataset_cfg = OmegaConf.to_container(cfg.task.dataset, resolve=True)
        dataset_cfg["dataset_path"] = str(Path(path).expanduser().resolve())
        root = zarr.open_group(dataset_cfg["dataset_path"], mode="r")
        policy_space = dataset_cfg.get("action_space", "eef")
        stored_space = root.attrs.get("action_space", "eef")
        if policy_space not in ACTION_DIMS or stored_space not in ACTION_DIMS:
            raise ValueError("Policy and dataset action_space must be eef or joint")
        if tuple(dataset_cfg["shape_meta"]["action"]["shape"]) != (ACTION_DIMS[policy_space],):
            raise ValueError("Checkpoint action shape does not match its action_space")
        # This source supplies observations only. Validate the stored action
        # labels against their own schema, while selecting observation keys
        # from the checkpoint. Shared caches may contain both joint and EEF
        # feedback; their unused labels must not select the deployed controller.
        # dataset_cfg is a resolved deep copy; model config/normalizer stay intact.
        dataset_cfg["action_space"] = stored_space
        dataset_cfg["shape_meta"]["action"]["shape"] = [ACTION_DIMS[stored_space]]
        if stored_space == "joint" and "robot_eef_pose" in dataset_cfg["shape_meta"]["obs"]:
            dataset_cfg["shape_meta"]["obs"]["robot_eef_pose"]["shape"] = [12]
        single = root["meta/episode_ends"].shape == (1,)
        provenance = root.attrs.get("episode_export")
        if mode == "auto":
            mode = "episode" if single else "full"
        if mode == "episode" and not single:
            raise ValueError("Dataset mode=episode requires exactly one source episode")
        if mode == "full" and provenance is not None:
            raise ValueError("An episode export must use dataset mode=episode or auto")
        original_seed, original_ratio = dataset_cfg.get("seed", 42), dataset_cfg.get(
            "val_ratio", 0
        )
        verified = mode == "full"
        if mode == "episode":
            if provenance is not None:
                if (
                    not isinstance(provenance, dict)
                    or provenance.get("version") != 1
                    or provenance.get("complete") is not True
                ):
                    raise ValueError("Invalid or incomplete episode export provenance")
                sources = provenance.get("source_episodes")
                name = provenance.get("source_episode")
                if (
                    not isinstance(sources, list)
                    or not sources
                    or not all(isinstance(s, str) for s in sources)
                    or len(set(sources)) != len(sources)
                    or name not in sources
                ):
                    raise ValueError(
                        "Invalid original source order in episode export provenance"
                    )
                start, end, index = (
                    provenance.get(key)
                    for key in (
                        "source_start_frame",
                        "source_end_frame",
                        "source_episode_index",
                    )
                )
                segments = provenance.get("source_segment_indices")
                if (
                    any(type(value) is not int for value in (start, end, index))
                    or start < 0
                    or end <= start
                    or index < 0
                    or not isinstance(segments, list)
                    or not segments
                    or any(type(i) is not int or i < 0 for i in segments)
                    or any(a >= b for a, b in zip(segments, segments[1:]))
                ):
                    raise ValueError(
                        "Invalid original frame or episode indices in episode export provenance"
                    )
                selected_val = get_val_mask(len(sources), original_ratio, original_seed)[
                    sources.index(name)
                ]
                original_split = "val" if selected_val else "train"
                if split is not None and split != original_split:
                    raise ValueError(
                        f"Episode {name} belongs to split={original_split}, not requested split={split}"
                    )
                split, verified = original_split, True
            elif split is not None:
                raise ValueError(
                    "Cannot verify train/val for an episode without export provenance; omit --dataset-split"
                )
            else:
                split = "unknown"
            # Validate shapes, clocks and boundaries through the existing Dataset,
            # but never try to partition a single downloaded episode again.
            dataset_cfg["val_ratio"] = 0
        else:
            split = split or "train"
        dataset = hydra.utils.instantiate(dataset_cfg)
        if not isinstance(dataset, TianjiRailImageDataset):
            raise ValueError("Dataset observations require TianjiRailImageDataset")
        if (
            dataset.n_obs_steps != n_obs_steps
            or not 1 <= n_action_steps <= dataset.horizon - n_obs_steps + 1
        ):
            raise ValueError("Checkpoint and dataset observation/action windows do not match")
        if mode == "episode" and provenance is not None:
            if (
                set(dataset.sources) != {provenance["source_episode"]}
                or len(dataset.data["action"])
                != provenance.get("source_end_frame", 0)
                - provenance.get("source_start_frame", 0)
                or len(dataset.segment_ends)
                != len(provenance.get("source_segment_indices", []))
            ):
                raise ValueError(
                    "Episode export provenance does not match its frames, sources or segments"
                )
        self.dataset = (
            dataset.get_validation_dataset() if mode == "full" and split == "val" else dataset
        )
        available = sorted(set(int(value) for value in self.dataset.indices[:, 1]))
        if not available:
            raise ValueError(
                f"No independent {split} windows in the checkpoint's dataset split"
            )
        if episode is not None:
            episode_ends = np.asarray(dataset.root["meta/episode_ends"][:])
            episode_ids = np.searchsorted(episode_ends, dataset.segment_ends)
            if str(episode).isdigit():
                if mode == "episode" and provenance is not None:
                    available = (
                        available if provenance["source_episode_index"] == int(episode) else []
                    )
                else:
                    available = [i for i in available if int(episode_ids[i]) == int(episode)]
            else:
                available = [i for i in available if dataset.sources[i] == str(episode)]
            if not available:
                raise ValueError(f"Episode {episode} has no windows in split={split}")
        if segment is not None:
            if segment not in available:
                raise ValueError(
                    f"Segment {segment} has no windows in split={split} for the selected episode"
                )
            chosen = segment
        else:
            source = dataset.sources[available[0]]
            candidates = [i for i in available if dataset.sources[i] == source]
            chosen = max(
                candidates,
                key=lambda i: int(dataset.segment_ends[i])
                - (0 if i == 0 else int(dataset.segment_ends[i - 1])),
            )
        low = 0 if chosen == 0 else int(dataset.segment_ends[chosen - 1])
        self.starts = [
            int(start)
            for start, index in self.dataset.indices
            if index == chosen and (int(start) - low) % n_action_steps == 0
        ][:max_windows]
        if not self.starts:
            raise ValueError("No complete dataset observation windows in the selected segment")
        self.n_obs_steps, self.n_action_steps = n_obs_steps, n_action_steps
        self.keys = dataset.rgb_keys + dataset.lowdim_keys
        self.metadata = {
            "kind": "dataset",
            "dataset_mode": mode,
            "dataset": dataset_cfg["dataset_path"],
            "split": split,
            "split_verified": verified,
            "split_seed": original_seed,
            "val_ratio": original_ratio,
            "source_episode": dataset.sources[chosen],
            "segment": chosen,
            "segment_start_frame": low,
            "segment_end_frame": int(dataset.segment_ends[chosen]),
            "first_window_start": self.starts[0],
            "last_window_start": self.starts[-1],
            "windows": len(self.starts),
            "window_stride": n_action_steps,
            "n_obs_steps": n_obs_steps,
            "n_action_steps": n_action_steps,
            "action_space": policy_space,
            "dataset_action_space": stored_space,
            "dataset_actions_used": False,
            "input_clock": "dataset timestamps; anchor_ns is the host monotonic read time",
        }
        if mode == "episode" and provenance is not None:
            self.metadata["episode_export"] = provenance
            self.metadata["source_segment"] = provenance["source_segment_indices"][chosen]
        self.reset()

    def reset(self):
        self.index = 0

    @property
    def exhausted(self):
        return self.index >= len(self.starts)

    def read(self):
        if self.exhausted:
            raise StopIteration("Dataset observation windows completed")
        start = self.starts[self.index]
        stop = start + self.n_obs_steps
        obs = {
            key: np.array(self.dataset.data[key][start:stop], copy=True) for key in self.keys
        }
        alignment = {
            "observation_source": "dataset",
            "split": self.metadata["split"],
            "source_episode": self.metadata["source_episode"],
            "segment": self.metadata["segment"],
            "dataset_window_index": self.index,
            "dataset_frames": list(range(start, stop)),
        }
        if "timestamp" in self.dataset.data:
            alignment["dataset_timestamp_s"] = np.asarray(
                self.dataset.data["timestamp"][start:stop]
            ).tolist()
        if "episode_export" in self.metadata:
            offset = self.metadata["episode_export"]["source_start_frame"]
            alignment["source_dataset_frames"] = list(range(offset + start, offset + stop))
        return obs, time.monotonic_ns(), alignment

    def advance(self):
        if self.exhausted:
            raise StopIteration("Dataset observation windows completed")
        self.index += 1
