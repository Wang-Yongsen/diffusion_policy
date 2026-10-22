"""Compare checkpoint predictions on explicit datasets and a recorded observation."""

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.file_util import file_sha256
from diffusion_policy.common.image_preprocessing import preprocess_observation
from diffusion_policy.common.policy_evaluation import evaluation_context
from diffusion_policy.env_runner.tianji_rail_offline_runner import action_errors
from diffusion_policy.real_world.tianji.checkpoint import load_policy


def compare_observations(left, right, shape_meta):
    """Report input differences in pixel values and native low-dimensional units."""
    report = {}
    for key, spec in shape_meta["obs"].items():
        first, second = np.asarray(left[key]), np.asarray(right[key])
        if first.shape != second.shape:
            report[key] = {"left_shape": list(first.shape), "right_shape": list(second.shape)}
            continue
        scale = 255.0 if spec.get("type", "low_dim") == "rgb" else 1.0
        difference = np.abs(first.astype(np.float64) - second.astype(np.float64)) * scale
        report[key] = {"mae": float(difference.mean()), "max": float(difference.max())}
    return report


def predict(policy, observation, seed):
    tensors = {
        key: torch.from_numpy(value).unsqueeze(0).to(policy.device)
        for key, value in observation.items()
    }
    with evaluation_context(policy, seed=seed):
        policy.reset()
        return policy.predict_action(tensors)["action"][0].cpu().numpy()


def inspect_dataset(policy, cfg, path, *, start, chunks, seed):
    dataset = hydra.utils.instantiate(
        cfg.task.dataset,
        dataset_path=str(path),
        val_ratio=0.0,
        image_augmentation=None,
    )
    if not 0 <= start < len(dataset):
        raise ValueError(f"Window index {start} is outside {path} ({len(dataset)} windows)")
    windows = []
    first_observation = None
    for index in range(start, min(len(dataset), start + chunks)):
        sample = dataset[index]
        observation = {key: value.numpy() for key, value in sample["obs"].items()}
        if first_observation is None:
            first_observation = observation
        prediction = predict(policy, observation, seed)
        offset = dataset.n_obs_steps - 1
        target = sample["action"][offset : offset + len(prediction)].numpy()
        errors = action_errors(prediction, target, dataset.action_space)
        windows.append(
            {
                "start_frame": int(dataset.indices[index, 0]),
                "errors": {key: float(value.mean()) for key, value in errors.items()},
                "first_action": prediction[0].tolist(),
                "last_action": prediction[-1].tolist(),
            }
        )
    return {"path": str(Path(path).resolve()), "windows": windows}, first_observation


def run(args):
    if args.chunks < 1 or args.start < 0:
        raise ValueError("chunks must be positive and start must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    policy, cfg, state_key = load_policy(args.checkpoint, args.device)
    inputs = {}
    observations = {}
    for name, path in (("train", args.train_zarr), ("validation", args.val_zarr)):
        inputs[name], observations[name] = inspect_dataset(
            policy,
            cfg,
            path,
            start=args.start,
            chunks=args.chunks,
            seed=args.seed,
        )
    report = {
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "state_dict": state_key,
        "device": args.device,
        "seed": args.seed,
        "action_space": cfg.task.dataset.get("action_space", "eef"),
        "datasets": inputs,
        "observation_difference": compare_observations(
            observations["train"],
            observations["validation"],
            cfg.task.shape_meta,
        ),
    }
    if args.reference_npz is not None:
        with np.load(args.reference_npz) as raw:
            reference = preprocess_observation(raw, cfg.task.shape_meta, cfg.task.dataset)
        report["reference_difference"] = compare_observations(
            observations["train"],
            reference,
            cfg.task.shape_meta,
        )
        report["reference_action"] = predict(policy, reference, args.seed).tolist()
    args.output.mkdir(parents=True)
    report["checkpoint_config"] = OmegaConf.to_container(cfg, resolve=True)
    target = args.output / "obs_gap_results.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"观测与预测对比：{target}")
    return 0
