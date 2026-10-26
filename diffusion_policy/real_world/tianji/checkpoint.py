"""Load a rail checkpoint and validate its inference contract."""

import time

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.image_preprocessing import preprocess_observation
from diffusion_policy.dataset.tianji_rail_image_dataset import ACTION_DIMS


def checkpoint_action_space(cfg):
    """Checkpoints trained before joint support have no action_space and are EEF."""
    return cfg.task.dataset.get("action_space", "eef")


def load_policy(checkpoint, device):
    # Local checkpoints contain the trusted training config and normalizer state.
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    torch.manual_seed(int(cfg.training.seed))
    np.random.seed(int(cfg.training.seed))
    action_space = checkpoint_action_space(cfg)
    if action_space not in ACTION_DIMS:
        raise ValueError(f"Unsupported checkpoint action_space={action_space}")
    arm_obs = {"eef": ("robot_eef_pose", 14), "joint": ("robot_joint", 14)}[action_space]
    if action_space == "eef" and list(cfg.task.shape_meta.action.shape) == [52]:
        raise ValueError(
            "Legacy 52-D EEF checkpoint uses rotvec observations/actions; "
            "retrain with 14-D quaternion observations and 58-D rotation-6D actions"
        )
    expected = {
        "camera_0": (3,),
        "camera_1": (3,),
        "camera_2": (3,),
        arm_obs[0]: (arm_obs[1],),
        "hand_joint": (40,),
    }
    dim = ACTION_DIMS[action_space]
    if set(cfg.task.shape_meta.obs) != set(expected) or list(
        cfg.task.shape_meta.action.shape
    ) != [dim]:
        raise ValueError(
            f"action_space={action_space} checkpoint must use three RGB + {arm_obs[1]}-D "
            f"{arm_obs[0]} + 40-D hand_joint / {dim}-D actions"
        )
    for key, prefix in expected.items():
        if tuple(cfg.task.shape_meta.obs[key].shape)[: len(prefix)] != prefix:
            raise ValueError(f"Checkpoint shape mismatch: {key}")
    if int(cfg.get("n_latency_steps", 0)) != 0:
        raise ValueError("Only checkpoints with n_latency_steps=0 are supported")
    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][key])
    policy.to(device).eval()
    return policy, cfg, key


def validate_policy_frequency(cfg, policy_hz):
    """Historical rail checkpoints without frequency metadata were trained at 30 Hz."""
    dataset_hz = cfg.task.dataset.get("sample_hz", None)
    checkpoint_hz = cfg.task.get("sample_hz", dataset_hz if dataset_hz is not None else 30)
    values = [checkpoint_hz, policy_hz] + ([] if dataset_hz is None else [dataset_hz])
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError(
            "Checkpoint and deployment sampling frequencies must be finite and positive"
        )
    if any(not np.isclose(value, checkpoint_hz, rtol=0, atol=1e-8) for value in values):
        raise ValueError(
            f"Checkpoint sample_hz={checkpoint_hz} does not match deployment policy_hz={policy_hz} "
            "or dataset sample_hz; train a model on matching-frequency data"
        )
    return float(checkpoint_hz)


def check_inference(policy, cfg, iterations=3, observation=None):
    obs = {} if observation is None else observation
    for key, spec in () if observation is not None else cfg.task.shape_meta.obs.items():
        if spec.get("type", "low_dim") == "rgb":
            obs[key] = np.zeros((policy.n_obs_steps, 480, 640, 3), np.uint8)
        else:
            obs[key] = np.zeros((policy.n_obs_steps, *spec.shape), np.float32)
            if key == "robot_eef_pose":
                obs[key][:, [6, 13]] = 1  # Identity quaternions in xyzw order.
    dim = ACTION_DIMS[checkpoint_action_space(cfg)]
    prepared = preprocess_observation(obs, cfg.task.shape_meta, cfg.task.dataset)
    tensor_obs = {
        key: torch.from_numpy(value).unsqueeze(0).to(policy.device)
        for key, value in prepared.items()
    }
    durations = []
    with torch.inference_mode():
        for i in range(iterations + 1):
            start = time.monotonic_ns()
            result = policy.predict_action(tensor_obs)["action"].detach().cpu().numpy()
            if result.shape != (1, policy.n_action_steps, dim) or not np.isfinite(result).all():
                raise ValueError("Checkpoint prediction shape/finite check failed")
            if i:
                durations.append((time.monotonic_ns() - start) / 1e6)
    policy.reset()
    return {
        "batch_size": 1,
        "n_obs_steps": policy.n_obs_steps,
        "n_action_steps": policy.n_action_steps,
        "image_shapes": {
            key: list(value.shape)
            for key, value in prepared.items()
            if key.startswith("camera")
        },
        "inference_ms": durations,
        "p95_ms": float(np.percentile(durations, 95)),
        "scope": ("dataset observations" if observation is not None else "synthetic images")
        + ", checkpoint preprocessing/normalizer/EMA, no device or end-to-end latency claim",
    }
