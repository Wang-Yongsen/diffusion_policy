"""Deployment configuration and explicit overrides."""

from pathlib import Path

import numpy as np
import yaml

from diffusion_policy.common.paths import PROJECT_ROOT

from .runtime import INFERENCE_MODES, SIDES


def apply_overrides(config, overrides):
    """Apply KEY=VALUE items (YAML values, dotted keys for nesting); only existing keys may be set."""
    for item in overrides:
        key, separator, text = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Override must be KEY=VALUE: {item}")
        *parents, leaf = key.split(".")
        node = config
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, dict) or leaf not in node:
            raise ValueError(f"Unknown deployment config key: {key}")
        node[leaf] = yaml.safe_load(text)
    return config


def load_deployment_config(path, overrides=()):
    with Path(path).open() as file:
        config = apply_overrides(yaml.safe_load(file), overrides)
    for key, value in config.items():
        if key == "trial_timeout_s" and value is None:
            continue
        if key.endswith(("_s", "_hz", "_rad", "_m", "_m_s", "_rad_s")):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
    if config["policy_hz"] not in (10, 30) or config["arm_hz"] != 200:
        raise ValueError("Deployment requires policy_hz in (10, 30) and arm_hz=200")
    if config.get("inference_mode") not in INFERENCE_MODES:
        raise ValueError(f"inference_mode must be one of {INFERENCE_MODES}")
    digest = config["expected_model_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("expected_model_sha256 must be a lowercase SHA256 digest")
    for key in ("replanning_steps", "record_queue_capacity"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for side in SIDES:
        box = config["workspace"][side]
        if box is not None:
            low, high = np.asarray(box["min"]), np.asarray(box["max"])
            if (
                low.shape != (3,)
                or high.shape != (3,)
                or not np.isfinite([low, high]).all()
                or np.any(low >= high)
            ):
                raise ValueError(f"workspace.{side} needs finite 3-D min < max")
    for key in ("arm_config", "hand_config", "camera_config"):
        config[key] = str((PROJECT_ROOT / config[key]).resolve())
    config["output_dir"] = str((PROJECT_ROOT / config["output_dir"]).resolve())
    return config
