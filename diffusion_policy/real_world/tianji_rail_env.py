"""Compatibility imports; new code uses the focused real_world.tianji modules."""

from diffusion_policy.common.paths import PROJECT_ROOT
from diffusion_policy.common.image_preprocessing import preprocess_observation
from diffusion_policy.real_world.tianji.config import apply_overrides, load_deployment_config
from diffusion_policy.real_world.tianji.env import TianjiRailEnv
from diffusion_policy.real_world.tianji.observation import (
    ObservationNotReady,
    align_observation,
    interpolate_state,
)

__all__ = [
    "PROJECT_ROOT",
    "preprocess_observation",
    "apply_overrides",
    "load_deployment_config",
    "TianjiRailEnv",
    "ObservationNotReady",
    "align_observation",
    "interpolate_state",
]

from diffusion_policy.real_world.tianji.observation import (
    CAMERA_KEYS,
    STATE_KEYS,
    ObservationWindowUnavailable,
    align_latest_observation,
)

__all__ += [
    "CAMERA_KEYS",
    "STATE_KEYS",
    "ObservationWindowUnavailable",
    "align_latest_observation",
]
