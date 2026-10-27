from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rail_common.core import (
    digest,
    difference,
    load_calibration,
    matrix,
    read_json,
    transform,
    vector,
)


def load_config(path):
    import yaml

    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config["move_above"].setdefault("position_move_timeout_s", 120.0)
    from .position_spiral import DEFAULT_POSITION_SPIRAL, validate_settings as validate_spiral
    from .preplace_alignment import (
        DEFAULT_PREPLACE_ALIGNMENT,
        validate_settings as validate_preplace,
    )
    from .release_return import DEFAULT_RELEASE_RETURN, validate_settings as validate_return

    config["position_spiral"] = dict(DEFAULT_POSITION_SPIRAL, **config.get("position_spiral", {}))
    config["preplace_alignment"] = dict(
        DEFAULT_PREPLACE_ALIGNMENT, **config.get("preplace_alignment", {})
    )
    config["release_return"] = dict(DEFAULT_RELEASE_RETURN, **config.get("release_return", {}))
    validate_spiral(config["position_spiral"])
    validate_preplace(config["preplace_alignment"])
    validate_return(config["release_return"])
    if config["release_return"]["path_tolerance_m"] < config["move_above"]["position_tolerance_m"]:
        raise ValueError("Release lift path tolerance must cover its arrival tolerance")
    for section in ("position_spiral", "preplace_alignment", "release_return"):
        if any(
            config[section][key] > config[key] for key in ("velocity_ratio", "acceleration_ratio")
        ):
            raise ValueError(f"{section} speed/acceleration ratios cannot exceed receiver limits")
    preplace = config["preplace_alignment"]
    if (
        preplace["position_tolerance_m"] > config["move_above"]["position_tolerance_m"]
        or preplace["rotation_tolerance_rad"] > config["position_spiral"]["rotation_tolerance_rad"]
    ):
        raise ValueError("Preplacement arrival tolerances must fit the spiral start tolerances")
    config.setdefault("sam_prompt_preplace", "sam_prompt_preplace.json")
    for key in (
        "arm_config",
        "hand_config",
        "hand_reference",
        "urdf",
        "manifest",
        "calibration",
        "mesh",
        "foundationpose_root",
        "sam_prompt",
        "sam_prompt_preplace",
    ):
        config[key] = str((path.parent / config[key]).resolve())
    if config["sam_prompt"] == config["sam_prompt_preplace"]:
        raise ValueError("Initial and preplacement SAM prompts must use different files")
    for key, value in config.items():
        if key.endswith(("_s", "_m", "_rad")) and not isinstance(value, (list, str)):
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
    for key in ("velocity_ratio", "acceleration_ratio"):
        if type(config[key]) is not int or not 1 <= config[key] <= 100:
            raise ValueError(f"{key} must be an integer from 1 to 100")
    vector(config["rail_midpoint_cad_m"], 3, "rail_midpoint_cad_m")
    vector(config.get("initial_offset_base_m", [0.0, 0.0, 0.0]), 3, "initial_offset_base_m")
    vector(config["initial_reference_joints_deg"], 7, "initial_reference_joints_deg")
    if "left_initial_pose" in config:
        vector(
            config["left_initial_pose"]["reference_joints_deg"],
            7,
            "left_initial_pose.reference_joints_deg",
        )
        vector(config["left_initial_pose"]["offset_base_m"], 3, "left_initial_pose.offset_base_m")
    parameters = config["hand_parameters"]
    if any(
        not np.isfinite(parameters[k]) or parameters[k] <= 0
        for k in ("kp", "kd", "current_limit_a")
    ):
        raise ValueError("Hand parameters must be finite and positive")
    config["config_path"] = str(path)
    return config


def hand_targets(config):
    from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import (
        JOINT_NAMES,
        JOINT_LIMITS_RAD,
    )

    data = read_json(config["hand_reference"])
    if (
        data.get("side") != "right"
        or data.get("unit") != "rad"
        or data.get("joint_order") != list(JOINT_NAMES)
    ):
        raise ValueError("Require an explicit right-hand profile in SDK joint order and radians")
    result = {}
    for name in ("open", "grasp"):
        q = vector(data[f"{name}_position_rad"], 20)
        if any(not low <= v <= high for v, (low, high) in zip(q, JOINT_LIMITS_RAD)):
            raise ValueError(f"{name}: hand target exceeds joint limits")
        result[name] = q
    return result


class ReceiveGeometry:
    """Convert the measured left-base camera calibration through verified robot FK."""

    def __init__(self, config, kinematics):
        from rail_common.geometry import RobotGeometry

        left = RobotGeometry(config, kinematics)
        calibration, T_L_C = load_calibration(config["calibration"])
        model_digest = calibration.get("sdk_model_sha256")
        if model_digest != kinematics.model.digest:
            raise ValueError("Camera calibration must identify the current SDK model")
        robot = left.robot
        q = np.zeros(54)
        T_R_W = matrix(kinematics.fk("right", tuple(q[7:14]))) @ np.linalg.inv(
            robot.flange("right", q, world=True)
        )
        for degrees in ([-30, -60, 34, -52, -30, 12, -4], [-25, -55, 30, -45, -20, 10, -2]):
            q[7:14] = np.radians(degrees)
            distance, angle = difference(
                T_R_W @ robot.flange("right", q, world=True),
                matrix(kinematics.fk("right", tuple(q[7:14]))),
            )
            if distance > 0.0001 or angle > 0.0002:
                raise ValueError("Right URDF base registration disagrees with official FK")
        self.T_B_C = transform(T_R_W @ np.linalg.inv(left.T_B_W) @ T_L_C)
        self.calibration = calibration
        self.mesh_digest = digest(config["mesh"])


# Translation and fixed-base rotations; no finger-joint keys.
JOG_KEYS = {
    "w": (0, 1),
    "s": (0, -1),
    "r": (1, 1),
    "f": (1, -1),
    "d": (2, 1),
    "a": (2, -1),
    "i": (3, 1),
    "k": (3, -1),
    "j": (4, 1),
    "l": (4, -1),
    "u": (5, 1),
    "o": (5, -1),
}


class KeyboardJog:
    def __init__(self, config, initial):
        self.config = config
        self.target = transform(initial).copy()
        self.scale = 1.0

    def propose(self, key):
        if key in ("+", "=", "-"):
            self.scale = float(np.clip(self.scale * (0.5 if key == "-" else 2), 0.25, 4))
            return None
        if key not in JOG_KEYS:
            return None
        c = self.config
        axis, sign = JOG_KEYS[key]
        target = self.target.copy()
        if axis < 3:
            target[axis, 3] += sign * self.scale * c["jog_step_m"]
        else:
            rotation = np.zeros(3)
            rotation[axis - 3] = sign * self.scale * c["jog_step_rad"]
            target[:3, :3] = Rotation.from_rotvec(rotation).as_matrix() @ target[:3, :3]
        self.target = target
        return target.copy()
