from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from rail_preinsert.core import digest, difference, load_calibration, matrix, read_json, transform, vector


def load_config(path, *, insertion_mode=None):
    import yaml
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    config.setdefault("arm_tracking_error_timeout_s", .5)
    config.setdefault("arm_tracking_hard_error_rad", 2*config["arm_tracking_error_rad"])
    config["move_above"].setdefault("position_move_timeout_s", 120.0)
    from .lowering import DEFAULT_LOWERING, validate_settings as validate_lowering
    from .insertion import DEFAULT_INSERTION, validate_settings as validate_insertion
    from .position_spiral import DEFAULT_POSITION_SPIRAL, validate_settings as validate_position_spiral
    from .preplace_alignment import DEFAULT_PREPLACE_ALIGNMENT, validate_settings as validate_preplace
    config["lowering"] = dict(DEFAULT_LOWERING, **config.get("lowering", {}))
    config["lowering"].setdefault("mode", "force_spiral")
    if insertion_mode is not None:
        config["lowering"]["mode"] = insertion_mode
    if config["lowering"]["mode"] not in ("force_spiral", "position_spiral", "position_steps"):
        raise ValueError("lowering.mode must be force_spiral, position_spiral or position_steps")
    validate_lowering(config["lowering"])
    config["insertion"] = dict(deepcopy(DEFAULT_INSERTION), **(config.get("insertion") or {}))
    # Missing field measurements disable N only, preserving receiving and M.
    config["insertion_missing"] = (validate_insertion(config["insertion"], require_ready=False)
        if config["lowering"]["mode"] == "force_spiral" else [])
    config["position_spiral"] = dict(DEFAULT_POSITION_SPIRAL, **(config.get("position_spiral") or {}))
    config["preplace_alignment"] = dict(DEFAULT_PREPLACE_ALIGNMENT, **(config.get("preplace_alignment") or {}))
    if config["lowering"]["mode"] == "position_spiral":
        validate_position_spiral(config["position_spiral"])
        validate_preplace(config["preplace_alignment"])
        if config["preplace_alignment"]["enabled"]:
            if any(config["preplace_alignment"][key] > config[key]
                   for key in ("velocity_ratio", "acceleration_ratio")):
                raise ValueError("Preplacement speed/acceleration ratios cannot exceed receiver limits")
            if (config["preplace_alignment"]["visual_position_tolerance_m"] > config["move_above"]["position_tolerance_m"]
                    or config["preplace_alignment"]["visual_rotation_tolerance_rad"] > min(
                        config["move_above"]["rotation_tolerance_rad"], config["position_spiral"]["rotation_tolerance_rad"])):
                raise ValueError("Preplacement visual tolerance must fit the position spiral start tolerance")
        if any(config["position_spiral"][key] > config[key]
               for key in ("velocity_ratio", "acceleration_ratio")):
            raise ValueError("Position spiral speed/acceleration ratios cannot exceed receiver limits")
    if any(config["lowering"][key] > config[key] for key in ("velocity_ratio", "acceleration_ratio")):
        raise ValueError("Lowering speed/acceleration ratios cannot exceed receiver limits")
    for key in ("arm_config", "hand_config", "hand_reference", "urdf", "manifest", "calibration",
                "mesh", "foundationpose_root", "sam_prompt"):
        config[key] = str((path.parent / config[key]).resolve())
    for key, value in config.items():
        if key.endswith(("_s", "_m", "_rad")) and not isinstance(value, (list, str)):
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
    if config["arm_tracking_hard_error_rad"] <= config["arm_tracking_error_rad"]:
        raise ValueError("arm_tracking_hard_error_rad must exceed arm_tracking_error_rad")
    for key in ("velocity_ratio", "acceleration_ratio"):
        if type(config[key]) is not int or not 1 <= config[key] <= 100:
            raise ValueError(f"{key} must be an integer from 1 to 100")
    if config["grasp_mode"] not in ("auto", "manual"):
        raise ValueError("grasp_mode must be auto or manual")
    for key in ("rail_midpoint_cad_m", "grasp_center_link_m", "grasp_half_extents_m"):
        vector(config[key], 3, key)
    vector(config.get("initial_offset_base_m", [0., 0., 0.]), 3, "initial_offset_base_m")
    vector(config["initial_reference_joints_deg"], 7, "initial_reference_joints_deg")
    if np.min(config["grasp_half_extents_m"]) <= 0:
        raise ValueError("Receiving volume must have positive half extents")
    parameters = config["hand_parameters"]
    if any(not np.isfinite(parameters[k]) or parameters[k] <= 0 for k in ("kp", "kd", "current_limit_a")):
        raise ValueError("Hand parameters must be finite and positive")
    roi = config["receive_roi_xyxy"]
    w, h = config["receive_image_size"]
    if (len(roi) != 4 or any(type(v) is not int for v in roi)
            or not (0 <= roi[0] < roi[2] <= w and 0 <= roi[1] < roi[3] <= h)):
        raise ValueError("Invalid fixed receiving ROI")
    if not 0 < config["cnos_detection"]["confidence_threshold"] <= 1:
        raise ValueError("Invalid CNOS threshold")
    config["config_path"] = str(path)
    detector = config["automatic_detection"]
    if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in detector.values()):
        raise ValueError("Automatic detection settings must be finite and positive")
    if type(detector["min_pixels"]) is not int or detector["min_pixels"] < 16:
        raise ValueError("Invalid automatic detection min_pixels")
    if detector["min_visible_fraction"] > 1:
        raise ValueError("Automatic detection min_visible_fraction must be <= 1")
    return config


def hand_targets(config):
    from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_NAMES, JOINT_LIMITS_RAD
    data = read_json(config["hand_reference"])
    if data.get("side") != "right" or data.get("unit") != "rad" or data.get("joint_order") != list(JOINT_NAMES):
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
        from rail_preinsert.geometry import RobotGeometry
        left = RobotGeometry(config, kinematics)
        calibration, T_L_C = load_calibration(config["calibration"])
        model_digest = calibration.get("sdk_model_sha256")
        if model_digest != kinematics.model.digest:
            raise ValueError("Camera calibration must identify the current SDK model")
        robot = left.robot
        q = np.zeros(54)
        T_R_W = matrix(kinematics.fk("right", tuple(q[7:14]))) @ np.linalg.inv(robot.flange("right", q, world=True))
        for degrees in ([-30, -60, 34, -52, -30, 12, -4], [-25, -55, 30, -45, -20, 10, -2]):
            q[7:14] = np.radians(degrees)
            distance, angle = difference(T_R_W @ robot.flange("right", q, world=True),
                                         matrix(kinematics.fk("right", tuple(q[7:14]))))
            if distance > .0001 or angle > .0002:
                raise ValueError("Right URDF base registration disagrees with official FK")
        self.T_B_C = transform(T_R_W @ np.linalg.inv(left.T_B_W) @ T_L_C)
        link = config["grasp_link"]
        chain = robot.chain(link)
        flange_index = next((i for i, j in enumerate(chain) if j.child == "TCP_Link_R"), None)
        if flange_index is None or any(j.kind != "fixed" for j in chain[flange_index + 1:]):
            raise ValueError("Receiving link must be rigidly attached to the right flange")
        self.T_F_G = np.eye(4)
        for joint in chain[flange_index + 1:]:
            self.T_F_G = self.T_F_G @ joint.origin
        self.T_F_G[:3, 3] += self.T_F_G[:3, :3] @ np.asarray(config["grasp_center_link_m"])
        self.calibration = calibration
        self.mesh_digest = digest(config["mesh"])


class DetectionGate:
    """Close once after distinct, fresh CNOS detections span one second."""
    def __init__(self, config, camera_serial, mesh_sha256):
        self.config, self.serial, self.mesh_digest = config, camera_serial, mesh_sha256
        self.first_stamp = self.last_stamp = None
        self.fired = False

    def reset(self):
        self.first_stamp = self.last_stamp = None

    def update(self, record, now_ns):
        c = self.config
        valid = (record and record.get("valid") is True
                 and record.get("detected_in_roi") is True and record.get("detector") == "cnos"
                 and not record.get("replay_only", False)
                 and record.get("camera_serial") == self.serial
                 and record.get("mesh_sha256") == self.mesh_digest
                 and record.get("roi_xyxy") == c["receive_roi_xyxy"]
                 and record.get("timestamp_source") == "sdk_global_timestamp_mapped_to_host_monotonic")
        if not valid:
            self.reset()
            return {"ready": False, "reason": "invalid_observation", "stable_s": 0.}
        stamp = int(record["timestamp_ns"])
        if not 0 <= now_ns-stamp <= c["detection_max_age_s"]*1e9:
            self.reset()
            return {"ready": False, "reason": "stale_observation", "stable_s": 0.}
        if self.fired:
            return {"ready": False, "reason": "already_triggered", "stable_s": 0.}
        if self.last_stamp is not None:
            if stamp == self.last_stamp:
                return {"ready": False, "reason": "waiting_new_frame",
                        "stable_s": (stamp-self.first_stamp)/1e9}
            if stamp < self.last_stamp or stamp-self.last_stamp > c["detection_max_gap_s"]*1e9:
                self.reset()
        if self.first_stamp is None:
            self.first_stamp = stamp
        self.last_stamp = stamp
        elapsed = (stamp-self.first_stamp)/1e9
        ready = elapsed >= c["detection_duration_s"]
        self.fired = bool(ready)
        return {"ready": ready, "reason": "detected" if ready else "detecting", "stable_s": elapsed}


# Translation and fixed-base rotations; no finger-joint keys.
JOG_KEYS = {"w": (0, 1), "s": (0, -1), "r": (1, 1), "f": (1, -1),
            "d": (2, 1), "a": (2, -1), "i": (3, 1), "k": (3, -1),
            "j": (4, 1), "l": (4, -1), "u": (5, 1), "o": (5, -1)}


class KeyboardJog:
    def __init__(self, config, initial):
        self.config = config
        self.target = transform(initial).copy()
        self.scale = 1.0

    def propose(self, key):
        if key in ("+", "=", "-"):
            self.scale = float(np.clip(self.scale * (0.5 if key == "-" else 2), .25, 4))
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
