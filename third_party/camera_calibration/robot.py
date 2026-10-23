"""Offline project adapter. No camera or robot connection is created here."""

from pathlib import Path

import numpy as np

from .storage import digest

URDF = Path(__file__).resolve().parents[1] / "rail_assets/generated/robot.urdf"


def load_model(expected_hash=None):
    from diffusion_policy.third_party.bimanual_teleop.kinematics import RobotModel

    actual = digest(URDF)
    if expected_hash and expected_hash != actual:
        raise ValueError(
            "Robot URDF changed since wrist calibration; recalibrate the wrist"
        )
    return RobotModel(URDF, yaw_deg=90), actual


def world_flange(model, side, joints):
    q = np.asarray(joints, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("Expected seven finite measured joint angles in radians")
    offset = 0 if side == "left" else 7
    if np.any(q < model.lower[offset : offset + 7] - 0.01) or np.any(
        q > model.upper[offset : offset + 7] + 0.01
    ):
        raise ValueError("Measured joints exceed URDF limits")
    positions = np.zeros(len(model.names))
    positions[offset : offset + 7] = q
    return model.flange(side, positions, world=True)
