"""Resolve the rail's approximate end-for-end pose ambiguity before planning.

The task prior, not exact mesh symmetry, distinguishes the two hypotheses.
The saved goal remains the reference CAD frame. Never rotate about its endpoint
or silently clamp an observed angle to the allowed range.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from .core import digest, transform, vector


def relative_rpy_deg(reference, observed):
    """Fixed XYZ roll/pitch/yaw of R_reference.T @ R_observed, in degrees."""
    relative = transform(reference)[:3, :3].T @ transform(observed)[:3, :3]
    with warnings.catch_warnings():
        # A gimbal-locked candidate has |pitch|=90 and fails the task bound.
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        return Rotation.from_matrix(relative).as_euler("xyz", degrees=True)


class RailPoseAlignment:
    def __init__(self, settings, mesh_sha256):
        self.center = vector(settings["center_cad_m"], 3, "pose_alignment.center_cad_m")
        if settings["mesh_sha256"] != mesh_sha256:
            raise ValueError("Pose alignment belongs to a different CAD mesh")
        self.limit = settings["max_axis_rotation_deg"]
        self.typical = settings["typical_axis_rotation_deg"]
        if (
            any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                for v in (self.limit, self.typical)
            )
            or not 0 < self.typical <= self.limit <= 45
        ):
            raise ValueError("Require 0 < typical_axis_rotation_deg <= max_axis_rotation_deg <= 45")
        # Preserve CAD Z (top/bottom); exchange both CAD X and CAD Y.
        flip = np.eye(4)
        flip[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        flip[:3, 3] = self.center - flip[:3, :3] @ self.center
        self.candidates = (np.eye(4), flip)

    @classmethod
    def from_config(cls, config):
        settings = config.get("pose_alignment")
        return cls(settings, digest(config["mesh"])) if settings is not None else None

    def check_motion(self, initial, goal):
        angles = relative_rpy_deg(goal, initial)
        if np.max(np.abs(angles)) >= self.limit:
            raise ValueError(
                f"Rail relative XYZ angles {angles.round(2).tolist()} deg exceed "
                f"the per-axis < {self.limit:g} deg prior; re-estimate before moving"
            )
        return angles

    def align(self, observed, reference, *, enforce_limit=True):
        observed, reference = transform(observed), transform(reference)
        poses = [observed @ symmetry for symmetry in self.candidates]
        angles = [relative_rpy_deg(reference, pose) for pose in poses]
        eligible = [i for i, value in enumerate(angles) if np.max(np.abs(value)) < self.limit]
        # Rank eligible hypotheses by true SO(3) angle, but gate each XYZ angle.
        rotation_errors = [
            float(Rotation.from_matrix(reference[:3, :3].T @ pose[:3, :3]).magnitude())
            for pose in poses
        ]
        index = min(eligible or range(len(poses)), key=lambda i: rotation_errors[i])
        if enforce_limit and not eligible:
            self.check_motion(poses[index], reference)
        report = {
            "hypothesis": ("identity", "cad_z_half_turn")[index],
            "approximate_symmetry": True,
            "center_cad_m": self.center.tolist(),
            "T_CAD_raw_CAD_aligned": self.candidates[index].tolist(),
            "relative_xyz_deg": angles[index].tolist(),
            "raw_relative_xyz_deg": angles[0].tolist(),
            "rotation_error_deg": float(np.degrees(rotation_errors[index])),
            "max_axis_rotation_deg": self.limit,
            "within_limit": bool(eligible),
            "within_typical_range": bool(np.max(np.abs(angles[index])) < self.typical),
        }
        return poses[index], report

    def align_record(self, record, reference, *, enforce_limit=True):
        raw = transform(record.get("T_C_CAD_raw", record["T_C_CAD"]))
        aligned, report = self.align(raw, reference, enforce_limit=enforce_limit)
        return {
            **record,
            "T_C_CAD_raw": raw.tolist(),
            "T_C_CAD": aligned.tolist(),
            "translation_m": aligned[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(aligned[:3, :3]).as_quat().tolist(),
            "pose_alignment": report,
        }
