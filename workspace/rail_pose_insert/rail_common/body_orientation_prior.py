"""Robot-body orientation prior inside FoundationPose's centered-mesh refinement.

base_link: +Z up, -Y forward, +X left. CAD: +Z open face, X width,
Y length. The two end-for-end branches remain available to visual scoring.
No temporal smoothing or target translation is used here.
"""

from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial.transform import Rotation

from .core import digest, load_calibration, transform
from .geometry import RobotGeometry


def camera_to_body(config, kinematics=None):
    if kinematics is None:
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import (
            TianjiKinematics,
        )

        kinematics = TianjiKinematics()
    calibration, T_L_C = load_calibration(config["calibration"])
    if calibration.get("sdk_model_sha256") != kinematics.model.digest:
        raise ValueError("Body prior requires camera calibration for the current SDK model")
    geometry = RobotGeometry(config, kinematics)
    T_W_body = geometry.robot.link_pose("base_link", np.zeros(54))
    # Explicitly remove sim_world -> base_link, including its 90-degree yaw.
    T_body_C = np.linalg.inv(T_W_body) @ np.linalg.inv(geometry.T_B_W) @ T_L_C
    return transform(T_body_C), calibration


class BodyOrientationPrior:
    def __init__(self, T_body_C, settings):
        self.T_body_C = transform(T_body_C)
        self.settings = dict(settings)
        if settings["frame"] != "base_link":
            raise ValueError("Orientation prior must use whole-robot base_link")
        self.strength = float(settings["strength"])
        if (
            isinstance(settings["strength"], bool)
            or not np.isfinite(self.strength)
            or not 0 <= self.strength < 1
        ):
            raise ValueError("Prior strength must be in [0, 1); locking orientation is forbidden")
        for key in (
            "search_tilt_deg",
            "search_heading_deg",
            "grid_step_deg",
            "max_visual_tilt_deg",
            "max_visual_heading_deg",
            "max_correction_deg",
        ):
            value = settings[key]
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value < 90:
                raise ValueError(f"Invalid body orientation prior: {key}")
        if (
            settings["search_tilt_deg"] > settings["max_visual_tilt_deg"]
            or settings["search_heading_deg"] > settings["max_visual_heading_deg"]
        ):
            raise ValueError("Prior search bounds must fit within visual validity bounds")
        # Columns are CAD X (forward), Y (left), Z (up), expressed in base_link.
        upright = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        self.references = np.stack((upright, upright @ np.diag([-1.0, -1.0, 1.0])))

    @classmethod
    def from_config(cls, config, camera_serial=None):
        settings = config.get("body_orientation_prior")
        if not settings or not settings.get("enabled", False):
            return None
        if settings["mesh_sha256"] != digest(config["mesh"]):
            raise ValueError("Body orientation prior belongs to a different CAD mesh")
        T_body_C, calibration = camera_to_body(config)
        if camera_serial is not None and camera_serial != calibration["camera_serial"]:
            raise ValueError("Body orientation prior camera does not match calibration")
        return cls(T_body_C, settings)

    def angles(self, camera_rotations):
        rotations = self.T_body_C[:3, :3] @ np.asarray(camera_rotations)
        tilt = np.degrees(np.arccos(np.clip(rotations[..., 2, 2], -1, 1)))
        # CAD Y is the long axis, whose horizontal projection is parallel to
        # body X in either direction. Top tilt is checked separately.
        length = rotations[..., :, 1]
        heading = np.degrees(np.arctan2(np.abs(length[..., 1]), np.abs(length[..., 0])))
        return tilt, heading

    def admissible(self, camera_rotations):
        tilt, heading = self.angles(camera_rotations)
        return (tilt <= self.settings["max_visual_tilt_deg"]) & (
            heading <= self.settings["max_visual_heading_deg"]
        )

    def rotation_grid(self):
        s = self.settings

        def samples(limit):
            return np.unique(np.r_[np.arange(-limit, limit + 1e-8, s["grid_step_deg"]), 0.0, limit])

        rows = []
        for reference in self.references:
            for xyz in itertools.product(
                samples(s["search_tilt_deg"]),
                samples(s["search_tilt_deg"]),
                samples(s["search_heading_deg"]),
            ):
                R = (
                    self.T_body_C[:3, :3].T
                    @ Rotation.from_euler("xyz", xyz, degrees=True).as_matrix()
                    @ reference
                )
                tilt, heading = self.angles(R)
                if (
                    tilt <= s["search_tilt_deg"] + 1e-7
                    and heading <= s["search_heading_deg"] + 1e-7
                ):
                    T = np.eye(4)
                    T[:3, :3] = R
                    rows.append(T)
        if not rows:
            raise ValueError("Body prior produced no initial rotation hypotheses")
        return np.stack(rows)

    def regularize_centered(self, poses):
        """Shrink orientation toward the nearest body-aligned branch, keep center.

        These poses are FoundationPose's CENTERED mesh poses, not the original
        CAD endpoint poses. The original endpoint is recovered by FoundationPose.
        Large contradictions remain unchanged and are rejected, never repaired.
        """
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 4, 4)
        if not np.isfinite(poses).all():
            raise ValueError("Nonfinite visual refinement cannot use the body prior")
        result = poses.copy()
        valid = self.admissible(poses[:, :3, :3])
        R_body = self.T_body_C[:3, :3] @ poses[:, :3, :3]
        for i in np.flatnonzero(valid):
            vectors = Rotation.from_matrix(self.references @ R_body[i].T).as_rotvec()
            vector = vectors[np.argmin(np.linalg.norm(vectors, axis=1))]
            correction = self.strength * vector
            norm = np.linalg.norm(correction)
            limit = np.radians(self.settings["max_correction_deg"])
            if norm > limit:
                correction *= limit / norm
            result[i, :3, :3] = (
                self.T_body_C[:3, :3].T @ Rotation.from_rotvec(correction).as_matrix() @ R_body[i]
            )
        return result, valid


class PriorRefiner:
    def __init__(self, delegate, prior):
        self.delegate, self.prior = delegate, prior
        self.raw = self.corrected = self.valid = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    @property
    def batch_size(self):
        return self.delegate.batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.delegate.batch_size = value

    def predict(self, *args, **kwargs):
        poses, vis = self.delegate.predict(*args, **kwargs)
        self.raw = poses.detach().cpu().numpy().copy()
        self.corrected, self.valid = self.prior.regularize_centered(self.raw)
        if not np.any(self.valid):
            tilt, heading = self.prior.angles(self.raw[:, :3, :3])
            raise ValueError(
                f"Visual pose contradicts body prior: min tilt {tilt.min():.2f} deg, "
                f"min heading {heading.min():.2f} deg"
            )
        return poses.new_tensor(self.corrected), vis


class PriorScorer:
    def __init__(self, delegate, refiner):
        self.delegate, self.refiner = delegate, refiner

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    @property
    def batch_size(self):
        return self.delegate.batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.delegate.batch_size = value

    def predict(self, *args, **kwargs):
        import torch

        scores, vis = self.delegate.predict(*args, **kwargs)
        # Score the rendered corrected candidates, excluding any hypothesis
        # whose unconstrained visual refinement contradicted the prior.
        scores = scores.clone()
        scores[~scores.new_tensor(self.refiner.valid, dtype=torch.bool)] = -float("inf")
        return scores, vis


class PriorPoseEngine:
    """Install the prior BEFORE scoring and internal tracking-state updates."""

    def __init__(self, engine, prior):
        self.engine, self.prior = engine, prior
        estimator = engine.estimator
        estimator.rot_grid = estimator.rot_grid.new_tensor(prior.rotation_grid())
        self.refiner = PriorRefiner(estimator.refiner, prior)
        estimator.refiner = self.refiner
        estimator.scorer = PriorScorer(estimator.scorer, self.refiner)

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def infer(self, frame, initialize=False):
        registration = initialize or not self.engine.initialized
        self.refiner.raw = None
        try:
            pose, record = self.engine.infer(frame, initialize=initialize)
            if self.refiner.raw is None:
                raise ValueError(
                    "FoundationPose skipped visual refinement; body prior cannot validate this pose"
                )
        except (ValueError, RuntimeError):
            self.engine.reset()
            raise
        index = int(self.engine.estimator.best_id) if registration else 0
        raw = self.refiner.raw[index].copy()
        center = np.asarray(self.engine.estimator.model_center)
        raw[:3, 3] -= raw[:3, :3] @ center
        raw_angles = self.prior.angles(raw[:3, :3])
        angles = self.prior.angles(pose[:3, :3])
        correction = Rotation.from_matrix(pose[:3, :3] @ raw[:3, :3].T).magnitude()
        record = dict(
            record,
            T_C_CAD_before_body_prior=raw.tolist(),
            orientation_prior={
                "frame": "base_link",
                "body_axes": {"up": "+Z", "forward": "-Y", "left": "+X"},
                "strength": self.prior.strength,
                "T_body_C": self.prior.T_body_C.tolist(),
                "visual_tilt_deg": float(raw_angles[0]),
                "visual_heading_deg": float(raw_angles[1]),
                "tilt_deg": float(angles[0]),
                "heading_deg": float(angles[1]),
                "correction_deg": float(np.degrees(correction)),
                "mesh_center_preserved": True,
                "note": "before_body_prior is this run's visual refinement, not an independent baseline",
            },
        )
        return pose, record
