from __future__ import annotations

from pathlib import Path
import os
import logging
from threading import Event
import time
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .common import REPOSITORY
from .geometry import validate_pose, pose_difference

WEIGHT_SETS = ("2023-10-28-18-33-37", "2024-01-11-20-02-45")


def check_weights():
    missing = [str(REPOSITORY / "weights" / stamp / name)
               for stamp in WEIGHT_SETS for name in ("config.yml", "model_best.pth")
               if not (REPOSITORY / "weights" / stamp / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required weights: " + ", ".join(missing))


class PoseEngine:
    def __init__(self, mesh_path: Path, config: dict, debug_directory: Path, low_memory=True):
        check_weights()
        import torch
        import nvdiffrast.torch as dr
        from estimater import FoundationPose
        from learning.training.predict_pose_refine import PoseRefinePredictor
        from learning.training.predict_score import ScorePredictor
        from .low_memory import ChunkedRefiner, ChunkedScorer
        if not torch.cuda.is_available():
            raise RuntimeError("FoundationPose requires a CUDA GPU")
        limit = os.environ.get("RAIL_CUDA_MEMORY_MIB")
        if limit:
            fraction = float(limit) * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory
            if not 0 < fraction <= 1:
                raise ValueError("invalid RAIL_CUDA_MEMORY_MIB allocator budget")
            torch.cuda.set_per_process_memory_fraction(fraction)
        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.config = config
        self.mesh = trimesh.load(mesh_path, force="mesh")
        self.mesh_path = mesh_path
        if not len(self.mesh.faces) or not np.isfinite(self.mesh.vertices).all():
            raise ValueError("invalid inference mesh")
        if self.mesh.extents.max() > 10:
            raise ValueError("inference mesh must be in metres; run prepare-model first")
        self.batch_size = int(config["pose_batch_size"])
        refiner = ChunkedRefiner(self.batch_size) if low_memory else PoseRefinePredictor()
        scorer = ChunkedScorer(self.batch_size) if low_memory else ScorePredictor()
        # Stable default for batch-size equivalence and GPUs without tensor cores.
        refiner.amp = scorer.amp = bool(config.get("use_amp", False))
        self.cancel_event = Event()
        if low_memory:
            refiner.cancel_event = scorer.cancel_event = self.cancel_event
        self.estimator = FoundationPose(
            model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals,
            mesh=self.mesh, refiner=refiner, scorer=scorer, glctx=dr.RasterizeCudaContext(),
            debug=0, debug_dir=str(debug_directory),
        )
        self.previous_pose = None
        self.low_memory = low_memory
        logging.getLogger().setLevel(logging.WARNING)

    @property
    def initialized(self):
        return self.previous_pose is not None

    def reset(self):
        self.previous_pose = None
        self.estimator.pose_last = None
        self.cancel_event.clear()

    def cancel_pending(self):
        self.cancel_event.set()

    def infer(self, frame, initialize=False):
        frame.validate(self.config["max_depth_m"])
        if np.count_nonzero(frame.depth_m >= .001) < self.config["min_valid_depth_pixels"]:
            self.reset()
            raise ValueError("frame has insufficient valid depth")
        if not self.initialized:
            initialize = True
        mask = frame.registration_mask(self.config["min_valid_depth_pixels"]) if initialize else None
        start = time.perf_counter()
        self.torch.cuda.reset_peak_memory_stats()
        while True:
            try:
                if initialize:
                    pose = self.estimator.register(K=frame.K, rgb=frame.rgb, depth=frame.depth_m.copy(),
                                                   ob_mask=mask, iteration=self.config["estimate_iterations"])
                else:
                    pose = self.estimator.track_one(rgb=frame.rgb, depth=frame.depth_m.copy(), K=frame.K,
                                                    iteration=self.config["track_iterations"])
                self.torch.cuda.synchronize()
                break
            except self.torch.OutOfMemoryError:
                if not initialize or not self.low_memory or self.batch_size == 1:
                    self.reset()
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                self.estimator.refiner.batch_size = self.batch_size
                self.estimator.scorer.batch_size = self.batch_size
                self.torch.cuda.empty_cache()
        pose = np.asarray(pose, dtype=np.float64)
        try:
            validate_pose(pose)
            # Check the object center, since the CAD origin may be at one end.
            center = pose[:3, :3] @ self.mesh.bounds.mean(axis=0) + pose[:3, 3]
            if not .001 < center[2] <= self.config["max_depth_m"]:
                raise ValueError("estimated object center is outside the camera depth range")
            if not initialize and self.previous_pose is not None:
                distance, angle = pose_difference(self.previous_pose, pose)
                if distance > self.config["max_translation_jump_m"] or angle > self.config["max_rotation_jump_deg"]:
                    raise ValueError(f"tracking jump: {distance:.3f} m, {angle:.1f} deg; reinitialize")
        except ValueError:
            self.reset()
            raise
        self.previous_pose = pose.copy()
        elapsed = time.perf_counter() - start
        record = {
            "frame_id": frame.frame_id, "timestamp_ns": int(frame.timestamp_ns),
            "frame": "T_C_CAD: original CAD coordinates to OpenCV camera coordinates",
            "valid": True, "mode": "register" if initialize else "track",
            "T_C_CAD": pose.tolist(), "translation_m": pose[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
            "processing_s": elapsed, "pose_batch_size": self.batch_size,
            "candidate_count": len(self.estimator.rot_grid),
            "torch_peak_allocated_mib": self.torch.cuda.max_memory_allocated() / 2 ** 20,
            "torch_peak_reserved_mib": self.torch.cuda.max_memory_reserved() / 2 ** 20,
        }
        return pose, record


def failure_record(frame, error):
    return {"frame_id": frame.frame_id, "timestamp_ns": int(frame.timestamp_ns),
            "valid": False, "reason": str(error), "T_C_CAD": None}
