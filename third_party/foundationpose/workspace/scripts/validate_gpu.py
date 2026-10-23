#!/usr/bin/env python3
"""Check the real CAD renderer and weighted chunk/full ranking equivalence."""
from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent)]

import numpy as np
import torch
import nvdiffrast.torch as dr
from rail_pose.common import save_json, load_config
from rail_pose.data import Scene
from rail_pose.engine import PoseEngine
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
from Utils import nvdiffrast_render, depth2xyzmap
from types import SimpleNamespace, MethodType

config = load_config()
scene = Scene(ROOT / "datasets/official-smoke")
frame = scene.frame(0)
mesh_path = ROOT / "datasets/official-download/mustard0/mesh/textured_simple.obj"
engine = PoseEngine(mesh_path, config, ROOT / "outputs/equivalence-debug")
poses = engine.estimator.generate_random_pose_hypo(frame.K, frame.rgb, frame.depth_m, frame.mask).cpu().numpy()
# Compare 17 poses spanning the complete grid, exercising a non-full final chunk.
poses = poses[np.linspace(0, len(poses)-1, 17).astype(int)]
kwargs = dict(rgb=frame.rgb, depth=frame.depth_m, K=frame.K, ob_in_cams=poses,
              mesh=engine.estimator.mesh, mesh_tensors=engine.estimator.mesh_tensors,
              glctx=engine.estimator.glctx, mesh_diameter=engine.estimator.diameter, get_vis=False)
with torch.inference_mode():
    original_scores, _ = ScorePredictor.predict(engine.estimator.scorer, **kwargs)
    chunk_scores, _ = engine.estimator.scorer.predict(**kwargs)
    score_error = float(torch.max(torch.abs(original_scores - chunk_scores)))
    score_top1_equal = int(original_scores.argmax()) == int(chunk_scores.argmax())
    refine_args = dict(kwargs, xyz_map=depth2xyzmap(frame.depth_m, frame.K), iteration=2)
    reference_poses, _ = PoseRefinePredictor.predict(engine.estimator.refiner, **refine_args)
    chunk_poses, _ = engine.estimator.refiner.predict(**refine_args)
    refine_error = float(torch.max(torch.abs(reference_poses - chunk_poses)))

# Compare complete registration (all 252 candidates, five iterations) with the
# SAME mesh, weights, diameter and renderer, avoiding constructor randomness.
refiner, scorer = engine.estimator.refiner, engine.estimator.scorer
try:
    engine.estimator.refiner = SimpleNamespace(predict=MethodType(PoseRefinePredictor.predict, refiner))
    engine.estimator.scorer = SimpleNamespace(predict=MethodType(ScorePredictor.predict, scorer))
    reference_pose = engine.estimator.register(K=frame.K, rgb=frame.rgb, depth=frame.depth_m,
                                               ob_mask=frame.mask, iteration=5)
    reference_best = int(engine.estimator.best_id)
finally:
    engine.estimator.refiner, engine.estimator.scorer = refiner, scorer
engine.reset()
chunk_pose, _ = engine.infer(frame)
registration_error = float(np.abs(reference_pose - chunk_pose).max())
registration_top1_equal = reference_best == int(engine.estimator.best_id)

# Render the supplied REAL CAD at a known pose; this validates geometry/rendering,
# not accuracy on a physical image and not a substitute for real RGB-D.
rail = PoseEngine(ROOT / "input/model/rail.obj", config, ROOT / "outputs/rail-render-debug")
known_pose = np.eye(4, dtype=np.float32)
known_pose[:3, 3] = [0, 0, 1]  # Renderer receives the internally centered mesh.
K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
rgb, depth, _ = nvdiffrast_render(K=K, H=480, W=640,
                                 ob_in_cams=torch.as_tensor(known_pose[None], device="cuda"),
                                 glctx=rail.estimator.glctx, mesh_tensors=rail.estimator.mesh_tensors,
                                 use_light=True)
from PIL import Image
Image.fromarray((rgb[0].cpu().numpy()*255).clip(0,255).astype(np.uint8)).save(ROOT / "outputs/rail-gpu-render.png")
render_pixels = int((depth[0] > 0).sum())
report = {"score_max_absolute_error": score_error, "score_top1_equal": score_top1_equal,
          "refine_max_absolute_error": refine_error, "real_cad_render_pixels": render_pixels,
          "full_registration_max_absolute_error": registration_error,
          "full_registration_top1_equal": registration_top1_equal,
          "full_candidate_count": len(engine.estimator.rot_grid),
          "physical_rgbd_accuracy_verified": False,
          "passed": score_error < .01 and score_top1_equal and refine_error < .002
                    and registration_error < .002 and registration_top1_equal and render_pixels > 100}
save_json(ROOT / "outputs/validation.json", report)
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 1)
