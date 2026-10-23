"""Rank tracking runs per frame with the FoundationPose scorer, including each run's end-to-end flip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rail_pose.common import REPOSITORY, load_config, resolve, save_json  # noqa: E402
from rail_pose.data import Scene  # noqa: E402


def load_run(path: Path) -> dict:
    poses = {}
    for line in (path / "poses.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["valid"]:
            poses[record["frame_id"]] = np.asarray(record["T_C_CAD"])
    return poses


def end_flip(mesh) -> np.ndarray:
    """180° about CAD Z through the box centre: swaps the rail ends and keeps its bounding box."""
    centre = mesh.bounds.mean(axis=0)
    flip = np.diag([-1., -1., 1., 1.])
    flip[:2, 3] = 2 * centre[:2]
    return flip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(REPOSITORY))
    from Utils import bilateral_filter_depth, erode_depth
    from rail_pose.engine import PoseEngine
    config = load_config()
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    engine = PoseEngine(resolve(config["model"]), config, output / "debug")
    estimator = engine.estimator
    flip = end_flip(engine.mesh)
    to_centred = np.eye(4)
    to_centred[:3, 3] = estimator.model_center
    scene = Scene(resolve(args.scene))
    runs = {Path(run).name: load_run(resolve(run)) for run in args.runs}
    with (output / "scores.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(0, len(scene), args.step):
            frame = scene.frame(index)
            candidates = {}
            for name, run in runs.items():
                if frame.frame_id in run:
                    candidates[name] = run[frame.frame_id]
                    candidates[name + "~flip"] = run[frame.frame_id] @ flip
            if not candidates:
                continue
            depth = bilateral_filter_depth(erode_depth(frame.depth_m, radius=2, device="cuda"), radius=2, device="cuda")
            scores, _ = estimator.scorer.predict(
                mesh=estimator.mesh, rgb=frame.rgb, depth=depth, K=frame.K,
                ob_in_cams=np.stack([pose @ to_centred for pose in candidates.values()]).astype(np.float32),
                normal_map=None, mesh_tensors=estimator.mesh_tensors, glctx=estimator.glctx,
                mesh_diameter=estimator.diameter, get_vis=False)
            scores = np.asarray(scores.detach().cpu() if hasattr(scores, "detach") else scores, dtype=float).ravel()
            row = {"index": index, "frame_id": frame.frame_id, "scores": dict(zip(candidates, scores.tolist())),
                   "best": list(candidates)[int(np.argmax(scores))]}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    save_json(output / "settings.json", {"scene": str(scene.directory), "runs": args.runs, "step": args.step,
                                           "end_flip_in_cad": flip.tolist()})


if __name__ == "__main__":
    main()
