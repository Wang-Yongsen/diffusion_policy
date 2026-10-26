"""Replay identical recorded RGB-D for a body-prior comparison. No device I/O.

Run baseline/prior in separate processes to release GPU memory. The supplied
mask must belong to the first frame; it is reused for reinitialization and fit
diagnostics only for a stationary scene. A fresh SAM mask is not generated here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from rail_preinsert.core import digest, mean_pose, write_json
from rail_preinsert.pose_alignment import RailPoseAlignment
from rail_receive.logic import load_config
from receive_perception_worker import pose_engine, pose_reference


def scene_digest(scene, mask):
    sha = hashlib.sha256()
    for path in [scene / "scene.json", mask] + sorted((scene / "rgb").glob("*.png")) + sorted((scene / "depth").glob("*.npy")):
        sha.update(path.name.encode()); sha.update(path.read_bytes())
    return sha.hexdigest()


def fit_metrics(engine, frame, pose, mask):
    """Visible-mask coverage and depth residual, not a ground-truth pose error."""
    from Utils import nvdiffrast_render
    import torch
    centered = pose.copy()
    centered[:3, 3] += pose[:3, :3] @ engine.estimator.model_center
    with torch.inference_mode():
        _, depth, _ = nvdiffrast_render(K=frame.K, H=frame.rgb.shape[0], W=frame.rgb.shape[1],
            ob_in_cams=torch.as_tensor(centered[None], dtype=torch.float32, device="cuda"),
            glctx=engine.estimator.glctx, mesh_tensors=engine.estimator.mesh_tensors, extra={})
    rendered = depth[0].cpu().numpy()
    overlap = mask & (frame.depth_m >= .001) & (rendered >= .001)
    residual = np.abs(rendered[overlap] - frame.depth_m[overlap]) * 1000
    return {"visible_mask_coverage": float(np.count_nonzero(mask & (rendered > 0))/mask.sum()),
            "overlap_depth_pixels": int(overlap.sum()),
            "depth_abs_median_mm": float(np.median(residual)) if len(residual) else None,
            "depth_abs_p90_mm": float(np.percentile(residual, 90)) if len(residual) else None}


def replay(args):
    config = load_config(args.config)
    if args.variant == "baseline":
        config["body_orientation_prior"]["enabled"] = False
    elif args.variant == "grid_only":
        config["body_orientation_prior"]["strength"] = 0.
    root = Path(config["foundationpose_root"])
    sys.path[:0] = [str(root), str(root / "workspace/src")]
    from rail_pose.data import Scene
    from rail_pose.geometry import overlay
    from Utils import set_seed
    set_seed(0)
    scene = Scene(args.scene)
    args.output.mkdir(parents=True, exist_ok=False)
    mask = np.asarray(Image.open(args.mask).convert("L")) > 0
    serial = scene.manifest["camera"]["serial"]
    reference, error = pose_reference(config, serial)
    if reference is None:
        raise ValueError(error)
    alignment = RailPoseAlignment.from_config(config)
    factory_args = argparse.Namespace(foundationpose_root=root, mesh=Path(config["mesh"]), camera_serial=serial)
    engine = pose_engine(factory_args, args.output, config)
    manifest = {"variant": args.variant, "scene": str(args.scene), "mask": str(args.mask),
        "input_sha256": scene_digest(args.scene, args.mask), "mesh_sha256": digest(config["mesh"]),
        "calibration_sha256": digest(config["calibration"]), "seed": 0,
        "prior": config["body_orientation_prior"], "engine_config": engine.config,
        "center_cad_m": engine.mesh.bounds.mean(axis=0).tolist(),
        "frames": len(scene), "robot_connected": False,
        "fit_note": "fixed visible mask, occluded geometry excluded from coverage denominator; no pose ground truth"}
    write_json(args.output / "manifest.json", manifest)
    with (args.output / "records.jsonl").open("w") as stream:
        for index in range(len(scene)):
            frame = replace(scene.frame(index), mask=mask)
            try:
                pose, record = engine.infer(frame)
                record = alignment.align_record(record, reference)
                pose = np.asarray(record["T_C_CAD"])
                if index % 10 == 0 or index == len(scene)-1:
                    record["fit"] = fit_metrics(engine, frame, pose, mask)
                    Image.fromarray(overlay(frame.rgb, engine.mesh, pose, frame.K,
                        label=f"{args.variant} frame {index}")).save(args.output / f"pose_{index:03d}.png")
            except (ValueError, RuntimeError) as error:
                record = {"valid": False, "reason": str(error), "frame_id": frame.frame_id,
                          "timestamp_ns": frame.timestamp_ns}
                engine.reset()
            record["index"] = index
            stream.write(json.dumps(record, allow_nan=False) + "\n"); stream.flush()
            if index % 20 == 0 or not record["valid"]:
                print(json.dumps({"variant": args.variant, "index": index,
                                  "valid": record["valid"], "reason": record.get("reason")}), flush=True)


def metrics(records, center, burn_in):
    valid = [r for r in records if r["valid"] and r["index"] >= burn_in]
    if len(valid) < 3:
        return {"valid": len(valid), "error": "insufficient valid poses"}
    poses = np.asarray([r["T_C_CAD"] for r in valid])
    centers = poses[:, :3, 3] + poses[:, :3, :3] @ center
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    angle = np.degrees((rotations * rotations.mean().inv()).magnitude())
    endpoint_spread = np.linalg.norm(poses[:, :3, 3]-poses[:, :3, 3].mean(axis=0), axis=1)*1000
    center_spread = np.linalg.norm(centers-centers.mean(axis=0), axis=1)*1000
    adjacent = np.diff([r["index"] for r in valid]) == 1
    step_angle = np.degrees((rotations[1:] * rotations[:-1].inv()).magnitude())[adjacent]
    step_center = np.linalg.norm(np.diff(centers, axis=0), axis=1)[adjacent]*1000
    step_endpoint = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)[adjacent]*1000
    triples = passed = 0
    for i in range(len(valid)-2):
        if valid[i+2]["index"]-valid[i]["index"] != 2:
            continue
        triples += 1
        try:
            mean_pose(poses[i:i+3]); passed += 1
        except ValueError:
            pass
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
    p95 = lambda x: float(np.percentile(x, 95)) if len(x) else None
    result = {"valid": len(valid), "burn_in": burn_in,
        "rotation_spread_rms_deg": rms(angle), "center_spread_rms_mm": rms(center_spread),
        "endpoint_spread_rms_mm": rms(endpoint_spread), "rotation_step_rms_deg": rms(step_angle),
        "rotation_step_p95_deg": p95(step_angle), "center_step_rms_mm": rms(step_center),
        "endpoint_step_rms_mm": rms(step_endpoint), "endpoint_step_p95_mm": p95(step_endpoint),
        "three_frame_passed": passed, "three_frame_windows": triples,
        "mean_center_camera_m": centers.mean(axis=0).tolist(),
        "mean_rotation_camera": rotations.mean().as_matrix().tolist()}
    fits = [r["fit"] for r in valid if "fit" in r]
    if fits:
        result["fit_mean"] = {k: float(np.mean([f[k] for f in fits if f[k] is not None]))
                              for k in fits[0] if any(f[k] is not None for f in fits)}
    return result


def summarize(args):
    report = {"limitations": ["Static scene only; no independent pose ground truth or robot motion.",
        "Three-frame metric is an offline pose-consistency proxy, not a live transfer success rate."], "runs": {}}
    hashes = set()
    for directory in args.runs:
        manifest = json.loads((directory / "manifest.json").read_text())
        records = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
        if [r["index"] for r in records] != list(range(manifest["frames"])):
            raise ValueError("Cannot summarize an incomplete or out-of-order replay")
        if manifest["variant"] in report["runs"]:
            raise ValueError("Each comparison run must have a distinct variant")
        hashes.add(manifest["input_sha256"])
        report["runs"][manifest["variant"]] = {"directory": str(directory),
            "total": len(records), "valid": sum(r["valid"] for r in records),
            "all": metrics(records, np.asarray(manifest["center_cad_m"]), 0),
            "steady": metrics(records, np.asarray(manifest["center_cad_m"]), args.burn_in)}
    if len(hashes) != 1:
        raise ValueError("Comparison requires exactly identical RGB-D and mask inputs")
    report["input_sha256"] = hashes.pop()
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("replay")
    run.add_argument("--config", type=Path, default=HERE / "receive.yaml")
    run.add_argument("--scene", type=Path, required=True)
    run.add_argument("--mask", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--variant", choices=["baseline", "grid_only", "prior"], required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("--runs", type=Path, nargs="+", required=True)
    summary.add_argument("--burn-in", type=int, default=10)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args) if args.command == "replay" else summarize(args)


if __name__ == "__main__":
    main()
