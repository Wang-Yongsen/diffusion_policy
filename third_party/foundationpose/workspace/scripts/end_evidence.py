"""Per frame and tracking run: does the camera see each rail end where the pose puts it?

An end is seen when the strip the rail would cover if it were `outside_mm` longer shows background
behind it while the camera does not see past the last `inside_mm` of the rendered rail. Rail surface in
that strip means the pose ends too early; background inside means it ends too late; nearer depth
outside means the end is occluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rail_pose.common import load_config, resolve, save_json  # noqa: E402
from rail_pose.data import Scene  # noqa: E402
from review_episode import DepthRenderer  # noqa: E402
from score_candidates import end_flip, load_run  # noqa: E402


def slid(pose, offset_m):
    moved = pose.copy()
    moved[:3, 3] += pose[:3, 1] * offset_m
    return moved


def end_evidence(observed, base, inward, outward, tolerance_m, background_m):
    """Pixel counts at one end; `inward`/`outward` are the depth renders shortened/lengthened at that end."""
    shell = cv2.dilate((base > 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    inside = cv2.erode(((base > 0) & (inward == 0)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    outside = (outward > 0) & ~shell
    measured_in, measured_out = inside & (observed > 0), outside & (observed > 0)
    gap_in, gap_out = observed - base, observed - outward
    return {"inside_px": int(inside.sum()), "inside_measured_px": int(measured_in.sum()),
            "inside_consistent_px": int((measured_in & (np.abs(gap_in) <= tolerance_m)).sum()),
            "inside_behind_px": int((measured_in & (gap_in > background_m)).sum()),
            "outside_px": int(outside.sum()), "outside_measured_px": int(measured_out.sum()),
            "outside_empty_px": int((measured_out & (gap_out > background_m)).sum()),
            "outside_rail_px": int((measured_out & (np.abs(gap_out) <= tolerance_m)).sum()),
            "outside_occluded_px": int((measured_out & (gap_out < -background_m)).sum())}


def verdict(evidence, min_px, min_share, min_inside_share):
    known = evidence["outside_empty_px"] + evidence["outside_rail_px"]
    measured = max(evidence["inside_measured_px"], 1)
    if evidence["outside_rail_px"] >= min_px and evidence["outside_rail_px"] / max(known, 1) >= min_share:
        return "rail_continues"
    if evidence["inside_behind_px"] >= min_px and evidence["inside_behind_px"] / measured > .5:
        return "rail_ends_earlier"
    # A hand covering the real end and the space beyond it also leaves background past a too-long pose.
    if (evidence["outside_empty_px"] >= min_px and evidence["outside_empty_px"] / max(known, 1) >= min_share
            and evidence["inside_consistent_px"] >= min_px
            and evidence["inside_consistent_px"] / measured >= min_inside_share):
        return "seen"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="tracking outputs; the first defines the CAD end convention")
    parser.add_argument("--flip-runs", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--mesh")
    parser.add_argument("--inside-mm", type=float, default=20.)
    parser.add_argument("--outside-mm", type=float, default=40.)
    parser.add_argument("--tolerance-mm", type=float, default=10.)
    parser.add_argument("--background-mm", type=float, default=15.)
    parser.add_argument("--min-px", type=int, default=30)
    parser.add_argument("--min-share", type=float, default=.8)
    parser.add_argument("--min-inside-share", type=float, default=.3,
                        help="share of measured end pixels on the CAD surface; thin metal edges keep it low")
    args = parser.parse_args()
    scene = Scene(resolve(args.scene))
    mesh = trimesh.load(resolve(args.mesh or load_config()["model"]), force="mesh")
    flip = end_flip(mesh)
    names = [Path(run).name for run in args.runs]
    flipped = {Path(run).name for run in args.flip_runs}
    runs = [{k: (v @ flip if name in flipped else v) for k, v in load_run(resolve(run)).items()}
            for name, run in zip(names, args.runs)]
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    width, height = scene.manifest["camera"]["width"], scene.manifest["camera"]["height"]
    renderer = DepthRenderer(mesh, scene.records[0]["K"], (height, width))
    inside, outside = args.inside_mm / 1000, args.outside_mm / 1000
    # CAD Y runs from end "y0" (Y = 0) to end "y480" (Y = max); lengthening an end moves the pose that way.
    ends = {"y0": (inside, -outside), "y480": (-inside, outside)}
    seen = {name: [] for name in names}
    with (output / "end_evidence.jsonl").open("w", encoding="utf-8") as stream:
        for index, record in enumerate(scene.records):
            frame_id, row = record["frame_id"], {"index": index, "frame_id": record["frame_id"], "runs": {}}
            present = [k for k, run in enumerate(runs) if frame_id in run]
            if present:
                observed = scene.frame(index).depth_m
                poses = []
                for k in present:
                    pose = runs[k][frame_id]
                    poses += [pose] + [slid(pose, offset) for end in ends.values() for offset in end]
                depth = renderer(poses).reshape(len(present), 1 + 2 * len(ends), height, width)
                for slot, k in enumerate(present):
                    result = {}
                    for e, (end, _) in enumerate(ends.items()):
                        evidence = end_evidence(observed, depth[slot, 0], depth[slot, 1 + 2 * e], depth[slot, 2 + 2 * e],
                                                args.tolerance_mm / 1000, args.background_mm / 1000)
                        result[end] = {"verdict": verdict(evidence, args.min_px, args.min_share, args.min_inside_share),
                                       **evidence}
                    row["runs"][names[k]] = result
                    if any(r["verdict"] == "seen" for r in result.values()):
                        seen[names[k]].append(index)
            stream.write(json.dumps(row) + "\n")
            if index % 100 == 0:
                print(f"{index}/{len(scene)}", flush=True)

    def spans(indices):
        result = []
        for i in indices:
            if result and i == result[-1][1] + 1:
                result[-1][1] = i
            else:
                result.append([i, i])
        return result

    summary = {"scene": str(scene.directory), "runs": dict(zip(names, map(str, args.runs))),
               "end_flipped_runs": sorted(flipped), "end_names": {"y0": "CAD Y = 0", "y480": "CAD Y = max"},
               "rule": {k: getattr(args, k) for k in ("inside_mm", "outside_mm", "tolerance_mm", "background_mm",
                                                       "min_px", "min_share", "min_inside_share")},
               "frames_with_seen_end": {n: len(v) for n, v in seen.items()},
               "seen_end_spans": {n: spans(v) for n, v in seen.items()}}
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
