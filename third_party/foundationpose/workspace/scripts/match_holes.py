"""Slide a rail pose along CAD Y and score CAD holes against dark spots in RGB."""
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
from score_candidates import end_flip, load_run  # noqa: E402


def hole_centers(mesh) -> np.ndarray:
    """Centres of the through-holes in the channel floor, original CAD metres."""
    section = mesh.section(plane_origin=[0, 0, -0.0153], plane_normal=[0, 0, 1])
    if section is None:
        raise ValueError("CAD midplane does not cut the channel floor")
    holes = []
    for loop in section.discrete:
        loop = np.asarray(loop)
        if np.ptp(loop[:, 1]) >= 0.10:
            continue
        holes.append(loop.mean(0))
    holes = np.asarray(sorted(holes, key=lambda p: p[1]))
    if len(holes) < 6:
        raise ValueError(f"expected the ten floor holes, found {len(holes)}")
    return holes


def slide(pose, offset_m):
    delta = np.eye(4)
    delta[1, 3] = offset_m
    return pose @ delta


def project(pose, K, xyz):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    pixels = camera @ K.T
    return pixels[:, :2] / pixels[:, 2:], camera[:, 2]


def sample(image, pixels):
    map_x = pixels[:, 0].astype(np.float32)
    map_y = pixels[:, 1].astype(np.float32)
    return cv2.remap(image, map_x.reshape(1, -1), map_y.reshape(1, -1),
                     cv2.INTER_LINEAR, borderValue=0)[0]


def hole_score(gray, pose, K, holes, rail_mask):
    """Higher when projected holes land on dark metal and the gaps stay bright."""
    pixels, depth = project(pose, K, holes)
    height, width = gray.shape
    inside = (depth > 0.001) & (pixels[:, 0] >= 2) & (pixels[:, 0] < width - 2) \
        & (pixels[:, 1] >= 2) & (pixels[:, 1] < height - 2)
    if inside.sum() < 2:
        return None
    ix = np.round(pixels[inside]).astype(int)
    on_rail = rail_mask[ix[:, 1], ix[:, 0]]
    if on_rail.sum() < 2:
        return None
    used = pixels[inside][on_rail]
    hole = sample(gray, used)
    # Four neighbours 8 mm along CAD Y, still on the floor: should be brighter metal.
    neighbours = []
    for sign in (-1, 1):
        shifted, _ = project(pose, K, holes[inside][on_rail] + np.array([0, sign * 0.008, 0]))
        neighbours.append(sample(gray, shifted))
    metal = np.mean(neighbours, axis=0)
    contrast = metal.astype(np.float32) - hole.astype(np.float32)
    # Gloves and shadows are dark everywhere; require the neighbour to look like metal.
    keep = metal > 70
    if keep.sum() < 2:
        return None
    return {"score": float(contrast[keep].mean()), "n_holes": int(keep.sum()),
            "hole": float(hole[keep].mean()), "metal": float(metal[keep].mean())}


def search(gray, pose, K, holes, mesh, flips, offsets):
    from review_episode import silhouette
    results = []
    for flipped, candidate in flips:
        mask = silhouette(mesh, candidate, K, gray.shape).astype(bool)
        for offset in offsets:
            trial = slide(candidate, offset)
            measured = hole_score(gray, trial, K, holes, mask)
            if measured is None:
                continue
            results.append({**measured, "offset_mm": float(offset * 1000), "flipped": flipped,
                            "pose": trial})
    if not results:
        return None
    best = max(results, key=lambda r: (r["score"], r["n_holes"]))
    return best, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--run", required=True, help="poses.jsonl directory in T_C_CAD")
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--range-mm", type=float, default=150)
    parser.add_argument("--step-mm", type=float, default=2)
    parser.add_argument("--try-flip", action="store_true")
    args = parser.parse_args()
    scene = Scene(resolve(args.scene))
    mesh = trimesh.load(resolve(load_config()["model"]), force="mesh")
    holes = hole_centers(mesh)
    poses = load_run(resolve(args.run))
    flip = end_flip(mesh)
    offsets = np.arange(-args.range_mm, args.range_mm + 1e-9, args.step_mm) / 1000
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()

    rows = []
    for index in args.frames:
        frame = scene.frame(index)
        if frame.frame_id not in poses:
            continue
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        pose = poses[frame.frame_id]
        variants = [(False, pose)]
        if args.try_flip:
            variants.append((True, pose @ flip))
        found = search(gray, pose, frame.K, holes, mesh, variants, offsets)
        if found is None:
            rows.append({"index": index, "frame_id": frame.frame_id, "valid": False})
            continue
        best, curve = found
        image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        pixels, depth = project(best["pose"], frame.K, holes)
        for pixel, z in zip(pixels, depth):
            if z > 0.001:
                cv2.circle(image, tuple(np.round(pixel).astype(int)), 4, (0, 255, 255), 1)
        label = f"{index} {'flip' if best['flipped'] else 'same'} {best['offset_mm']:+.0f}mm score {best['score']:.1f} n={best['n_holes']}"
        cv2.putText(image, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(output / "frames" / f"{index:04d}.jpg"), image)
        rows.append({"index": index, "frame_id": frame.frame_id, "valid": True,
                     "offset_mm": best["offset_mm"], "flipped": best["flipped"],
                     "score": best["score"], "n_holes": best["n_holes"],
                     "hole": best["hole"], "metal": best["metal"],
                     "curve": [{"offset_mm": c["offset_mm"], "flipped": c["flipped"],
                                "score": c["score"], "n_holes": c["n_holes"]} for c in curve]})
    save_json(output / "holes.json", {"holes_cad_m": holes.tolist(), "frames": rows})
    print(json.dumps({
        "output": str(output),
        "holes": len(holes),
        "frames": [{k: r[k] for k in r if k != "curve"} for r in rows],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
