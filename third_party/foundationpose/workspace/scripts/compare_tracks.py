"""Draw two tracking runs on every frame so a reviewer can tell which one sits on the rail.

Each run gets its own contour colour; a filled dot marks its CAD Y = 0 end and a ring its Y = max end,
so an end-flipped run shows dot and ring swapped. Writes compare.mp4 and contact sheets of rail crops.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rail_pose.common import load_config, resolve  # noqa: E402
from rail_pose.data import Scene  # noqa: E402
from review_episode import silhouette  # noqa: E402
from score_candidates import end_flip, load_run  # noqa: E402

COLORS = [(60, 220, 60), (0, 160, 255), (255, 80, 200)]


def project(K, pose, point):
    p = K @ (pose[:3, :3] @ point + pose[:3, 3])
    return tuple(np.round(p[:2] / p[2]).astype(int)) if p[2] > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="first run defines the CAD end convention")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--flip-runs", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--mesh")
    parser.add_argument("--sheet-from", type=int, default=0)
    parser.add_argument("--sheet-to", type=int)
    parser.add_argument("--sheet-step", type=int, default=20)
    parser.add_argument("--sheet-columns", type=int, default=2)
    parser.add_argument("--sheet-rows", type=int, default=4)
    args = parser.parse_args()
    scene = Scene(resolve(args.scene))
    mesh = trimesh.load(resolve(args.mesh or load_config()["model"]), force="mesh")
    flip = end_flip(mesh)
    labels = args.labels or [Path(run).name for run in args.runs]
    flipped = {Path(run).name for run in args.flip_runs}
    runs = [{k: (v @ flip if Path(run).name in flipped else v) for k, v in load_run(resolve(run)).items()}
            for run in args.runs]
    (x0, y0, z0), (x1, y1, z1) = mesh.bounds
    ends = np.array([[(x0 + x1) / 2, y0, (z0 + z1) / 2], [(x0 + x1) / 2, y1, (z0 + z1) / 2]])
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    width, height = scene.manifest["camera"]["width"], scene.manifest["camera"]["height"]
    video = cv2.VideoWriter(str(output / "compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            scene.manifest["camera"].get("fps", 30), (width, height))
    last = len(scene) - 1 if args.sheet_to is None else args.sheet_to
    sheet_frames = set(range(args.sheet_from, last + 1, args.sheet_step))
    tiles = []
    for index, record in enumerate(scene.records):
        frame = scene.frame(index)
        image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        clean, box = image.copy(), []
        for run, color in zip(runs, COLORS):
            pose = run.get(record["frame_id"])
            if pose is None:
                continue
            mask = silhouette(mesh, pose, frame.K, image.shape[:2])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 1, cv2.LINE_AA)
            start, end = (project(frame.K, pose, p) for p in ends)
            if start:
                cv2.circle(image, start, 5, color, -1, cv2.LINE_AA)
            if end:
                cv2.circle(image, end, 7, color, 2, cv2.LINE_AA)
            box += [c.reshape(-1, 2) for c in contours]
        cv2.rectangle(image, (0, 0), (width, 24), (30, 30, 30), -1)
        cv2.putText(image, f"{index:4d}", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        x = 60
        for label, color in zip(labels, COLORS):
            cv2.putText(image, label, (x, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)
            x += 12 * len(label) + 20
        video.write(image)
        if index in sheet_frames:
            if box:
                points = np.concatenate(box)
                (bx0, by0), (bx1, by1) = points.min(0) - 50, points.max(0) + 50
            else:
                bx0, by0, bx1, by1 = 0, 0, width, height
            bx0, by0 = max(int(bx0), 0), max(int(by0), 25)
            bx1, by1 = min(int(bx1), width), min(int(by1), height)
            side = min(max(bx1 - bx0, by1 - by0), width, height)
            cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
            ax0, ay0 = int(np.clip(cx - side // 2, 0, max(width - side, 0))), int(np.clip(cy - side // 2, 0, max(height - side, 0)))
            pair = [cv2.resize(img[ay0:ay0 + side, ax0:ax0 + side], (360, 360)) for img in (clean, image)]
            tile = np.concatenate(pair, axis=1)
            cv2.putText(tile, f"{index}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
    video.release()
    per_sheet = args.sheet_columns * args.sheet_rows
    for start in range(0, len(tiles), per_sheet):
        group = tiles[start:start + per_sheet]
        group += [np.zeros_like(tiles[0])] * (per_sheet - len(group))
        rows = [np.concatenate(group[r * args.sheet_columns:(r + 1) * args.sheet_columns], axis=1)
                for r in range(args.sheet_rows)]
        cv2.imwrite(str(output / f"sheet_{start // per_sheet:02d}.jpg"), np.concatenate(rows, axis=0))
    print(output)


if __name__ == "__main__":
    main()
