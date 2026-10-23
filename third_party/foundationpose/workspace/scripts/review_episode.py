"""Merge offline tracking runs of one scene, compare their overlaps and render a review video."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rail_pose.common import load_config, resolve, save_json  # noqa: E402
from rail_pose.data import Scene  # noqa: E402
from score_candidates import end_flip, load_run  # noqa: E402


class DepthRenderer:
    def __init__(self, mesh, K, shape):
        import torch
        import nvdiffrast.torch as dr
        self.torch, self.dr = torch, dr
        self.context = dr.RasterizeCudaContext()
        self.vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device="cuda")
        self.faces = torch.as_tensor(mesh.faces, dtype=torch.int32, device="cuda")
        self.K, self.shape = np.asarray(K, dtype=np.float64), shape

    def __call__(self, poses):
        """Metric Z of the nearest CAD surface per OpenCV pixel centre; 0 outside the silhouette."""
        torch = self.torch
        height, width = self.shape
        poses = torch.as_tensor(np.asarray(poses), dtype=torch.float32, device="cuda")
        camera = self.vertices @ poses[:, :3, :3].transpose(1, 2) + poses[:, None, :3, 3]
        X, Y, Z = camera.unbind(-1)
        (fx, _, cx), (_, fy, cy) = self.K[:2]
        near, far = .01, 10.
        clip = torch.stack([(fx * X + (cx + .5) * Z) * 2 / width - Z, (fy * Y + (cy + .5) * Z) * 2 / height - Z,
                            Z * (far + near) / (far - near) - 2 * far * near / (far - near), Z], -1)
        raster, _ = self.dr.rasterize(self.context, clip.contiguous(), self.faces, (height, width))
        depth, _ = self.dr.interpolate(Z[..., None].contiguous(), raster, self.faces)
        return torch.where(raster[..., 3:] > 0, depth, 0)[..., 0].cpu().numpy()


def depth_agreement(observed, rendered, tolerance_m):
    inside = cv2.erode((rendered > 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    measured = inside & (observed > 0)
    difference = observed - rendered
    count = max(int(inside.sum()), 1)
    return {"silhouette_px": int(inside.sum()), "measured_px": int(measured.sum()),
            "consistent": float((measured & (np.abs(difference) <= tolerance_m)).sum() / count),
            "occluded": float((measured & (difference < -tolerance_m)).sum() / count),
            "contradicted": float((measured & (difference > tolerance_m)).sum() / max(int(measured.sum()), 1))}


def silhouette(mesh, pose, K, shape):
    camera = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
    mask = np.zeros(shape, np.uint8)
    if np.any(camera[:, 2] <= .001):
        return mask
    pixels = camera @ K.T
    pixels = np.round(pixels[:, :2] / pixels[:, 2:] * 4).astype(np.int32)
    cv2.fillPoly(mask, list(pixels[mesh.faces]), 1, cv2.LINE_8, 2)
    return mask


def exclusive_against(observed, rendered, tolerance_m):
    """Per candidate: pixels only it claims where the camera sees past its surface."""
    claims = [r > 0 for r in rendered]
    result = []
    for k, depth in enumerate(rendered):
        others = np.any([c for j, c in enumerate(claims) if j != k], axis=0) if len(claims) > 1 else np.zeros_like(claims[k])
        others = cv2.dilate(others.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        result.append(int((claims[k] & ~others & (observed > 0) & (observed - depth > tolerance_m)).sum()))
    return result


def select_tracks(available, against, spreads, margin, agree_mm):
    """Keep the current track until another has `margin` fewer contradicting pixels, then move the
    switch back to the latest crossing (centres within `agree_mm`) since the last decisive evidence,
    or to the closest approach when the tracks never crossed."""
    choice = np.full(len(available), -1)
    current, decisive = None, 0
    for i, candidates in enumerate(available):
        if not candidates:
            current = None
            continue
        best = min(candidates, key=lambda k: against[i][k])
        if current not in candidates:
            current, decisive = best, i
        elif against[i][current] - against[i][best] > margin:
            gap = [j for j in range(decisive, i + 1) if best in available[j] and current in available[j]]
            crossings = [j for j in gap if spreads[j][current, best] <= agree_mm]
            switch = (crossings[-1] if crossings else min(gap, key=lambda j: spreads[j][current, best])) if gap else i
            choice[switch:i] = best
            current = best
        if against[i][current] + margin < max(against[i][k] for k in candidates):
            decisive = i
        choice[i] = current
    return choice


def spread(poses, centre):
    """Largest box-centre distance [mm] and unsigned long-axis angle [deg] between any two poses."""
    centres = [p[:3, :3] @ centre + p[:3, 3] for p in poses]
    axes = [p[:3, 1] for p in poses]
    distance = max((np.linalg.norm(a - b) for a in centres for b in centres), default=0.)
    angle = max((np.degrees(np.arccos(min(1., abs(a @ b)))) for a in axes for b in axes), default=0.)
    return float(distance * 1000), float(angle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="tracking outputs; the first defines the CAD end convention")
    parser.add_argument("--flip-runs", nargs="*", default=[], help="runs initialised with the opposite rail end")
    parser.add_argument("--not-visible", nargs="*", default=[], metavar="FIRST:LAST",
                        help="index ranges reviewed as showing no part of the target; no pose is reported")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mesh")
    parser.add_argument("--switch-margin-px", type=int, default=50)
    parser.add_argument("--tolerance-mm", type=float, default=10.)
    parser.add_argument("--agree-mm", type=float, default=20.)
    parser.add_argument("--agree-deg", type=float, default=5.)
    parser.add_argument("--max-contradicted", type=float, default=.15,
                        help="largest share of measured silhouette pixels whose depth lies behind the CAD surface")
    args = parser.parse_args()
    scene = Scene(resolve(args.scene))
    mesh = trimesh.load(resolve(args.mesh or load_config()["model"]), force="mesh")
    flip, centre = end_flip(mesh), mesh.bounds.mean(axis=0)
    names = [Path(run).name for run in args.runs]
    flipped = {Path(run).name for run in args.flip_runs}
    runs = [{k: (v @ flip if name in flipped else v) for k, v in load_run(resolve(run)).items()}
            for name, run in zip(names, args.runs)]
    hidden = set()
    for span in args.not_visible:
        first, last = map(int, span.split(":"))
        hidden.update(range(first, last + 1))
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)

    ids = [x["frame_id"] for x in scene.records]
    width, height = scene.manifest["camera"]["width"], scene.manifest["camera"]["height"]
    renderer = DepthRenderer(mesh, scene.records[0]["K"], (height, width))
    available, against, spreads = [], [], []
    for index, frame_id in enumerate(ids):
        candidates = [] if index in hidden else [k for k, run in enumerate(runs) if frame_id in run]
        evidence, pairs = {}, {}
        if candidates:
            rendered = renderer([runs[k][frame_id] for k in candidates])
            evidence = dict(zip(candidates, exclusive_against(scene.frame(index).depth_m, rendered, args.tolerance_mm / 1000)))
            for a in candidates:
                for b in candidates:
                    pairs[a, b] = spread([runs[a][frame_id], runs[b][frame_id]], centre)[0]
        available.append(candidates)
        against.append(evidence)
        spreads.append(pairs)
    choice = select_tracks(available, against, spreads, args.switch_margin_px, args.agree_mm)

    video = cv2.VideoWriter(str(output / "review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            scene.manifest["camera"].get("fps", 30), (width, height))
    colors = {"confirmed": (40, 220, 80), "depth_selected": (0, 230, 230), "ambiguous": (255, 120, 40),
              "single_track": (0, 200, 255), "contradicted": (40, 40, 255)}
    merged = []
    with (output / "poses.jsonl").open("w", encoding="utf-8") as stream:
        for index, frame_id in enumerate(ids):
            frame = scene.frame(index)
            image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            row = {"index": index, "frame_id": frame_id, "timestamp_ns": frame.timestamp_ns,
                   "frame": "T_C_CAD: original CAD coordinates to OpenCV camera coordinates",
                   "valid": bool(choice[index] >= 0), "quality": "not_visible" if index in hidden else "no_pose"}
            if choice[index] >= 0:
                source = int(choice[index])
                pose = runs[source][frame_id]
                others = [run[frame_id] for run in runs if frame_id in run]
                distance, angle = spread(others, centre)
                agreement = depth_agreement(frame.depth_m, renderer([pose])[0], args.tolerance_mm / 1000)
                if agreement["contradicted"] > args.max_contradicted:
                    quality = "contradicted"
                elif len(others) < 2:
                    quality = "single_track"
                elif distance <= args.agree_mm and angle <= args.agree_deg:
                    quality = "confirmed"
                else:
                    rivals = [against[index][k] for k in available[index] if k != source]
                    quality = ("depth_selected" if min(rivals) - against[index][source] > args.switch_margin_px
                               else "ambiguous")
                contours, _ = cv2.findContours(silhouette(mesh, pose, frame.K, image.shape[:2]),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, colors[quality], 1, cv2.LINE_AA)
                origin = frame.K @ pose[:3, 3]
                if origin[2] > 0:
                    cv2.circle(image, tuple(np.round(origin[:2] / origin[2]).astype(int)), 4, (255, 0, 255), -1)
                row.update(quality=quality, source_run=names[source], tracks=len(others),
                           track_spread_mm=distance, track_spread_deg=angle, depth_agreement=agreement,
                           exclusive_contradicting_px={names[k]: v for k, v in against[index].items()},
                           T_C_CAD=pose.tolist(), translation_m=pose[:3, 3].tolist(),
                           quaternion_xyzw=Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
                           box_centre_m=(pose[:3, :3] @ centre + pose[:3, 3]).tolist())
            label = f"{index:4d} {frame_id} {row['quality']}"
            if row["valid"]:
                label += f" spread {row['track_spread_mm']:.0f}mm/{row['track_spread_deg']:.1f}deg"
            cv2.rectangle(image, (0, 0), (width, 24), (30, 30, 30), -1)
            cv2.putText(image, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            video.write(image)
            if index % 40 == 0 or index == len(scene) - 1:
                (output / "frames").mkdir(exist_ok=True)
                cv2.imwrite(str(output / "frames" / f"{index:04d}.jpg"), image)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            merged.append(row)
    video.release()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seconds = (np.array([r["timestamp_ns"] for r in merged]) - merged[0]["timestamp_ns"]) / 1e9
    figure, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    centres = np.array([r["box_centre_m"] if r["valid"] else [np.nan] * 3 for r in merged])
    for axis, label in enumerate("XYZ"):
        axes[0].plot(seconds, centres[:, axis], label=f"camera {label}")
    axes[0].set_ylabel("rail box centre [m]")
    axes[0].legend(loc="upper right")
    axes[1].plot(seconds, [r.get("track_spread_mm", np.nan) for r in merged], label="centre spread [mm]")
    axes[1].plot(seconds, [r.get("track_spread_deg", np.nan) for r in merged], label="long-axis spread [deg]")
    axes[1].axhline(args.agree_mm, color="gray", lw=.6, ls=":")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("independent tracks")
    axes[1].legend(loc="upper right", fontsize=8)
    for key in ("consistent", "occluded", "contradicted"):
        axes[2].plot(seconds, [r["depth_agreement"][key] if r["valid"] else np.nan for r in merged], label=key, lw=.8)
    axes[2].axhline(args.max_contradicted, color="red", lw=.6, ls=":")
    axes[2].set_ylabel("depth share")
    axes[2].set_xlabel("episode time [s]")
    axes[2].legend(loc="upper right", fontsize=8)
    for axis in axes:
        for r in merged:
            if r["quality"] in ("ambiguous", "contradicted", "no_pose", "not_visible"):
                axis.axvspan(seconds[r["index"]] - 1 / 60, seconds[r["index"]] + 1 / 60,
                             color="orange" if r["quality"] == "ambiguous" else "red", alpha=.08, lw=0)
    figure.tight_layout()
    figure.savefig(output / "trajectory.png", dpi=110)

    summary = {"scene": str(scene.directory), "runs": dict(zip(names, map(str, args.runs))),
               "end_flipped_runs": sorted(flipped), "end_flip_in_cad": flip.tolist(),
               "frames": len(merged), "valid_frames": sum(r["valid"] for r in merged),
               "frames_by_run": {n: sum(r.get("source_run") == n for r in merged) for n in names},
               "frames_by_quality": {q: sum(r["quality"] == q for r in merged)
                                     for q in ("confirmed", "depth_selected", "ambiguous", "single_track",
                                               "contradicted", "not_visible", "no_pose")},
               "not_visible_ranges": args.not_visible,
               "switches": [{"index": i, "from": names[choice[i - 1]], "to": names[choice[i]]}
                            for i in range(1, len(choice)) if choice[i] >= 0 and choice[i - 1] >= 0 and choice[i] != choice[i - 1]],
               "quality_rule": {"agree_mm": args.agree_mm, "agree_deg": args.agree_deg,
                                "depth_tolerance_mm": args.tolerance_mm, "max_contradicted": args.max_contradicted,
                                "silhouette_erosion_px": 2, "switch_margin_px": args.switch_margin_px,
                                "selection": "fewest exclusive-silhouette pixels with camera depth behind the CAD surface"},
               "physical_accuracy_verified": False}
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
