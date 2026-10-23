"""Slide a rail pose along its long axis (CAD Y) and match hole and end edges with the RGB image.

Hole regions are rendered exactly by closing every bottom-plate hole with two caps inside the plate:
a pixel shows a cap only where the line of sight passes through a hole.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rail_pose.common import load_config, resolve, save_json  # noqa: E402
from rail_pose.data import Scene  # noqa: E402
from score_candidates import end_flip, load_run  # noqa: E402

PLATE_SECTION_M = -16.1e-3
CAP_Z_M = (-16.9e-3, -15.3e-3)


def hole_caps(mesh):
    loops = [l for l in mesh.section(plane_origin=[0, 0, PLATE_SECTION_M], plane_normal=[0, 0, 1]).discrete
             if np.ptp(l[:, 1]) < .1]
    vertices, faces = [], []
    for loop in loops:
        points = loop[:-1] if np.allclose(loop[0], loop[-1]) else loop
        centre = points.mean(axis=0)
        for z in CAP_Z_M:
            base = len(vertices)
            vertices += [[centre[0], centre[1], z]] + [[p[0], p[1], z] for p in points]
            faces += [[base, base + 1 + i, base + 1 + (i + 1) % len(points)] for i in range(len(points))]
    return np.asarray(vertices), np.asarray(faces), len(loops)


class SlideMatcher:
    def __init__(self, mesh, K, shape, occlusion_m=.015):
        import torch
        import nvdiffrast.torch as dr
        self.torch, self.dr = torch, dr
        caps, cap_faces, self.holes = hole_caps(mesh)
        self.mesh_faces = len(mesh.faces)
        vertices = np.vstack([mesh.vertices, caps])
        faces = np.vstack([mesh.faces, cap_faces + len(mesh.vertices)])
        self.vertices = torch.as_tensor(vertices, dtype=torch.float32, device="cuda")
        self.faces = torch.as_tensor(faces, dtype=torch.int32, device="cuda")
        self.context = dr.RasterizeCudaContext()
        self.K, self.shape, self.occlusion_m = np.asarray(K, float), shape, occlusion_m
        self.centre = mesh.bounds.mean(axis=0)
        x = torch.arange(-3, 4, device="cuda", dtype=torch.float32)
        g = torch.exp(-x ** 2 / 2)
        self.gauss = (g / g.sum())

    def _blur(self, image):
        F = self.torch.nn.functional
        image = image[:, None]
        image = F.conv2d(F.pad(image, (3, 3, 0, 0), mode="replicate"), self.gauss.view(1, 1, 1, 7))
        image = F.conv2d(F.pad(image, (0, 0, 3, 3), mode="replicate"), self.gauss.view(1, 1, 7, 1))
        return image[:, 0]

    def _along(self, image, u):
        """Absolute derivative along the image direction u, i.e. edges crossing the rail axis."""
        F = self.torch.nn.functional
        padded = F.pad(image[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
        gx = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / 2
        gy = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / 2
        return (gx * u[0] + gy * u[1]).abs()

    def _render(self, poses):
        torch = self.torch
        height, width = self.shape
        poses = torch.as_tensor(poses, dtype=torch.float32, device="cuda")
        camera = self.vertices @ poses[:, :3, :3].transpose(1, 2) + poses[:, None, :3, 3]
        X, Y, Z = camera.unbind(-1)
        (fx, _, cx), (_, fy, cy) = self.K[:2]
        near, far = .01, 10.
        clip = torch.stack([(fx * X + (cx + .5) * Z) * 2 / width - Z, (fy * Y + (cy + .5) * Z) * 2 / height - Z,
                            Z * (far + near) / (far - near) - 2 * far * near / (far - near), Z], -1)
        raster, _ = self.dr.rasterize(self.context, clip.contiguous(), self.faces, (height, width))
        depth, _ = self.dr.interpolate(Z[..., None].contiguous(), raster, self.faces)
        triangle = raster[..., 3]
        return triangle > 0, triangle > self.mesh_faces, depth[..., 0]

    def direction(self, pose):
        a = self.K @ (pose[:3, :3] @ self.centre + pose[:3, 3])
        b = self.K @ (pose[:3, :3] @ (self.centre + [0, .05, 0]) + pose[:3, 3])
        u = b[:2] / b[2] - a[:2] / a[2]
        return u / np.linalg.norm(u)

    def scores(self, frame, poses, batch=48):
        torch = self.torch
        u = torch.as_tensor(self.direction(poses[len(poses) // 2]), dtype=torch.float32, device="cuda")
        gray = torch.as_tensor(frame.rgb, dtype=torch.float32, device="cuda").mean(-1)[None] / 255
        image = self._along(self._blur(gray), u)[0]
        observed = torch.as_tensor(frame.depth_m, dtype=torch.float32, device="cuda")
        result = []
        for start in range(0, len(poses), batch):
            silhouette, hole, depth = self._render(np.asarray(poses[start:start + batch]))
            material = (silhouette & ~hole).float()
            rendered = self._along(self._blur(material), u)
            near = torch.nn.functional.max_pool2d(silhouette.float()[:, None], 5, 1, 2)[:, 0] > 0
            occluded = (observed > 0) & silhouette & (observed < depth - self.occlusion_m)
            occluded = torch.nn.functional.max_pool2d(occluded.float()[:, None], 5, 1, 2)[:, 0] > 0
            valid = (near & ~occluded).float()
            num = (image * rendered * valid).sum((1, 2))
            den = torch.sqrt((rendered ** 2 * valid).sum((1, 2)) * (image ** 2 * valid).sum((1, 2))) + 1e-9
            result.append((num / den).cpu().numpy())
        return np.concatenate(result)


def slid(pose, offsets_m):
    poses = np.repeat(pose[None], len(offsets_m), axis=0)
    poses[:, :3, 3] += np.outer(offsets_m, pose[:3, 1])
    return poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--tracks", nargs="+", required=True, help="run or result directories with poses.jsonl")
    parser.add_argument("--flip-tracks", nargs="*", default=[])
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--range-mm", type=float, default=150.)
    parser.add_argument("--step-mm", type=float, default=1.)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scene = Scene(resolve(args.scene))
    mesh = trimesh.load(resolve(load_config()["model"]), force="mesh")
    flip = end_flip(mesh)
    flipped = {Path(t).name for t in args.flip_tracks}
    tracks = {Path(t).name: {k: (v @ flip if Path(t).name in flipped else v) for k, v in load_run(resolve(t)).items()}
              for t in args.tracks}
    matcher = SlideMatcher(mesh, scene.records[0]["K"], (scene.manifest["camera"]["height"], scene.manifest["camera"]["width"]))
    offsets = np.arange(-args.range_mm, args.range_mm + 1e-9, args.step_mm) / 1000
    rows = []
    for index in args.frames:
        frame = scene.frame(index)
        row = {"index": index, "frame_id": frame.frame_id, "tracks": {}}
        for name, track in tracks.items():
            if frame.frame_id not in track:
                continue
            pose = track[frame.frame_id]
            entry = {}
            for label, base in (("kept", pose), ("flipped", pose @ flip)):
                score = matcher.scores(frame, slid(base, offsets))
                best = int(np.argmax(score))
                far = np.abs(offsets - offsets[best]) > .02
                entry[label] = {"best_offset_mm": float(offsets[best] * 1000), "best_score": float(score[best]),
                                "score_at_zero": float(score[np.argmin(np.abs(offsets))]),
                                "runner_up_beyond_20mm": float(score[far].max()) if far.any() else None,
                                "curve": score.round(5).tolist()}
            row["tracks"][name] = entry
        rows.append(row)
        print(json.dumps({"index": index, **{n: {k: (v["best_offset_mm"], round(v["best_score"], 4), round(v["runner_up_beyond_20mm"], 4))
                                                   for k, v in e.items()} for n, e in row["tracks"].items()}}))
    output = resolve(args.output)
    save_json(output, {"scene": str(scene.directory), "holes": matcher.holes, "offsets_mm": (offsets * 1000).tolist(),
                       "frames": rows})


if __name__ == "__main__":
    main()
