"""Recover the rail position along its long axis from arm kinematics while a hand holds it.

While grasp g of arm a lasts, T_C_rail(t) = X_a @ F_a(t) @ Y_g with F_a the flange pose from FK of the
measured joints, X_a the fixed camera-to-arm-base transform and Y_g the fixed flange-to-rail grasp. The
camera tracks fix rotation and the box-centre position across the rail axis in every trusted frame,
but the position along the axis only in anchor frames; overlapping grasps must agree along the axis.
The rail frame here is the CAD frame moved to the box centre, so the end flip only flips its X and Y.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rail_pose.common import resolve, save_json  # noqa: E402

TRUSTED = {"confirmed", "depth_selected", "ambiguous"}
ARM = np.dtype([("t", "<i8"), ("seq", "<i8"), ("v", "<f8", 13)])


def span(text):
    first, last = map(int, text.split(":"))
    return range(first, last + 1)


def matrix(rotvec, translation):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_rotvec(rotvec).as_matrix(), translation
    return T


def flange_poses(kinematics, path, side, times_ns):
    """Base-to-flange transforms at the given host times from linearly interpolated measured joints."""
    state = np.fromfile(path, dtype=ARM)
    joints = np.stack([np.interp(times_ns, state["t"], state["v"][:, j]) for j in range(7)], axis=1)
    result = []
    for q in joints:
        pose = kinematics.fk(side, tuple(q))
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(pose.orientation_xyzw).as_matrix()
        T[:3, 3] = pose.position_m
        result.append(T)
    return np.array(result)


class Problem:
    def __init__(self, grasps, rail, flanges, anchors, sigma_mm, sigma_deg):
        self.grasps, self.rail, self.flanges, self.anchors = grasps, rail, flanges, anchors
        self.arms = sorted({g["arm"] for g in grasps})
        self.position, self.angle = sigma_mm / 1000, np.radians(sigma_deg)
        self.overlaps = [(a, b, sorted(set(ga["frames"]) & set(gb["frames"])))
                         for a, ga in enumerate(grasps) for b, gb in enumerate(grasps) if a < b and ga["arm"] != gb["arm"]]

    def unpack(self, x):
        x = x.reshape(-1, 6)
        X = {arm: matrix(*np.split(x[k], 2)) for k, arm in enumerate(self.arms)}
        Y = [matrix(*np.split(x[len(self.arms) + g], 2)) for g in range(len(self.grasps))]
        return X, Y

    def predict(self, X, Y, g, frames):
        grasp = self.grasps[g]
        return X[grasp["arm"]] @ self.flanges[grasp["arm"]][frames] @ Y[g]

    def residuals(self, x):
        X, Y = self.unpack(x)
        out = []
        for g, grasp in enumerate(self.grasps):
            frames = np.array(grasp["frames"])
            T, R = self.predict(X, Y, g, frames), self.rail[frames]
            axis = R[:, :3, 1]
            out.append(((T[:, :3, 2] - R[:, :3, 2]) / self.angle).ravel())
            out.append((np.cross(T[:, :3, 1], axis) / self.angle).ravel())
            gap = T[:, :3, 3] - R[:, :3, 3]
            along = np.einsum("ij,ij->i", gap, axis)
            out.append(((gap - along[:, None] * axis) / self.position).ravel())
            anchored = np.isin(frames, self.anchors)
            out.append(along[anchored] / self.position)
        for a, b, frames in self.overlaps:
            if frames:
                gap = self.predict(X, Y, a, frames)[:, :3, 3] - self.predict(X, Y, b, frames)[:, :3, 3]
                out.append(np.einsum("ij,ij->i", gap, self.rail[frames][:, :3, 1]) / self.position)
        return np.concatenate(out)


def initial_guess(problem, stride=5, min_angle_deg=5.):
    """Per arm: R_X from rotation axes of frame-pair motions (rail motion = R_X flange motion R_X^T),
    then the grasp rotations, then all translations by linear least squares on the tracked centres."""
    X, Y = {}, [np.eye(4) for _ in problem.grasps]
    for arm in problem.arms:
        owned = [g for g, grasp in enumerate(problem.grasps) if grasp["arm"] == arm]
        seen, moved = [], []
        for g in owned:
            frames = problem.grasps[g]["frames"][::stride]
            for a, i in enumerate(frames):
                for j in frames[a + 1:]:
                    rail = Rotation.from_matrix(problem.rail[i][:3, :3] @ problem.rail[j][:3, :3].T).as_rotvec()
                    flange = Rotation.from_matrix(problem.flanges[arm][i][:3, :3] @ problem.flanges[arm][j][:3, :3].T).as_rotvec()
                    if np.linalg.norm(flange) > np.radians(min_angle_deg):
                        seen.append(rail)
                        moved.append(flange)
        R_X = Rotation.align_vectors(np.array(seen), np.array(moved))[0].as_matrix()
        rows, rhs = [], []
        for slot, g in enumerate(owned):
            frames = problem.grasps[g]["frames"]
            F, C = problem.flanges[arm][frames], problem.rail[frames]
            Y[g][:3, :3] = Rotation.from_matrix(np.einsum("nji,jk,nkl->nil", F[:, :3, :3], R_X.T, C[:, :3, :3])).mean().as_matrix()
            for f, c in zip(F, C):
                row = np.zeros((3, 3 + 3 * len(owned)))
                row[:, :3] = np.eye(3)
                row[:, 3 + 3 * slot:6 + 3 * slot] = R_X @ f[:3, :3]
                rows.append(row)
                rhs.append(c[:3, 3] - R_X @ f[:3, 3])
        solution = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)[0]
        X[arm] = np.eye(4)
        X[arm][:3, :3], X[arm][:3, 3] = R_X, solution[:3]
        for slot, g in enumerate(owned):
            Y[g][:3, 3] = solution[3 + 3 * slot:6 + 3 * slot]
    pack = lambda T: np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])
    return np.concatenate([pack(X[arm]) for arm in problem.arms] + [pack(T) for T in Y])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="merged review output (poses.jsonl)")
    parser.add_argument("--tracks", nargs="+", default=[], metavar="NAME=RUN[:flip]",
                        help="tracking runs whose along-axis offset from the kinematic prediction is reported")
    parser.add_argument("--episode", required=True, help="recorded episode directory with raw_spool/streams")
    parser.add_argument("--teleop", required=True, help="bimanual-teleop checkout providing the Tianji FK")
    parser.add_argument("--grasp", nargs="+", required=True, metavar="ARM:FIRST:LAST")
    parser.add_argument("--anchor", nargs="+", required=True, metavar="FIRST:LAST",
                        help="frames whose tracked position along the axis is trusted")
    parser.add_argument("--holdout", nargs="*", default=[], metavar="FIRST:LAST",
                        help="trusted frames compared with the prediction but not fitted")
    parser.add_argument("--time-offsets-ms", type=float, nargs="+", default=[0.])
    parser.add_argument("--sigma-mm", type=float, default=5.)
    parser.add_argument("--sigma-deg", type=float, default=2.)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.teleop).resolve()))
    from bimanual_teleop.devices.tianji.model import TianjiKinematics
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_candidates import end_flip, load_run
    import trimesh
    from rail_pose.common import load_config

    rows = [json.loads(line) for line in resolve(args.result).joinpath("poses.jsonl").open()]
    mesh = trimesh.load(resolve(load_config()["model"]), force="mesh")
    centre = np.eye(4)
    centre[:3, 3] = mesh.bounds.mean(axis=0)
    rail = np.array([np.array(r["T_C_CAD"]) @ centre if r["valid"] else np.full((4, 4), np.nan) for r in rows])
    trusted = {r["index"] for r in rows if r["valid"] and r["quality"] in TRUSTED}
    anchors = sorted({i for s in args.anchor for i in span(s)} & trusted)
    holdout = sorted({i for s in args.holdout for i in span(s)} & trusted)
    grasps = []
    for text in args.grasp:
        arm, first, last = text.split(":")
        grasps.append({"arm": arm, "span": [int(first), int(last)],
                       "frames": sorted(set(range(int(first), int(last) + 1)) & trusted - set(holdout))})
    times = np.array([r["timestamp_ns"] for r in rows], dtype=np.float64)
    kinematics = TianjiKinematics()
    streams = Path(args.episode) / "raw_spool" / "streams"

    best = None
    for offset in args.time_offsets_ms:
        flanges = {arm: flange_poses(kinematics, streams / f"arms__{arm}.bin", arm, times + offset * 1e6)
                   for arm in {g["arm"] for g in grasps}}
        problem = Problem(grasps, rail, flanges, anchors, args.sigma_mm, args.sigma_deg)
        fit = least_squares(problem.residuals, initial_guess(problem), loss="soft_l1", f_scale=1.)
        cost = float(np.mean(np.minimum(fit.fun ** 2, 9.)))
        print(f"time offset {offset:+6.1f} ms: mean clipped squared residual {cost:.3f}", flush=True)
        if best is None or cost < best[0]:
            best = (cost, offset, problem, fit)
    cost, offset, problem, fit = best
    X, Y = problem.unpack(fit.x)

    tracks = {}
    flip = end_flip(mesh)
    for text in args.tracks:
        name, run = text.split("=", 1)
        flipped = run.endswith(":flip")
        poses = load_run(resolve(run.removesuffix(":flip")))
        tracks[name] = {k: (v @ flip if flipped else v) @ centre for k, v in poses.items()}

    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=False)
    per_frame = []
    with (output / "kinematic.jsonl").open("w", encoding="utf-8") as stream:
        for r in rows:
            i = r["index"]
            row = {"index": i, "frame_id": r["frame_id"], "result_quality": r["quality"], "grasps": {}}
            for g, grasp in enumerate(grasps):
                if not grasp["span"][0] <= i <= grasp["span"][1]:
                    continue
                T = problem.predict(X, Y, g, [i])[0]
                entry = {"T_C_rail_centre": T.tolist()}
                axis = T[:3, 1]
                if r["valid"]:
                    gap = T[:3, 3] - rail[i][:3, 3]
                    entry["result_along_mm"] = float(-gap @ axis * 1000)
                    entry["result_across_mm"] = float(np.linalg.norm(gap - (gap @ axis) * axis) * 1000)
                    cos = np.clip(abs(T[:3, 1] @ rail[i][:3, 1]), -1, 1)
                    entry["axis_angle_deg"] = float(np.degrees(np.arccos(cos)))
                for name, poses in tracks.items():
                    if r["frame_id"] in poses:
                        entry[f"{name}_along_mm"] = float((poses[r["frame_id"]][:3, 3] - T[:3, 3]) @ axis * 1000)
                row["grasps"][f"{grasp['arm']}:{grasp['span'][0]}:{grasp['span'][1]}"] = entry
            stream.write(json.dumps(row) + "\n")
            per_frame.append(row)

    def stats(values):
        values = np.abs(np.asarray(values, dtype=float))
        if values.size == 0:
            return None
        return {"frames": int(values.size), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
                "max": float(values.max())}

    report = {"time_offset_ms": offset, "anchors": [anchors[0], anchors[-1]] if anchors else [],
              "holdout": [holdout[0], holdout[-1]] if holdout else [], "sigma_mm": args.sigma_mm,
              "sigma_deg": args.sigma_deg, "grasps": [], "X_C_base": {a: X[a].tolist() for a in X}}
    for g, grasp in enumerate(grasps):
        key = f"{grasp['arm']}:{grasp['span'][0]}:{grasp['span'][1]}"
        frames = [row for row in per_frame if key in row["grasps"] and "result_along_mm" in row["grasps"][key]]
        fitted = [row for row in frames if row["index"] in grasp["frames"]]
        entry = {"grasp": key, "fitted_frames": len(grasp["frames"]),
                 "across_mm": stats([row["grasps"][key]["result_across_mm"] for row in fitted]),
                 "axis_angle_deg": stats([row["grasps"][key]["axis_angle_deg"] for row in fitted]),
                 "anchor_along_mm": stats([row["grasps"][key]["result_along_mm"] for row in fitted if row["index"] in anchors]),
                 "Y_flange_rail_centre": Y[g].tolist()}
        for name in tracks:
            values = [(row["index"], row["grasps"][key].get(f"{name}_along_mm")) for row in frames]
            entry[f"{name}_along_mm_by_40"] = {str(i): round(v, 1) for i, v in values if v is not None and i % 40 == 0}
        report["grasps"].append(entry)
    save_json(output / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "X_C_base"}, indent=1))


if __name__ == "__main__":
    main()
