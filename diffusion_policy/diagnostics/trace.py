"""Offline measurements and plots from saved deployment traces."""

import json
from pathlib import Path

import numpy as np

from diffusion_policy.real_world.tianji.trace import SIDES


def load_trace(path):
    path = Path(path)
    meta = json.loads((path / "trace_meta.json").read_text())
    return {
        "meta": meta,
        "decisions": np.load(path / "trace_decisions.npz"),
        "feedback": np.load(path / "trace_feedback.npz"),
        "commands": np.load(path / "trace_commands.npz"),
        "hands": np.load(path / "trace_hands.npz"),
    }


def alignment_series(commands, feedback, fk):
    """Request, command FK, and offline FK of controller target and measured joints.

    ``fk(side, q)`` returns a 3-vector in metres, or None when it cannot.
    ``position_error_m`` is not used as a measured position.
    """
    stamps = np.array(commands["started_ns"], copy=True)
    fallback = np.array(commands["sink_received_ns"], copy=True)
    stamps[stamps < 0] = fallback[stamps < 0]
    origin = _time_origin(stamps, feedback["received_ns"])
    request = _pose_series(
        stamps, commands["requested_xyz"][:, 0], commands["requested_present"][:, 0], origin
    )
    command = _pose_series(
        stamps, commands["command_fk_xyz"][:, 0], commands["command_fk_present"][:, 0], origin
    )
    measured = _joint_series(
        feedback["received_ns"],
        feedback["q"][:, 0],
        feedback["q_present"][:, 0],
        origin,
        fk,
        "left",
    )
    controller = _joint_series(
        feedback["received_ns"],
        feedback["controller_target"][:, 0],
        feedback["controller_present"][:, 0],
        origin,
        fk,
        "left",
    )
    return {
        "origin_ns": origin,
        "request": request,
        "command_fk": command,
        "measured_fk": measured,
        "controller_fk": controller,
    }


def inference_motion_mm(feedback_ns, measured_xyz, anchor_ns, returned_ns):
    """Millimetres of measured motion from the input anchor until prediction return."""
    finite = np.isfinite(measured_xyz).all(axis=-1)
    result = np.full(len(anchor_ns), np.nan)
    for index, (anchor, returned) in enumerate(zip(anchor_ns, returned_ns)):
        if anchor < 0 or returned < anchor:
            continue
        chosen = (feedback_ns >= anchor) & (feedback_ns <= returned) & finite
        if int(chosen.sum()) < 2:
            continue
        points = measured_xyz[chosen]
        result[index] = float(np.linalg.norm(points[-1] - points[0]) * 1000)
    return result


def replay_follow_mm(demo_eef, real_eef, real_present):
    """Left-arm position gap at the paired decision. Absent real rows stay missing."""
    result = np.full(len(real_present), np.nan)
    for index, present in enumerate(real_present):
        if present:
            result[index] = float(
                np.linalg.norm(real_eef[index, -1, :3] - demo_eef[index, -1, :3]) * 1000
            )
    return result


def hand_tracking_deg(
    command_ns,
    command_q,
    command_present,
    feedback_ns,
    feedback_q,
    feedback_present,
    tolerance_ns=15_000_000,
):
    """Max joint error to the nearest feedback inside the tolerance. No match stays missing."""
    result = np.full(len(command_ns), np.nan)
    usable = np.flatnonzero(feedback_present)
    if not len(usable):
        return result
    stamps = feedback_ns[usable]
    for index, (stamp, present) in enumerate(zip(command_ns, command_present)):
        if not present or stamp < 0:
            continue
        nearest = int(np.argmin(np.abs(stamps - stamp)))
        if abs(int(stamps[nearest]) - int(stamp)) > tolerance_ns:
            continue
        delta = command_q[index] - feedback_q[usable[nearest]]
        if not np.isfinite(delta).all():
            continue
        result[index] = float(np.rad2deg(np.max(np.abs(delta))))
    return result


def compare_action_pair(demo_actions, real_actions):
    """Seed-major arrays (seed, step, 52). Same position in the stack used the same initial noise."""
    demo_actions = np.asarray(demo_actions, dtype=np.float64)
    real_actions = np.asarray(real_actions, dtype=np.float64)
    if (
        demo_actions.shape != real_actions.shape
        or demo_actions.shape[-1] != 52
        or demo_actions.ndim != 3
    ):
        raise ValueError("Paired actions must share shape (seed, step, 52)")
    from diffusion_policy.env_runner.tianji_rail_offline_runner import action_errors

    per_seed = [
        action_errors(demo_actions[index], real_actions[index])
        for index in range(len(demo_actions))
    ]
    summary = {
        key: {
            "mean": float(np.mean([item[key].mean() for item in per_seed])),
            "max": float(np.max([item[key].max() for item in per_seed])),
        }
        for key in per_seed[0]
    }
    demo_dir = demo_actions[:, -1, :3] - demo_actions[:, 0, :3]
    real_dir = real_actions[:, -1, :3] - real_actions[:, 0, :3]
    cosines = [_cosine(demo_dir[index], real_dir[index]) for index in range(len(demo_actions))]
    summary["left_direction_cosine"] = cosines
    summary["demo_final_left_xyz_std_mm"] = (demo_actions[:, -1, :3].std(axis=0) * 1000).tolist()
    summary["real_final_left_xyz_std_mm"] = (real_actions[:, -1, :3].std(axis=0) * 1000).tolist()
    return summary


def request_to_command_mm(commands):
    """Position gap from the requested flange to the command FK. This is not measured tracking."""
    both = commands["requested_present"] & commands["command_fk_present"]
    result = {}
    for index, side in enumerate(SIDES):
        chosen = both[:, index]
        if not np.any(chosen):
            result[side] = None
            continue
        delta = (
            np.linalg.norm(
                commands["requested_xyz"][chosen, index]
                - commands["command_fk_xyz"][chosen, index],
                axis=-1,
            )
            * 1000
        )
        result[side] = {
            "max": float(delta.max()),
            "median": float(np.median(delta)),
            "count": int(chosen.sum()),
        }
    return result


def assess_trace(meta, gaps):
    """Numbers only. An invalid or incomplete trace does not select a cause."""
    if not meta.get("valid", False):
        return {
            "root_cause": None,
            "reason": "记录无效，不能判定根因",
            "invalid_reason": meta.get("invalid_reason"),
            "dropped": meta.get("dropped"),
        }
    return {
        "root_cause": None,
        "reason": "只报告各层差值；要等成功回放和真实闭环两段都完整后再选分支",
        "gaps": gaps,
    }


def write_alignment_plot(path, series, hand_series, inference_mm):
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    layers = (
        ("request", "request"),
        ("command_fk", "command FK"),
        ("controller_fk", "controller FK"),
        ("measured_fk", "measured FK"),
    )
    for axis, component in zip(axes, (0, 2)):
        for key, label in layers:
            item = series[key]
            if len(item["time_s"]):
                axis.plot(item["time_s"], item["xyz_mm"][:, component], label=label, linewidth=1)
        axis.set_ylabel("left X mm" if component == 0 else "left Z mm")
        axis.grid(True, linewidth=0.4)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("seconds from first sample")
    figure.tight_layout()
    figure.savefig(path / "alignment_left.png", dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 3))
    for side, item in hand_series.items():
        finite = np.isfinite(item["error_deg"])
        if finite.any():
            axis.plot(item["time_s"][finite], item["error_deg"][finite], label=side, linewidth=1)
    axis.set_xlabel("seconds from first sample")
    axis.set_ylabel("max hand joint error deg")
    axis.grid(True, linewidth=0.4)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path / "alignment_hands.png", dpi=120)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 3))
    finite = np.isfinite(inference_mm)
    if finite.any():
        axis.bar(np.flatnonzero(finite), inference_mm[finite], width=0.8)
    axis.set_xlabel("decision")
    axis.set_ylabel("motion during inference mm")
    axis.grid(True, axis="y", linewidth=0.4)
    figure.tight_layout()
    figure.savefig(path / "inference_motion.png", dpi=120)
    plt.close(figure)


def _pose_series(stamps, xyz, present, origin):
    chosen = present & np.isfinite(xyz).all(axis=-1) & (stamps >= 0)
    return {"time_s": (stamps[chosen] - origin) / 1e9, "xyz_mm": xyz[chosen] * 1000}


def _joint_series(stamps, joints, present, origin, fk, side):
    if fk is None:
        return {"time_s": np.zeros(0), "xyz_mm": np.zeros((0, 3))}
    chosen = np.flatnonzero(present & np.isfinite(joints).all(axis=-1) & (stamps >= 0))
    points, times = [], []
    for index in chosen:
        point = fk(side, joints[index])
        if point is None:
            continue
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            continue
        points.append(point)
        times.append(stamps[index])
    if not points:
        return {"time_s": np.zeros(0), "xyz_mm": np.zeros((0, 3))}
    return {"time_s": (np.asarray(times) - origin) / 1e9, "xyz_mm": np.vstack(points) * 1000}


def _time_origin(*groups):
    stamps = [int(group[group >= 0].min()) for group in groups if np.any(group >= 0)]
    return min(stamps) if stamps else 0


def _cosine(left, right):
    length = np.linalg.norm(left) * np.linalg.norm(right)
    if length < 1e-12:
        return None
    return float(np.dot(left, right) / length)
