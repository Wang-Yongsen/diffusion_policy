"""Read teleoperation recordings and export candidate hand poses; no device access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def read_numeric(episode: Path, name: str, width: int):
    path = episode / "raw_spool" / "streams" / f"{name}.bin"
    dtype = np.dtype([("t", "<i8"), ("sequence", "<i8"), ("v", "<f8", width)])
    if path.stat().st_size % dtype.itemsize:
        raise ValueError(f"Partial numeric record: {path}")
    return np.fromfile(path, dtype=dtype)


def camera_metadata(episode: Path, camera: str):
    path = episode / "raw_spool" / "cameras" / f"{camera}_rgb.bin"
    return np.fromfile(path, dtype=[("t", "<i8"), ("sequence", "<i8"), ("source_ms", "<f8")])


def frame_at(episode: Path, camera: str, timestamp: int):
    metadata = camera_metadata(episode, camera)
    index = int(np.argmin(np.abs(metadata["t"] - timestamp)))
    cap = cv2.VideoCapture(str(episode / f"{camera}.mp4"))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count != len(metadata):
        cap.release()
        raise ValueError(f"{camera}: {count} video frames != {len(metadata)} timestamps")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read {camera} frame {index}")
    # Video already includes the recording pipeline's camera orientation correction.
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), {
        "camera": camera, "frame_index": index,
        "timestamp_ns": int(metadata["t"][index]),
        "time_error_ms": float((metadata["t"][index] - timestamp) / 1e6),
    }


def save_sheet(episode: Path, times, labels, output: Path, cameras):
    columns = 1 if len(cameras) >= 3 else min(3, len(times))
    tile_w, tile_h = 320 * len(cameras), 266
    sheet = Image.new("RGB", (tile_w * columns, tile_h * ((len(times) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, (stamp, label) in enumerate(zip(times, labels)):
        x, y = (slot % columns) * tile_w, (slot // columns) * tile_h
        draw.text((x + 6, y + 5), label, fill="black")
        for i, camera in enumerate(cameras):
            frame, _ = frame_at(episode, camera, int(stamp))
            sheet.paste(frame.resize((320, 240)), (x + i * 320, y + 26))
    sheet.save(output)


def extract_candidate(episode, info, label, start_s, end_s, output):
    origin = info["start_ns"]
    bounds = [origin + round(s * 1e9) for s in (start_s, end_s)]
    hands = read_numeric(episode, "hands__left", 20)
    command = read_numeric(episode, "hand_commands__left", 20)
    arms = read_numeric(episode, "arms__left", 13)
    selection = hands[(hands["t"] >= bounds[0]) & (hands["t"] <= bounds[1])]
    if len(selection) < 2 or not np.isfinite(selection["v"]).all():
        raise ValueError(f"{label}: too few finite hand samples")
    median = np.median(selection["v"], axis=0)
    # An actual sample near the median preserves the recorded joint configuration.
    representative = selection[np.argmin(np.linalg.norm(selection["v"] - median, axis=1))]
    stamp = int(representative["t"])
    cmd = command[np.argmin(np.abs(command["t"] - stamp))]
    arm = arms[np.argmin(np.abs(arms["t"] - stamp))]
    duration = (selection["t"][-1] - selection["t"][0]) / 1e9
    images = []
    for camera in ("camera_0", "camera_1", "camera_2"):
        frame, record = frame_at(episode, camera, stamp)
        path = output / f"{label}_{camera}.png"
        frame.save(path)
        images.append(dict(record, path=str(path)))
    values = selection["v"]
    force = arm["v"][7:]
    return {
        "label": label,
        "source_episode": str(episode),
        "episode_recording_status": info["status"],
        "validation": "candidate selected from teleoperation; physical grasp not verified",
        "window_relative_to_episode_start_s": [start_s, end_s],
        "representative_relative_to_episode_start_s": (stamp - origin) / 1e9,
        "representative_timestamp_ns": stamp,
        "sample_count": len(selection),
        "observed_duration_s": duration,
        "side": "left", "units": "radian",
        "joint_order": [f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)],
        "measured_position_rad": representative["v"].tolist(),
        "command_position_rad": cmd["v"].tolist(),
        "command_timestamp_delta_ms": (int(cmd["t"]) - stamp) / 1e6,
        "measured_window_median_rad": median.tolist(),
        "measured_window_p05_rad": np.quantile(values, .05, axis=0).tolist(),
        "measured_window_p95_rad": np.quantile(values, .95, axis=0).tolist(),
        "max_joint_p95_minus_p05_rad": float(np.ptp(np.quantile(values, [.05, .95], axis=0), axis=0).max()),
        "command_minus_measured_rad": (cmd["v"] - representative["v"]).tolist(),
        "recorded_arm_wrench_sensor_axes_N_Nm": [float(x) if np.isfinite(x) else None for x in force],
        "wrench_note": "Raw sensor axes; no added gravity or tool compensation. Not a contact-force threshold.",
        "recorded_hand_parameters": info["metadata"]["wuji_config"]["parameters"],
        "images": images,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--window", action="append", default=[], metavar="LABEL:START_S:END_S")
    args = parser.parse_args()
    episode = args.episode.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = json.loads((episode / "episode.json").read_text())
    if info["status"] in {"failed", "discarded", "capturing"}:
        raise ValueError(f"Episode has status {info['status']}")
    frames = camera_metadata(episode, "camera_0")
    indices = np.linspace(0, len(frames) - 1, args.samples).astype(int)
    times = frames["t"][indices]
    labels = [f"{episode.name} t={(int(t)-info['start_ns'])/1e9:.2f}s cam0 | cam1" for t in times]
    save_sheet(episode, times, labels, output / "overview.png", ["camera_0", "camera_1"])
    result = []
    for window in args.window:
        label, start, end = window.split(":")
        if not label.replace("_", "").isalnum():
            raise ValueError("Window label must be alphanumeric or underscore")
        result.append(extract_candidate(episode, info, label, float(start), float(end), output))
    if result:
        payload = {"source": str(episode), "candidates": result,
                   "scope": "left hand pose references only; not an executable robot trajectory"}
        (output / "hand_pose_candidates.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        labels = [f"{c['label']} t={c['representative_relative_to_episode_start_s']:.3f}s cam0 | cam1 | cam2" for c in result]
        save_sheet(episode, [c["representative_timestamp_ns"] for c in result], labels,
                   output / "candidate_frames.png", ["camera_0", "camera_1", "camera_2"])
    print(json.dumps({"episode": episode.name, "output": str(output), "candidates": len(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
