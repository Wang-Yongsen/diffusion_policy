"""Convert complete raw episodes to the official DP ReplayBuffer layout.

All vectors concatenate left before right. Poses are xyz (m) followed by
rotation vectors (rad); joints are radians and wrench is Fx,Fy,Fz (N),
Tx,Ty,Tz (N m). Actions concatenate both arms, then both hands. Cartesian
actions use the recorded controller input goals, never FK of joint commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile

from diffusion_policy.third_party.bimanual_teleop.common.console import EpisodeProgress

STATE_GAP_NS = 50_000_000
COMMAND_AGE_NS = 50_000_000
CAMERA_TOLERANCE_NS = 20_000_000
SIDES = ("left", "right")


@dataclass
class _Stream:
    times: object
    values: dict
    rows: object
    raw_count: int


def _read_stream(raw, name, fields, start_ns, end_ns):
    import numpy as np

    if name not in raw:
        raise ValueError(f"Missing required raw stream: {name}")
    group = raw[name]
    times = np.asarray(group["time_ns"][:])
    sequence = np.asarray(group["sequence"][:])
    if (times.ndim != 1 or times.dtype != np.dtype("int64")
            or sequence.shape != times.shape or sequence.dtype != np.dtype("int64")):
        raise ValueError(f"{name}: time_ns and sequence must be equally sized int64 vectors")
    if len(times) > 1 and np.any(times[1:] <= times[:-1]):
        raise ValueError(f"{name}: time_ns must be strictly increasing")
    rows = np.flatnonzero((times >= start_ns) & (times < end_ns))
    values = {}
    for field, width in fields.items():
        array = group[field]
        if array.shape != (len(times), width):
            raise ValueError(f"{name}/{field}: expected shape ({len(times)}, {width})")
        values[field] = np.asarray(array[:], dtype=np.float64)[rows]
    if name.startswith("cameras/"):
        if group["source_time_ms"].shape != times.shape:
            raise ValueError(f"{name}: source_time_ms length differs from timestamps")
    return _Stream(times[rows], values, rows, len(times))


def _brackets(times, query, max_gap_ns=STATE_GAP_NS, bounds=None):
    import numpy as np

    if len(times) == 0:
        return np.zeros(len(query), int), np.zeros(len(query), int), np.zeros(len(query), bool)
    right = np.searchsorted(times, query, side="left")
    hi = np.clip(right, 0, len(times) - 1)
    exact = times[hi] == query
    lo = np.where(exact, hi, np.clip(right - 1, 0, len(times) - 1))
    valid = (query >= times[0]) & (query <= times[-1])
    valid &= times[hi] - times[lo] <= max_gap_ns
    if bounds is not None:
        valid &= (times[lo] >= bounds[:, 0]) & (times[hi] < bounds[:, 1])
    return lo, hi, valid


def _interpolate(stream, query, field, *, pose=False, max_gap_ns=STATE_GAP_NS, bounds=None):
    import numpy as np

    values = stream.values[field]
    width = 7 if pose else values.shape[1]
    if not len(stream.times):
        return np.zeros((len(query), width)), np.zeros(len(query), bool)
    lo, hi, valid = _brackets(stream.times, query, max_gap_ns, bounds)
    a, b = values[lo], values[hi]
    valid &= np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    gap = stream.times[hi] - stream.times[lo]
    weight = np.divide(query - stream.times[lo], gap,
                       out=np.zeros(len(query), float), where=gap != 0)
    weight = np.clip(weight, 0., 1.)
    result = a + weight[:, None] * (b - a)
    if pose:
        # Pairwise shortest-path SLERP also handles q and -q identically.
        qa, qb = a[:, 3:].copy(), b[:, 3:].copy()
        na, nb = np.linalg.norm(qa, axis=1), np.linalg.norm(qb, axis=1)
        good = valid & np.isfinite(na) & np.isfinite(nb) & (na > 1e-8) & (nb > 1e-8)
        qa[~good] = qb[~good] = (0., 0., 0., 1.)
        qa /= np.linalg.norm(qa, axis=1)[:, None]
        qb /= np.linalg.norm(qb, axis=1)[:, None]
        dot = np.sum(qa * qb, axis=1)
        qb[dot < 0] *= -1
        theta = np.arccos(np.clip(np.abs(dot), 0., 1.))
        denom = np.sin(theta)
        wa, wb = 1. - weight, weight.copy()
        curved = denom > 1e-6
        wa[curved] = np.sin((1. - weight[curved]) * theta[curved]) / denom[curved]
        wb[curved] = np.sin(weight[curved] * theta[curved]) / denom[curved]
        quat = wa[:, None] * qa + wb[:, None] * qb
        quat /= np.linalg.norm(quat, axis=1)[:, None]
        result[:, 3:] = quat
        valid = good
    return result, valid


def _hold(stream, query, field, max_age_ns=COMMAND_AGE_NS, bounds=None):
    import numpy as np

    values = stream.values[field]
    if not len(stream.times):
        return np.zeros((len(query), values.shape[1])), np.zeros(len(query), bool)
    indices = np.searchsorted(stream.times, query, side="right") - 1
    safe = np.maximum(indices, 0)
    result = values[safe]
    valid = (indices >= 0) & (query - stream.times[safe] <= max_age_ns)
    if bounds is not None:
        valid &= (stream.times[safe] >= bounds[:, 0]) & (stream.times[safe] < bounds[:, 1])
    valid &= np.isfinite(result).all(axis=1)
    return result, valid


def _pose_vector(poses):
    import numpy as np
    from scipy.spatial.transform import Rotation

    norms = np.linalg.norm(poses[:, 3:], axis=1)
    valid = np.isfinite(poses).all(axis=1) & np.isfinite(norms) & (norms > 1e-8)
    result = np.zeros((len(poses), 6), dtype=np.float64)
    result[valid, :3] = poses[valid, :3]
    if np.any(valid):
        result[valid, 3:] = Rotation.from_quat(poses[valid, 3:]).as_rotvec()
    return result, valid


def _nearest(stream, query, tolerance_ns=CAMERA_TOLERANCE_NS, bounds=None):
    import numpy as np

    if not len(stream.times):
        return np.zeros(len(query), int), np.zeros(len(query), bool)
    right = np.clip(np.searchsorted(stream.times, query), 0, len(stream.times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(stream.times[left] - query) <= abs(stream.times[right] - query), left, right)
    valid = abs(stream.times[indices] - query) <= tolerance_ns
    if bounds is not None:
        valid &= (stream.times[indices] >= bounds[:, 0]) & (stream.times[indices] < bounds[:, 1])
    return stream.rows[indices], valid


def _conversion_config(config):
    """A small validated mapping; no new configuration framework."""
    import math
    from collections.abc import Mapping

    if config is None:
        return None
    if isinstance(config, (str, Path)):
        import yaml
        config = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("conversion_config must be a YAML mapping or its path")
    result = dict(mode="repair", target_fps=30, max_missing_camera_frames=2,
                  camera_match_tolerance_ms=20, max_state_interp_gap_ms=100,
                  max_command_age_ms=50, pause_policy="checked_compress",
                  pause_offset_tolerance_ms=100, pause_reviews={})
    unknown = set(config) - set(result)
    if unknown:
        raise ValueError(f"Unknown conversion options: {sorted(unknown)}")
    result.update(config)
    if result["mode"] not in ("strict", "repair"):
        raise ValueError("mode must be strict or repair")
    if result["pause_policy"] not in ("keep", "checked_compress"):
        raise ValueError("pause_policy must be keep or checked_compress")
    for key in ("target_fps", "camera_match_tolerance_ms", "max_state_interp_gap_ms",
                "max_command_age_ms", "pause_offset_tolerance_ms"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
    if result["target_fps"] > 1000:
        raise ValueError("target_fps must not exceed 1000")
    value = result["max_missing_camera_frames"]
    if type(value) is not int or value < 0:
        raise ValueError("max_missing_camera_frames must be a nonnegative integer")
    if not isinstance(result["pause_reviews"], Mapping):
        raise ValueError("pause_reviews must map source_episode to reviewed seams")
    for source, reviews in result["pause_reviews"].items():
        if not isinstance(reviews, Mapping):
            raise ValueError(f"pause_reviews/{source} must be a mapping")
        for row, review in reviews.items():
            if (not str(row).isdigit() or not isinstance(review, Mapping)
                    or set(review) != {"merge", "reason"} or type(review["merge"]) is not bool
                    or not isinstance(review["reason"], str) or not review["reason"].strip()):
                raise ValueError("Each pause review requires a frame index, boolean merge and nonempty reason")
    return result


def _repair_grid(raw, cameras, start, end, config):
    """Separate pauses using all cameras, then grid each shared recording block."""
    import numpy as np

    splits, offsets = [], []
    threshold = config["pause_offset_tolerance_ms"]
    for i, camera in enumerate(cameras):
        source = np.asarray(raw[f"cameras/camera_{i}/rgb/source_time_ms"][:])[camera.rows]
        if not len(source) or not np.isfinite(source).all() or np.any(np.diff(source) <= 0):
            raise ValueError(f"camera_{i}: missing, invalid or reset source timestamps")
        offset = source - camera.times / 1e6
        if np.any(np.diff(offset) < -threshold):
            raise ValueError(f"camera_{i}: source clock or recording time moved backwards")
        splits.append(np.r_[0, np.flatnonzero(np.diff(offset) > threshold) + 1, len(source)])
        offsets.append(offset)
    if len({len(x) for x in splits}) != 1:
        raise ValueError("Ambiguous pause ownership: camera pause counts differ")
    query_parts, id_parts, ranges, pauses = [], [], [], []
    baseline_offset = offsets[0][0]
    source_shifts = []
    for block in range(len(splits[0]) - 1):
        low = start if block == 0 else max(int(c.times[splits[i][block]]) for i, c in enumerate(cameras))
        high = end if block == len(splits[0]) - 2 else min(int(c.times[splits[i][block+1]-1]) + 1 for i, c in enumerate(cameras))
        if low >= high:
            raise ValueError("Ambiguous pause ownership: no shared recording interval")
        ranges.append((low, high))
        main = cameras[0].times
        available = main[(main >= low) & (main < high)]
        if not len(available):
            raise ValueError("Recording block contains no main camera frames")
        count = int(np.ceil((available[-1] - available[0]) * config["target_fps"] / 1e9)) + 1
        times = available[0] + np.rint(np.arange(count) * 1e9 / config["target_fps"]).astype(np.int64)
        times = times[times <= available[-1]]
        count = len(times)
        query_parts.append(times)
        id_parts.append(np.full(count, block, np.int64))
        source_shifts.append(round((offsets[0][splits[0][block]] - baseline_offset) * 1e6))
        if block:
            row = splits[0][block]
            shifts = [offsets[i][splits[i][block]] - offsets[i][splits[i][block]-1] for i in range(3)]
            if max(shifts) - min(shifts) > threshold:
                raise ValueError("Ambiguous pause ownership: camera pause durations differ")
            pauses.append(dict(after_main_row=int(cameras[0].rows[row]), block=block,
                               removed_wait_ns=int(source_shifts[-1] - source_shifts[-2])))
    query = np.concatenate(query_parts)
    block_ids = np.concatenate(id_parts)
    return query, block_ids, np.asarray(ranges, np.int64)[block_ids], np.asarray(source_shifts)[block_ids], pauses


def _repair_camera(stream, query, block_ids, bounds, config):
    import numpy as np

    rows, valid = _nearest(stream, query, round(config["camera_match_tolerance_ms"] * 1e6), bounds)
    reused = np.zeros(len(query), bool)
    max_age = (config["max_missing_camera_frames"] + 1) * 1e9 / config["target_fps"]
    for block in np.unique(block_ids):
        indices = np.flatnonzero(block_ids == block)
        missing = indices[~valid[indices]]
        runs = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)
        camera_times = stream.times[(stream.times >= bounds[indices[0], 0]) & (stream.times < bounds[indices[0], 1])]
        if not len(camera_times):
            continue
        for run in runs:
            if not len(run) or len(run) > config["max_missing_camera_frames"]:
                continue
            # Do not extend coverage or keep the first few frames of a long hole.
            if query[run[0]] < camera_times[0] or query[run[-1]] > camera_times[-1]:
                continue
            previous = np.searchsorted(stream.times, query[run], side="right") - 1
            safe = np.maximum(previous, 0)
            good = (previous >= 0) & (stream.times[safe] >= bounds[run, 0]) & (query[run] - stream.times[safe] <= max_age)
            if not np.all(good):
                continue
            rows[run] = stream.rows[safe]
            valid[run] = reused[run] = True
    selected = np.searchsorted(stream.rows, rows)
    return rows, valid, reused, stream.times[selected] - query


def _segments(valid, times, max_gap_ns=STATE_GAP_NS, block_ids=None):
    import numpy as np

    keep = np.flatnonzero(valid)
    if not len(keep):
        return []
    breaks = (np.diff(keep) != 1) | (np.diff(times[keep]) > max_gap_ns)
    if block_ids is not None:
        breaks |= np.diff(block_ids[keep]) != 0
    return np.split(keep, np.flatnonzero(breaks) + 1)


def _append(data, key, values):
    from numcodecs import Blosc

    if key not in data:
        data.create_dataset(key, shape=(0,) + values.shape[1:], dtype=values.dtype,
                            chunks=(1024,) + values.shape[1:],
                            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE))
    if data[key].shape[1:] != values.shape[1:]:
        raise ValueError(f"Output shape changed for {key}")
    data[key].append(values, axis=0)


def _image_array(data, key, total, shape, dtype):
    from numcodecs import Blosc

    if key not in data:
        return data.create_dataset(key, shape=(total,) + tuple(shape), dtype=dtype,
                                   chunks=(1,) + tuple(shape),
                                   compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.NOSHUFFLE))
    array = data[key]
    if array.shape[1:] != tuple(shape) or array.dtype != dtype:
        raise ValueError(f"Output image format changed for {key}")
    array.resize((total,) + tuple(shape))
    return array


def _copy_video(path, expected_count, source_rows, data, key, offset, progress):
    import av
    import numpy as np

    destinations = {}
    for i, source_row in enumerate(source_rows):
        destinations.setdefault(int(source_row), []).append(offset + i)
    count = 0
    array = None
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            count += 1
            progress.advance()
            if index not in destinations:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if array is None:
                array = _image_array(data, key, offset + len(source_rows), rgb.shape, np.dtype("uint8"))
            if rgb.shape != array.shape[1:]:
                raise ValueError(f"Video resolution changes within {path}")
            for target in destinations[index]:
                array[target] = rgb
    if count != expected_count:
        raise ValueError(f"{path}: decoded {count} frames, timestamp table has {expected_count}")


def _convert_episode(episode, descriptor, raw, output, action_space, report, source_name, *, include_depth, config=None,
                     progress):
    import numpy as np

    start, end = descriptor["start_ns"], descriptor["end_ns"]
    if (not isinstance(start, int) or not isinstance(end, int) or end <= start):
        raise ValueError(f"{episode}: invalid episode start_ns/end_ns")
    cameras = [_read_stream(raw, f"cameras/camera_{i}/rgb", {}, start, end) for i in range(3)]
    main = cameras[0]
    repair = config is not None and config["mode"] == "repair"
    query, bounds, block_ids, source_shifts, pauses = main.times, None, None, None, []
    if repair:
        query, block_ids, bounds, source_shifts, pauses = _repair_grid(raw, cameras, start, end, config)
    state_gap = round(config["max_state_interp_gap_ms"] * 1e6) if repair else STATE_GAP_NS
    command_age = round(config["max_command_age_ms"] * 1e6) if repair else COMMAND_AGE_NS
    valid = np.ones(len(query), bool)
    reasons = {}

    def require(name, mask):
        nonlocal valid
        reasons[name] = int(np.count_nonzero(~mask))
        valid &= mask

    image_rows, image_reused, image_offsets = [], [], []
    for i, camera in enumerate(cameras):
        if repair:
            rows, mask, reused, delta = _repair_camera(camera, query, block_ids, bounds, config)
            image_reused.append(reused)
            image_offsets.append(delta)
        elif i == 0:
            rows, mask = main.rows, np.ones(len(query), bool)
        else:
            rows, mask = _nearest(camera, query)
        image_rows.append(rows)
        require(f"camera_{i}_unmatched", mask)
    values = {key: [] for key in ("robot_joint", "robot_eef_pose", "hand_joint", "wrench")}
    arm_actions, hand_actions, state_interpolated, command_ages = [], [], [], []
    for side in SIDES:
        arm = _read_stream(raw, f"arms/{side}", {"joint_pos": 7, "eef_pose": 7, "wrench": 6}, start, end)
        hand = _read_stream(raw, f"hands/{side}", {"joint_pos": 20}, start, end)
        for key, stream, field in (("robot_joint", arm, "joint_pos"), ("robot_eef_pose", arm, "eef_pose"),
                                   ("wrench", arm, "wrench"), ("hand_joint", hand, "joint_pos")):
            result, mask = _interpolate(stream, query, field, pose=field == "eef_pose",
                                        max_gap_ns=state_gap, bounds=bounds)
            if repair:
                lo, hi, _ = _brackets(stream.times, query, state_gap, bounds)
                state_interpolated.append((lo != hi) & mask)
            if field == "eef_pose":
                result, pose_valid = _pose_vector(result)
                mask &= pose_valid
            values[key].append(result)
            require(f"{side}_{key}_invalid_or_gap", mask)
        command_field, width = ("eef_pose", 7) if action_space == "eef" else ("joint_pos", 7)
        arm_command = _read_stream(raw, f"arm_commands/{side}", {command_field: width}, start, end)
        action, mask = _hold(arm_command, query, command_field, command_age, bounds)
        if repair:
            indices = np.maximum(np.searchsorted(arm_command.times, query, side="right") - 1, 0)
            command_ages.append(query - arm_command.times[indices] if len(arm_command.times) else np.zeros(len(query), np.int64))
        if action_space == "eef":
            action, pose_valid = _pose_vector(action)
            mask &= pose_valid
        arm_actions.append(action)
        require(f"{side}_arm_command_invalid_or_stale", mask)
        hand_command = _read_stream(raw, f"hand_commands/{side}", {"joint_pos": 20}, start, end)
        action, mask = _hold(hand_command, query, "joint_pos", command_age, bounds)
        if repair:
            indices = np.maximum(np.searchsorted(hand_command.times, query, side="right") - 1, 0)
            command_ages.append(query - hand_command.times[indices] if len(hand_command.times) else np.zeros(len(query), np.int64))
        hand_actions.append(action)
        require(f"{side}_hand_command_invalid_or_stale", mask)

    depth = depth_rows = None
    if include_depth and "cameras/camera_0/depth" in raw:
        depth = _read_stream(raw, "cameras/camera_0/depth", {}, start, end)
        depth_rows, mask = _nearest(depth, query,
                                   round(config["camera_match_tolerance_ms"] * 1e6) if repair else CAMERA_TOLERANCE_NS, bounds)
        image = raw["cameras/camera_0/depth/image"]
        if len(image.shape) != 3 or image.shape[0] != depth.raw_count or image.dtype != np.dtype("uint16"):
            raise ValueError(f"{episode}: depth must be uint16 (N,H,W), matching its timestamp table")
        require("camera_0_depth_unmatched", mask)

    runs = _segments(valid, query, round(1.5e9 / config["target_fps"]) if repair else STATE_GAP_NS, block_ids)
    training_query = query.copy()
    if repair:
        reviews = {str(k): v for k, v in config["pause_reviews"].get(source_name, {}).items()}
        known_rows = {str(pause["after_main_row"]) for pause in pauses}
        if set(reviews) - known_rows:
            raise ValueError(f"{source_name}: pause review references an undetected seam")
        for pause in pauses:
            block = pause["block"]
            review = reviews.get(str(pause["after_main_row"]))
            left = np.flatnonzero(valid & (block_ids == block - 1))
            right = np.flatnonzero(valid & (block_ids == block))
            merge = (config["pause_policy"] == "checked_compress" and review is not None
                     and review["merge"] and len(left) > 0 and len(right) > 0)
            pause.update(merged=bool(merge), reason=review["reason"] if review else "Not reviewed; keep boundary")
            if merge:
                shift = training_query[left[-1]] + round(1e9 / config["target_fps"]) - training_query[right[0]]
                training_query[block_ids >= block] += shift
                pause["training_shift_ns"] = int(shift)
                # Join only the two adjacent runs; retain other invalid gaps.
                for index in range(1, len(runs)):
                    if runs[index-1][-1] == left[-1] and runs[index][0] == right[0]:
                        runs[index-1:index+1] = [np.r_[runs[index-1], runs[index]]]
                        break
        report["pauses"] = pauses
    report.update(reference_frames=len(query), valid_frames=int(valid.sum()), segments=len(runs),
                  main_camera_gaps=int(np.count_nonzero(np.diff(main.times) > STATE_GAP_NS)),
                  invalid_reasons={key: count for key, count in reasons.items() if count})
    if not runs:
        return []
    keep = np.flatnonzero(valid)
    data = output["data"]
    offset = data["timestamp"].shape[0] if "timestamp" in data else 0
    for key, parts in values.items():
        _append(data, key, np.concatenate(parts, axis=1)[keep].astype(np.float32))
    _append(data, "action", np.concatenate(arm_actions + hand_actions, axis=1)[keep].astype(np.float32))
    _append(data, "timestamp", (training_query[keep] - start).astype(np.float64) / 1e9)
    if repair:
        meta = output["meta"]
        _append(meta, "source_time_ns", (query + source_shifts)[keep])
        _append(meta, "recording_time_ns", query[keep])
        _append(meta, "recording_block", block_ids[keep])
        _append(meta, "camera_source_row", np.stack(image_rows, axis=1)[keep])
        _append(meta, "camera_reused", np.stack(image_reused, axis=1)[keep])
        _append(meta, "camera_time_offset_ns", np.stack(image_offsets, axis=1)[keep])
        captures = [np.asarray(raw[f"cameras/camera_{i}/rgb/source_time_ms"][:])[rows] for i, rows in enumerate(image_rows)]
        _append(meta, "camera_source_time_ms", np.stack(captures, axis=1)[keep])
        _append(meta, "state_interpolated", np.stack(state_interpolated, axis=1)[keep])
        _append(meta, "command_age_ns", np.stack(command_ages, axis=1)[keep])
        report["camera_repairs"] = [dict(camera=f"camera_{i}", reused_frames=int(image_reused[i][keep].sum()),
                                         reused_ratio=float(image_reused[i][keep].mean()),
                                         max_image_age_ms=float(np.maximum(-image_offsets[i][keep], 0).max() / 1e6)) for i in range(3)]
    progress.total(sum(camera.raw_count for camera in cameras) + (len(keep) if depth is not None else 0))
    for i, camera in enumerate(cameras):
        _copy_video(episode / f"camera_{i}.mp4", camera.raw_count, image_rows[i][keep],
                    data, f"camera_{i}", offset, progress)
    if depth is not None:
        source = raw["cameras/camera_0/depth/image"]
        target = _image_array(data, "camera_0_depth", offset + len(keep), source.shape[1:], source.dtype)
        for i, row in enumerate(depth_rows[keep]):
            target[offset + i] = source[int(row)]
            progress.advance()
    result = []
    for run in runs:
        length = len(run)
        result.append({"source_episode": source_name, "reference_frame_start": int(image_rows[0][run[0]]),
                       "reference_frame_end": int(image_rows[0][run[-1]]) + 1,
                       "start_ns": int(query[run[0]]), "end_ns": int(query[run[-1]]),
                       "output_start": offset, "output_end": offset + length})
        offset += length
    return result


def convert_recordings(input_path, output_path, *, action_space, include_depth=False, conversion_config=None,
                       progress=None):
    """Create a new dataset; never overwrite an existing file or directory.

    Episode bounds are [start_ns,end_ns). Without conversion_config, state
    brackets and command age are limited to 50 ms and real camera times are
    used. Repair configuration selects a fixed grid, bounded image reuse and
    per-recording-block interpolation; only explicitly reviewed seams merge.
    No interpolation, held command or image match crosses an episode boundary.
    Depth is ignored unless explicitly included. episode_ends counts source
    demonstrations; segment_ends marks continuous runs for training sampling.
    Returns the quality report, also stored in meta.attrs['quality_report'].
    ``progress`` counts decoded RGB frames plus copied depth frames.
    """
    import numpy as np
    import zarr

    if action_space not in ("eef", "joint"):
        raise ValueError("action_space must be 'eef' or 'joint'")
    if type(include_depth) is not bool:
        raise ValueError("include_depth must be a boolean")
    config = _conversion_config(conversion_config)
    source, destination = Path(input_path).expanduser().resolve(), Path(output_path).expanduser().absolute()
    if not source.is_dir():
        raise ValueError(f"Input is not a directory: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    manifests = [source / "episode.json"] if (source / "episode.json").is_file() else sorted(source.rglob("episode.json"))
    if not manifests:
        raise ValueError(f"No episode.json found under {source}")
    progress = EpisodeProgress(enabled=False) if progress is None else progress
    progress.start(len(manifests))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.converting-", dir=destination.parent))
    report = {"episodes": [], "action_space": action_space, "include_depth": include_depth,
              "conversion_config": config}
    try:
        output = zarr.open_group(str(temporary), mode="w")
        output.create_group("data")
        meta = output.create_group("meta")
        segments = []
        episode_ends = []
        has_depth = None
        for manifest in manifests:
            descriptor = json.loads(manifest.read_text(encoding="utf-8"))
            name = str(manifest.parent.relative_to(source))
            item = {"source_episode": name, "status": descriptor.get("status", "unknown")}
            report["episodes"].append(item)
            if item["status"] != "complete":
                progress.finish_episode()
                continue
            if descriptor.get("schema_version") != 1:
                raise ValueError(f"Unsupported raw schema in {manifest}")
            raw = zarr.open_group(str(manifest.parent / "raw.zarr"), mode="r")
            this_depth = include_depth and "cameras/camera_0/depth" in raw
            if has_depth is not None and this_depth != has_depth:
                raise ValueError("Complete episodes must consistently include or omit camera_0 depth")
            has_depth = this_depth
            item["metadata"] = descriptor.get("metadata", raw.attrs.get("metadata", {}))
            progress.episode(manifest.parent.name if name == "." else name)
            runs = _convert_episode(manifest.parent, descriptor, raw, output, action_space, item, name,
                                    include_depth=include_depth, config=config, progress=progress)
            segments.extend(runs)
            if runs:
                episode_ends.append(runs[-1]["output_end"])
            progress.finish_episode()
        if not segments:
            raise ValueError("No valid frames in complete episodes")
        ends = np.asarray(episode_ends, dtype=np.int64)
        segment_ends = np.asarray([segment["output_end"] for segment in segments], dtype=np.int64)
        meta.create_dataset("episode_ends", data=ends, compressor=None)
        meta.create_dataset("segment_ends", data=segment_ends, compressor=None)
        report.update(output_episodes=len(ends), output_segments=len(segments), output_frames=int(ends[-1]))
        meta.attrs.update(segments=segments, quality_report=report)
        output.attrs.update(schema_version=2, format="diffusion_policy_replay_buffer", action_space=action_space,
                            include_depth=include_depth, episode_ends_semantics="source_demonstrations",
                            sampling_boundaries="meta/segment_ends",
                            side_order=list(SIDES), eef_pose_format="xyz_m+rotvec_rad", joint_unit="rad",
                            wrench_format="Fx,Fy,Fz [N]; Tx,Ty,Tz [N*m]",
                            action_layout="left_arm,right_arm,left_hand,right_hand",
                            timestamp="seconds since source episode start; camera_0 real frame times")
        if config is not None and config["mode"] == "repair":
            output.attrs.update(timestamp="seconds on per-block training grids; reviewed pauses may be compressed",
                                target_fps=config["target_fps"], conversion_config=config)
            meta.attrs.update(source_time_ns="recording time_ns plus accumulated removed pause duration",
                              state_interpolated_fields=[f"{side}_{key}" for side in SIDES for key in ("robot_joint", "robot_eef_pose", "wrench", "hand_joint")],
                              command_age_fields=[f"{side}_{kind}" for side in SIDES for kind in ("arm", "hand")])
        # Exclusively reserve the name, then atomically replace our empty
        # reservation with the complete directory on the same filesystem.
        destination.mkdir()
        try:
            os.replace(temporary, destination)
        except BaseException:
            try:
                destination.rmdir()  # Never remove contents written by someone else.
            except OSError:
                pass
            raise
        return report
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
