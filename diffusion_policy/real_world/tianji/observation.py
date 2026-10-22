"""Align measured states and camera frames on the host clock."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .runtime import SIDES


class ObservationNotReady(RuntimeError):
    pass


def interpolate_state(history, queries, *, pose=False, max_gap_ns=50_000_000):
    stamps = np.asarray([item[0] for item in history], dtype=np.int64)
    if (
        len(stamps) < 2
        or np.any(np.diff(stamps) <= 0)
        or queries[0] < stamps[0]
        or queries[-1] > stamps[-1]
    ):
        raise ObservationNotReady("State history does not cover requested observation times")
    values = np.stack([item[1] for item in history])
    result, used = [], []
    for query in queries:
        upper = int(np.searchsorted(stamps, query, side="left"))
        if stamps[upper] == query:
            result.append(values[upper])
            used.append((int(stamps[upper]), int(stamps[upper])))
            continue
        lower = upper - 1
        gap = int(stamps[upper] - stamps[lower])
        if gap > max_gap_ns:
            raise ObservationNotReady("State interpolation interval is too large")
        fraction = (int(query) - int(stamps[lower])) / gap
        value = values[lower] + fraction * (values[upper] - values[lower])
        if pose:
            rotations = Rotation.from_rotvec(values[[lower, upper], 3:6])
            value[3:6] = Slerp([0.0, 1.0], rotations)([fraction]).as_rotvec()[0]
        result.append(value)
        used.append((int(stamps[lower]), int(stamps[upper])))
    return np.asarray(result, dtype=np.float32), used


def align_observation(histories, n_obs_steps, config, *, now_ns):
    if any(not history for history in histories.values()):
        raise ObservationNotReady("Waiting for all seven observation streams")
    state_keys = [f"{kind}/{side}" for kind in ("arms", "hands") for side in SIDES]
    state_end = min(histories[key][-1][0] for key in state_keys)
    anchors = [item[0] for item in histories["camera_0"] if item[0] <= state_end]
    if not anchors:
        raise ObservationNotReady("No camera anchor covered by measured states")
    anchor = anchors[-1]
    if anchor > now_ns:
        raise ObservationNotReady("Observation anchor is in the future")
    queries = anchor - np.rint(
        np.arange(n_obs_steps - 1, -1, -1) * 1e9 / config["policy_hz"]
    ).astype(np.int64)
    if now_ns - anchor > config["max_camera_age_s"] * 1e9:
        raise ObservationNotReady("Aligned observation is too old")
    obs, metadata = {}, {"query_ns": queries.tolist(), "camera_ns": {}, "state_brackets_ns": {}}
    camera_stamps = []
    for key in ("camera_0", "camera_1", "camera_2"):
        history = histories[key]
        if now_ns - history[-1][0] > config["max_camera_age_s"] * 1e9:
            raise ObservationNotReady(f"{key} is stale")
        stamps = np.asarray([item[0] for item in history], dtype=np.int64)
        indices = [int(np.abs(stamps - query).argmin()) for query in queries]
        selected = stamps[indices]
        if np.any(np.abs(selected - queries) > config["max_image_skew_s"] * 1e9):
            raise ObservationNotReady(f"{key} has no image near the query times")
        obs[key] = np.stack([history[i][1] for i in indices])
        metadata["camera_ns"][key] = selected.tolist()
        camera_stamps.append(selected)
    if np.any(np.ptp(np.asarray(camera_stamps), axis=0) > config["max_camera_span_s"] * 1e9):
        raise ObservationNotReady("Three-camera timestamp span exceeds budget")
    outputs = [("arms", "robot_eef_pose"), ("hands", "hand_joint")]
    if "arm_joints/left" in histories:
        outputs.append(("arm_joints", "robot_joint"))
    for kind, output in outputs:
        values = []
        for side in SIDES:
            key = f"{kind}/{side}"
            if now_ns - histories[key][-1][0] > config["max_state_age_s"] * 1e9:
                raise ObservationNotReady(f"{key} is stale")
            value, brackets = interpolate_state(
                histories[key],
                queries,
                pose=kind == "arms",
                max_gap_ns=round(config["max_state_gap_s"] * 1e9),
            )
            values.append(value)
            metadata["state_brackets_ns"][key] = brackets
        obs[output] = np.concatenate(values, axis=-1)
    return obs, anchor, metadata
