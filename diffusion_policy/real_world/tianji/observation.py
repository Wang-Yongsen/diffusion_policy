"""Align measured states and camera frames on the host clock."""

from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .runtime import SIDES

CAMERA_KEYS = ("camera_0", "camera_1", "camera_2")
STATE_KEYS = tuple(f"{kind}/{side}" for kind in ("arms", "hands") for side in SIDES)


class ObservationWindowUnavailable(RuntimeError):
    """A fixed window cannot be recovered by appending newer samples."""


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


def align_observation(histories, n_obs_steps, config, *, now_ns, query_ns=None):
    if any(not history for history in histories.values()):
        raise ObservationNotReady("Waiting for all seven observation streams")
    fixed = query_ns is not None
    if fixed:
        queries = _fixed_queries(query_ns, n_obs_steps)
        _check_fixed_coverage(histories, queries, config)
        anchor = int(queries[-1])
    else:
        state_end = min(histories[key][-1][0] for key in STATE_KEYS)
        anchors = [item[0] for item in histories["camera_0"] if item[0] <= state_end]
        if not anchors:
            raise ObservationNotReady("No camera anchor covered by measured states")
        anchor = anchors[-1]
        queries = anchor - np.rint(
            np.arange(n_obs_steps - 1, -1, -1) * 1e9 / config["policy_hz"]
        ).astype(np.int64)
    if anchor > now_ns:
        raise ObservationNotReady("Observation anchor is in the future")
    if not fixed and now_ns - anchor > config["max_camera_age_s"] * 1e9:
        raise ObservationNotReady("Aligned observation is too old")
    obs, metadata = {}, {"query_ns": queries.tolist(), "camera_ns": {}, "state_brackets_ns": {}}
    camera_indices = _fixed_camera_indices(histories, queries, config) if fixed else {}
    camera_stamps = []
    for key in CAMERA_KEYS:
        history = histories[key]
        if now_ns - history[-1][0] > config["max_camera_age_s"] * 1e9:
            raise ObservationNotReady(f"{key} is stale")
        stamps = np.asarray([item[0] for item in history], dtype=np.int64)
        indices = (
            camera_indices[key]
            if fixed
            else [int(np.abs(stamps - query).argmin()) for query in queries]
        )
        selected = stamps[indices]
        if np.any(np.abs(selected - queries) > config["max_image_skew_s"] * 1e9):
            raise ObservationNotReady(f"{key} has no image near the query times")
        camera_indices[key] = indices
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
            try:
                value, brackets = interpolate_state(
                    histories[key],
                    queries,
                    pose=kind == "arms",
                    max_gap_ns=round(config["max_state_gap_s"] * 1e9),
                )
            except ObservationNotReady as error:
                if fixed:
                    raise ObservationWindowUnavailable(
                        f"{key}: fixed state interpolation failed: {error}"
                    ) from error
                raise
            values.append(value)
            metadata["state_brackets_ns"][key] = brackets
        obs[output] = np.concatenate(values, axis=-1)
    for key, indices in camera_indices.items():
        obs[key] = np.stack([histories[key][i][1] for i in indices])
    return obs, anchor, metadata


def _fixed_queries(query_ns, n_obs_steps):
    queries = np.asarray(query_ns)
    if (
        n_obs_steps not in (1, 2)
        or queries.shape != (n_obs_steps,)
        or queries.dtype.kind not in "iu"
        or np.any(queries[1:] <= queries[:-1])
    ):
        raise ValueError(
            "Fixed observation requires n_obs_steps in (1, 2) and matching, increasing integer query_ns"
        )
    return queries.astype(np.int64)


def _check_fixed_coverage(histories, queries, config):
    """Timestamp-only preview; do not copy images while a stream is still behind."""
    skew = round(config["max_image_skew_s"] * 1e9)
    for key in CAMERA_KEYS + STATE_KEYS:
        history = histories[key]
        if not history:
            raise ObservationNotReady(f"Waiting for {key}")
        earliest = int(queries[0]) + skew if key in CAMERA_KEYS else int(queries[0])
        if history[0][0] > earliest:
            raise ObservationWindowUnavailable(f"{key}: fixed observation history was lost")
    if any(histories[key][-1][0] < queries[-1] for key in CAMERA_KEYS + STATE_KEYS):
        raise ObservationNotReady("Waiting for all seven streams to cover fixed query_ns")


def _fixed_camera_indices(histories, queries, config):
    """Closest one- or two-frame selections within image and three-camera budgets."""
    skew = round(config["max_image_skew_s"] * 1e9)
    span = round(config["max_camera_span_s"] * 1e9)
    options = []
    for key in CAMERA_KEYS:
        stamps = np.asarray([item[0] for item in histories[key]], dtype=np.int64)
        candidates = [np.flatnonzero(np.abs(stamps - query) <= skew) for query in queries]
        selections = [
            tuple(map(int, indices))
            for indices in product(*candidates)
            if all(stamps[i] < stamps[j] for i, j in zip(indices, indices[1:]))
        ]
        if not selections:
            if stamps[-1] >= queries[-1] + skew or (
                not len(candidates[0]) and stamps[-1] >= queries[0] + skew
            ):
                requirement = "image" if len(queries) == 1 else "distinct images"
                raise ObservationWindowUnavailable(
                    f"{key}: no {requirement} for fixed query_ns"
                )
            requirement = "an image" if len(queries) == 1 else "two distinct images"
            raise ObservationNotReady(f"{key}: waiting for {requirement} near fixed query_ns")
        options.append(
            [
                (
                    indices,
                    stamps[list(indices)],
                    int(np.abs(stamps[list(indices)] - queries).sum()),
                )
                for indices in selections
            ]
        )
    best, best_cost = None, None
    for combination in product(*options):
        if np.any(np.ptp(np.stack([item[1] for item in combination]), axis=0) > span):
            continue
        cost = sum(item[2] for item in combination)
        if best_cost is None or cost < best_cost:
            best, best_cost = combination, cost
    if best is None:
        if all(histories[key][-1][0] >= queries[-1] + skew for key in CAMERA_KEYS):
            raise ObservationWindowUnavailable(
                "Fixed three-camera timestamp span exceeds budget"
            )
        raise ObservationNotReady("Waiting for fixed three-camera timestamp span within budget")
    return {key: list(item[0]) for key, item in zip(CAMERA_KEYS, best)}


def align_latest_observation(histories, n_obs_steps, config, *, now_ns):
    """Newest usable main-camera anchor with the checkpoint's one or two observations.

    Try recent anchors from newest to oldest on one immutable history snapshot.
    A missing frame or state gap can be bypassed by a newer window on the next read.
    """
    if n_obs_steps not in (1, 2):
        raise ValueError("Synchronous inference requires checkpoint n_obs_steps in (1, 2)")
    if any(not histories[key] for key in CAMERA_KEYS + STATE_KEYS):
        raise ObservationNotReady("Waiting for all seven observation streams")
    covered_until = min(now_ns, *(histories[key][-1][0] for key in CAMERA_KEYS + STATE_KEYS))
    oldest_anchor = now_ns - round(config["max_camera_age_s"] * 1e9)
    period = round(1e9 / config["policy_hz"])
    reason = "No recent camera anchor covered by all seven streams"
    for anchor, _ in reversed(histories["camera_0"]):
        if anchor > covered_until:
            continue
        if anchor < oldest_anchor:
            break
        try:
            return align_observation(
                histories,
                n_obs_steps,
                config,
                now_ns=now_ns,
                query_ns=tuple(
                    int(anchor) - step * period for step in range(n_obs_steps - 1, -1, -1)
                ),
            )
        except (ObservationNotReady, ObservationWindowUnavailable) as error:
            reason = str(error)
    raise ObservationNotReady(f"Latest aligned observation is not ready: {reason}")
