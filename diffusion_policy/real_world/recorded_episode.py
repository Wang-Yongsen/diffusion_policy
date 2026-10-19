"""Step-locked windows from a converted 30 Hz Tianji rail dataset.

Observations stay on the demonstration clock. The next window begins
``n_action_steps`` frames later, which is the same advance a synchronous
execution chunk makes. Nothing here talks to hardware.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import zarr

CAMERAS = ("camera_0", "camera_1", "camera_2")
POLICY_HZ = 30
# Approach stays under the strict deployment speeds so a loose trial config
# cannot turn the move-to-start into a fast Cartesian slide.
APPROACH_SPEEDS = {
    "max_position_speed_m_s": 0.15,
    "max_rotation_speed_rad_s": 0.8,
    "max_hand_speed_rad_s": 2.0,
}
APPROACH_MAX_POSITION_M = 0.8
APPROACH_MAX_ROTATION_RAD = np.deg2rad(90)


def playback_starts(n_frames, n_obs_steps, n_action_steps):
    """Window starts that advance one executed chunk at a time."""
    for name, value in (("n_frames", n_frames), ("n_obs_steps", n_obs_steps), ("n_action_steps", n_action_steps)):
        if type(value) is not int:
            raise ValueError(f"{name} must be an integer")
    if n_obs_steps < 1 or n_action_steps < 1 or n_frames < n_obs_steps:
        raise ValueError("Need a positive window and at least n_obs_steps frames")
    return list(range(0, n_frames - n_obs_steps + 1, n_action_steps))


def pose_delta(current, goal):
    """Millimetres, degrees, and the largest hand-joint degree error."""
    current, goal = _action_row(current), _action_row(goal)
    result = {}
    for index, side in enumerate(("left", "right")):
        result[f"{side}_position_mm"] = float(np.linalg.norm(goal[6 * index:6 * index + 3] - current[6 * index:6 * index + 3]) * 1000)
        current_r = Rotation.from_rotvec(current[6 * index + 3:6 * index + 6])
        goal_r = Rotation.from_rotvec(goal[6 * index + 3:6 * index + 6])
        result[f"{side}_rotation_deg"] = float(np.rad2deg((goal_r * current_r.inv()).magnitude()))
    result["hand_max_deg"] = float(np.rad2deg(np.max(np.abs(goal[12:] - current[12:]))))
    return result


def approach_within_limits(current, goal):
    delta = pose_delta(current, goal)
    position = max(delta["left_position_mm"], delta["right_position_mm"]) / 1000
    rotation = np.deg2rad(max(delta["left_rotation_deg"], delta["right_rotation_deg"]))
    return position <= APPROACH_MAX_POSITION_M and rotation <= APPROACH_MAX_ROTATION_RAD


def approach_actions(start, goal, *, hz=POLICY_HZ, speeds=None):
    """30 Hz path from the current target to ``goal``. The current target itself is not included.

    Position and finger joints are linear. Flange rotation uses slerp. Each step
    stays a little under the given speed caps so a later exact speed check passes.
    """
    start, goal = _action_row(start), _action_row(goal)
    speeds = APPROACH_SPEEDS if speeds is None else speeds
    hz = float(hz)
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be positive")
    steps = 1
    for index in range(2):
        distance = np.linalg.norm(goal[6 * index:6 * index + 3] - start[6 * index:6 * index + 3])
        angle = (Rotation.from_rotvec(goal[6 * index + 3:6 * index + 6])
                 * Rotation.from_rotvec(start[6 * index + 3:6 * index + 6]).inv()).magnitude()
        steps = max(steps, _steps(distance, speeds["max_position_speed_m_s"], hz),
                    _steps(angle, speeds["max_rotation_speed_rad_s"], hz))
    steps = max(steps, _steps(np.max(np.abs(goal[12:] - start[12:])), speeds["max_hand_speed_rad_s"], hz))
    path = np.empty((steps, 52), dtype=np.float64)
    start_r = [Rotation.from_rotvec(start[6 * index + 3:6 * index + 6]) for index in range(2)]
    goal_r = [Rotation.from_rotvec(goal[6 * index + 3:6 * index + 6]) for index in range(2)]
    for step in range(1, steps + 1):
        fraction = step / steps
        row = start + fraction * (goal - start)
        for index in range(2):
            blend = Slerp([0., 1.], Rotation.concatenate([start_r[index], goal_r[index]]))([fraction])
            row[6 * index + 3:6 * index + 6] = blend.as_rotvec()[0]
        path[step - 1] = row
    path[-1] = goal
    return path


def list_segments(path):
    root = zarr.open_group(str(Path(path).expanduser()), mode="r")
    if root.attrs.get("action_space", "eef") != "eef":
        raise ValueError("Recorded playback requires action_space=eef")
    data, meta = root["data"], root["meta"]
    count = int(data["action"].shape[0])
    if data["action"].shape[1:] != (52,):
        raise ValueError("Recorded actions must be 52-D")
    episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    segment_ends = np.asarray(meta["segment_ends"][:] if "segment_ends" in meta else episode_ends, dtype=np.int64)
    _check_ends(episode_ends, count, "episode_ends")
    _check_ends(segment_ends, count, "segment_ends")
    if not np.isin(episode_ends, segment_ends).all():
        raise ValueError("segment_ends must preserve episode boundaries")
    described = meta.attrs.get("segments", [])
    if described and len(described) != len(segment_ends):
        raise ValueError("Segment metadata count differs from segment_ends")
    starts = np.r_[0, segment_ends[:-1]]
    result = []
    for index, (low, high) in enumerate(zip(starts, segment_ends)):
        episode_index = int(np.searchsorted(episode_ends, int(high) - 1, side="left"))
        source = described[index].get("source_episode") if described else f"episode_{episode_index:06d}"
        result.append({"index": index, "episode_index": episode_index, "source": str(source),
                       "start": int(low), "end": int(high), "frames": int(high - low)})
    return result


def open_recorded_segment(path, *, episode=0, segment=None):
    """Open one continuous segment. ``episode`` is a source name or a 0-based episode index."""
    items = list_segments(path)
    if segment is not None:
        if type(segment) is not int or not 0 <= segment < len(items):
            raise ValueError(f"segment must be an index in [0, {len(items)})")
        chosen = items[segment]
    else:
        if isinstance(episode, str) and not episode.isdigit():
            matched = [item for item in items if item["source"] == episode]
        else:
            matched = [item for item in items if item["episode_index"] == int(episode)]
        if not matched:
            raise ValueError(f"No segment for episode {episode}")
        chosen = max(matched, key=lambda item: item["frames"])
    return RecordedSegment(path, chosen)


class RecordedSegment:
    def __init__(self, path, selected):
        self.path = str(Path(path).expanduser())
        self.info = dict(selected)
        self.start, self.end = selected["start"], selected["end"]
        self.root = zarr.open_group(self.path, mode="r")
        data = self.root["data"]
        self.action = np.asarray(data["action"][self.start:self.end], dtype=np.float64)
        self.robot_eef_pose = np.asarray(data["robot_eef_pose"][self.start:self.end], dtype=np.float32)
        self.hand_joint = np.asarray(data["hand_joint"][self.start:self.end], dtype=np.float32)
        self._images = {key: data[key] for key in CAMERAS}
        if "timestamp" in data and self.n_frames > 1:
            dt = np.diff(np.asarray(data["timestamp"][self.start:self.end], dtype=np.float64))
            if np.median(np.abs(dt - 1 / POLICY_HZ)) > 0.002:
                raise ValueError("Recorded timestamps are not a 30 Hz grid")

    @property
    def n_frames(self):
        return self.end - self.start

    @property
    def source(self):
        return self.info["source"]

    def observation(self, window_start, n_obs_steps):
        stop = _window_stop(window_start, n_obs_steps, self.n_frames)
        absolute = self.start + window_start
        obs = {key: np.asarray(array[absolute:absolute + n_obs_steps]) for key, array in self._images.items()}
        obs["robot_eef_pose"] = self.robot_eef_pose[window_start:stop].copy()
        obs["hand_joint"] = self.hand_joint[window_start:stop].copy()
        return obs

    def label_actions(self, window_start, n_obs_steps, n_action_steps):
        """Demonstration actions aligned with policy execution, or None when the label runs off the segment."""
        first = _window_stop(window_start, n_obs_steps, self.n_frames) - 1
        last = first + n_action_steps
        if type(n_action_steps) is not int or n_action_steps < 1 or last > self.n_frames:
            return None
        return self.action[first:last].copy()


def _window_stop(window_start, n_obs_steps, n_frames):
    if type(window_start) is not int or type(n_obs_steps) is not int or window_start < 0 or n_obs_steps < 1:
        raise ValueError("window_start and n_obs_steps must be nonnegative / positive integers")
    stop = window_start + n_obs_steps
    if stop > n_frames:
        raise ValueError("Observation window runs past the recorded segment")
    return stop


def _action_row(value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (52,) or not np.isfinite(value).all():
        raise ValueError("Require one finite 52-D action")
    return value


def _steps(distance, speed, hz):
    if distance < 0 or not np.isfinite(distance) or not np.isfinite(speed) or speed <= 0:
        raise ValueError("Approach distance and speed must be finite and nonnegative / positive")
    return int(np.ceil(distance / (speed / hz * 0.98) - 1e-12))


def _check_ends(ends, count, name):
    if (ends.ndim != 1 or not len(ends) or ends[0] <= 0 or np.any(np.diff(ends) <= 0) or int(ends[-1]) != count):
        raise ValueError(f"Invalid {name}")
