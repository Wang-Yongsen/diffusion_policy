"""Recorded-observation windows and the slow move onto the first predicted target."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from diffusion_policy.real_world.tianji.recorded_episode import (
    approach_actions,
    approach_within_limits,
    list_segments,
    open_recorded_segment,
    playback_starts,
    pose_delta,
)
from diffusion_policy.real_world.tianji.runtime import validate_actions


def write_dataset(path):
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(schema_version=2, action_space="eef")
    data, meta = root.create_group("data"), root.create_group("meta")
    meta.array("episode_ends", np.asarray([25, 45], np.int64))
    meta.array("segment_ends", np.asarray([20, 25, 45], np.int64))
    meta.attrs["segments"] = [
        {"source_episode": name} for name in ("episode_000000", "episode_000000", "episode_000001")
    ]
    frames = 45
    data.array("timestamp", np.arange(frames, dtype=np.float64) / 30)
    data.array(
        "action", np.arange(frames, dtype=np.float32)[:, None] + np.zeros((1, 52), np.float32)
    )
    data.array(
        "robot_eef_pose",
        np.arange(frames, dtype=np.float32)[:, None] + np.zeros((1, 12), np.float32),
    )
    data.array(
        "hand_joint",
        np.arange(frames, dtype=np.float32)[:, None] * 0.01 + np.zeros((1, 40), np.float32),
    )
    images = np.arange(frames, dtype=np.uint8)[:, None, None, None] + np.zeros(
        (1, 4, 5, 3), np.uint8
    )
    for key in ("camera_0", "camera_1", "camera_2"):
        data.array(key, images, chunks=(1, 4, 5, 3))


class RecordedEpisodeTest(unittest.TestCase):
    def test_playback_starts_follow_executed_chunks(self):
        self.assertEqual(playback_starts(10, 2, 8), [0, 8])
        self.assertEqual(playback_starts(3, 2, 8), [0])
        with self.assertRaises(ValueError):
            playback_starts(1, 2, 8)

    def test_segment_window_matches_execution_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.zarr"
            write_dataset(path)
            listed = list_segments(path)
            self.assertEqual([item["frames"] for item in listed], [20, 5, 20])
            segment = open_recorded_segment(path, episode="episode_000000")
            self.assertEqual(segment.info["index"], 0)
            self.assertEqual(segment.n_frames, 20)
            obs = segment.observation(8, 2)
            self.assertEqual(obs["camera_0"].shape, (2, 4, 5, 3))
            self.assertEqual(int(obs["camera_0"][0, 0, 0, 0]), 8)
            self.assertEqual(int(obs["camera_0"][1, 0, 0, 0]), 9)
            np.testing.assert_allclose(obs["robot_eef_pose"][:, 0], [8, 9])
            labels = segment.label_actions(8, 2, 4)
            np.testing.assert_allclose(labels[:, 0], [9, 10, 11, 12])
            self.assertIsNone(segment.label_actions(14, 2, 8))

    def test_episode_index_selects_longest_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.zarr"
            write_dataset(path)
            self.assertEqual(open_recorded_segment(path, episode=1).source, "episode_000001")

    def test_approach_reaches_goal_under_strict_speed(self):
        start = np.zeros(52)
        goal = np.zeros(52)
        goal[0] = 0.5
        goal[3:6] = Rotation.from_euler("z", 40, degrees=True).as_rotvec()
        goal[12] = 0.4
        self.assertTrue(approach_within_limits(start, goal))
        path = approach_actions(start, goal)
        self.assertGreater(len(path), 8)
        np.testing.assert_allclose(path[-1], goal)
        config = {
            "max_position_jump_m": 0.04,
            "max_rotation_jump_rad": 0.3,
            "max_hand_jump_rad": 0.3,
            "max_position_speed_m_s": 0.15,
            "max_rotation_speed_rad_s": 0.8,
            "max_hand_speed_rad_s": 2.0,
            "workspace": {
                side: {"min": [-2.0, -2.0, -2.0], "max": [2.0, 2.0, 2.0]}
                for side in ("left", "right")
            },
        }
        reference = start
        for offset in range(0, len(path), 8):
            chunk = path[offset : offset + 8]
            times = 1_000_000_000 + np.arange(len(chunk)) * round(1e9 / 30)
            validate_actions(chunk, times, reference, config)
            reference = chunk[-1]
        delta = pose_delta(start, goal)
        self.assertAlmostEqual(delta["left_position_mm"], 500, places=3)
        self.assertAlmostEqual(delta["left_rotation_deg"], 40, places=3)

    def test_far_target_is_not_an_automatic_approach(self):
        start, goal = np.zeros(52), np.zeros(52)
        goal[0] = 1.2
        self.assertFalse(approach_within_limits(start, goal))


if __name__ == "__main__":
    unittest.main()
