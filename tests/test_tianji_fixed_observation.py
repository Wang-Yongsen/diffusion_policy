"""Explicit timestamp alignment and latest synchronous observations, without hardware."""

from collections import deque
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from diffusion_policy.common.tianji_rail_pose import eef_action_to_6d
import torch

from diffusion_policy.env_runner.tianji_rail_real_runner import TianjiRailRealRunner
from diffusion_policy.real_world.decision_trace import SessionTrace
from diffusion_policy.real_world.tianji_policy_runtime import TianjiPolicyRuntime
from diffusion_policy.real_world.tianji_rail_env import (
    CAMERA_KEYS,
    STATE_KEYS,
    ObservationNotReady,
    ObservationWindowUnavailable,
    TianjiRailEnv,
    align_observation,
    align_latest_observation,
)
from tests.test_tianji_rail_real_eval import (
    FakeArm,
    FakeHand,
    checkpoint_cfg,
    config,
    histories,
)


ENV_MODULE = "diffusion_policy.real_world.tianji.env"
QUERIES = (1_000_000_000, 1_033_333_333)


def memory_env(n_obs_steps=2):
    env = TianjiRailEnv.__new__(TianjiRailEnv)
    env.config = config()
    env.config["inference_mode"] = "sync"
    env.n_obs_steps, env.motion = n_obs_steps, False
    env._lock = threading.Lock()
    env.error, env.recorder = None, None
    env.histories = {key: deque(series) for key, series in histories().items()}
    env._shadow_chunk_end_ns = (
        QUERIES[-1] + 2_000_000
    )  # Completion does not prescribe observation timestamps.
    env._shadow_source_chunk_id, env._shadow_chunk_id, env._shadow_prediction_id = 7, 7, 3
    env.runtime = SimpleNamespace(active=False, fault=None, pause=lambda *args, **kwargs: None)
    env.health_problem = lambda: None
    return env


def append_window(env, anchor, fill=100):
    period = round(1e9 / env.config["policy_hz"])
    for key in CAMERA_KEYS:
        for index, stamp in enumerate((anchor - period, anchor)):
            if stamp > env.histories[key][-1][0]:
                env.histories[key].append((stamp, np.full((16, 16, 3), fill + index, np.uint8)))
    for key in STATE_KEYS:
        samples = env.histories[key]
        while samples[-1][0] < anchor + 10_000_000:
            stamp = samples[-1][0] + 10_000_000
            samples.append((stamp, np.full_like(samples[-1][1], stamp / 1e9)))


class FixedObservationTests(unittest.TestCase):
    def test_old_window_keeps_images_and_interpolates_all_states_at_fixed_times(self):
        series = histories()
        for key, samples in series.items():
            if key in CAMERA_KEYS:
                samples.append((1_300_000_000, np.full_like(samples[-1][1], 200)))
            else:
                series[key] = [
                    (stamp, np.full_like(value, (stamp - QUERIES[0]) / 1e9))
                    for stamp, value in samples
                ]
                series[key].append((1_300_000_000, np.full_like(samples[-1][1], 0.3)))
        obs, anchor, meta = align_observation(
            series, 2, config(), now_ns=1_300_000_000, query_ns=QUERIES
        )
        self.assertEqual(anchor, QUERIES[-1])
        self.assertEqual(meta["query_ns"], list(QUERIES))
        for index, key in enumerate(CAMERA_KEYS):
            self.assertEqual(meta["camera_ns"][key], list(QUERIES))
            np.testing.assert_array_equal(obs[key][:, 0, 0, 0], [index, index + 10])
        for key in ("robot_eef_pose", "hand_joint"):
            np.testing.assert_allclose(obs[key][0], 0, atol=1e-7)
            np.testing.assert_allclose(obs[key][1], 1 / 30, atol=1e-7)
        for key in STATE_KEYS:
            self.assertEqual(
                meta["state_brackets_ns"][key],
                [(QUERIES[0], QUERIES[0]), (1_030_000_000, 1_040_000_000)],
            )

    def test_camera_matching_uses_closest_distinct_pair_instead_of_repeating_nearest(self):
        series = histories()
        # Independent nearest selection would use the middle frame for both queries.
        for key in CAMERA_KEYS:
            image = series[key][0][1]
            series[key] = [(990_000_000, image), (1_016_000_000, image), (1_060_000_000, image)]
        _, _, meta = align_observation(
            series, 2, config(), now_ns=1_060_000_000, query_ns=QUERIES
        )
        for key in CAMERA_KEYS:
            self.assertEqual(meta["camera_ns"][key], [990_000_000, 1_016_000_000])

    def test_camera_matching_can_choose_an_alternative_pair_to_meet_span(self):
        series = histories()
        cfg = config()
        cfg["max_camera_span_s"] = 0.005
        for key in CAMERA_KEYS:
            image = series[key][0][1]
            series[key] = [(1_010_000_000, image), (1_043_333_333, image)]
        image = series["camera_0"][0][1]
        series["camera_0"] = [
            (QUERIES[0], image),
            (1_010_000_000, image),
            (QUERIES[1], image),
            (1_043_333_333, image),
        ]
        _, _, meta = align_observation(series, 2, cfg, now_ns=1_060_000_000, query_ns=QUERIES)
        for key in CAMERA_KEYS:
            self.assertEqual(meta["camera_ns"][key], [1_010_000_000, 1_043_333_333])

    def test_duplicate_frame_waits_then_fails_when_new_frames_cannot_fill_the_window(self):
        series = histories()
        image = series["camera_2"][0][1]
        series["camera_2"] = [(1_034_000_000, image)]
        with self.assertRaisesRegex(ObservationNotReady, "two distinct images"):
            align_observation(series, 2, config(), now_ns=1_050_000_000, query_ns=QUERIES)
        series["camera_2"].append((1_100_000_000, image))
        with self.assertRaisesRegex(ObservationWindowUnavailable, "no distinct images"):
            align_observation(series, 2, config(), now_ns=1_100_000_000, query_ns=QUERIES)

    def test_irrecoverable_three_camera_span_is_fatal(self):
        series = histories()
        cfg = config()
        cfg["max_camera_span_s"] = 0.005
        series["camera_2"] = [
            (stamp + 10_000_000, image) for stamp, image in series["camera_2"]
        ]
        for key in CAMERA_KEYS:
            series[key].append((1_100_000_000, series[key][-1][1]))
        with self.assertRaisesRegex(ObservationWindowUnavailable, "span exceeds"):
            align_observation(series, 2, cfg, now_ns=1_100_000_000, query_ns=QUERIES)

    def test_state_gap_is_fatal_but_uncovered_future_state_is_retryable(self):
        series = histories()
        series["hands/right"] = series["hands/right"][:2]
        with self.assertRaisesRegex(ObservationNotReady, "cover fixed"):
            align_observation(series, 2, config(), now_ns=1_100_000_000, query_ns=QUERIES)
        series["hands/right"] = [(990_000_000, np.zeros(20)), (1_060_000_000, np.zeros(20))]
        with self.assertRaisesRegex(
            ObservationWindowUnavailable, "interpolation interval is too large"
        ):
            align_observation(series, 2, config(), now_ns=1_060_000_000, query_ns=QUERIES)

    def test_evicted_history_fails_in_timestamp_preview_without_copying_images(self):
        for key in CAMERA_KEYS + STATE_KEYS:
            with self.subTest(key=key):
                env = memory_env()
                env.histories[key] = deque([(1_100_000_000, object())])
                with self.assertRaisesRegex(ObservationWindowUnavailable, "history was lost"):
                    env.observation_ready(QUERIES)

    def test_preview_waits_for_each_of_seven_streams(self):
        for key in CAMERA_KEYS + STATE_KEYS:
            with self.subTest(key=key):
                env = memory_env()
                env.histories[key] = deque([(QUERIES[0], object())])
                with patch(
                    f"{ENV_MODULE}.np.stack",
                    side_effect=AssertionError("image copy in preview"),
                ):
                    self.assertFalse(env.observation_ready(QUERIES))

    def test_latest_stream_freshness_checks_still_apply_to_old_fixed_windows(self):
        for key in CAMERA_KEYS + STATE_KEYS:
            with self.subTest(key=key):
                series = histories()
                for name, samples in series.items():
                    if name != key:
                        samples.append((1_300_000_000, samples[-1][1]))
                with self.assertRaisesRegex(ObservationNotReady, f"{key} is stale"):
                    align_observation(
                        series, 2, config(), now_ns=1_300_000_000, query_ns=QUERIES
                    )
        env = memory_env()
        env.error = "collector failed"
        with self.assertRaisesRegex(RuntimeError, "collector failed"):
            env.observation_ready(QUERIES)
        env.error = None
        env.health_problem = lambda: "camera_2: missing/stale history"
        with self.assertRaisesRegex(RuntimeError, "missing/stale history"):
            env.observation_ready(QUERIES)

    def test_fixed_query_validation(self):
        for queries in ([QUERIES[0]], QUERIES[::-1], [1.0, 2.0], [True, False]):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                align_observation(
                    histories(), 2, config(), now_ns=1_100_000_000, query_ns=queries
                )
        for queries in ([], QUERIES, [1.0], [True], 1):
            with self.subTest(queries=queries), self.assertRaises(ValueError):
                align_observation(
                    histories(), 1, config(), now_ns=1_100_000_000, query_ns=queries
                )
        for steps in (0, 3):
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "n_obs_steps"):
                TianjiRailEnv({"inference_mode": "sync"}, steps)
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "n_obs_steps"):
                align_latest_observation(histories(), steps, config(), now_ns=1_100_000_000)

    def test_sync_constructor_accepts_both_counts_before_loading_device_config(self):
        for steps in (1, 2):
            with self.subTest(steps=steps), patch(
                f"{ENV_MODULE}.load_arm_config",
                side_effect=RuntimeError("offline config boundary"),
            ):
                with self.assertRaisesRegex(RuntimeError, "offline config boundary"):
                    TianjiRailEnv(config() | {"inference_mode": "sync"}, steps)

    def test_single_step_needs_only_one_image_and_aligns_joint_states_at_current_time(self):
        env = memory_env(n_obs_steps=1)
        append_window(env, 1_100_000_000)
        for key in CAMERA_KEYS:
            env.histories[key] = deque([env.histories[key][-1]])
        for side in ("left", "right"):
            env.histories[f"arm_joints/{side}"] = deque(
                (stamp, np.full(7, stamp / 1e9)) for stamp, _ in env.histories[f"arms/{side}"]
            )
        self.assertTrue(env.observation_ready([1_100_000_000]))
        with patch(f"{ENV_MODULE}.time.monotonic_ns", return_value=1_120_000_000):
            obs, anchor, meta = env.get_obs()
        self.assertEqual(anchor, 1_100_000_000)
        self.assertEqual(meta["query_ns"], [anchor])
        for key in CAMERA_KEYS:
            self.assertEqual(obs[key].shape, (1, 16, 16, 3))
            self.assertEqual(meta["camera_ns"][key], [anchor])
            np.testing.assert_array_equal(obs[key][:, 0, 0, 0], [101])
        for key, width in (("robot_eef_pose", 12), ("robot_joint", 14), ("hand_joint", 40)):
            self.assertEqual(obs[key].shape, (1, width))
            np.testing.assert_allclose(obs[key], 1.1, atol=1e-6)
        for key, brackets in meta["state_brackets_ns"].items():
            self.assertEqual(len(brackets), 1, key)
            self.assertLessEqual(brackets[0][0], anchor)
            self.assertGreaterEqual(brackets[0][1], anchor)

    def test_single_step_camera_matching_can_choose_an_alternative_to_meet_span(self):
        series = histories()
        query = QUERIES[-1]
        cfg = config() | {"max_camera_span_s": 0.005}
        for key in CAMERA_KEYS:
            image = series[key][0][1]
            later = query if key == "camera_0" else query + 20_000_000
            series[key] = [(query - 10_000_000, image), (later, image)]
        _, _, meta = align_observation(series, 1, cfg, now_ns=1_070_000_000, query_ns=[query])
        for key in CAMERA_KEYS:
            self.assertEqual(meta["camera_ns"][key], [query - 10_000_000])

    def test_single_step_falls_back_from_a_state_gap_and_recovers(self):
        series = histories()
        series["hands/right"] = [(QUERIES[0], np.zeros(20)), (1_060_000_000, np.ones(20))]
        _, anchor, meta = align_latest_observation(series, 1, config(), now_ns=1_070_000_000)
        self.assertEqual(anchor, QUERIES[0])
        self.assertEqual(meta["query_ns"], [anchor])
        series["hands/right"].insert(1, (1_030_000_000, np.full(20, 0.5)))
        _, anchor, meta = align_latest_observation(series, 1, config(), now_ns=1_070_000_000)
        self.assertEqual(anchor, QUERIES[-1])
        self.assertEqual(meta["query_ns"], [anchor])

    def test_single_step_retains_age_skew_span_and_state_gap_limits(self):
        cases = (
            ({}, 1_300_000_000, None),
            ({"max_image_skew_s": 0.001}, 1_070_000_000, "camera_skew"),
            ({"max_camera_span_s": 0.001}, 1_070_000_000, "camera_skew"),
            ({}, 1_070_000_000, "state_gap"),
        )
        for overrides, now, problem in cases:
            with self.subTest(overrides=overrides, problem=problem):
                series = histories()
                if problem == "camera_skew":
                    series["camera_2"] = [
                        (stamp + 5_000_000, image) for stamp, image in series["camera_2"]
                    ]
                elif problem == "state_gap":
                    series["hands/right"] = [
                        (990_000_000, np.zeros(20)),
                        (1_060_000_000, np.zeros(20)),
                    ]
                with self.assertRaises(ObservationNotReady):
                    align_latest_observation(series, 1, config() | overrides, now_ns=now)

    def test_latest_observation_uses_current_and_one_period_of_history(self):
        env = memory_env()
        append_window(env, 1_100_000_000)
        obs, anchor, meta = align_latest_observation(
            env.histories, 2, env.config, now_ns=1_120_000_000
        )
        self.assertEqual(anchor, 1_100_000_000)
        self.assertEqual(meta["query_ns"], [1_066_666_667, 1_100_000_000])
        for key in CAMERA_KEYS:
            self.assertEqual(meta["camera_ns"][key], meta["query_ns"])
            np.testing.assert_array_equal(obs[key][:, 0, 0, 0], [100, 101])
        for key in ("robot_eef_pose", "hand_joint"):
            np.testing.assert_allclose(obs[key][:, 0], [1.066666667, 1.1], atol=1e-6)
        for key in STATE_KEYS:
            for query, (low, high) in zip(meta["query_ns"], meta["state_brackets_ns"][key]):
                self.assertLessEqual(low, query)
                self.assertGreaterEqual(high, query)

    def test_latest_anchor_waits_for_camera_and_state_coverage(self):
        for key in ("camera_2", "hands/right"):
            with self.subTest(key=key):
                env = memory_env()
                old = list(env.histories[key])
                append_window(env, 1_100_000_000)
                new = env.histories[key]
                env.histories[key] = deque(old)
                _, anchor, _ = align_latest_observation(
                    env.histories, 2, env.config, now_ns=1_110_000_000
                )
                self.assertEqual(anchor, QUERIES[-1])
                env.histories[key] = new
                _, anchor, _ = align_latest_observation(
                    env.histories, 2, env.config, now_ns=1_110_000_000
                )
                self.assertEqual(anchor, 1_100_000_000)

    def test_latest_selection_falls_back_to_newest_valid_pair_and_recovers(self):
        env = memory_env()
        env.config["max_image_skew_s"] = 0.005
        append_window(env, 1_100_000_000)
        missing = env.histories["camera_2"][-2]
        del env.histories["camera_2"][-2]
        _, anchor, _ = align_latest_observation(
            env.histories, 2, env.config, now_ns=1_110_000_000
        )
        self.assertEqual(anchor, QUERIES[-1])
        env.histories["camera_2"].insert(len(env.histories["camera_2"]) - 1, missing)
        _, anchor, _ = align_latest_observation(
            env.histories, 2, env.config, now_ns=1_110_000_000
        )
        self.assertEqual(anchor, 1_100_000_000)

    def test_latest_window_keeps_age_limit_and_requires_two_different_frames(self):
        with self.assertRaises(ObservationNotReady):
            align_latest_observation(histories(), 2, config(), now_ns=1_300_000_000)
        env = memory_env()
        env.histories["camera_2"].popleft()
        with self.assertRaises(ObservationNotReady):
            align_latest_observation(env.histories, 2, env.config, now_ns=1_070_000_000)

    def test_runner_retries_latest_observation_after_stability_wait_and_freezes_input(self):
        env = memory_env()
        env.histories["camera_2"].popleft()
        reads, gates, predictions, submitted, events = [], [], [], [], []
        original_read, original_window = env.get_obs, env.sync_observation_window
        window_reads = []

        def window():
            window_reads.append(1)
            return None if len(window_reads) == 1 else original_window()

        def read():
            reads.append(1)
            self.assertLess(len(reads), 5, "latest observation never became ready")
            return original_read()

        def stability_wait():
            gates.append(1)
            if len(gates) == 2:
                append_window(env, 1_100_000_000, fill=200)
            return None

        def predict(obs):
            predictions.append(obs)
            np.testing.assert_allclose(obs["camera_0"][0, :, 0, 0, 0], [200 / 255, 201 / 255])
            append_window(env, 1_166_666_666, fill=240)
            # New camera data cannot alter the snapshot already passed to this prediction.
            np.testing.assert_allclose(obs["camera_0"][0, :, 0, 0, 0], [200 / 255, 201 / 255])
            return {
                "action": torch.from_numpy(eef_action_to_6d(np.zeros((1, 8, 52), np.float32)))
            }

        def execute(actions, anchor, prediction_id=None):
            submitted.append((actions, anchor, prediction_id))
            runner._stop.set()
            return 0

        env.sync_observation_window, env.get_obs, env.exec_actions = window, read, execute
        env.emit = lambda event, **values: events.append((event, values))
        env.observation_ready = lambda *args: self.fail(
            "latest mode used a fixed-window preview"
        )
        policy = SimpleNamespace(
            device=torch.device("cpu"),
            n_action_steps=8,
            reset=lambda: None,
            predict_action=predict,
        )
        trace = SessionTrace(decision_capacity=2)
        initial_obs, _, _ = align_observation(histories(), 2, config(), now_ns=1_050_000_000)
        trace.allocate_decisions(initial_obs, 8)
        with tempfile.TemporaryDirectory() as root, patch(
            f"{ENV_MODULE}.time.monotonic_ns", return_value=1_120_000_000
        ):
            runner = TianjiRailRealRunner(
                env, policy, checkpoint_cfg(), {}, Path(root) / "session", trace=trace
            )
            runner._before_observation = stability_wait
            runner._begin()
            runner._infer()
            self.assertIsNone(runner.error)
            self.assertEqual(
                (len(reads), len(gates), len(predictions), len(submitted)), (2, 2, 1, 1)
            )
            self.assertEqual(len(window_reads), 2)
            decision = next(values for event, values in events if event == "prediction")
            self.assertEqual(decision["chunk_completed_ns"], env._shadow_chunk_end_ns)
            self.assertEqual(decision["anchor_ns"], 1_100_000_000)
            self.assertEqual(decision["alignment"]["source_chunk_id"], 7)
            self.assertEqual(decision["alignment"]["source_prediction_id"], 3)
            self.assertEqual(decision["alignment"]["query_ns"], [1_066_666_667, 1_100_000_000])
            self.assertEqual(trace.decision_count, 1)
            self.assertEqual(
                trace.meta[0]["alignment"], json.loads(json.dumps(decision["alignment"]))
            )
            np.testing.assert_array_equal(trace.images["camera_0"][0, :, 0, 0, 0], [200, 201])
            self.assertEqual(submitted[0][2], 0)
            runner._end("shadow", "test complete")

    def test_runner_rejects_short_checkpoint_before_creating_output(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "session"
            with self.assertRaisesRegex(ValueError, "n_action_steps >= 2"):
                TianjiRailRealRunner(
                    memory_env(), SimpleNamespace(n_action_steps=1), {}, {}, output
                )
            self.assertFalse(output.exists())

    def test_stop_during_latest_observation_inference_discards_prediction(self):
        env = memory_env()
        entered, release = threading.Event(), threading.Event()
        submitted = []

        def predict(obs):
            entered.set()
            release.wait(2)
            return {
                "action": torch.from_numpy(eef_action_to_6d(np.zeros((1, 8, 52), np.float32)))
            }

        env.exec_actions = lambda *args, **kwargs: submitted.append(args)
        policy = SimpleNamespace(
            device=torch.device("cpu"),
            n_action_steps=8,
            reset=lambda: None,
            predict_action=predict,
        )
        with tempfile.TemporaryDirectory() as root, patch(
            f"{ENV_MODULE}.time.monotonic_ns", return_value=1_070_000_000
        ):
            runner = TianjiRailRealRunner(
                env, policy, checkpoint_cfg(), {}, Path(root) / "session"
            )
            runner._begin()
            worker = threading.Thread(target=runner._infer)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                runner._end("shadow", "stop during inference")
                runner._stop.set()
            finally:
                release.set()
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertIsNone(runner.error)
            self.assertEqual(submitted, [])

    def test_new_trial_discards_previous_completion_context(self):
        env = memory_env()
        reads, events = [], []
        original_read = env.get_obs

        def read():
            reads.append(1)
            self.assertLess(len(reads), 4)
            if len(reads) == 1:
                runner._end("shadow", "restart while waiting")
                runner._begin()
                self.assertIsNone(env._shadow_source_chunk_id)
                env._shadow_source_chunk_id, env._shadow_chunk_end_ns = 8, 1_100_000_000
                append_window(env, 1_100_000_000)
                raise ObservationNotReady("new trial")
            return original_read()

        def execute(*args, **kwargs):
            runner._stop.set()
            return 0

        env.get_obs, env.exec_actions = read, execute
        env.emit = lambda event, **values: events.append((event, values))
        policy = SimpleNamespace(
            device=torch.device("cpu"),
            n_action_steps=8,
            reset=lambda: None,
            predict_action=lambda obs: {
                "action": torch.from_numpy(eef_action_to_6d(np.zeros((1, 8, 52), np.float32)))
            },
        )
        with tempfile.TemporaryDirectory() as root, patch(
            f"{ENV_MODULE}.time.monotonic_ns", return_value=1_120_000_000
        ):
            runner = TianjiRailRealRunner(
                env, policy, checkpoint_cfg(), {}, Path(root) / "session"
            )
            runner._begin()
            runner._infer()
            self.assertIsNone(runner.error)
            self.assertEqual(len(reads), 2)
            decision = next(values for event, values in events if event == "prediction")
            self.assertEqual(decision["alignment"]["query_ns"], [1_066_666_667, 1_100_000_000])
            self.assertEqual(decision["alignment"]["source_chunk_id"], 8)
            runner._end("shadow", "test complete")

    def test_trial_timeout_ends_wait_for_latest_usable_window(self):
        env = memory_env()
        env.histories["camera_2"].popleft()
        env.config["trial_timeout_s"] = 0.02
        reads = []
        original_read = env.get_obs

        def read():
            reads.append(1)
            return original_read()

        env.get_obs = read
        policy = SimpleNamespace(
            n_action_steps=8,
            reset=lambda: None,
            predict_action=lambda obs: self.fail("inference without complete data"),
        )
        with tempfile.TemporaryDirectory() as root, patch(
            f"{ENV_MODULE}.time.monotonic_ns", return_value=1_070_000_000
        ), patch(
            "diffusion_policy.env_runner.tianji_rail_real_runner.sys.stdin.isatty",
            return_value=False,
        ):
            runner = TianjiRailRealRunner(
                env, policy, checkpoint_cfg(), {}, Path(root) / "session"
            )
            with patch.object(runner, "_end", wraps=runner._end) as end:
                runner.run()
                end.assert_any_call("shadow", "Trial time limit")
            self.assertIsNone(runner.error)
            self.assertTrue(reads)
            trial = json.loads((Path(root) / "session/trial_0001/trial.json").read_text())
            self.assertEqual(trial["outcome"], "device_abort")
            self.assertIsNone(env._shadow_source_chunk_id)


class CompletionContextTests(unittest.TestCase):
    def execute_steps(self, runtime):
        times = runtime._times.copy()
        for stamp in times:
            self.now = int(stamp)
            runtime.tick("arm")
            runtime.tick("hand")
        return int(times[-1]) + round(1e9 / 30)

    def runtime(self):
        self.now, self.events = 1_000_000_000, []
        cfg = config()
        cfg["inference_mode"] = "sync"
        runtime = TianjiPolicyRuntime(
            FakeArm(),
            {side: FakeHand(side) for side in ("left", "right")},
            cfg,
            lambda: None,
            lambda event, **values: self.events.append((event, values)),
            clock_ns=lambda: self.now,
        )
        runtime.seed(np.zeros(52), arm_ready=True)
        self.addCleanup(runtime.close)
        return runtime

    def test_completion_context_does_not_prescribe_observation_times(self):
        runtime = self.runtime()
        runtime.submit(np.zeros((8, 52)), 0, prediction_id=4)
        schedule = self.events[-1][1]
        self.assertNotIn("next_observation_query_ns", schedule)
        self.assertIsNone(runtime.sync_observation_window())
        self.now = self.execute_steps(runtime) + 20_000_000
        runtime.tick("arm")
        window = runtime.sync_observation_window()
        self.assertNotIn("query_ns", window)
        self.assertEqual(window["completed_ns"], self.now)
        self.assertEqual(window["source_chunk_id"], schedule["generation"])
        self.assertEqual(window["source_prediction_id"], 4)
        self.now += 300_000_000
        runtime.tick("arm")
        self.assertEqual(runtime.sync_observation_window(), window)
        window["source_chunk_id"] = -1
        self.assertEqual(
            runtime.sync_observation_window()["source_chunk_id"], schedule["generation"]
        )

    def test_pause_seed_and_next_chunk_clear_previous_context(self):
        runtime = self.runtime()
        self.assertIsNone(runtime.sync_observation_window()["source_chunk_id"])
        runtime.submit(np.zeros((8, 52)), 0)
        self.now = self.execute_steps(runtime)
        runtime.tick("arm")
        first = runtime.sync_observation_window()
        runtime.submit(np.zeros((8, 52)), 0)
        self.assertIsNone(runtime.sync_observation_window())
        self.now = self.execute_steps(runtime)
        runtime.tick("arm")
        second = runtime.sync_observation_window()
        self.assertGreater(second["completed_ns"], first["completed_ns"])
        self.assertGreater(second["source_chunk_id"], first["source_chunk_id"])
        runtime.pause("operator")
        self.assertIsNone(runtime.sync_observation_window())
        runtime.seed(np.zeros(52), arm_ready=True)
        self.assertIsNone(runtime.sync_observation_window()["source_chunk_id"])

    def test_sync_rejects_single_action_without_activating_chunk(self):
        runtime = self.runtime()
        with self.assertRaisesRegex(ValueError, "at least two action steps"):
            runtime.submit(np.zeros((1, 52)), 0)
        self.assertFalse(runtime.active)
        self.assertIsNone(runtime.sync_observation_window()["source_chunk_id"])


if __name__ == "__main__":
    unittest.main()
