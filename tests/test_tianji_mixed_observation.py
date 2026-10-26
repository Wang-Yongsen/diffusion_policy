"""Mixed-input routing, fresh live windows, and stop/trace behavior without devices."""

from contextlib import redirect_stdout
import importlib.util
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from diffusion_policy.common.tianji_rail_pose import eef_action_to_6d
import torch
import zarr

from diffusion_policy.real_world.tianji import replay
from diffusion_policy.real_world.recorded_episode import (
    OBSERVATION_MODES,
    compose_observation,
    observation_sources,
    open_recorded_segment,
)
from diffusion_policy.real_world.tianji_rail_env import (
    ObservationNotReady,
    PROJECT_ROOT,
    load_deployment_config,
)
from tests.test_recorded_episode import write_dataset


def observation(value):
    return {
        **{f"camera_{i}": np.full((2, 4, 5, 3), value + i, np.uint8) for i in range(3)},
        "robot_eef_pose": np.full((2, 12), value / 1000, np.float32),
        "hand_joint": np.full((2, 40), value / 100, np.float32),
    }


def write_full_dataset(path):
    frames = 1549
    root = zarr.open_group(str(path), mode="w")
    root.attrs["action_space"] = "eef"
    data, meta = root.create_group("data"), root.create_group("meta")
    meta.array("episode_ends", np.asarray([frames], np.int64))
    meta.array("segment_ends", np.asarray([frames], np.int64))
    data.array("timestamp", np.arange(frames, dtype=np.float64) / 30)
    for key, width in (("action", 52), ("robot_eef_pose", 12), ("hand_joint", 40)):
        data.array(key, np.zeros((frames, width), np.float32))
    pixels = (np.arange(frames) % 250).astype(np.uint8)[:, None, None, None] + np.zeros(
        (1, 4, 5, 3), np.uint8
    )
    for i in range(3):
        data.array(f"camera_{i}", pixels, chunks=(1, 4, 5, 3))


def load_batch_adapter():
    spec = importlib.util.spec_from_file_location(
        "mixed_batch_adapter", PROJECT_ROOT / "scripts/run_tianji_final_batch_slot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MixedObservationTests(unittest.TestCase):
    def test_all_three_cameras_and_both_state_fields_follow_the_selected_source(self):
        demo, live = observation(10), observation(80)
        for mode in OBSERVATION_MODES:
            with self.subTest(mode=mode):
                obs = compose_observation(demo, live, mode)
                sources = observation_sources(mode)
                for key in obs:
                    source = sources["images"] if key.startswith("camera_") else sources[key]
                    expected = live if source == "live" else demo
                    np.testing.assert_array_equal(obs[key], expected[key])
                    self.assertFalse(np.shares_memory(obs[key], expected[key]))
        copied = compose_observation(demo, live, "dataset-images-live-state")
        live["hand_joint"].fill(99)
        demo["camera_0"].fill(99)
        self.assertEqual(copied["camera_0"][0, 0, 0, 0], 10)
        self.assertAlmostEqual(float(copied["hand_joint"][0, 0]), 0.8)

    def test_missing_live_window_and_mismatched_history_are_rejected(self):
        for mode in OBSERVATION_MODES[1:]:
            with self.assertRaisesRegex(ValueError, "valid live"):
                compose_observation(observation(1), None, mode)
        live = observation(2)
        live["hand_joint"] = live["hand_joint"][:1]
        with self.assertRaisesRegex(ValueError, "hand_joint"):
            compose_observation(observation(1), live, "dataset-images-live-state")
        live = observation(2)
        live["robot_eef_pose"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "robot_eef_pose"):
            compose_observation(observation(1), live, "dataset-images-live-state")
        with self.assertRaisesRegex(ValueError, "Unknown"):
            compose_observation(observation(1), observation(2), "typo")

    def test_live_alignment_retries_but_operator_stop_and_deadline_interrupt(self):
        env = SimpleNamespace(runtime=SimpleNamespace(fault=None), error=None)
        env.get_obs = unittest.mock.Mock(
            side_effect=[ObservationNotReady("waiting"), (observation(80), 123, {})]
        )
        with patch.object(replay, "_read_command", return_value=None), patch.object(
            replay.time, "sleep"
        ):
            captured = replay._wait_replay_observation(env)
        self.assertEqual(env.get_obs.call_count, 2)
        self.assertEqual(captured["anchor"], 123)
        with patch.object(replay, "_read_command", return_value="p"):
            with self.assertRaises(replay.ReplayStopped):
                replay._wait_replay_observation(env)
        with patch.object(replay, "_read_command", return_value=None):
            with self.assertRaises(replay.ReplayTimedOut):
                replay._wait_replay_observation(env, deadline=0)
            with self.assertRaises(TimeoutError):
                replay._wait_replay_observation(env, timeout_s=0)
        self.assertEqual(env.get_obs.call_count, 2)

    def test_deadline_prevents_submission_and_interrupts_an_active_chunk(self):
        env = SimpleNamespace(
            exec_actions=unittest.mock.Mock(),
            chunk_completed_ns=lambda: None,
            runtime=SimpleNamespace(active=True, fault=None),
            error=None,
        )
        with self.assertRaises(replay.ReplayTimedOut):
            replay._submit_path(env, np.zeros((8, 52)), 8, deadline=0)
        env.exec_actions.assert_not_called()
        with patch.object(replay.time, "monotonic", side_effect=[1.0, 3.0]):
            with self.assertRaises(replay.ReplayTimedOut):
                replay._submit_path(env, np.zeros((8, 52)), 8, deadline=2.0)
        self.assertEqual(env.exec_actions.call_count, 1)

    def test_manifest_has_two_paired_conditions_and_keeps_existing_batch(self):
        batch = load_batch_adapter()
        manifest = PROJECT_ROOT / "diffusion_policy/config/tianji_rail_mixed_obs_batch.json"
        payload = json.loads(manifest.read_text())
        with patch.object(
            batch,
            "read_json",
            side_effect=lambda path: payload if Path(path) == manifest else {"frozen": True},
        ):
            plan, root, frozen = batch.load_batch(manifest)
        self.assertEqual([slot["slot_id"] for slot in plan["slots"]], ["M01", "M02"])
        self.assertEqual(
            [batch.slot_observation_mode(slot) for slot in plan["slots"]],
            list(OBSERVATION_MODES[1:]),
        )
        self.assertEqual({slot["seed"] for slot in plan["slots"]}, {42})
        self.assertEqual(plan["execution_scope"], "full_segment")
        self.assertIsNone(plan["limits"]["max_decisions"])
        self.assertIsNone(plan["limits"]["active_duration_s"])
        self.assertFalse(plan["comparison_rules"]["return_to_initial_pose"])
        self.assertFalse(plan["comparison_rules"]["initialization_check_required"])
        # Resolve the common freeze directory without requiring ignored local data.
        self.assertEqual(root.name, "tianji_rail_final_batch_20261018")
        self.assertTrue(all(item["frozen"] for item in frozen.values()))

    def test_batch_calibration_does_not_install_initialization_callbacks(self):
        batch = load_batch_adapter()
        manifest = json.loads(
            (
                PROJECT_ROOT / "diffusion_policy/config/tianji_rail_mixed_obs_batch.json"
            ).read_text()
        )
        for item in manifest["slots"]:
            with self.subTest(slot=item["slot_id"]), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                args = SimpleNamespace(
                    checkpoint=output / "model.ckpt", dataset=output / "demo.zarr", device="cpu"
                )
                with patch(
                    "diffusion_policy.real_world.tianji.recorded_episode.open_recorded_segment"
                ), patch.object(
                    replay, "run_motion", return_value=0
                ) as motion, redirect_stdout(
                    io.StringIO()
                ):
                    self.assertEqual(
                        batch.run_calibration(
                            args, manifest, output, {}, item, output, {"device": "cpu"}
                        ),
                        0,
                    )
                namespace = motion.call_args.args[0]
                self.assertFalse(hasattr(namespace, "prepare_hands"))
                self.assertFalse(hasattr(namespace, "initialization_check"))
                self.assertTrue(namespace.skip_approach)
                self.assertTrue(namespace.full_segment)
                self.assertIsNone(namespace.max_windows)
                self.assertIsNone(namespace.active_limit_s)

    def test_closed_loop_starts_without_robot_repositioning_or_frozen_pose_gate(self):
        batch = load_batch_adapter()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = SimpleNamespace(
                device="cpu", checkpoint=output / "model.ckpt", config=output / "limits.yaml"
            )
            plan = {"limits": {"max_decisions": 24}, "checkpoint_sha256": "fake"}
            frozen = {"stability_gate.json": {"parameters": {}}}
            env = SimpleNamespace(
                start=unittest.mock.Mock(),
                enable_trace=unittest.mock.Mock(),
                close=unittest.mock.Mock(),
            )
            inference = SimpleNamespace(
                preflight={"n_obs_steps": 2, "n_action_steps": 8}, close=unittest.mock.Mock()
            )
            runner = SimpleNamespace(run=unittest.mock.Mock())
            with patch(
                "diffusion_policy.real_world.tianji.inference.PolicyInferenceProcess",
                return_value=inference,
            ), patch(
                "diffusion_policy.real_world.tianji.env.TianjiRailEnv", return_value=env
            ), patch(
                "diffusion_policy.real_world.tianji.trace.SessionTrace"
            ), patch(
                "diffusion_policy.real_world.tianji.trace.write_provenance"
            ), patch.object(
                batch, "provenance_files", return_value={}
            ), patch.object(
                batch, "StabilityGate"
            ), patch(
                "diffusion_policy.env_runner.tianji_rail_real_runner.TianjiRailRealRunner",
                return_value=runner,
            ) as make_runner, redirect_stdout(
                io.StringIO()
            ):
                self.assertEqual(
                    batch.run_closed_loop(
                        args, plan, output, frozen, {"seed": 42}, output, {"device": "cpu"}
                    ),
                    0,
                )
            runner.run.assert_called_once_with()
            self.assertIsNone(runner.batch_hooks.check_init)
            self.assertFalse(make_runner.call_args.args[3]["initialization_check_required"])
            env.close.assert_called_once_with()
            inference.close.assert_called_once_with()

    def test_slot_reports_unchecked_initialization_without_marking_it_as_failure(self):
        batch = load_batch_adapter()
        plan = json.loads(
            (
                PROJECT_ROOT / "diffusion_policy/config/tianji_rail_mixed_obs_batch.json"
            ).read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ("initialization.json", "visual_candidate.json", "stability_gate.json"):
                (output / name).write_text("{}")
            with patch(
                "diffusion_policy.real_world.tianji.trace.write_provenance"
            ), patch.object(batch, "provenance_files", return_value={}):
                payload = batch.write_slot(
                    output,
                    plan["slots"][0],
                    plan,
                    {},
                    output,
                    {"outcome": "success", "trace_valid": True},
                    None,
                )
            self.assertEqual(payload["initialization_mode"], "current_pose")
            self.assertIsNone(payload["initial_state_comparable"])
            self.assertFalse(payload["initialization_check_required"])
            self.assertEqual(payload["outcome"], "success")

    def test_motion_loop_refreshes_live_input_and_logs_the_actual_mixed_input(self):
        cases = [
            (mode, full, False)
            for mode, full in itertools.product(OBSERVATION_MODES[1:], (False, True))
        ]
        cases.extend((mode, True, True) for mode in OBSERVATION_MODES[1:])
        for mode, full_segment, early_stop in cases:
            with self.subTest(
                mode=mode, full_segment=full_segment, early_stop=early_stop
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (write_full_dataset if full_segment else write_dataset)(root / "demo.zarr")
                segment = open_recorded_segment(root / "demo.zarr")
                output = root / "output"
                output.mkdir()
                args = SimpleNamespace(
                    checkpoint=root / "model.ckpt",
                    batch_seed=42,
                    skip_approach=False,
                    trace=True,
                    trace_capacity=2,
                    max_windows=None if full_segment else 2,
                    active_limit_s=None if full_segment else 10.0,
                    observation_mode=mode,
                    full_segment=full_segment,
                    prepare_hands=unittest.mock.Mock(
                        side_effect=AssertionError("must not reposition hands")
                    ),
                    initialization_check=unittest.mock.Mock(
                        return_value={"comparable": False, "deltas": {}}
                    ),
                )
                config = load_deployment_config(
                    PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval_loose.yaml"
                )
                config["inference_mode"] = "sync"
                inferred, events, lifecycle = [], [], []

                class Inference:
                    preflight = {"n_obs_steps": 2, "n_action_steps": 8}
                    last_timing = None

                    def reset(self):
                        pass

                    def reseed(self, seed):
                        lifecycle.append(("seed", seed, len(inferred)))

                    def predict_observation(self, obs):
                        inferred.append({key: value.copy() for key, value in obs.items()})
                        return {
                            "action": torch.from_numpy(
                                eef_action_to_6d(np.zeros((1, 8, 52), np.float32))
                            )
                        }

                    def close(self):
                        lifecycle.append("inference_closed")

                class Env:
                    runtime = SimpleNamespace(reference_action=np.zeros(52), fault=None)
                    error = None
                    recorder = None
                    count = 0

                    def enable_trace(self, trace):
                        self.trace = trace

                    def start(self):
                        pass

                    def get_obs(self):
                        self.count += 1
                        return (
                            observation(50 + (self.count % 10) * 10),
                            self.count,
                            {"query_ns": [self.count - 1, self.count]},
                        )

                    def engage(self):
                        lifecycle.append("engaged")

                    def emit(self, event, **values):
                        events.append(values)
                        self.recorder.event(event, **values)

                    def pause(self, reason):
                        lifecycle.append("paused")

                    def close(self):
                        lifecycle.append("env_closed")

                env = Env()
                original_trace = replay.SessionTrace
                clock = itertools.count().__next__ if full_segment else replay.time.monotonic

                def trace_factory(**kwargs):
                    # No servo/driver producers in this test; retain real decision
                    # buffers, file mapping and export while keeping numeric buffers tiny.
                    kwargs.update(feedback_capacity=4, command_capacity=4, hand_capacity=4)
                    return original_trace(**kwargs)

                with patch.object(
                    replay, "PolicyInferenceProcess", return_value=Inference()
                ), patch.object(replay, "TianjiRailEnv", return_value=env), patch.object(
                    replay, "_checkpoint_hash", return_value="fake"
                ), patch.object(
                    replay, "write_provenance"
                ), patch.object(
                    replay, "runtime_files", return_value={}
                ), patch.object(
                    replay, "_wait_live_pose", return_value=np.zeros(52)
                ), patch.object(
                    replay, "_wait_for_start", return_value=True
                ), patch.object(
                    replay, "_read_command", return_value=None
                ), patch.object(
                    replay, "SessionTrace", side_effect=trace_factory
                ), patch.object(
                    replay.time, "monotonic", side_effect=clock
                ), patch.object(
                    replay,
                    "_submit_path",
                    side_effect=[None] * 25 + ["p"] if early_stop else None,
                    return_value=None,
                ) as submit, redirect_stdout(
                    io.StringIO()
                ):
                    self.assertEqual(replay.run_motion(args, segment, config, "cpu", output), 0)
                expected_windows = 26 if early_stop else 194 if full_segment else 2
                self.assertEqual(
                    len(inferred), expected_windows + 1
                )  # One preview before formal decisions.
                args.prepare_hands.assert_not_called()
                args.initialization_check.assert_not_called()
                self.assertIn("engaged", lifecycle)
                self.assertIn(("seed", 42, 1), lifecycle)
                self.assertEqual(
                    submit.call_count, expected_windows
                )  # No approach, even though skip_approach=False.
                self.assertEqual(
                    [item["window_start"] for item in events],
                    list(range(0, expected_windows * 8, 8)),
                )
                if full_segment:
                    self.assertEqual(len(submit.call_args.args[1]), 8 if early_stop else 4)
                    self.assertIsNone(submit.call_args.kwargs["deadline"])
                    self.assertFalse((output / "trace_image_cache").exists())
                expected0 = compose_observation(
                    segment.observation(0, 2), observation(80), mode
                )
                expected1 = compose_observation(
                    segment.observation(8, 2), observation(90), mode
                )
                for key in expected0:
                    np.testing.assert_array_equal(inferred[1][key], expected0[key])
                    np.testing.assert_array_equal(inferred[2][key], expected1[key])
                trial = json.loads((output / "trial_0001/trial.json").read_text())
                self.assertEqual(trial["observation_sources"], observation_sources(mode))
                self.assertEqual(trial["initialization_mode"], "current_pose")
                self.assertFalse(trial["initialization_check_required"])
                self.assertEqual(trial["completed_windows"], expected_windows - int(early_stop))
                self.assertEqual(
                    trial["full_segment_completed"], full_segment and not early_stop
                )
                if full_segment:
                    self.assertEqual(
                        trial["completed_action_steps"], 200 if early_stop else 1548
                    )
                    self.assertEqual(
                        trial["outcome"], "takeover" if early_stop else "playback_complete"
                    )
                meta = json.loads((output / "trial_0001/trace_meta.json").read_text())
                self.assertEqual(meta["decisions"][0]["control_input"], mode)
                self.assertEqual(meta["decisions"][1]["real_alignment"], {"query_ns": [3, 4]})
                with np.load(output / "trial_0001/trace_decisions.npz") as recorded:
                    np.testing.assert_array_equal(
                        recorded["camera_0"][0], expected0["camera_0"]
                    )
                    np.testing.assert_array_equal(
                        recorded["robot_eef_pose"][0], expected0["robot_eef_pose"]
                    )
                    np.testing.assert_array_equal(
                        recorded["real_camera_0"][0], observation(80)["camera_0"]
                    )
                    if full_segment:
                        self.assertEqual(
                            recorded["action_rows"].tolist(),
                            [8] * 26 if early_stop else [8] * 193 + [4],
                        )
                        np.testing.assert_array_equal(
                            recorded["window_start"], np.arange(expected_windows) * 8
                        )
                        self.assertEqual(meta["counts"]["decisions"], expected_windows)
                        self.assertTrue(meta["valid"])
                self.assertEqual(lifecycle[-3:], ["paused", "env_closed", "inference_closed"])


if __name__ == "__main__":
    unittest.main()
