"""Timing, real-contract alignment, stop races and bounded event logging, without hardware."""

import json
import importlib.util
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.third_party.bimanual_teleop.types import JointTarget, Submission
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_NAMES
from diffusion_policy.real_world.tianji_policy_runtime import (
    TianjiPolicyRuntime, future_actions, split_action, validate_actions,
)
from diffusion_policy.real_world.tianji_rail_env import (
    PROJECT_ROOT, TianjiRailEnv, ObservationNotReady, align_observation, interpolate_state,
    load_deployment_config, preprocess_observation,
)
from diffusion_policy.env_runner.tianji_rail_real_runner import TrialRecorder, TianjiRailRealRunner, summarize_trials, timing_report


def config():
    result = load_deployment_config(PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval.yaml")
    # Existing regression cases exercise historical 30 Hz queue boundaries.
    result["policy_hz"], result["replanning_steps"] = 30, 2
    result["workspace"] = {side: {"min": [-1., -1., -1.], "max": [1., 1., 1.]} for side in ("left", "right")}
    return result


def histories():
    result = {f"camera_{i}": [(1_000_000_000, np.full((16, 16, 3), i, np.uint8)),
                              (1_033_333_333, np.full((16, 16, 3), 10 + i, np.uint8))] for i in range(3)}
    for kind, size in (("arms", 6), ("hands", 20)):
        for side in ("left", "right"):
            result[f"{kind}/{side}"] = [(stamp, np.zeros(size)) for stamp in range(990_000_000, 1_050_000_001, 10_000_000)]
    return result


class FakeArm:
    def __init__(self):
        self.driver = SimpleNamespace(engaged=True, profile=SimpleNamespace(profile_id="arm"))
        self.commands = []
        self.stopped = threading.Event()

    def submit(self, target):
        self.commands.append(target)
        return Submission(target.command_id, True)

    def request_hold(self, reason):
        self.driver.engaged = False
        self.stopped.set()


class FakeHand:
    def __init__(self, side):
        self.enabled = True
        self.device_id = side
        self.profile = SimpleNamespace(profile_id="hand")
        self.last_target = JointTarget(JOINT_NAMES, (0.,) * 20)
        self.commands = []
        self.accepted = True

    def submit(self, target):
        self.commands.append(target)
        if self.accepted:
            self.last_target = target.payload
        return Submission(target.command_id, self.accepted, None if self.accepted else "fake reject")

    def request_hold(self, reason):
        pass

    def health(self, **kwargs):
        return SimpleNamespace(ready=True)


class RealEvaluationTests(unittest.TestCase):
    def test_10hz_observation_alignment_action_timing_and_expiry(self):
        cfg = load_deployment_config(PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval.yaml")
        self.assertEqual((cfg["policy_hz"], cfg["replanning_steps"]), (10, 1))
        history = histories()
        for key in ("camera_0", "camera_1", "camera_2"):
            history[key] = [(1_000_000_000 + i * 33_333_333, np.zeros((16, 16, 3), np.uint8))
                            for i in range(7)]
        for kind in ("arms", "hands"):
            for side in ("left", "right"):
                size = 6 if kind == "arms" else 20
                history[f"{kind}/{side}"] = [(t, np.zeros(size))
                    for t in range(990_000_000, 1_220_000_001, 10_000_000)]
        _, anchor, meta = align_observation(history, 2, cfg, now_ns=1_220_000_000)
        self.assertEqual(meta["query_ns"], [anchor - 100_000_000, anchor])
        actions, times, dropped = future_actions(np.zeros((8, 52)), anchor, anchor + 150_000_000, hz=10)
        self.assertEqual((len(actions), dropped), (6, 2))
        self.assertEqual(times.tolist(), [anchor + i * 100_000_000 for i in range(2, 8)])
        with self.assertRaisesRegex(RuntimeError, "no future"):
            future_actions(np.zeros((8, 52)), anchor, anchor + 690_000_000, hz=10)
        cfg["workspace"] = config()["workspace"]
        runtime = TianjiPolicyRuntime(FakeArm(), {side: FakeHand(side) for side in ("left", "right")},
                                    cfg, lambda: None, lambda *args, **kwargs: None, clock_ns=lambda: anchor)
        self.addCleanup(runtime.close)
        runtime.seed(np.zeros(52), arm_ready=True)
        runtime.submit(np.zeros((8, 52)), anchor)
        runtime._target(anchor + 799_999_999)
        with self.assertRaisesRegex(RuntimeError, "coverage expired"):
            runtime._target(anchor + 800_000_000)

    def test_checkpoint_frequency_mismatch_fails_before_inference_or_devices(self):
        import eval_tianji_rail as entry
        from omegaconf import OmegaConf
        current = OmegaConf.create({"task": {"sample_hz": 10, "dataset": {"sample_hz": 10}}})
        legacy = OmegaConf.create({"task": {"dataset": {}}})
        self.assertEqual(entry.validate_policy_frequency(current, 10), 10)
        self.assertEqual(entry.validate_policy_frequency(legacy, 30), 30)
        with self.assertRaisesRegex(ValueError, "Checkpoint sample_hz=30"):
            entry.validate_policy_frequency(legacy, 10)
        with self.assertRaisesRegex(ValueError, "does not match"):
            entry.validate_policy_frequency(current, 30)
        current.task.dataset.sample_hz = 30
        with self.assertRaisesRegex(ValueError, "dataset sample_hz"):
            entry.validate_policy_frequency(current, 10)
        with tempfile.TemporaryDirectory() as root, \
             patch.object(entry, "load_policy", return_value=(None, legacy, "ema_model")), \
             patch.object(entry, "check_inference") as infer, patch.object(entry, "TianjiRailEnv") as env:
            with self.assertRaisesRegex(ValueError, "Checkpoint sample_hz=30"):
                entry.main(["--checkpoint", "unused.ckpt", "--output", str(Path(root) / "session")])
            infer.assert_not_called()
            env.assert_not_called()

    def test_10hz_replanning_waits_for_100ms_of_new_observations(self):
        import torch
        cfg = config()
        cfg.update(policy_hz=10, replanning_steps=1)
        anchors = iter([1_000_000_000, 1_033_333_333, 1_066_666_667, 1_100_000_000])
        submitted = []
        policy = SimpleNamespace(reset=lambda: None, last_timing={
            "preprocessed_ns": 0, "inputs_ready_ns": 0},
            predict_observation=lambda obs: {"action": torch.zeros(1, 8, 52)})
        env = SimpleNamespace(config=cfg, motion=False, get_obs=lambda: ({}, next(anchors), {}),
                              emit=lambda *args, **kwargs: None,
                              runtime=SimpleNamespace(pause=lambda *args, **kwargs: None))
        with tempfile.TemporaryDirectory() as root:
            runner = TianjiRailRealRunner(env, policy, None, {}, Path(root) / "session")
            def submit(actions, anchor):
                submitted.append(anchor)
                if len(submitted) == 2:
                    runner._stop.set()
                return 1
            env.exec_actions = submit
            runner._enabled.set()
            runner._infer()
            self.assertIsNone(runner.error)
        self.assertEqual(submitted, [1_000_000_000, 1_100_000_000])

    def test_shadow_lifecycle_never_configures_enables_or_submits(self):
        module = "diffusion_policy.real_world.tianji_rail_env"
        with patch(f"{module}.TianjiKinematics") as kine_type, patch(f"{module}.TianjiDriver") as arm_type, \
             patch(f"{module}.WujiSdkSession"), patch(f"{module}.WujiHandDriver") as hand_type, \
             patch(f"{module}.CameraRig") as camera_type:
            arm_type.return_value.get_latest.return_value = None
            kine_type.return_value.model.digest = config()["expected_model_sha256"]
            hand_type.return_value.get_latest.return_value = None
            camera_type.return_value.poll.return_value = []
            env = TianjiRailEnv(config(), 2)
            try:
                env.start()
                with self.assertRaisesRegex(RuntimeError, "Shadow"):
                    env.engage()
            finally:
                env.close()
            for device in (arm_type.return_value, hand_type.return_value):
                device.start.assert_called_with()
                device.configure.assert_not_called()
                device.engage.assert_not_called()
                device.submit.assert_not_called()
            self.assertFalse(camera_type.call_args.args[0].main_depth)

    def test_model_mismatch_is_rejected_before_device_creation(self):
        module = "diffusion_policy.real_world.tianji_rail_env"
        with patch(f"{module}.TianjiKinematics") as kine_type, patch(f"{module}.TianjiDriver") as arm_type:
            kine_type.return_value.model.digest = "f" * 64
            with self.assertRaisesRegex(ValueError, "FK model differs"):
                TianjiRailEnv(config(), 2)
            arm_type.assert_not_called()

    def runtime(self):
        self.now = 1_000_000_000
        self.events = []
        arm = FakeArm()
        hands = {side: FakeHand(side) for side in ("left", "right")}
        runtime = TianjiPolicyRuntime(arm, hands, config(), lambda: None,
                                    lambda event, **values: self.events.append((event, values)), clock_ns=lambda: self.now)
        runtime.seed(np.zeros(52), arm_ready=True)
        self.addCleanup(runtime.close)
        return runtime

    def test_late_actions_keep_original_times_and_all_expired_fails(self):
        actions = np.zeros((8, 52))
        selected, stamps, dropped = future_actions(actions, 1_000_000_000, 1_070_000_000)
        self.assertEqual(dropped, 3)
        self.assertEqual(int(stamps[0]), 1_100_000_000)
        self.assertEqual(len(selected), 5)
        with self.assertRaisesRegex(RuntimeError, "no future"):
            future_actions(actions, 1_000_000_000, 1_300_000_000)

    def test_expiry_keeps_submission_margin_and_reports_observation_age(self):
        actions = np.zeros((8, 52))
        selected, stamps, dropped = future_actions(actions, 1_000_000_000, 1_223_333_332)
        self.assertEqual((len(selected), dropped), (1, 7))
        self.assertEqual(int(stamps[0]), 1_233_333_333)
        with self.assertRaisesRegex(RuntimeError, r"observation age=223.3 ms.*submit margin=10.0 ms"):
            future_actions(actions, 1_000_000_000, 1_223_333_333)

    def test_deployment_limits_blas_threads_before_loading_model(self):
        import eval_tianji_rail as entry
        from threadpoolctl import threadpool_info
        def blas_threads():
            return {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()
                    if pool["user_api"] == "blas"}
        previous = blas_threads()
        self.assertTrue(previous)
        def load(*args, **kwargs):
            self.assertTrue(all(count == 1 for count in blas_threads().values()))
            raise RuntimeError("offline model boundary")
        with tempfile.TemporaryDirectory() as root, patch.object(entry, "load_policy", side_effect=load), \
             patch.object(entry, "TianjiRailEnv") as env:
            with self.assertRaisesRegex(RuntimeError, "offline model boundary"):
                entry.main(["--checkpoint", "unused.ckpt", "--output", str(Path(root) / "session")])
            env.assert_not_called()
        self.assertEqual(blas_threads(), previous)

    def test_stage_timing_report_accepts_new_and_legacy_prediction_events(self):
        prediction = {"event": "prediction", "anchor_ns": 980_000_000,
                      "started_ns": 1_000_000_000, "observed_ns": 1_001_000_000,
                      "preprocessed_ns": 1_003_000_000, "inputs_ready_ns": 1_006_000_000,
                      "returned_ns": 1_070_000_000}
        legacy = {key: value for key, value in prediction.items()
                  if key not in ("observed_ns", "preprocessed_ns", "inputs_ready_ns")}
        with tempfile.TemporaryDirectory() as root:
            Path(root, "events.jsonl").write_text("\n".join(json.dumps(item) for item in (prediction, legacy)) + "\n")
            report = timing_report(root)
        for name, expected in (("observation_read_ms", 1), ("preprocess_ms", 2),
                               ("input_transfer_ms", 3), ("inference_output_transfer_ms", 64)):
            self.assertEqual(report[name]["count"], 1)
            self.assertEqual(report[name]["median"], expected)
        self.assertEqual(report["preprocess_inference_transfer_ms"]["count"], 2)
        self.assertEqual(report["preprocess_inference_transfer_ms"]["median"], 70)
        self.assertEqual(report["observation_age_at_processing_start_ms"]["median"], 20)

    def test_action_sides_frames_units_and_joint_names(self):
        action = np.zeros(52)
        action[:3] = [.1, .2, .3]
        action[6:9] = [-.1, -.2, -.3]
        action[5] = np.pi / 2
        action[12:32], action[32:] = .1, .2
        poses, hands = split_action(action)
        self.assertEqual(poses["left"].parent_frame, "tianji_left_base")
        np.testing.assert_allclose(Rotation.from_quat(poses["left"].orientation_xyzw).apply([1., 0, 0]), [0., 1., 0], atol=1e-6)
        self.assertEqual(hands["right"].joint_names, JOINT_NAMES)
        np.testing.assert_allclose(hands["right"].position_rad, .2)

    def test_rotation_interpolates_on_so3_and_never_extrapolates(self):
        values = [(0, np.r_[0., 0, 0, 0, 0, np.deg2rad(170)]),
                  (10, np.r_[2., 0, 0, 0, 0, np.deg2rad(-170)])]
        pose, _ = interpolate_state(values, np.array([5]), pose=True)
        self.assertAlmostEqual(float(pose[0, 0]), 1.)
        self.assertAlmostEqual(abs(float(pose[0, 5])), np.pi, places=5)
        with self.assertRaises(ObservationNotReady):
            interpolate_state(values, np.array([11]), pose=True)
        with self.assertRaisesRegex(ObservationNotReady, "too large"):
            interpolate_state(values, np.array([5]), pose=True, max_gap_ns=9)

    def test_three_cameras_and_both_states_have_common_covered_history(self):
        obs, anchor, meta = align_observation(histories(), 2, config(), now_ns=1_050_000_000)
        self.assertEqual(anchor, 1_033_333_333)
        self.assertEqual(obs["hand_joint"].shape, (2, 40))
        self.assertEqual(obs["robot_eef_pose"].shape, (2, 12))
        self.assertEqual(meta["query_ns"], [1_000_000_000, 1_033_333_333])
        bad = histories()
        bad["camera_2"] = [(stamp + 60_000_000, image) for stamp, image in bad["camera_2"]]
        with self.assertRaises(ObservationNotReady):
            align_observation(bad, 2, config(), now_ns=1_100_000_000)
        with self.assertRaisesRegex(ObservationNotReady, "too old"):
            align_observation(histories(), 2, config(), now_ns=1_300_000_000)

    def test_runtime_replans_future_queue_and_pause_clears_it(self):
        runtime = self.runtime()
        actions = np.zeros((8, 52))
        actions[:, 0] = np.arange(8) * .001
        self.assertEqual(runtime.submit(actions, self.now), 1)
        self.now += 40_000_000
        runtime.tick("arm")
        self.assertAlmostEqual(runtime.arm.commands[-1].tool_poses["left"].position_m[0], .001)
        replacement = np.zeros((8, 52))
        replacement[:, 0] = .005
        runtime.submit(replacement, self.now)
        self.now += 40_000_000
        runtime.tick("arm")
        self.assertAlmostEqual(runtime.arm.commands[-1].tool_poses["left"].position_m[0], .005)
        count = len(runtime.arm.commands)
        runtime.pause("operator")
        self.now += 50_000_000
        runtime.tick("arm")
        self.assertEqual(len(runtime.arm.commands), count)
        self.assertEqual(len(runtime._times), 0)

    def test_no_inference_progress_independently_stops_and_hands_hold(self):
        runtime = self.runtime()
        runtime.submit(np.zeros((8, 52)), self.now)
        runtime.start()
        self.now += 300_000_000
        self.assertTrue(runtime.arm.stopped.wait(1.))
        self.assertFalse(runtime.active)
        self.assertIn("expired", runtime.fault)
        previous = runtime.hands["left"].last_target
        runtime.tick("hand")
        self.assertEqual(runtime.hands["left"].last_target, previous)

    def test_partial_hand_rejection_stops_all_groups(self):
        runtime = self.runtime()
        runtime.submit(np.zeros((8, 52)), self.now)
        runtime.hands["right"].accepted = False
        runtime.start()
        self.assertTrue(runtime.arm.stopped.wait(1.))
        self.assertFalse(runtime.active)
        self.assertIn("rejected", runtime.fault)

    def test_workspace_joint_limits_jump_and_speed_fail_before_submission(self):
        cfg = config()
        actions = np.zeros((2, 52))
        stamps = np.array([100, 33_333_433])
        for index, value in ((0, 2.), (12, 3.)):
            bad = actions.copy()
            bad[0, index] = value
            with self.assertRaises(ValueError):
                validate_actions(bad, stamps, np.zeros(52), cfg)
        actions[1, 0] = .01
        with self.assertRaisesRegex(ValueError, "too fast"):
            validate_actions(actions, stamps, np.zeros(52), cfg)
        cfg["workspace"]["left"] = None
        with self.assertRaisesRegex(ValueError, "measured workspace"):
            validate_actions(np.zeros((2, 52)), stamps, np.zeros(52), cfg)

    def test_event_log_without_videos_and_fixed_motion_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = TrialRecorder(Path(root) / "trial_0001", {"mode": "motion"})
            recorder.event("submission", kind="arm", side=None, mode="policy", planned_ns=1_000_000_000,
                           started_ns=1_001_000_000, returned_ns=1_003_000_000)
            result = recorder.close("success")
            self.assertIsNone(result["reason"])
            self.assertEqual(result["dropped_events"], 0)
            trial = Path(root) / "trial_0001"
            self.assertEqual(sorted(path.name for path in trial.iterdir()), ["events.jsonl", "timing.json", "trial.json"])
            timing = json.loads((trial / "timing.json").read_text())
            self.assertEqual(timing["arm_submission_call_ms"]["median"], 2)
            self.assertEqual(timing["arm_target_age_at_submission_ms"]["median"], 1)
            aborted = TrialRecorder(Path(root) / "trial_0002", {"mode": "motion"})
            aborted.close("device_abort", "fake disconnect")
            shadow = TrialRecorder(Path(root) / "trial_0003", {"mode": "shadow"})
            self.assertEqual(shadow.close("shadow")["outcome"], "device_abort")
            report = summarize_trials(root)
            self.assertEqual(report["motion_trials"], 2)
            self.assertEqual(report["success_rate"], .5)

    def test_full_event_queue_drops_and_counts_without_raising(self):
        real_dumps, entered, release = json.dumps, threading.Event(), threading.Event()
        def blocking_dumps(*args, **kwargs):
            entered.set()
            release.wait(2)
            return real_dumps(*args, **kwargs)
        with tempfile.TemporaryDirectory() as root:
            recorder = TrialRecorder(Path(root) / "trial", {"mode": "motion"}, capacity=1)
            with patch("json.dumps", blocking_dumps):
                recorder.event("pause", reason="writing")
                self.assertTrue(entered.wait(1))
                recorder.event("pause", reason="queued")
                recorder.event("pause", reason="dropped")
                release.set()
            result = recorder.close("success")
            lines = (Path(root) / "trial/events.jsonl").read_text().splitlines()
        self.assertEqual((result["outcome"], result["dropped_events"], result["record_error"]), ("success", 1, None))
        self.assertEqual([json.loads(line)["reason"] for line in lines], ["writing", "queued"])

    def test_event_writer_failure_keeps_operator_outcome(self):
        def failing_dumps(*args, **kwargs):
            raise OSError("disk failure")
        with tempfile.TemporaryDirectory() as root:
            recorder = TrialRecorder(Path(root) / "trial", {"mode": "motion"})
            with patch("json.dumps", failing_dumps):
                recorder.event("pause", reason="operator")
                recorder._thread.join(1)
            recorder.event("pause", reason="after writer failure")
            result = recorder.close("success")
        self.assertEqual((result["status"], result["outcome"], result["dropped_events"]), ("complete", "success", 1))
        self.assertIn("disk failure", result["record_error"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires unified model environment")
    def test_stop_during_inference_discards_inflight_prediction(self):
        import torch
        entered, release = threading.Event(), threading.Event()
        submitted = []
        obs, anchor, alignment = align_observation(histories(), 2, config(), now_ns=1_050_000_000)
        def predict(_):
            entered.set()
            release.wait(2)
            return {"action": torch.zeros(1, 8, 52)}
        policy = SimpleNamespace(device=torch.device("cpu"), reset=lambda: None, predict_action=predict)
        shape_meta = {"obs": {key: {"shape": [3, 16, 16], "type": "rgb"} for key in ("camera_0", "camera_1", "camera_2")}}
        shape_meta["obs"].update(robot_eef_pose={"shape": [12]}, hand_joint={"shape": [40]})
        cfg = SimpleNamespace(task=SimpleNamespace(shape_meta=shape_meta, dataset={"resize_enabled": False}))
        env = SimpleNamespace(config=config(), motion=False, recorder=None,
            get_obs=lambda: (obs, anchor, alignment), pause=lambda reason: None,
            emit=lambda *args, **kwargs: None, exec_actions=lambda *args: submitted.append(args),
            runtime=SimpleNamespace(pause=lambda *args, **kwargs: None))
        with tempfile.TemporaryDirectory() as root:
            runner = TianjiRailRealRunner(env, policy, cfg, {}, Path(root) / "session")
            runner._begin()
            worker = threading.Thread(target=runner._infer)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                runner._end("shadow", "operator stop during inference")
                runner._stop.set()
            finally:
                release.set()
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(submitted, [])


if __name__ == "__main__":
    unittest.main()
