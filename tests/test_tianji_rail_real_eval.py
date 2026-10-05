"""Timing, real-contract alignment, stop races and bounded recording, without hardware."""

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
from diffusion_policy.env_runner.tianji_rail_real_runner import TrialRecorder, TianjiRailRealRunner, summarize_trials


def config():
    result = load_deployment_config(PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval.yaml")
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

    def test_recording_actual_rgb_videos_and_fixed_motion_denominator(self):
        import av
        from diffusion_policy.third_party.bimanual_teleop.recording.sink import Record
        with tempfile.TemporaryDirectory() as root:
            recorder = TrialRecorder(Path(root) / "trial_0001", {"mode": "motion"})
            for i in range(3):
                for frame in range(4):
                    image = np.full((16, 16, 3), 40 + frame, np.uint8)
                    recorder.camera(f"camera_{i}", image, Record(f"camera_{i}", frame * 33_333_333, frame, {}))
            result = recorder.close("success")
            self.assertIsNone(result["reason"])
            for i in range(3):
                with av.open(str(Path(root) / f"trial_0001/camera_{i}.mp4")) as video:
                    self.assertEqual(len(list(video.decode(video=0))), 4)
            aborted = TrialRecorder(Path(root) / "trial_0002", {"mode": "motion"})
            aborted.close("device_abort", "fake disconnect")
            shadow = TrialRecorder(Path(root) / "trial_0003", {"mode": "shadow"})
            shadow.close("shadow")
            report = summarize_trials(root)
            self.assertEqual(report["motion_trials"], 2)
            self.assertEqual(report["success_rate"], .5)

    def test_video_writer_failure_is_not_reported_as_success(self):
        from diffusion_policy.third_party.bimanual_teleop.recording.sink import Record
        with tempfile.TemporaryDirectory() as root, patch("av.open", side_effect=OSError("disk failure")):
            recorder = TrialRecorder(Path(root) / "trial", {"mode": "motion"})
            recorder.camera("camera_0", np.zeros((16, 16, 3), np.uint8), Record("camera_0", 0, 0, {}))
            result = recorder.close("success")
            self.assertEqual(result["outcome"], "device_abort")
            self.assertIn("disk failure", result["reason"])

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
