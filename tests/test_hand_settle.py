"""核对前摆手：插值到冻结起点，并一直保持到运行时接管。"""

import importlib.util
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_NAMES
from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile, JointTarget, Submission
from diffusion_policy.real_world.hand_settle import HandSettleCancelled, settle_and_hold
from diffusion_policy.real_world.tianji_rail_env import (
    PROJECT_ROOT, TianjiRailEnv, load_deployment_config,
)


def _load_batch():
    spec = importlib.util.spec_from_file_location(
        "run_tianji_final_batch_slot", ROOT / "scripts/run_tianji_final_batch_slot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


class FakeHand:
    def __init__(self, side, position, clock, *, track=True):
        self.side, self.clock, self.track = side, clock, track
        self.position = tuple(position)
        self.timeout_ns = 10 ** 12
        self.enabled = False
        self.profile = None
        self.last_target = None
        self.device_id = f"hand-{side}"
        self.commands = []

    def health(self, check_latch=True):
        return SimpleNamespace(ready=True, detail="")

    def configure(self, profile):
        self.profile = profile

    def engage(self):
        self.enabled = True
        self.last_target = JointTarget(JOINT_NAMES, self.position)

    def get_latest(self):
        return SimpleNamespace(
            header=SimpleNamespace(valid=True, received_monotonic_ns=self.clock(), ref="sample"),
            payload=SimpleNamespace(position_rad=self.position))

    def submit(self, command):
        self.commands.append(command.payload.position_rad)
        if self.track:
            self.position = command.payload.position_rad
            self.last_target = command.payload
        return Submission(command.command_id, True)


def _env(clock, start, *, track=True):
    hands = {side: FakeHand(side, start, clock, track=track) for side in ("left", "right")}
    return SimpleNamespace(
        hands=hands, error=None, _hand_hold=None, config={"hand_hz": 120},
        hand_profile=ControlProfile("settle", "mit", {}),
        runtime=SimpleNamespace(_device_locks={side: threading.RLock() for side in ("left", "right")}))


def _stop(env):
    hold = env._hand_hold
    if hold is not None:
        hold.stop()


class HandSettleTests(unittest.TestCase):
    def test_blend_starts_at_measurement_and_hold_keeps_the_goal(self):
        clock = Clock()
        start, goal = (0.1,) * 20, (0.4,) * 20
        env = _env(clock, start)
        try:
            worst = settle_and_hold(env, {"left": goal, "right": goal}, math.radians(3),
                                    lambda _timeout: None, duration_s=0.05, settle_timeout_s=0.05,
                                    clock_ns=clock, sleep=clock.sleep)
        finally:
            _stop(env)
        self.assertLessEqual(worst, math.radians(3))
        left = env.hands["left"].commands
        self.assertEqual(left[0], start)
        self.assertEqual(left[-1], goal)
        self.assertGreater(len(left), 2)
        self.assertTrue(env.hands["left"].enabled)
        self.assertIsNotNone(env._hand_hold)

    def test_already_close_hands_are_held_without_a_blend(self):
        clock = Clock()
        goal = (0.2,) * 20
        env = _env(clock, goal)
        try:
            settle_and_hold(env, {"left": goal, "right": goal}, math.radians(3),
                            lambda _timeout: None, clock_ns=clock, sleep=clock.sleep)
        finally:
            _stop(env)
        self.assertTrue(env.hands["left"].commands)
        self.assertTrue(all(command == goal for command in env.hands["left"].commands))

    def test_cancel_before_motion_does_not_start_a_hold(self):
        clock = Clock()
        env = _env(clock, (0.1,) * 20)
        with self.assertRaises(HandSettleCancelled):
            settle_and_hold(env, {"left": (0.4,) * 20, "right": (0.4,) * 20}, math.radians(3),
                            lambda _timeout: "q", duration_s=0.05, settle_timeout_s=0.05,
                            clock_ns=clock, sleep=clock.sleep)
        self.assertIsNone(env._hand_hold)
        self.assertEqual(env.hands["left"].commands, [])

    def test_timeout_reports_the_remaining_error(self):
        clock = Clock()
        env = _env(clock, (0.1,) * 20, track=False)
        with self.assertRaisesRegex(RuntimeError, "未在容差内到位"):
            settle_and_hold(env, {"left": (0.4,) * 20, "right": (0.4,) * 20}, math.radians(3),
                            lambda _timeout: None, duration_s=0.02, settle_timeout_s=0.02,
                            clock_ns=clock, sleep=clock.sleep)
        self.assertIsNone(env._hand_hold)

    def test_goal_outside_limits_is_rejected_before_enable(self):
        clock = Clock()
        env = _env(clock, (0.1,) * 20)
        with self.assertRaisesRegex(ValueError, "超出关节限位"):
            settle_and_hold(env, {"left": (10.,) * 20, "right": (0.1,) * 20}, math.radians(3),
                            lambda _timeout: None, clock_ns=clock, sleep=clock.sleep)
        self.assertFalse(env.hands["left"].enabled)

    def test_queued_enter_does_not_confirm_hand_motion(self):
        batch = _load_batch()

        class Keys:
            def __init__(self, values):
                self.values = list(values)

            def read(self, _timeout):
                if not self.values:
                    return None
                return self.values.pop(0)

        self.assertFalse(batch.confirm_hand_settle(Keys(["q"])))
        self.assertTrue(batch.confirm_hand_settle(Keys(["\n", None, "\n"])))

    def test_engage_stops_the_hold_before_the_runtime_commands(self):
        order = []
        module = "diffusion_policy.real_world.tianji_rail_env"
        cfg = load_deployment_config(PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval.yaml")
        cfg["workspace"] = {side: {"min": [-1., -1., -1.], "max": [1., 1., 1.]} for side in ("left", "right")}
        with patch(f"{module}.TianjiKinematics") as kinematics, patch(f"{module}.TianjiDriver"), \
                patch(f"{module}.WujiSdkSession"), patch(f"{module}.WujiHandDriver"), \
                patch(f"{module}.CameraRig"):
            kinematics.return_value.model.digest = cfg["expected_model_sha256"]
            env = TianjiRailEnv(cfg, 2, motion=True)
        pose = SimpleNamespace(position_m=(0.1, 0.2, 0.3), orientation_xyzw=(0., 0., 0., 1.))
        env.driver.profile = SimpleNamespace(profile_id=env.arm_profile.profile_id)
        env.arm = SimpleNamespace(
            engagement_poses={"left": pose, "right": pose},
            engage=lambda: order.append("arm"),
            configure=lambda profile: order.append("configure"))
        for hand in env.hands.values():
            hand.profile = SimpleNamespace(profile_id="hand")
            hand.enabled = True
            hand.last_target = JointTarget(JOINT_NAMES, (0.2,) * 20)
            hand.engage = lambda: order.append("hand")

        class Hold:
            error = None

            def stop(self):
                order.append("stop")

        env._hand_hold = Hold()
        env.get_obs = lambda: (order.append("obs") or {
            "robot_eef_pose": np.zeros((2, 12)), "hand_joint": np.zeros((2, 40))}, 0, {})
        seed = env.runtime.seed

        def record_seed(action, arm_ready=False):
            order.append(f"seed-{arm_ready}")
            return seed(action, arm_ready=arm_ready)

        env.runtime.seed = record_seed
        try:
            env.engage()
        finally:
            env.close()
        self.assertEqual(order, ["obs", "stop", "seed-False", "hand", "hand", "arm", "seed-True"])
        self.assertIsNone(env._hand_hold)


if __name__ == "__main__":
    unittest.main()
