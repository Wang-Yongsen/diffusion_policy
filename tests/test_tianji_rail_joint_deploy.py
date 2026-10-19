"""Joint-space deployment contract, joint following limits and config overrides, without hardware."""

import math
from types import SimpleNamespace
import unittest

import numpy as np
from omegaconf import OmegaConf

from diffusion_policy.real_world.tianji_joint_executor import (
    PINNED_WRIST, JointArmTarget, JointFollower, TianjiJointExecutor, joints_within_limits, model_joint_limits,
)
from diffusion_policy.real_world.tianji_policy_runtime import TianjiPolicyRuntime, split_action, validate_actions
from diffusion_policy.real_world.tianji_rail_env import (
    PROJECT_ROOT, align_observation, load_deployment_config, preprocess_observation,
)
from diffusion_policy.third_party.bimanual_teleop.types import Submission
from tests.test_tianji_rail_real_eval import FakeArm, FakeHand, config, histories

CONFIG = PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval.yaml"
# M6 rows: positive deg, negative deg, velocity deg/s, acceleration deg/s^2.
LIMITS = ((170, -170, 180, 450), (120, -120, 180, 450), (170, -170, 180, 900), (60, -145, 180, 900),
          (170, -170, 180, 900), (60, -60, 180, 900), (90, -90, 180, 900))
ARM = SimpleNamespace(limits_native=LIMITS, bd67_native=PINNED_WRIST,
                      lower_rad=tuple(math.radians(row[1]) for row in LIMITS),
                      upper_rad=tuple(math.radians(row[0]) for row in LIMITS),
                      max_joint_velocity_rad_s=tuple(math.radians(row[2]) for row in LIMITS))
MODEL = SimpleNamespace(arm=lambda side: ARM)
SIDES = ("left", "right")


def joint_config():
    cfg = config()
    cfg["action_space"] = "joint"
    return cfg


class FakeJointDriver:
    def __init__(self, q):
        arms = {side: SimpleNamespace(velocity_ratio=15, acceleration_ratio=15) for side in SIDES}
        self.profile = SimpleNamespace(profile_id="arm", active_arms=SIDES, arms=arms, model=MODEL)
        self.engaged, self.engagement_sample, self.commands, self.holds = False, None, [], []
        self.sample = SimpleNamespace(header=SimpleNamespace(ref="ref"), payload=SimpleNamespace(arms={
            side: SimpleNamespace(joints=SimpleNamespace(position_rad=tuple(q))) for side in SIDES}))

    def get_latest(self):
        return self.sample

    def health(self):
        return SimpleNamespace(ready=True)

    def engage(self):
        self.engaged, self.engagement_sample = True, self.sample

    def submit(self, command):
        self.commands.append(command)
        return Submission(command.command_id, True)

    def request_hold(self, reason):
        self.engaged = False
        self.holds.append(reason)


class JointDeploymentTests(unittest.TestCase):
    def test_overrides_replace_the_former_loose_config_and_reject_unknown_keys(self):
        cfg = load_deployment_config(CONFIG, ["policy_hz=30", "replanning_steps=2", "workspace.left.min=[-1,-1,-1]"])
        self.assertEqual((cfg["policy_hz"], cfg["replanning_steps"]), (30, 2))
        self.assertEqual(cfg["workspace"]["left"]["min"], [-1, -1, -1])
        self.assertEqual((cfg["max_arm_joint_jump_rad"], cfg["max_arm_joint_speed_rad_s"]), (0.6, 2.0))
        for bad, message in (("polcy_hz=30", "Unknown"), ("workspace.top.min=[0,0,0]", "Unknown"),
                             ("policy_hz", "KEY=VALUE"), ("policy_hz=20", "policy_hz in")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, message):
                load_deployment_config(CONFIG, [bad])

    def test_joint_action_layout_limits_jump_and_speed(self):
        action = np.r_[np.arange(14) / 100, np.full(20, .01), np.full(20, .02)]
        arms, hands = split_action(action, "joint")
        self.assertEqual(arms["right"], tuple(np.arange(7, 14) / 100))
        self.assertEqual(hands["left"].position_rad, (.01,) * 20)
        self.assertEqual(hands["right"].position_rad, (.02,) * 20)
        with self.assertRaisesRegex(ValueError, "54-D"):
            split_action(np.zeros(52), "joint")
        cfg, limits = joint_config(), {side: model_joint_limits(MODEL, side) for side in SIDES}
        stamps = np.array([0, 100_000_000])
        validate_actions(np.zeros((2, 54)), stamps, np.zeros(54), cfg, limits)
        cases = [({(0, 3): math.radians(61)}, "outside M6"),  # J4 upper limit is 60 deg
                 # Each wrist joint is inside its own range; only 1.025*|J6| + |J7| exceeds 110.5 deg.
                 ({(0, 5): math.radians(50), (0, 6): math.radians(60)}, "outside M6"),
                 ({(0, 7): .7}, "jumps too far"),
                 ({(1, 0): .3}, "changes too fast")]  # 0.3 rad in 0.1 s
        for values, message in cases:
            actions = np.zeros((2, 54))
            for index, value in values.items():
                actions[index] = value
            with self.subTest(message=message, values=values), self.assertRaisesRegex(ValueError, message):
                validate_actions(actions, stamps, np.zeros(54), cfg, limits)
        with self.assertRaisesRegex(ValueError, "requires 54-D"):
            validate_actions(np.zeros((2, 52)), stamps, np.zeros(52), cfg, limits)
        hands_out = np.zeros((2, 54))
        hands_out[:, 14] = 10.
        with self.assertRaisesRegex(ValueError, "Hand2"):
            validate_actions(hands_out, stamps, np.zeros(54), cfg, limits)

    def test_runtime_routes_joint_goals_and_hand_offsets(self):
        now = [1_000_000_000]
        arm, hands = FakeArm(), {side: FakeHand(side) for side in SIDES}
        runtime = TianjiPolicyRuntime(arm, hands, joint_config(), lambda: None, lambda *a, **k: None,
                                      clock_ns=lambda: now[0], joint_limits={s: model_joint_limits(MODEL, s) for s in SIDES})
        self.addCleanup(runtime.close)
        runtime.seed(np.zeros(54), arm_ready=True)
        actions = np.zeros((8, 54))
        actions[:, 0], actions[:, 34:] = .01, .02
        runtime.submit(actions, now[0])
        now[0] += 40_000_000
        runtime.tick("arm")
        runtime.tick("hand")
        target = arm.commands[-1]
        self.assertIsInstance(target, JointArmTarget)
        self.assertAlmostEqual(target.joints["left"][0], .01)
        self.assertEqual(len(target.joints["right"]), 7)
        self.assertEqual(hands["right"].commands[-1].payload.position_rad, (.02,) * 20)
        self.assertEqual(hands["left"].commands[-1].payload.position_rad, (0.,) * 20)
        with self.assertRaisesRegex(ValueError, "requires 54-D"):
            runtime.submit(np.zeros((8, 52)), now[0])

    def test_follower_respects_speed_acceleration_goal_and_limits(self):
        follower = JointFollower(MODEL, "left", velocity_ratio=15, acceleration_ratio=15)
        follower.reset(np.zeros(7))
        goal, dt = np.r_[.2, np.zeros(6)], .005
        previous_v = np.zeros(7)
        for _ in range(int(3 / dt)):
            q, v = follower.propose(goal, dt)
            self.assertTrue(np.all(np.abs(v) <= follower.vmax + 1e-12))
            self.assertTrue(np.all(np.abs(v - previous_v) <= follower.amax * dt + 1e-12))
            self.assertLessEqual(q[0], goal[0] + 1e-5)
            follower.accept(q, v)
            previous_v = v
        self.assertLess(abs(follower.q[0] - .2), 1e-3)
        with self.assertRaisesRegex(ValueError, "outside"):
            follower.propose(np.r_[0, 0, 0, math.radians(61), 0, 0, 0], dt)
        # Moving toward the coupled wrist boundary: the step is shortened, not sent outside.
        follower.reset(np.radians([0, 0, 0, 0, 0, 53, 56.1]))
        follower.v = follower.vmax * np.r_[0, 0, 0, 0, 0, 1, 1]
        q, _ = follower.propose(follower.q, dt)
        self.assertTrue(joints_within_limits(q, follower.lower, follower.upper))

    def test_executor_engages_from_measured_joints_and_submits_limited_steps(self):
        start = np.radians([10, -20, 5, -30, 0, 10, 0])
        driver = FakeJointDriver(start)
        executor = TianjiJointExecutor(driver, clock_ns=lambda: 2_000_000_000)
        executor.engage()
        self.assertEqual(executor.engagement_joints["left"], tuple(start))
        goal = start + np.r_[.3, np.zeros(6)]
        result = executor.submit(JointArmTarget("c1", {side: tuple(goal) for side in SIDES}, (),
                                                1_999_000_000, 2_050_000_000, "arm"))
        self.assertTrue(result.accepted)
        sent = np.asarray(driver.commands[-1].payload.targets["left"])
        self.assertGreater(sent[0], start[0])
        self.assertLessEqual(sent[0] - start[0], math.radians(180) * .15 * .005 + 1e-12)
        np.testing.assert_allclose(sent[1:], start[1:])
        bad = executor.submit(JointArmTarget("c2", {"left": tuple(goal)}, (), 1_999_000_000, 2_050_000_000, "arm"))
        self.assertFalse(bad.accepted)
        self.assertTrue(driver.holds)

    def test_observation_adds_robot_joint_and_preprocess_selects_checkpoint_keys(self):
        history = histories()
        for side, offset in (("left", 0.), ("right", 1.)):
            history[f"arm_joints/{side}"] = [(stamp, np.full(7, offset)) for stamp, _ in history["arms/left"]]
        obs, _, _ = align_observation(history, 2, config(), now_ns=1_050_000_000)
        self.assertEqual(obs["robot_joint"].shape, (2, 14))
        np.testing.assert_allclose(obs["robot_joint"][:, 7:], 1.)
        shape_meta = {"obs": {f"camera_{i}": {"shape": [3, 16, 16], "type": "rgb"} for i in range(3)}}
        shape_meta["obs"].update(robot_joint={"shape": [14]}, hand_joint={"shape": [40]})
        prepared = preprocess_observation(obs, shape_meta, {"resize_enabled": False})
        self.assertEqual(set(prepared), set(shape_meta["obs"]))
        del obs["robot_joint"]
        with self.assertRaisesRegex(ValueError, "not provided"):
            preprocess_observation(obs, shape_meta, {"resize_enabled": False})

    def test_entry_reads_action_space_and_preflights_joint_checkpoint(self):
        import torch
        import eval_tianji_rail as entry
        image = [3, 480, 640]
        cfg = OmegaConf.create({"task": {"shape_meta": {"obs": {
            **{f"camera_{i}": {"shape": image, "type": "rgb"} for i in range(3)},
            "robot_joint": {"shape": [14], "type": "low_dim"}, "hand_joint": {"shape": [40], "type": "low_dim"}},
            "action": {"shape": [54]}}, "dataset": {"resize_enabled": False, "action_space": "joint"}}})
        self.assertEqual(entry.checkpoint_action_space(cfg), "joint")
        self.assertEqual(entry.checkpoint_action_space(OmegaConf.create({"task": {"dataset": {}}})), "eef")
        policy = SimpleNamespace(n_obs_steps=2, n_action_steps=8, device="cpu", reset=lambda: None,
                                 predict_action=lambda obs: {"action": torch.zeros(1, 8, 54)})
        self.assertEqual(entry.check_inference(policy, cfg, iterations=1)["n_action_steps"], 8)


if __name__ == "__main__":
    unittest.main()
