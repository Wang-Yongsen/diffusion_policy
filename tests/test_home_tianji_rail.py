"""回位目标：机械臂沿用 ready_pose，灵巧手使用示范起点。"""

import math
import unittest
from unittest.mock import Mock

from diffusion_policy.real_world.tianji import home
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import (
    load_config,
)
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import (
    JOINT_LIMITS_RAD,
    JOINT_NAMES,
)

READY_DEG = {
    "left": [30, -60, -34, -52, 30, 12, 4],
    "right": [-30, -60, 34, -52, -30, 12, -4],
}


class HomeTargetTests(unittest.TestCase):
    def test_arm_targets_are_the_configured_ready_pose(self):
        settings = load_config()
        source, order, targets = home.ready_joint_targets(settings, "both")
        self.assertEqual(order, ("left", "right"))
        for side, expected in READY_DEG.items():
            self.assertEqual(source["target_deg"][side], expected)
            self.assertEqual([round(math.degrees(q), 6) for q in targets[side]], expected)
        profile = home.ready_profile(settings, source, order)
        self.assertEqual(profile.parameters["active_arms"], ["left", "right"])
        self.assertEqual(profile.parameters["arms"]["left"]["velocity_ratio"], 25)
        self.assertEqual(profile.parameters["arms"]["right"]["acceleration_ratio"], 25)

    def test_hand_targets_match_dataset_start_and_joint_limits(self):
        self.assertEqual(home.HAND_HOME_DEG["left"][:4], (3.47, 11.81, 17.38, 25.56))
        self.assertEqual(home.HAND_HOME_DEG["right"][12:16], (13.08, -2.92, 22.83, 7.87))
        for side in ("left", "right"):
            radians = home.hand_home_rad(side)
            self.assertEqual(len(radians), len(JOINT_NAMES))
            for q, (low, high) in zip(radians, JOINT_LIMITS_RAD):
                self.assertGreaterEqual(q, low)
                self.assertLessEqual(q, high)

    def test_hand_blend_starts_at_measurement_and_ends_at_goal(self):
        start = tuple(math.radians(q) for q in home.HAND_HOME_DEG["left"])
        goal = tuple(math.radians(q) for q in home.HAND_HOME_DEG["right"])
        self.assertEqual(home.blend_joints(start, goal, 0, 3), start)
        end = home.blend_joints(start, goal, 3, 3)
        self.assertTrue(all(math.isclose(a, b) for a, b in zip(end, goal)))
        mid = home.blend_joints(start, goal, 1.5, 3)
        self.assertTrue(all(math.isclose(q, (a + b) / 2) for q, a, b in zip(mid, start, goal)))
        self.assertEqual(home.blend_joints(start, goal, 9, 3), end)

    def test_connected_arm_home_uses_the_existing_driver(self):
        settings = load_config()
        driver = Mock()
        driver.engaged = False
        terminal = Mock()
        terminal.read.return_value = None
        home.home_connected_arms(driver, settings, terminal)
        driver.start.assert_not_called()
        driver.close.assert_not_called()
        driver.clear_errors.assert_called_once()
        driver.configure.assert_called_once()
        self.assertEqual(
            [call.args[0] for call in driver.move_joints.call_args_list], ["left", "right"]
        )
        profile = driver.configure.call_args.args[0]
        self.assertEqual(profile.profile_id, settings["profile"]["profile_id"] + "-ready")
        self.assertEqual(profile.parameters["arms"]["left"]["velocity_ratio"], 25)


if __name__ == "__main__":
    unittest.main()
