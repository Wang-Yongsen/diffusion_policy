"""Position insertion uses joint feedback without enabling six-axis force data."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

from rail_receive.hardware import RightHardware
from rail_receive.logic import KeyboardJog
from test_above import ToyKinematics
import test_receive


class PositionSpiralHardwareTests(unittest.TestCase):
    def hardware(self, mode="position_spiral"):
        config = test_receive.config()
        if mode is None:
            config["lowering"].pop("mode", None)
        else:
            config["lowering"]["mode"] = mode
        executor = Mock()
        driver_module = "diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver"
        hand_module = "diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter"
        with patch(driver_module + ".TianjiDriver") as driver_factory, \
                patch(hand_module + ".WujiSdkSession"), patch(hand_module + ".WujiHandDriver"):
            hardware = RightHardware(config, ToyKinematics(), Mock(), motion=True, executor=executor)
        driver, hand = hardware.driver, hardware.hand
        driver.health.return_value = NS(ready=True)
        # Deliberately omit wrench: even accessing the unavailable sensor field
        # would fail, rather than merely proving that None can pass through.
        arm = NS(joints=NS(position_rad=np.zeros(7)), state=1, low_speed=1,
                 impedance_type=0, source_sequence=1)
        driver.get_latest.side_effect = lambda: NS(
            header=NS(received_monotonic_ns=time.monotonic_ns()), payload=NS(arms={"right": arm}))
        hand.get_latest.side_effect = lambda: NS(
            header=NS(valid=True, received_monotonic_ns=time.monotonic_ns()),
            payload=NS(position_rad=np.full(20, .2), motor_current_a=np.zeros(20)))
        return hardware, driver_factory, arm, executor

    def test_force_stream_is_enabled_only_for_force_mode_including_legacy_default(self):
        for mode, expected in (("position_spiral", False), ("position_steps", False),
                               ("force_spiral", True), (None, True)):
            with self.subTest(mode=mode):
                hardware, factory, _, _ = self.hardware(mode)
                self.assertIs(factory.call_args.kwargs["record_force"], expected)
                self.assertIs(hardware.record_force, expected)
                factory.return_value.start.assert_not_called()

    def test_position_snapshot_requires_no_wrench_field(self):
        for mode in ("position_spiral", "position_steps"):
            with self.subTest(mode=mode):
                hardware, _, _, _ = self.hardware(mode)
                state = hardware.snapshot()
                self.assertIsNone(state["wrench"])
                np.testing.assert_array_equal(state["joints_rad"], np.zeros(7))
                self.assertEqual(state["arm_control_state"], 1)
                hardware.driver.health.assert_called_with(sides=("right",))

    def test_force_mode_still_exposes_the_sensor_measurement(self):
        hardware, _, arm, _ = self.hardware("force_spiral")
        arm.wrench = (1., 2., 3., .1, .2, .3)
        self.assertEqual(hardware.snapshot()["wrench"], arm.wrench)

    def test_position_move_completes_without_wrench_or_impedance_submissions(self):
        hardware, _, arm, executor = self.hardware()
        hardware.arm_enabled = hardware.hand_enabled = True
        hardware.jogger = KeyboardJog(hardware.config, hardware.snapshot()["T_B_F"])
        target = np.full(7, .01)

        def move(side, goal, *, cancel, timeout_s):
            self.assertEqual(side, "right")
            self.assertFalse(cancel.is_set())
            arm.joints.position_rad = np.asarray(goal)
            arm.source_sequence += 1

        hardware.driver.move_joints.side_effect = move
        hardware.move_to_position(np.zeros(7), target, timeout_s=20.,
                                  velocity_ratio=1, acceleration_ratio=1)
        self.assertFalse(hardware.position_move_finished())
        hardware._position_step()
        self.assertTrue(hardware.position_move_finished())
        state = hardware.snapshot()
        self.assertIsNone(state["wrench"])
        np.testing.assert_allclose(state["joints_rad"], target)
        hardware.driver.move_joints.assert_called_once_with(
            "right", tuple(target), cancel=hardware.position_cancel, timeout_s=20.)
        executor.configure.assert_not_called()
        executor.engage.assert_not_called()
        executor.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
