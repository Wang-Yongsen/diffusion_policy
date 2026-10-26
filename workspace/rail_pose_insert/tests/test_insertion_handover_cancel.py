"""Cancellable position-to-impedance handover, using only an in-memory SDK."""
from copy import deepcopy
import ctypes
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from diffusion_policy.third_party.bimanual_teleop.control.arm.cartesian import TianjiCartesianExecutor
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import TianjiDriver
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import MotionProfile
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.sdk import ControlSDK, FeedbackSnapshot
from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile, Pose


def profile(identifier="insertion-test"):
    return ControlProfile(identifier, "cartesian_impedance", {
        "active_arms": ["right"], "arms": {"right": {
            "stiffness": [100.] * 6, "damping": [.5] * 6,
            "nullspace_stiffness": 10., "nullspace_damping": .5,
            "tool_dyn10": [0.] * 10, "velocity_ratio": 1, "acceleration_ratio": 1,
        }},
    })


class MemoryBinding:
    """Exercise ControlSDK's actual serialized build/cancel/send implementation."""
    def __init__(self):
        self.sent = 0
        self.on_build = lambda name: None

    def clear_set(self):
        return True

    def send_cmd(self):
        self.sent += 1
        return True

    def __getattr__(self, name):
        if not name.startswith("set_"):
            raise AttributeError(name)
        def build(*args):
            self.on_build(name)
            return True
        return build


class MemorySDK(ControlSDK):
    def __init__(self, driver):
        # Deliberately bypass ControlSDK.__init__: no shared library or device.
        self.driver = driver
        self.robot = MemoryBinding()
        self._lock, self._opened = threading.RLock(), True
        self.cancelled = lambda: (driver._closing.is_set() or
            driver._command_cancel is not None and driver._command_cancel.is_set())
        self.report_configuration = self.report_engagement = True
        self.configuration_sent, self.engagement_sent = threading.Event(), threading.Event()
        self.after_engage = lambda: None
        self.commands, self.holds = [], []
        self.packet = FeedbackSnapshot(state=[1, 1], low_speed=[1, 1])
        self.publish()

    def publish(self):
        self.packet.packet_index += 1
        self.packet.sequence = [self.packet.packet_index] * 2
        self.packet.received_ns = time.monotonic_ns()
        self.driver._on_feedback(deepcopy(self.packet))

    def configure(self, mask, profiles):
        submitted = super().configure(mask, profiles)
        if self.report_configuration:
            for index, _ in self._arms(mask):
                arm = profiles[index]
                channels = {
                    "cart_k": (*arm.stiffness, arm.nullspace_stiffness),
                    "cart_d": (*arm.damping, arm.nullspace_damping),
                    "tool_pose": (0.,) * 6, "tool_dynamics": arm.tool_dyn10,
                    "impedance_rotation": (0.,) * 7,
                }
                for name, values in channels.items():
                    width = len(values)
                    getattr(self.packet, name)[index*width:(index+1)*width] = [
                        ctypes.c_float(value).value for value in values]
                self.packet.velocity_ratio[index] = arm.velocity_ratio
                self.packet.acceleration_ratio[index] = arm.acceleration_ratio
                self.packet.force_type[index] = 1
            self.publish()
        self.configuration_sent.set()
        return submitted

    def submit(self, mask, targets, expiry, *, engage=False):
        submitted = super().submit(mask, targets, expiry, engage=engage)
        self.commands.append((engage, tuple(targets)))
        if engage:
            if self.report_engagement:
                self.packet.state[1], self.packet.impedance_type[1] = 3, 2
                self.publish()
            self.engagement_sent.set()
            self.after_engage()
        return submitted

    def hold(self, mask):
        self.holds.append(mask)
        self.publish()
        return time.monotonic_ns()


class ToyKinematics:
    def fk(self, side, joints):
        return Pose(f"tianji_{side}_base", f"tianji_{side}_flange",
                    (joints[0], joints[1], joints[2]), (0., 0., 0., 1.))

    def jacobian(self, side, joints):
        return np.eye(6, 7)


class HandoverCancellationTests(unittest.TestCase):
    def fixture(self):
        # This fixture publishes on demand rather than running the real receiver.
        # Keep scheduling/NumPy startup delays out of the cancellation tests.
        driver = TianjiDriver("127.0.0.1", watchdog_s=2., engagement_timeout_s=1.)
        driver.profile = MotionProfile.from_control_profile(profile("before-handover"))
        sdk = MemorySDK(driver)
        driver._sdk = sdk
        return driver, sdk, threading.Event()

    def interrupt_wait(self, operation, started, cancel):
        errors = []
        def run():
            try:
                operation()
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(started.wait(.5), "operation did not submit its initial command")
        finally:
            cancel.set()
            thread.join(.5)
        self.assertFalse(thread.is_alive(), "cancel must interrupt confirmation, not await its 1 s timeout")
        self.assertEqual(len(errors), 1)
        self.assertIn("cancelled", str(errors[0]))

    def test_precancelled_configuration_and_engagement_never_build_commands(self):
        for operation in ("configure", "engage"):
            with self.subTest(operation=operation):
                driver, sdk, cancel = self.fixture()
                cancel.set()
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    if operation == "configure":
                        driver.configure(profile(), cancel=cancel)
                    else:
                        driver.engage(cancel=cancel)
                self.assertEqual(sdk.robot.sent, 0)
                self.assertFalse(driver.engaged)

    def test_cancellation_during_sdk_build_prevents_submission(self):
        for operation, setter in (("configure", "set_tool"), ("engage", "set_state")):
            with self.subTest(operation=operation):
                driver, sdk, cancel = self.fixture()
                sdk.robot.on_build = lambda name: cancel.set() if name == setter else None
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    if operation == "configure":
                        driver.configure(profile(), cancel=cancel)
                    else:
                        driver.engage(cancel=cancel)
                self.assertEqual(sdk.robot.sent, 0)
                self.assertIsNone(driver._command_cancel)
                self.assertFalse(driver.engaged)

    def test_configuration_readback_wait_is_cancellable_and_does_not_commit_profile(self):
        driver, sdk, cancel = self.fixture()
        sdk.report_configuration = False
        self.interrupt_wait(lambda: driver.configure(profile(), cancel=cancel), sdk.configuration_sent, cancel)
        self.assertEqual(driver.profile.profile_id, "before-handover")
        self.assertEqual(sdk.robot.sent, 1)
        self.assertIsNone(driver._command_cancel)
        self.assertFalse(sdk.cancelled())

    def test_engagement_mode_wait_is_cancellable_and_stops_the_accepted_seed(self):
        driver, sdk, cancel = self.fixture()
        sdk.report_engagement = False
        self.interrupt_wait(lambda: driver.engage(cancel=cancel), sdk.engagement_sent, cancel)
        self.assertEqual(sdk.holds, [2])
        self.assertFalse(driver.engaged)
        self.assertFalse(driver._mode_confirmed)
        self.assertIn("cancelled", driver.motion_stop["reason"])
        self.assertIsNone(driver._command_cancel)

    def test_cancel_after_mode_ack_does_not_cancel_measured_hold(self):
        driver, sdk, cancel = self.fixture()
        sdk.packet.q[7:] = [2.] * 7
        sdk.publish()
        sdk.after_engage = cancel.set
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            driver.engage(cancel=cancel)
        self.assertEqual([engage for engage, _ in sdk.commands], [True, False])
        self.assertEqual(sdk.robot.sent, 2)
        np.testing.assert_allclose(sdk.commands[-1][1][7:], [2.] * 7)
        self.assertFalse(driver.engaged)
        self.assertFalse(sdk.cancelled())

    def test_legacy_configuration_and_engagement_still_confirm_feedback(self):
        driver, sdk, _ = self.fixture()
        driver.configure(profile())
        driver.engage()
        self.assertEqual(driver.profile.profile_id, "insertion-test")
        self.assertTrue(driver.engaged)
        self.assertTrue(driver._mode_confirmed)
        self.assertEqual(sdk.robot.sent, 2)
        self.assertIsNone(driver._command_cancel)

    def test_executor_uses_actual_engagement_sample_after_preflight_changes(self):
        driver, sdk, cancel = self.fixture()
        executor = TianjiCartesianExecutor(driver, ToyKinematics())
        resolve = executor._resolve_reference
        calls = []
        def changing_reference(sample):
            result = resolve(sample)
            calls.append(sample)
            if len(calls) == 1:
                sdk.packet.q[7:] = [1.] * 7
                sdk.publish()
            return result
        with patch.object(executor, "_resolve_reference", side_effect=changing_reference):
            executor.engage(cancel=cancel)
        self.assertNotEqual(calls[0].header.ref, executor.engagement_ref)
        self.assertEqual(executor.engagement_ref, driver.engagement_sample.header.ref)
        np.testing.assert_allclose(executor.engagement_poses["right"].position_m,
                                   [np.radians(1.)] * 3)

    def test_executor_preflight_cancellation_prevents_engagement(self):
        driver, sdk, cancel = self.fixture()
        executor = TianjiCartesianExecutor(driver, ToyKinematics())
        resolve = executor._resolve_reference
        def cancelled_reference(sample):
            result = resolve(sample)
            cancel.set()
            return result
        with patch.object(executor, "_resolve_reference", side_effect=cancelled_reference):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                executor.engage(cancel=cancel)
        self.assertEqual(sdk.robot.sent, 0)
        self.assertEqual(executor.engagement_poses, {})

    def test_executor_cancellation_during_reference_initialization_stops_without_publishing(self):
        driver, sdk, cancel = self.fixture()
        executor = TianjiCartesianExecutor(driver, ToyKinematics())
        resolve = executor._resolve_reference
        def cancelled_reference(sample):
            result = resolve(sample)
            if driver.engaged:
                cancel.set()
            return result
        with patch.object(executor, "_resolve_reference", side_effect=cancelled_reference):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                executor.engage(cancel=cancel)
        self.assertFalse(driver.engaged)
        self.assertEqual(executor.engagement_poses, {})
        self.assertEqual(executor._reference, {})
        self.assertEqual([engage for engage, _ in sdk.commands], [True, False])

    def test_executor_configuration_forwards_cancel_and_discards_previous_servo(self):
        driver, sdk, cancel = self.fixture()
        executor = TianjiCartesianExecutor(driver, ToyKinematics())
        executor._reference = {"right": (0.,) * 7}
        sdk.report_configuration = False
        self.interrupt_wait(lambda: executor.configure(profile(), cancel=cancel),
                            sdk.configuration_sent, cancel)
        self.assertEqual(executor._reference, {})
        self.assertIsNone(driver._command_cancel)


if __name__ == "__main__":
    unittest.main()
