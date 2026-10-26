"""Insertion ownership, handover and last-moment guards with fake devices."""
from copy import deepcopy
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from diffusion_policy.third_party.bimanual_teleop.types import Pose, Submission
from rail_preinsert.core import matrix
from rail_receive.logic import KeyboardJog
from test_insertion import synthetic_settings
import test_receive


class InsertionHardwareTests(unittest.TestCase):
    def hardware(self):
        hw, driver, hand, executor = test_receive.HardwareTests().hardware()
        hw.arm_enabled = hw.hand_enabled = True
        hw.arm_mode = "position"
        hw.position_done.set()
        hw.hand_target = np.full(20, .2)
        hw.jogger = KeyboardJog(hw.config, np.eye(4))
        state = {"joints_rad": np.zeros(7), "T_B_F": np.eye(4),
            "hand_position_rad": np.full(20, .2), "wrench": np.zeros(6),
            "arm_control_state": 1, "arm_low_speed": 1, "arm_impedance_type": 0,
            "right_sequence": 0, "timestamp_ns": time.monotonic_ns()}
        def snapshot():
            state.update(timestamp_ns=time.monotonic_ns(), right_sequence=state["right_sequence"]+1)
            return deepcopy(state)
        hw.snapshot = snapshot
        def engage(**kw):
            state.update(arm_control_state=3, arm_impedance_type=2)
            state["T_B_F"] = matrix(executor.engagement_poses["right"])
        executor.engage.side_effect = engage
        executor.tracking_status = {"right": {"max_joint_tracking_error_rad": 0., "limited": False}}
        def submit(target, **kw):
            kw["before_submit"]()
            executor.applied_poses = dict(target.tool_poses)
            return Submission(target.command_id, True)
        executor.submit.side_effect = submit
        goal = np.eye(4); goal[1, 3] = -.02
        request = dict(settings=synthetic_settings(), grasp=np.eye(4), goal=goal, above=np.eye(4))
        return hw, driver, hand, executor, state, request

    def test_queue_never_writes_and_locks_old_motion_and_hand_paths(self):
        hw, driver, hand, executor, state, request = self.hardware()
        hw.queue_insertion(request)
        executor.configure.assert_not_called()
        executor.engage.assert_not_called()
        executor.submit.assert_not_called()
        driver.move_joints.assert_not_called()
        self.assertIsNone(hw.jog("w"))
        with self.assertRaisesRegex(RuntimeError, "Hand posture"):
            hw.begin_hand(np.zeros(20))
        with self.assertRaises(RuntimeError):
            hw.queue_insertion(request)
        with self.assertRaisesRegex(RuntimeError, "Insertion owns"):
            hw.move_to_position(np.zeros(7), np.zeros(7), timeout_s=1.)
        hand.disable.assert_not_called()

    def test_owner_reseeds_from_engagement_measurement_and_new_profile(self):
        hw, driver, hand, executor, state, request = self.hardware()
        hw.queue_insertion(request)
        seed = Pose("tianji_right_base", "tianji_right_flange", (.0002, 0, 0), (0, 0, 0, 1))
        executor.engagement_poses = {"right": seed}
        owner_names = []
        submit = executor.submit.side_effect
        def owner_submit(*args, **kwargs):
            owner_names.append(threading.current_thread().name)
            return submit(*args, **kwargs)
        executor.submit.side_effect = owner_submit
        publish = hw._publish_insertion
        def published(*args):
            publish(*args)
            hw.stop.set()
        hw._publish_insertion = published
        thread = threading.Thread(target=hw._arm_loop, name="test-arm-owner")
        thread.start(); thread.join(2.)
        self.assertFalse(thread.is_alive())
        self.assertFalse(hw.paused, hw.error)
        self.assertEqual(owner_names, ["test-arm-owner"])
        command = executor.submit.call_args.args[0]
        np.testing.assert_allclose(matrix(command.tool_poses["right"]), matrix(seed))
        self.assertEqual(command.control_profile_id, "rail-right-force-insertion")
        profile = executor.configure.call_args.args[0]
        self.assertEqual(profile.parameters["active_arms"], ["right"])
        self.assertEqual(profile.parameters["arms"]["right"]["stiffness"], request["settings"]["impedance"]["stiffness"])
        self.assertIs(executor.engage.call_args.kwargs["cancel"], hw.insertion_cancel)
        self.assertEqual(hw.insertion_snapshot()["status"], "baseline")
        driver.move_joints.assert_not_called()
        hand.disable.assert_not_called()

    def test_start_rejects_incomplete_configuration_or_stale_geometry_before_sdk(self):
        for cause in ("missing", "depth", "moved", "force", "busy", "not_stationary", "read_only"):
            with self.subTest(cause=cause):
                hw, _, _, executor, state, request = self.hardware()
                if cause == "missing": request["settings"]["T_F_S"] = None
                elif cause == "depth": request["settings"]["max_descent_m"] = .001
                elif cause == "moved": state["T_B_F"][0, 3] = .01
                elif cause == "force": state["wrench"] = None
                elif cause == "busy": hw.position_active = True
                elif cause == "not_stationary": state["arm_low_speed"] = 0
                else: hw.motion = False
                with self.assertRaises((ValueError, RuntimeError)):
                    hw.queue_insertion(request)
                executor.configure.assert_not_called()
                executor.engage.assert_not_called()
                self.assertIsNone(hw.insertion_snapshot())

    def test_pause_at_each_handover_boundary_prevents_search_submission(self):
        for boundary in ("queued", "configure", "engage", "first_submit"):
            with self.subTest(boundary=boundary):
                hw, _, hand, executor, state, request = self.hardware()
                hw.queue_insertion(request)
                if boundary == "queued": hw.pause("operator pause")
                elif boundary == "configure":
                    executor.configure.side_effect = lambda *a, **k: hw.pause("operator pause")
                elif boundary == "engage":
                    original = executor.engage.side_effect
                    def engage(**kw):
                        original(**kw)
                        hw.pause("operator pause")
                    executor.engage.side_effect = engage
                else:
                    def submit(target, **kw):
                        hw.pause("operator pause")
                        kw["before_submit"]()
                    executor.submit.side_effect = submit
                with self.assertRaises(RuntimeError):
                    hw._insertion_step(hw.snapshot(), time.monotonic())
                self.assertTrue(hw.insertion_cancel.is_set())
                self.assertEqual(hw.insertion_snapshot()["status"], "paused")
                self.assertFalse(hw.insertion_snapshot()["insertion_executed"])
                hand.disable.assert_not_called()

    def test_overload_during_servo_computation_is_checked_before_device_write(self):
        hw, _, _, executor, state, request = self.hardware()
        hw.queue_insertion(request)
        hw._insertion_step(hw.snapshot(), time.monotonic())
        hw.insertion_controller.baseline = np.zeros(6)
        sent = []
        def submit(target, **kw):
            state["wrench"] = np.array([request["settings"]["hard_force_n"]*2, 0, 0, 0, 0, 0])
            kw["before_submit"]()
            sent.append(target)
            return Submission(target.command_id, True)
        executor.submit.side_effect = submit
        with self.assertRaisesRegex(RuntimeError, "hard limit") as failure:
            hw._submit_insertion_target(np.eye(4))
        self.assertFalse(sent)
        hw.pause(str(failure.exception))
        report = hw.insertion_snapshot()
        self.assertEqual(report["raw_wrench"][0], request["settings"]["hard_force_n"]*2)
        self.assertIn("hard limit", report["hard_fault"])
        self.assertEqual(report["trace"][-1]["fault_sample"]["wrench"][0], report["raw_wrench"][0])

    def test_owner_uses_time_after_reading_feedback(self):
        hw, _, _, executor, state, request = self.hardware()
        hw.queue_insertion(request)
        hw._insertion_step(hw.snapshot(), time.monotonic())
        original_snapshot = hw.snapshot
        def snapshot_after_feedback_arrival():
            time.sleep(.001)
            return original_snapshot()
        hw.snapshot = snapshot_after_feedback_arrival
        original_publish = hw._publish_insertion
        def publish(*args):
            original_publish(*args)
            hw.stop.set()
        hw._publish_insertion = publish
        hw._arm_loop()
        self.assertFalse(hw.paused, hw.error)
        self.assertEqual(hw.insertion_snapshot()["status"], "baseline")

    def test_cancel_after_driver_acceptance_never_claims_no_motion(self):
        hw, _, _, executor, state, request = self.hardware()
        hw.queue_insertion(request)
        hw._insertion_step(hw.snapshot(), time.monotonic())
        accepted_by_sdk = []
        def cancelled_after_sdk_acceptance(target, **kw):
            kw["before_submit"]()
            accepted_by_sdk.append(target)
            hw.pause("operator pause after SDK acceptance")
            # CartesianExecutor can reject its reference publication after a
            # concurrent hold changed generation, even though SDK accepted it.
            return Submission(target.command_id, False, "engagement ended during submission")
        executor.submit.side_effect = cancelled_after_sdk_acceptance
        target = np.eye(4); target[1, 3] = -.0001
        with self.assertRaisesRegex(RuntimeError, "ended during submission"):
            hw._submit_insertion_target(target)
        report = hw.insertion_snapshot()
        self.assertEqual(len(accepted_by_sdk), 1)
        self.assertTrue(report["motion_submission_attempted"])
        self.assertTrue(report["motion_acceptance_unknown"])
        self.assertIsNone(report["insertion_executed"])
        self.assertIsNone(report["trace"][-1]["insertion_executed"])

    def test_trace_pairs_latest_guard_wrench_with_its_feedback_timestamp(self):
        hw, _, _, executor, state, request = self.hardware()
        hw.queue_insertion(request)
        hw._insertion_step(hw.snapshot(), time.monotonic())
        c = hw.insertion_controller
        c.baseline = np.zeros(6)
        c.filtered = np.zeros(6)
        c.status = "awaiting_confirmation"
        previous = executor.submit.side_effect
        def force_changes_during_servo(target, **kw):
            state["wrench"] = np.array([0., 1., 0., 0., 0., 0.])
            return previous(target, **kw)
        executor.submit.side_effect = force_changes_during_servo
        hw._insertion_step(hw.snapshot(), time.monotonic())
        report = hw.insertion_snapshot()
        self.assertEqual(report["raw_wrench"][1], 1.)
        self.assertEqual(report["wrench_at_hole_midpoint_base"][1], 1.)
        self.assertGreaterEqual(report["host_monotonic_ns"], report["timestamp_ns"])
        self.assertLessEqual(report["filtered_wrench_timestamp_ns"], report["timestamp_ns"])

    def test_terminal_hold_cannot_resume_jog_and_trace_is_bounded_and_copied(self):
        hw, _, _, executor, state, request = self.hardware()
        hw.queue_insertion(request)
        hw._insertion_step(hw.snapshot(), time.monotonic())
        controller = hw.insertion_controller
        controller.baseline = np.zeros(6)
        controller.filtered = np.zeros(6)
        controller.status = "awaiting_confirmation"
        controller.target = state["T_B_F"].copy()
        hw._insertion_step(hw.snapshot(), time.monotonic())
        report = hw.insertion_snapshot()
        self.assertEqual(report["status"], "awaiting_confirmation")
        self.assertFalse(report["assembly_verified"])
        self.assertFalse(report["automatic_release"])
        self.assertIsNone(hw.jog("f"))
        cursor = report["trace"][-1]["trace_seq"]
        self.assertEqual(hw.insertion_snapshot(after_seq=cursor)["trace"], [])
        report["status"] = "tampered"
        self.assertEqual(hw.insertion_snapshot()["status"], "awaiting_confirmation")
        self.assertEqual(hw._insertion_trace.maxlen, 2048)

    def test_complete_contact_search_insert_and_hold_with_measured_feedback(self):
        hw, driver, hand, executor, state, request = self.hardware()
        request["goal"][1, 3] = -.005
        clock = [time.monotonic()]
        def snapshot():
            state.update(timestamp_ns=round(clock[0]*1e9), right_sequence=state["right_sequence"]+1)
            return deepcopy(state)
        hw.snapshot = snapshot
        submit = executor.submit.side_effect
        def follow(target, **kw):
            result = submit(target, **kw)
            state["T_B_F"] = matrix(target.tool_poses["right"])
            return result
        executor.submit.side_effect = follow
        stages, search_cycles = set(), 0
        with patch("time.monotonic", side_effect=lambda: clock[0]), patch("time.monotonic_ns", side_effect=lambda: round(clock[0]*1e9)):
            hw.queue_insertion(request)
            for _ in range(800):
                clock[0] += .02
                c = hw.insertion_controller
                stage = c.status if c is not None else "switching_to_insertion"
                stages.add(stage)
                if stage == "searching":
                    search_cycles += 1
                reaction = 0.
                if state["T_B_F"][1, 3] <= -.001:
                    reaction = 2. if search_cycles < 10 else .2
                if state["T_B_F"][1, 3] <= -.00495:
                    reaction = 2.
                state["wrench"] = np.array([0., reaction, 0., 0., 0., 0.])
                hw._insertion_step(hw.snapshot(), clock[0])
                if hw.insertion_controller.status == "awaiting_confirmation":
                    break
            report = hw.insertion_snapshot()
            self.assertEqual(report["status"], "awaiting_confirmation", report)
            self.assertTrue({"baseline", "approaching", "searching", "capture", "inserting"}.issubset(stages))
            self.assertTrue(report["insertion_executed"])
            self.assertTrue(report["contact_verified"])
            self.assertFalse(report["assembly_verified"])
            final = state["T_B_F"].copy()
            for _ in range(5):
                clock[0] += .02
                hw._insertion_step(hw.snapshot(), clock[0])
            np.testing.assert_allclose(state["T_B_F"], final)
            hand.disable.assert_not_called()
            driver.move_joints.assert_not_called()


if __name__ == "__main__":
    unittest.main()
