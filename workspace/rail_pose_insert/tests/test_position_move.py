"""Position handover, cancellation and arrival semantics, with no device I/O."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
from rail_preinsert.core import matrix
from rail_preinsert.planner import JointPath
from rail_receive.logic import KeyboardJog
import test_above
from test_above import ToyKinematics, plan_above
import test_receive


class PositionMoveTests(unittest.TestCase):
    def hardware(self):
        hw, driver, hand, executor = test_receive.HardwareTests().hardware()
        hw.kinematics = ToyKinematics()
        hw.arm_enabled = hw.hand_enabled = True
        hw.jogger = KeyboardJog(hw.config, np.eye(4))
        hw.hand_target = np.full(20, .2)
        state = {"joints_rad": np.zeros(7), "T_B_F": np.eye(4), "arm_control_state": 3}
        hw.snapshot = lambda: dict(state)
        def move(side, target, *, cancel, timeout_s):
            self.assertEqual(side, "right")
            self.assertFalse(cancel.is_set())
            self.assertEqual(timeout_s, 120.)
            state.update(joints_rad=np.array(target),
                         T_B_F=matrix(hw.kinematics.fk(side, target)), arm_control_state=1)
        driver.move_joints.side_effect = move
        return hw, driver, hand, executor, state

    def start(self, hw, *, hand=False):
        for function in ([hw._arm_loop, hw._hand_loop] if hand else [hw._arm_loop]):
            thread = threading.Thread(target=function, daemon=True)
            hw.threads.append(thread)
            thread.start()

    def test_only_one_right_position_target_and_no_cartesian_resubmission(self):
        hw, driver, hand, executor, state = self.hardware()
        goal = np.array([.2, .1, .3, .1, .2, .1, 0.])
        hw.move_to_position(np.zeros(7), goal, timeout_s=120.)
        driver.move_joints.assert_not_called()  # Queueing is not a hardware write.
        self.assertFalse(hw.position_move_finished())
        self.start(hw)
        try:
            self.assertTrue(hw.position_done.wait(1.))
            self.assertTrue(hw.position_move_finished())
            driver.move_joints.assert_called_once_with("right", tuple(goal),
                                                      cancel=hw.position_cancel, timeout_s=120.)
            executor.request_hold.assert_called_once()
            for _ in range(3):
                hw._position_step()
            executor.submit.assert_not_called()
            executor.engage.assert_not_called()
            np.testing.assert_allclose(state["joints_rad"], goal)
            hand.disable.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "Cannot stream"):
                hw.follow_path(JointPath([goal, goal], .1, .2))
        finally:
            hw.close()

    def test_pause_cancels_blocking_position_move_and_hand_keeps_holding(self):
        hw, driver, hand, executor, _ = self.hardware()
        entered, cancelled, held_after_pause = threading.Event(), threading.Event(), threading.Event()
        def move(side, target, *, cancel, timeout_s):
            entered.set()
            if not cancel.wait(1.):
                raise RuntimeError("test cancellation deadline")
            cancelled.set()
            raise RuntimeError("SDK position move cancelled")
        driver.move_joints.side_effect = move
        original_submit = hand.submit.side_effect
        def submit(command):
            if hw.paused:
                held_after_pause.set()
            return original_submit(command)
        hand.submit.side_effect = submit
        hw.move_to_position(np.zeros(7), np.full(7, .1), timeout_s=120.)
        self.start(hw, hand=True)
        try:
            self.assertTrue(entered.wait(1.))
            self.assertFalse(hw.position_done.is_set())
            with self.assertRaisesRegex(RuntimeError, "already active"):
                hw.move_to_position(np.zeros(7), np.full(7, .2), timeout_s=120.)
            hw.pause("Operator pause")
            self.assertTrue(cancelled.wait(1.))
            self.assertTrue(held_after_pause.wait(1.))
            self.assertEqual(hw.error, "Operator pause")
            self.assertFalse(hw.position_done.is_set())
            executor.submit.assert_not_called()
            hand.disable.assert_not_called()
            np.testing.assert_allclose(hand.submit.call_args.args[0].payload.position_rad, .2)
            with self.assertRaisesRegex(RuntimeError, "Operator pause"):
                hw.move_to_position(np.zeros(7), np.full(7, .2), timeout_s=120.)
        finally:
            hw.close()

    def test_close_cancels_position_wait_before_closing_devices(self):
        hw, driver, _, executor, _ = self.hardware()
        entered, returned = threading.Event(), threading.Event()
        def move(side, target, *, cancel, timeout_s):
            entered.set()
            if not cancel.wait(1.):
                raise RuntimeError("test cancellation deadline")
            returned.set()
            raise RuntimeError("SDK move cancelled on close")
        driver.move_joints.side_effect = move
        driver.close.side_effect = lambda: self.assertTrue(returned.is_set())
        hw.move_to_position(np.zeros(7), np.full(7, .1), timeout_s=120.)
        self.start(hw)
        try:
            self.assertTrue(entered.wait(1.))
        finally:
            hw.close()
        self.assertTrue(all(not thread.is_alive() for thread in hw.threads))
        self.assertFalse(hw.position_done.is_set())
        executor.submit.assert_not_called()

    def test_limits_changed_start_and_read_only_reject_before_any_mode_switch(self):
        for case in ("limits", "moved", "read_only", "paused"):
            with self.subTest(case=case):
                hw, driver, _, executor, state = self.hardware()
                target = np.full(7, .1)
                if case == "limits": target[5:7] = [1., 1.]
                elif case == "moved": state["joints_rad"] = np.full(7, .04)
                elif case == "read_only": hw.motion = False
                elif case == "paused": hw.paused = True
                with self.assertRaises((RuntimeError, ValueError)):
                    hw.move_to_position(np.zeros(7), target, timeout_s=120.)
                driver.move_joints.assert_not_called()
                executor.request_hold.assert_not_called()

    def test_sdk_failure_pauses_and_does_not_claim_completion(self):
        hw, driver, _, executor, _ = self.hardware()
        paused = threading.Event()
        hw.emit.side_effect = lambda event, **kw: paused.set() if event == "paused" else None
        driver.move_joints.side_effect = RuntimeError("position mode was not reported")
        hw.move_to_position(np.zeros(7), np.full(7, .1), timeout_s=120.)
        self.start(hw)
        try:
            self.assertTrue(paused.wait(1.))
            self.assertFalse(hw.position_done.is_set())
            self.assertIn("position mode was not reported", hw.error)
            executor.submit.assert_not_called()
        finally:
            hw.close()

    def test_post_arrival_jog_stays_in_position_mode_and_does_not_queue_repeats(self):
        hw, driver, _, executor, _ = self.hardware()
        hw.move_to_position(np.zeros(7), np.full(7, .1), timeout_s=120.)
        hw._position_step()
        before = hw.snapshot()["T_B_F"].copy()
        target = hw.jog("w")
        self.assertAlmostEqual(target[0, 3]-before[0, 3], .001)
        self.assertIsNone(hw.jog("w"))
        hw._position_step()
        self.assertEqual(driver.move_joints.call_count, 2)
        self.assertEqual(hw.arm_mode, "position")
        executor.engage.assert_not_called()
        executor.submit.assert_not_called()


class PositionArrivalTests(unittest.TestCase):
    def test_plan_solves_only_one_target_and_never_samples_a_host_path(self):
        c = test_receive.HardwareTests().hardware()[0].config
        k = ToyKinematics()
        with patch.object(k, "ik", wraps=k.ik) as ik, \
                patch.object(JointPath, "at", side_effect=AssertionError("must not sample")):
            target = np.eye(4); target[:3, 3] = [.2, .1, .3]
            path, report = plan_above(k, c, np.zeros(7), np.eye(4), target)
        ik.assert_called_once()
        self.assertEqual(len(path.points), 2)
        self.assertNotIn("segment_durations_s", report)
        self.assertEqual(report["path_type"], "sdk_position_point_to_point")

    def test_arrival_requires_sdk_completion_and_measured_pose_not_nominal_duration(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            transfer, hw = test_above.AboveTests().fixture(directory)
            transfer.status, transfer.planning_started = "planning_above", 10.
            transfer.path = JointPath([np.zeros(7), np.full(7, .1)], .01, .01)
            transfer.path.duration = 10000.  # Must have no effect on SDK execution.
            transfer.report = {"T_R_F_above": np.eye(4).tolist()}
            transfer.planning_done.set()
            state = {"T_B_F": np.eye(4)}
            hw.position_move_finished.return_value = False
            with patch("rail_receive.above.time.monotonic", return_value=10.):
                transfer.advance(hw, state, None)
            hw.move_to_position.assert_called_once()
            hw.follow_path.assert_not_called()
            with patch("rail_receive.above.time.monotonic", return_value=11.):
                transfer.advance(hw, state, None)
            self.assertEqual(transfer.status, "moving_above")  # Accepted, not completed.
            hw.position_move_finished.return_value = True
            state["T_B_F"][0, 3] = .004
            with patch("rail_receive.above.time.monotonic", return_value=12.):
                transfer.advance(hw, state, None)
            self.assertEqual(transfer.status, "moving_above")  # Outside 3 mm tolerance.
            state["T_B_F"][0, 3] = 0.
            for now in (13., 14.):
                with patch("rail_receive.above.time.monotonic", return_value=now):
                    transfer.advance(hw, state, None)
            self.assertEqual(transfer.status, "verifying_above")
            hw.begin_hand.assert_not_called()


if __name__ == "__main__":
    unittest.main()
