"""Slow lowering geometry, measured progress and keyboard routing; no devices."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from queue import Queue
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import receive
import test_above
import test_position_move
from rail_preinsert.core import matrix
from rail_receive.logic import load_config
from rail_receive.lowering import Lowering, plan_lowering, validate_settings


class LoweringTests(unittest.TestCase):
    def geometry(self):
        c, k = load_config(HERE/"receive.yaml"), test_above.ToyKinematics()
        grasp = test_above.T([.02, -.2, .04], [.1, -.2, .3])
        goal = test_above.T([.12, .05, .6], [.2, .1, -.1])
        above = goal.copy(); above[1, 3] += .05
        target = above @ np.linalg.inv(grasp)
        from rail_preinsert.core import pose
        q = np.array(k.ik("right", pose(target, "right"), np.zeros(7)))
        return c, k, q, grasp, goal, above

    def fixture(self, directory):
        c, k, q, grasp, goal, above = self.geometry()
        lowering = Lowering(c, k, directory, q, grasp, goal, above)
        self.assertTrue(lowering.done.wait(1.))
        self.assertIsNone(lowering.error)
        state = dict(joints_rad=q.copy(), T_B_F=matrix(k.fk("right", q)))
        hw = Mock(paused=False)
        hw.position_move_finished.return_value = False
        return lowering, hw, state

    def advance(self, lowering, hw, state, now):
        with patch("rail_receive.lowering.time.monotonic", return_value=now), redirect_stdout(io.StringIO()):
            return lowering.advance(hw, state)

    def test_vertical_targets_preserve_rotation_grasp_and_end_exactly_at_goal(self):
        c, k, q, grasp, goal, above = self.geometry()
        report = plan_lowering(k, c, q, grasp, goal, above)
        self.assertEqual(report["steps"], 10)
        targets = np.array(report["rail_targets"])
        np.testing.assert_allclose(targets[:, [0, 2], 3], np.tile(goal[[0, 2], 3], (10, 1)))
        for rail, joints in zip(targets, report["joint_waypoints_rad"][1:]):
            np.testing.assert_allclose(rail[:3, :3], goal[:3, :3])
            np.testing.assert_allclose(matrix(k.fk("right", joints)) @ grasp, rail, atol=1e-10)
        np.testing.assert_allclose(np.diff(np.r_[above[1, 3], targets[:, 1, 3]]), -.005)
        np.testing.assert_allclose(targets[-1], goal)

    def test_last_step_handles_non_multiple_height_and_rejects_ik_branch_jump(self):
        c, k, q, grasp, goal, above = self.geometry()
        goal[1, 3] += .003
        report = plan_lowering(k, c, q, grasp, goal, above)
        self.assertEqual(report["steps"], 10)
        np.testing.assert_allclose(report["rail_targets"][-1], goal)
        original = k.ik
        def jump(*args):
            result = list(original(*args)); result[-1] = .1
            return result
        with patch.object(k, "ik", side_effect=jump), self.assertRaisesRegex(ValueError, "joint branch"):
            plan_lowering(k, c, q, grasp, goal, above)

    def test_next_step_requires_sdk_completion_pose_tolerance_and_settle(self):
        with tempfile.TemporaryDirectory() as directory:
            lowering, hw, state = self.fixture(directory)
            now = lowering.started
            self.advance(lowering, hw, state, now)
            hw.move_to_position.assert_called_once()
            self.advance(lowering, hw, state, now+.1)
            self.assertEqual(hw.move_to_position.call_count, 1)
            hw.position_move_finished.return_value = True
            self.advance(lowering, hw, state, now+.2)  # Accepted but actual still at start.
            self.assertEqual(hw.move_to_position.call_count, 1)
            state.update(joints_rad=np.array(lowering.report["joint_waypoints_rad"][1]),
                         T_B_F=np.array(lowering.report["flange_targets"][0]))
            self.advance(lowering, hw, state, now+.3)
            self.advance(lowering, hw, state, now+.4)
            self.assertEqual(hw.move_to_position.call_count, 1)
            self.advance(lowering, hw, state, now+.6)
            self.assertEqual(hw.move_to_position.call_count, 2)
            self.assertEqual(hw.move_to_position.call_args.kwargs["velocity_ratio"], 1)
            self.assertEqual(hw.move_to_position.call_args.kwargs["acceleration_ratio"], 1)

    def test_full_lowering_holds_hand_and_repeated_advance_cannot_descend_again(self):
        with tempfile.TemporaryDirectory() as directory:
            lowering, hw, state = self.fixture(directory)
            now = lowering.started
            self.advance(lowering, hw, state, now)
            hw.position_move_finished.return_value = True
            for index in range(lowering.report["steps"]):
                state.update(joints_rad=np.array(lowering.report["joint_waypoints_rad"][index+1]),
                             T_B_F=np.array(lowering.report["flange_targets"][index]))
                now += 1.
                self.advance(lowering, hw, state, now)
                self.advance(lowering, hw, state, now+.3)
            self.assertEqual(lowering.status, "placement_reached")
            self.advance(lowering, hw, state, now+1.)
            self.assertEqual(hw.move_to_position.call_count, 10)
            hw.begin_hand.assert_not_called()
            result = json.loads((Path(directory)/"result.json").read_text())
            self.assertEqual(result["hand_action"], "keep_grasp")
            self.assertFalse(result["contact_verified"])
            self.assertFalse(result["automatic_release"])

    def test_deviation_overshoot_timeout_and_pause_never_send_next_step(self):
        for cause in ("lateral", "rotation", "overshoot", "timeout", "paused"):
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory:
                lowering, hw, state = self.fixture(directory)
                now = lowering.started
                self.advance(lowering, hw, state, now)
                if cause == "lateral": state["T_B_F"][0, 3] += .004
                elif cause == "rotation":
                    state["T_B_F"][:3, :3] = test_above.T(rpy=[.1, 0., 0.])[:3, :3] @ state["T_B_F"][:3, :3]
                elif cause == "overshoot": state["T_B_F"][1, 3] -= .007
                elif cause == "timeout": now += 21.
                elif cause == "paused": hw.raise_if_fault.side_effect = RuntimeError("paused")
                with self.assertRaises((RuntimeError, TimeoutError)):
                    self.advance(lowering, hw, state, now+.1)
                self.assertEqual(hw.move_to_position.call_count, 1)

    def test_n_is_manual_single_use_and_rejects_early_unverified_or_displaced_start(self):
        for stage in ("waiting_move_key", "moving_above", "placement_reached"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                transfer, hw = test_above.AboveTests().fixture(directory)
                transfer.config["lowering"]["mode"] = "position_steps"
                transfer.status = stage
                transfer.advance(hw, {}, None, lower_requested=True)
                self.assertIsNone(transfer.lowering)
                self.assertIn("N 未执行", transfer.manual_lower_message)
                hw.move_to_position.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            transfer, hw = test_above.AboveTests().fixture(directory)
            transfer.config["lowering"]["mode"] = "position_steps"
            transfer.status = "above_target"
            transfer.report = {"T_R_F_above": np.eye(4)}
            transfer.cached_grasp = transfer.above.copy()
            state = dict(T_B_F=np.eye(4), joints_rad=np.zeros(7))
            hw.position_move_finished.return_value = True
            transfer.advance(hw, state, None)
            self.assertIsNone(transfer.lowering)  # No auto descent.
            state["T_B_F"][0, 3] = .004
            transfer.advance(hw, state, None, lower_requested=True)
            self.assertIn("偏离", transfer.manual_lower_message)
            self.assertIsNone(transfer.lowering)
            state["T_B_F"][0, 3] = 0.
            transfer.advance(hw, state, None, lower_requested=True)
            self.assertTrue(transfer.lowering.done.wait(1.))
            before = transfer.lowering
            transfer.advance(hw, state, None, lower_requested=True)
            self.assertIs(transfer.lowering, before)
            hw.move_to_position.assert_called_once()  # Existing plan still advances.

    def test_repeated_n_during_motion_cannot_bypass_deviation_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            lowering, hw, state = self.fixture(directory)
            self.advance(lowering, hw, state, lowering.started)
            transfer, _ = test_above.AboveTests().fixture(directory)
            transfer.config["lowering"]["mode"] = "position_steps"
            transfer.lowering, transfer.status = lowering, "lowering"
            state["T_B_F"][0, 3] += .004
            with self.assertRaisesRegex(RuntimeError, "vertical corridor"):
                transfer.advance(hw, state, None, lower_requested=True)
            self.assertEqual(hw.move_to_position.call_count, 1)

    def test_n_terminal_reaches_main_loop_including_paused_states(self):
        for stage in ("above_target", "moving_above", "lowering", "paused"):
            for key in ("n", "N", "nn"):
                with self.subTest(stage=stage, key=key):
                    end, commands = threading.Event(), Queue(maxsize=1)
                    terminal, hw = Mock(), Mock(paused=stage == "paused")
                    def read(_):
                        end.set()
                        return key
                    terminal.read.side_effect = read
                    receive.keyboard(terminal, hw, [stage], commands, end)
                    self.assertEqual(commands.empty(), key == "nn")
                    if key != "nn": self.assertEqual(commands.get_nowait(), "n")

    def test_lowering_ratios_are_confirmed_before_move_and_not_reconfigured_each_step(self):
        hw, driver, _, executor, _ = test_position_move.PositionMoveTests().hardware()
        hw.move_to_position(np.zeros(7), np.full(7, .1), timeout_s=120.,
                            velocity_ratio=1, acceleration_ratio=1)
        hw._position_step()
        driver.configure.assert_called_once()
        profile = driver.configure.call_args.args[0]
        self.assertEqual(profile.parameters["active_arms"], ["right"])
        self.assertEqual(profile.parameters["arms"]["right"]["velocity_ratio"], 1)
        self.assertEqual([call[0] for call in driver.mock_calls], ["configure", "move_joints"])
        hw.move_to_position(np.full(7, .1), np.full(7, .105), timeout_s=120.,
                            velocity_ratio=1, acceleration_ratio=1)
        hw._position_step()
        driver.configure.assert_called_once()
        executor.engage.assert_not_called()
        executor.submit.assert_not_called()

    def test_configuration_failure_or_pause_during_configuration_prevents_motion(self):
        for cause in ("error", "pause"):
            with self.subTest(cause=cause):
                hw, driver, _, _, _ = test_position_move.PositionMoveTests().hardware()
                if cause == "error": driver.configure.side_effect = RuntimeError("ratio readback failed")
                else: driver.configure.side_effect = lambda _: hw.pause("operator pause")
                hw.move_to_position(np.zeros(7), np.full(7, .1), timeout_s=120.,
                                    velocity_ratio=1, acceleration_ratio=1)
                with self.assertRaises(RuntimeError): hw._position_step()
                driver.move_joints.assert_not_called()

    def test_invalid_lowering_settings_are_rejected(self):
        c = load_config(HERE/"receive.yaml")
        for change in ({"step_m": .01}, {"step_m": 0.}, {"velocity_ratio": True},
                       {"velocity_ratio": 0}, {"position_tolerance_m": .005},
                       {"settle_s": float("nan")}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_settings(dict(c["lowering"], **change))


if __name__ == "__main__":
    unittest.main()
