"""Position-only spiral geometry, dispatch gates and failures; no devices."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import test_above
from rail_preinsert.core import matrix, pose
from rail_receive.logic import load_config
from rail_receive.position_spiral import (DEFAULT_POSITION_SPIRAL, MAX_STEPS,
                                         PositionSpiral, plan_position_spiral, validate_settings)


class PositionSpiralTests(unittest.TestCase):
    def geometry(self, start_offset=None):
        config, kinematics = load_config(HERE/"receive.yaml"), test_above.ToyKinematics()
        config["position_spiral"] = dict(DEFAULT_POSITION_SPIRAL)
        # Neither force settings nor legacy insertion settings are a dependency.
        config.pop("insertion", None)
        grasp = test_above.T([.02, -.2, .04], [.1, -.2, .3])
        goal = test_above.T([.12, .05, .6], [.2, .1, -.1])
        above = goal.copy(); above[1, 3] += .05
        actual = above.copy()
        if start_offset is not None:
            actual[:3, 3] += start_offset
        joints = np.asarray(kinematics.ik("right", pose(actual @ np.linalg.inv(grasp), "right"), np.zeros(7)))
        return config, kinematics, joints, grasp, goal, above

    def fixture(self, directory, start_offset=None):
        config, kinematics, joints, grasp, goal, above = self.geometry(start_offset)
        insertion = PositionSpiral(config, kinematics, directory, joints, grasp, goal, above)
        self.assertTrue(insertion.done.wait(3.))
        self.assertIsNone(insertion.error)
        state = dict(joints_rad=joints.copy(), T_B_F=matrix(kinematics.fk("right", joints)))
        hardware = Mock()
        hardware.position_move_finished.return_value = False
        return insertion, hardware, state

    def advance(self, insertion, hardware, state, now):
        with patch("rail_receive.position_spiral.time.monotonic", return_value=now), redirect_stdout(io.StringIO()):
            return insertion.advance(hardware, state)

    def reach(self, insertion, state):
        state.update(joints_rad=np.asarray(insertion.report["joint_waypoints_rad"][insertion.index+1]),
                     T_B_F=np.asarray(insertion.report["flange_targets"][insertion.index]))

    def test_geometry_is_right_base_spiral_with_bounded_steps_and_exact_goal(self):
        for offset in (None, [.002, 0., 0.], [.001, .001, -.001]):
            with self.subTest(offset=offset):
                config, k, q, grasp, goal, above = self.geometry(offset)
                report = plan_position_spiral(k, config, q, grasp, goal, above)
                initial = matrix(k.fk("right", q)) @ grasp
                targets = np.asarray(report["rail_targets"])
                path = np.concatenate([initial[None, :3, 3], targets[:, :3, 3]])
                self.assertTrue(np.all(np.linalg.norm(np.diff(path, axis=0), axis=1)
                                       <= config["position_spiral"]["max_step_m"]+1e-12))
                self.assertTrue(np.all(np.diff(path[:, 1]) < 0))
                self.assertTrue(np.all(targets[:, 1, 3] >= goal[1, 3]))
                np.testing.assert_array_equal(targets[-1], goal)
                for i, (rail, joints) in enumerate(zip(targets, report["joint_waypoints_rad"][1:]), 1):
                    np.testing.assert_array_equal(rail[:3, :3], goal[:3, :3])
                    np.testing.assert_allclose(matrix(k.fk("right", joints)) @ grasp, rail, atol=1e-10)
                    u = i/report["steps"]
                    center = (1-u)*initial[:3, 3]+u*goal[:3, 3]
                    radius = config["position_spiral"]["radius_m"]*np.sin(np.pi*u)
                    theta = 2*np.pi*config["position_spiral"]["turns"]*u
                    np.testing.assert_allclose((rail[:3, 3]-center)[[0, 2]],
                                               radius*np.array([np.cos(theta), np.sin(theta)]), atol=1e-12)
                self.assertLessEqual(report["steps"], MAX_STEPS)
                self.assertFalse(report["force_sensor_used"])
                self.assertFalse(report["force_controlled"])

    def test_small_final_height_and_fractional_turns_still_end_exactly(self):
        config, k, q, grasp, goal, above = self.geometry()
        goal[1, 3] += .047
        config["position_spiral"]["turns"] = 1.25
        report = plan_position_spiral(k, config, q, grasp, goal, above)
        np.testing.assert_array_equal(report["rail_targets"][-1], goal)
        self.assertGreater(report["steps"], 2)

    def test_all_ik_is_preplanned_and_joint_branch_jumps_fail_before_motion(self):
        config, k, q, grasp, goal, above = self.geometry()
        original = k.ik
        def jump(*args):
            result = list(original(*args)); result[-1] = .1
            return result
        with tempfile.TemporaryDirectory() as directory, patch.object(k, "ik", side_effect=jump):
            insertion = PositionSpiral(config, k, directory, q, grasp, goal, above)
            self.assertTrue(insertion.done.wait(3.))
            hardware = Mock()
            with self.assertRaisesRegex(ValueError, "joint branch"):
                self.advance(insertion, hardware,
                             dict(joints_rad=q, T_B_F=matrix(k.fk("right", q))), insertion.started)
            hardware.move_to_position.assert_not_called()
            self.assertFalse((Path(directory)/"position_spiral_plan.json").exists())

    def test_invalid_settings_and_excessive_point_count_are_rejected(self):
        invalid = [dict(radius_m=value) for value in (True, None, "0.001", float("nan"), float("inf"), 0., -.001)]
        invalid += [dict(max_step_m=.002), dict(radius_m=.006), dict(turns=101),
                    dict(velocity_ratio=1.5), dict(acceleration_ratio=True), dict(velocity_ratio=101),
                    dict(position_tolerance_m=.00025), dict(path_tolerance_m=.0001),
                    dict(settle_s=20.), dict(step_timeout_s=121.), dict(total_timeout_s=10.),
                    dict(max_step_m=1e-8, position_tolerance_m=1e-10), dict(max_descent_m=.2)]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_settings(dict(DEFAULT_POSITION_SPIRAL, **change))
        config, k, q, grasp, goal, above = self.geometry()
        config["position_spiral"].update(radius_m=.000001, max_step_m=.000001,
                                        position_tolerance_m=.0000001)
        with self.assertRaisesRegex(ValueError, "planning budget"):
            plan_position_spiral(k, config, q, grasp, goal, above)

    def test_wrong_start_geometry_descent_and_rotation_are_rejected(self):
        for cause in ("below_goal", "descent_limit", "lateral", "rotation", "displaced"):
            config, k, q, grasp, goal, above = self.geometry()
            if cause == "below_goal": goal[1, 3] = above[1, 3]+.001
            elif cause == "descent_limit": config["position_spiral"]["max_descent_m"] = .01
            elif cause == "lateral": goal[0, 3] += .001
            elif cause == "rotation": goal[:3, :3] = test_above.T(rpy=[.2, 0., 0.])[:3, :3]
            elif cause == "displaced": q[0] += .01
            with self.subTest(cause=cause), self.assertRaises(ValueError):
                plan_position_spiral(k, config, q, grasp, goal, above)

    def test_sdk_measured_pose_and_continuous_settle_are_all_required(self):
        with tempfile.TemporaryDirectory() as directory:
            insertion, hw, state = self.fixture(directory, [.002, 0., 0.])
            now = insertion.started
            self.advance(insertion, hw, state, now)
            hw.move_to_position.assert_called_once()
            hw.position_move_finished.return_value = True
            self.advance(insertion, hw, state, now+.2)  # SDK reports done but still at start.
            hw.move_to_position.assert_called_once()
            self.reach(insertion, state)
            hw.position_move_finished.return_value = False
            self.advance(insertion, hw, state, now+.3)
            hw.position_move_finished.return_value = True
            self.advance(insertion, hw, state, now+.4)
            self.advance(insertion, hw, state, now+.45)
            hw.move_to_position.assert_called_once()
            # A single non-arrived observation resets the settle interval.
            state["T_B_F"][0, 3] += .0003
            self.advance(insertion, hw, state, now+.46)
            self.reach(insertion, state)
            self.advance(insertion, hw, state, now+.5)
            self.advance(insertion, hw, state, now+.55)
            hw.move_to_position.assert_called_once()
            self.advance(insertion, hw, state, now+.7)
            self.assertEqual(hw.move_to_position.call_count, 2)
            self.assertTrue(insertion.evidence[0]["endpoint_verified"])
            self.assertEqual(hw.move_to_position.call_args.kwargs["velocity_ratio"], 1)
            self.assertEqual(hw.move_to_position.call_args.kwargs["acceleration_ratio"], 1)

    def test_completion_keeps_grasp_saves_evidence_and_never_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            insertion, hw, state = self.fixture(directory)
            now = insertion.started
            self.advance(insertion, hw, state, now)
            hw.position_move_finished.return_value = True
            for _ in range(insertion.report["steps"]):
                self.reach(insertion, state)
                now += .3
                self.advance(insertion, hw, state, now)
                self.advance(insertion, hw, state, now+.15)
            self.assertEqual(insertion.status, "awaiting_confirmation")
            self.advance(insertion, hw, state, now+1000.)
            self.assertEqual(hw.move_to_position.call_count, insertion.report["steps"])
            hw.begin_hand.assert_not_called()
            hw.start_insertion.assert_not_called()
            self.assertTrue((Path(directory)/"position_spiral_plan.json").exists())
            summary = json.loads((Path(directory)/"result.json").read_text())
            self.assertEqual(summary, json.loads((Path(directory)/"insertion_result.json").read_text()))
            self.assertEqual(summary["completed_steps"], insertion.report["steps"])
            self.assertIs(summary["insertion_executed"], True)
            self.assertEqual(summary["hand_action"], "keep_grasp")
            for flag in ("force_controlled", "force_sensor_used", "contact_verified", "assembly_verified", "automatic_release"):
                self.assertFalse(summary[flag])
            self.assertTrue(all(e["endpoint_verified"] for e in summary["execution_evidence"]))
            self.assertLess(summary["position_error_m"], 1e-10)

    def test_faults_never_dispatch_another_segment_and_paused_summary_is_durable(self):
        causes = ("corridor", "rotation", "overshoot", "upward", "step_timeout", "total_timeout", "hardware")
        for cause in causes:
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory:
                insertion, hw, state = self.fixture(directory)
                now = insertion.started
                self.advance(insertion, hw, state, now)
                if cause == "corridor": state["T_B_F"][0, 3] += .004
                elif cause == "rotation":
                    state["T_B_F"][:3, :3] = test_above.T(rpy=[.1, 0., 0.])[:3, :3] @ state["T_B_F"][:3, :3]
                elif cause == "overshoot": state["T_B_F"][1, 3] -= .002
                elif cause == "upward": state["T_B_F"][1, 3] += .0015
                elif cause == "step_timeout":
                    # Even an endpoint arriving after its deadline cannot trigger the next move.
                    self.reach(insertion, state)
                    hw.position_move_finished.return_value = True
                    now += 21.
                elif cause == "total_timeout": now += 601.
                elif cause == "hardware": hw.raise_if_fault.side_effect = RuntimeError("paused")
                with self.assertRaises((RuntimeError, TimeoutError)) as caught:
                    self.advance(insertion, hw, state, now+.1)
                result = insertion.paused_summary(state, str(caught.exception))
                self.assertEqual(result["status"], "paused")
                self.assertEqual(result["completed_steps"], 0)
                self.assertEqual(result["queued_steps"], 1)
                self.assertEqual(result["hand_action"], "retain_last_target")
                self.assertEqual(result["previous_status"], "position_spiral")
                self.assertIsNone(result["insertion_executed"])  # Queued commands alone prove no motion.
                self.assertTrue(result["reason"])
                self.advance(insertion, hw, state, now+1.)
                hw.move_to_position.assert_called_once()
                self.assertEqual(json.loads((Path(directory)/"result.json").read_text()), result)

    def test_motion_or_joint_drift_during_planning_prevents_first_command(self):
        for cause in ("pose", "joint", "timeout"):
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory:
                insertion, hw, state = self.fixture(directory)
                now = insertion.started
                if cause == "pose": state["T_B_F"][0, 3] += .0004
                elif cause == "joint": state["joints_rad"][-1] += .1
                else: now += insertion.config["move_above"]["planning_timeout_s"]+1
                with self.assertRaises((RuntimeError, TimeoutError)):
                    self.advance(insertion, hw, state, now)
                hw.move_to_position.assert_not_called()

    def test_paused_summary_survives_missing_or_invalid_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            insertion, hw, state = self.fixture(directory)
            self.advance(insertion, hw, state, insertion.started)
            expected = insertion.last_actual.copy()
            state["T_B_F"][:] = np.nan
            result = insertion.paused_summary(state, "invalid feedback")
            np.testing.assert_allclose(result["T_R_O_from_fk"], expected)
            self.assertEqual(insertion.paused_summary({}, "still paused")["reason"], "still paused")

    def test_immediate_pause_or_dispatch_failure_does_not_claim_execution(self):
        for failure in ("before_dispatch", "dispatch_error", "queued"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                insertion, hw, state = self.fixture(directory)
                if failure == "dispatch_error":
                    hw.move_to_position.side_effect = RuntimeError("queue failed")
                    with self.assertRaisesRegex(RuntimeError, "queue failed"):
                        self.advance(insertion, hw, state, insertion.started)
                elif failure == "queued":
                    self.advance(insertion, hw, state, insertion.started)
                result = insertion.paused_summary(state, "cancelled")
                self.assertEqual(result["queued_steps"], int(failure == "queued"))
                if failure == "before_dispatch":
                    self.assertIs(result["insertion_executed"], False)
                else:
                    self.assertIsNone(result["insertion_executed"])


if __name__ == "__main__":
    unittest.main()
