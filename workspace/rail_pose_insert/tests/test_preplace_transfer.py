"""Fresh vision -> measured correction -> N routing, without devices."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import test_above
from rail_preinsert.core import difference, matrix
from rail_receive.logic import load_config


class PreplaceTransferTests(unittest.TestCase):
    def fixture(self, directory, *, position=.012, degrees=2.):
        t, hw = test_above.AboveTests().fixture(directory)
        t.config["lowering"]["mode"] = "position_spiral"
        t.config["preplace_alignment"]["settle_s"] = .00001
        q = np.array([.2, .1, .5, .05, -.05, .03, 0.])
        flange = matrix(t.kinematics.fk("right", q))
        state = dict(T_B_F=flange, joints_rad=q)
        t.cached_grasp = np.linalg.inv(flange) @ t.above
        observed = t.above.copy()
        observed[0, 3] += position
        observed[:3, :3] = Rotation.from_euler("y", degrees, degrees=True).as_matrix() @ observed[:3, :3]
        grasp = np.linalg.inv(flange) @ observed
        t.report = dict(T_R_F_above=flange.tolist(), joint_waypoints_rad=[q.tolist(), q.tolist()])
        hw.position_move_finished.return_value = True
        t._finish_position_arrival(state, (0., 0.))
        return t, hw, state, grasp

    def record(self, t, hw, state, grasp, *, generation="fresh", stamp=None):
        stamp = time.monotonic_ns() if stamp is None else stamp
        hw.history = [(stamp-1_000_000, state["joints_rad"].copy()),
                      (stamp+1_000_000, state["joints_rad"].copy())]
        record = test_above.AboveTests().record(t, stamp,
            np.linalg.inv(t.geometry.T_B_C) @ state["T_B_F"] @ grasp)
        record.update(generation=generation, reestimate_request=True,
                      pose_source="preplace_registration", body_orientation_prior_enabled=False)
        return record

    def advance(self, t, hw, state, record=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return t.advance(hw, state, record, **kwargs)

    def sample(self, t, hw, state, grasp, *, generation="fresh", **kwargs):
        t.perception_initialized(generation)
        for _ in range(t.config["preplace_alignment"]["samples"]):
            self.advance(t, hw, state, self.record(t, hw, state, grasp, generation=generation), **kwargs)

    def execute_plan(self, t, hw, state):
        self.assertEqual(t.status, "planning_preplace")
        self.assertTrue(t.preplace.done.wait(10), "mock IK planner did not finish")
        self.advance(t, hw, state)
        self.assertEqual(t.status, "aligning_preplace")
        for _ in range(500):
            state["joints_rad"] = hw.move_to_position.call_args.args[1].copy()
            state["T_B_F"] = matrix(t.kinematics.fk("right", state["joints_rad"]))
            self.advance(t, hw, state, lower_requested=True)
            if t.status != "aligning_preplace":
                break
            t.preplace.settled_since = time.monotonic()-1
            self.advance(t, hw, state)
            if t.status != "aligning_preplace":
                break
        else:
            self.fail("position correction did not complete")

    def test_default_arrival_requests_fresh_pose_with_correct_camera_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            t, hw, state, _ = self.fixture(directory)
            self.assertEqual(t.status, "reestimating_preplace")
            self.assertTrue(t.needs_reinitialize)
            np.testing.assert_allclose(t.reestimate_predicted_pose,
                np.linalg.inv(t.geometry.T_B_C) @ state["T_B_F"] @ t.cached_grasp)
            saved = json.loads((Path(directory)/"result.json").read_text())
            self.assertEqual(saved["status"], "reestimating_preplace")
            self.advance(t, hw, state, lower_requested=True)
            self.assertIn("再次按 N", t.manual_lower_message)
            hw.move_to_position.assert_not_called()
            self.assertIsNone(t.lowering)

    def test_old_generation_old_frames_duplicates_and_missing_history_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            t, hw, state, grasp = self.fixture(directory)
            record = self.record(t, hw, state, grasp)
            self.advance(t, hw, state, record)  # no request acknowledged yet
            self.assertEqual(len(t.records), 0)
            t.perception_initialized("fresh")
            self.advance(t, hw, state, dict(record, generation="old"))
            self.advance(t, hw, state, dict(record, timestamp_ns=t.not_before_ns-1))
            self.advance(t, hw, state, dict(record, timestamp_ns=time.monotonic_ns()-2_000_000_000))
            hw.history = []
            self.advance(t, hw, state, record)
            self.assertEqual(len(t.records), 0)
            record = self.record(t, hw, state, grasp)
            self.advance(t, hw, state, record)
            self.advance(t, hw, state, record)
            self.assertEqual(len(t.records), 1)
            hw.move_to_position.assert_not_called()

    def test_full_correction_reestimate_then_n_uses_new_grasp_and_original_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            t, hw, state, grasp = self.fixture(directory)
            original_goal, original_above, old_grasp = t.goal.copy(), t.above.copy(), t.cached_grasp.copy()
            self.sample(t, hw, state, grasp)
            self.assertEqual(t.status, "planning_preplace")
            hw.move_to_position.assert_not_called()  # Full planning precedes dispatch.
            self.execute_plan(t, hw, state)
            self.assertEqual(t.status, "reestimating_preplace")
            self.assertTrue(t.needs_reinitialize)
            self.assertEqual(t.preplace.corrections, 1)
            self.sample(t, hw, state, grasp, generation="second", lower_requested=True)
            self.assertEqual(t.status, "above_target")
            self.assertIsNone(t.lowering)  # N during estimation is not queued.
            np.testing.assert_allclose(t.cached_grasp, grasp, atol=1e-10)
            self.assertGreater(difference(old_grasp, t.cached_grasp)[0], .001)
            np.testing.assert_allclose(t.goal, original_goal)
            np.testing.assert_allclose(t.above, original_above)
            np.testing.assert_allclose(t.report["T_R_F_above"], t.above @ np.linalg.inv(grasp), atol=1e-10)
            self.assertNotIn("joint_waypoints_rad", t.report)
            self.assertTrue(all(e["endpoint_verified"] for e in t.preplace.evidence))
            for call in hw.move_to_position.call_args_list:
                self.assertEqual(call.kwargs["velocity_ratio"], 1)
            with patch("rail_receive.position_spiral.PositionSpiral") as spiral:
                self.advance(t, hw, state, lower_requested=True)
                spiral.assert_called_once()
                np.testing.assert_allclose(spiral.call_args.args[4], grasp, atol=1e-10)
                np.testing.assert_allclose(spiral.call_args.args[5], original_goal)
                self.assertTrue(spiral.return_value.preplace_alignment["converged"])
            hw.queue_insertion.assert_not_called()
            hw.begin_hand.assert_not_called()

    def test_visual_subdegree_residual_is_corrected_before_fixed_attitude_spiral(self):
        with tempfile.TemporaryDirectory() as directory:
            t, hw, state, grasp = self.fixture(directory, position=0., degrees=.9)
            self.sample(t, hw, state, grasp)
            self.assertEqual(t.status, "planning_preplace")
            self.assertTrue(t.preplace.finalizing)
            self.assertFalse(t.preplace.converged)
            self.execute_plan(t, hw, state)
            self.assertEqual(t.status, "above_target")
            self.assertFalse(t.needs_reinitialize)
            self.assertLess(difference(state["T_B_F"] @ t.cached_grasp, t.above)[1], 1e-6)
            self.assertLessEqual(t.preplace.report["max_rotation_step_rad"], np.radians(.5)+1e-9)
            self.assertTrue(t.preplace.summary()["final_pose_lock"])

    def test_no_command_for_outlier_or_exhausted_correction_budget(self):
        for case in ("position", "rotation", "budget"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                t, hw, state, grasp = self.fixture(directory,
                    position=.05 if case == "position" else .012,
                    degrees=11. if case == "rotation" else 2.)
                if case == "budget":
                    t.preplace.corrections = t.config["preplace_alignment"]["max_corrections"]
                with self.assertRaisesRegex((ValueError, RuntimeError), "limits|converge"):
                    self.sample(t, hw, state, grasp)
                hw.move_to_position.assert_not_called()
                self.assertIsNone(t.lowering)

    def test_n_rejects_rotation_jog_after_alignment_instead_of_rotating_in_first_spiral_step(self):
        with tempfile.TemporaryDirectory() as directory:
            t, hw, state, grasp = self.fixture(directory, position=0., degrees=0.)
            self.sample(t, hw, state, grasp)
            self.assertTrue(t.preplace.converged)
            actual = state["T_B_F"] @ t.cached_grasp
            actual[:3, :3] = Rotation.from_euler("y", .9, degrees=True).as_matrix() @ actual[:3, :3]
            state["T_B_F"] = actual @ np.linalg.inv(t.cached_grasp)
            with patch("rail_receive.position_spiral.PositionSpiral") as spiral:
                self.advance(t, hw, state, lower_requested=True)
                spiral.assert_not_called()
            self.assertIn("偏离", t.manual_lower_message)

    def test_pause_and_timeouts_never_dispatch(self):
        for case in ("paused", "perception", "total", "drift"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                t, hw, state, _ = self.fixture(directory)
                if case == "paused":
                    hw.paused = True
                elif case == "perception":
                    t.preplace.sample_started -= 181
                elif case == "total":
                    t.preplace.started -= 901
                else:
                    state["T_B_F"][0, 3] += .003
                with self.assertRaises((RuntimeError, TimeoutError)):
                    self.advance(t, hw, state, lower_requested=True)
                hw.move_to_position.assert_not_called()

    def test_sdk_and_fk_both_required_and_stopped_summary_preserves_queued_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            t, hw, state, grasp = self.fixture(directory)
            self.sample(t, hw, state, grasp)
            self.assertTrue(t.preplace.done.wait(10))
            self.advance(t, hw, state)
            hw.move_to_position.assert_called_once()
            # SDK completion alone does not advance the original measured FK.
            self.advance(t, hw, state)
            hw.move_to_position.assert_called_once()
            state["joints_rad"] = hw.move_to_position.call_args.args[1]
            state["T_B_F"] = matrix(t.kinematics.fk("right", state["joints_rad"]))
            hw.position_move_finished.return_value = False
            self.advance(t, hw, state)
            hw.move_to_position.assert_called_once()
            result = t.preplace.stopped("Receiver exit")
            self.assertEqual(result["status"], "paused")
            self.assertEqual(result["previous_status"], "aligning_preplace")
            self.assertTrue(result["execution_evidence"][0]["command_queued"])
            self.assertFalse(result["execution_evidence"][0]["endpoint_verified"])

    def test_measured_geometry_guard_catches_tilt_within_loose_angle_corridor(self):
        with tempfile.TemporaryDirectory() as directory:
            t, hw, state, grasp = self.fixture(directory, degrees=0.)
            self.sample(t, hw, state, grasp)
            self.assertTrue(t.preplace.done.wait(10))
            self.advance(t, hw, state)
            start = np.asarray(t.preplace.report["T_R_O_waypoints"][0])
            corners = np.asarray(t.preplace.report["mesh_bound_corners_cad_m"])
            goal_y = (corners @ t.goal[:3, :3].T+t.goal[:3, 3])[:, 1]
            options = []
            for axis in ("x", "z"):
                for sign in (-1, 1):
                    pose = start.copy()
                    pose[:3, :3] = Rotation.from_euler(axis, sign*.9, degrees=True).as_matrix() @ start[:3, :3]
                    options.append((np.min((corners @ pose[:3, :3].T+pose[:3, 3])[:, 1]-goal_y), pose))
            actual = min(options, key=lambda item: item[0])[1]
            state["T_B_F"] = actual @ np.linalg.inv(t.cached_grasp)
            with self.assertRaisesRegex(RuntimeError, "geometry"):
                self.advance(t, hw, state)
            hw.move_to_position.assert_called_once()


if __name__ == "__main__":
    unittest.main()
