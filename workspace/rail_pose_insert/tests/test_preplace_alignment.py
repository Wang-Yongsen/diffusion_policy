"""Offline geometry, clearance and bounded-IK tests for visual correction."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import test_above
from rail_preinsert.core import difference, matrix, pose
from rail_receive.logic import load_config
from rail_receive.preplace_alignment import (
    DEFAULT_PREPLACE_ALIGNMENT, MAX_STEPS, plan_preplace_alignment, validate_settings)


class PreplaceAlignmentTests(unittest.TestCase):
    def geometry(self):
        config, k = load_config(HERE/"receive.yaml"), test_above.ToyKinematics()
        config["preplace_alignment"] = dict(DEFAULT_PREPLACE_ALIGNMENT)
        camera = test_above.T([.2, -.1, .05], [.1, -.15, .2])
        goal = camera @ test_above.T([.12, .05, .6], [.2, .1, -.1])
        grasp = test_above.T([.02, -.2, .04], [.1, -.2, .3])
        above = goal.copy(); above[1, 3] += .015
        actual = above.copy(); actual[:3, 3] += [.012, -.004, .008]
        actual[:3, :3] = Rotation.from_rotvec([.02, -.01, .03]).as_matrix() @ actual[:3, :3]
        q = self.joints(k, actual, grasp)
        return config, k, q, grasp, goal, above

    @staticmethod
    def joints(k, rail, grasp):
        return np.asarray(k.ik("right", pose(rail @ np.linalg.inv(grasp), "right"), np.zeros(7)))

    def test_full_se3_correction_preserves_goal_and_uses_updated_grasp(self):
        config, k, q, grasp, goal, above = self.geometry()
        originals = [value.copy() for value in (grasp, goal, above, q)]
        report = plan_preplace_alignment(k, config, q, grasp, goal, above)
        rails, flanges = map(np.asarray, (report["T_R_O_waypoints"], report["T_R_F_waypoints"]))
        np.testing.assert_allclose(rails[0], matrix(k.fk("right", q)) @ grasp, atol=1e-12)
        np.testing.assert_array_equal(rails[-1], above)
        np.testing.assert_allclose(flanges[-1], above @ np.linalg.inv(grasp), atol=1e-12)
        self.assertEqual(len(rails), report["steps"]+1)
        self.assertEqual(len(report["phase_by_step"]), report["steps"])
        self.assertLessEqual(report["steps"], MAX_STEPS)
        for rail, flange, joints in zip(rails, flanges, report["joint_waypoints_rad"]):
            np.testing.assert_allclose(matrix(k.fk("right", joints)), flange, atol=1e-10)
            np.testing.assert_allclose(flange @ grasp, rail, atol=1e-10)
        for unchanged, expected in zip((grasp, goal, above, q), originals):
            np.testing.assert_array_equal(unchanged, expected)
        for flag in ("collision_checked", "force_sensor_used", "force_controlled", "assembly_verified", "automatic_release"):
            self.assertIs(report[flag], False)
        json.dumps(report, allow_nan=False)

    def test_long_rail_tilt_lifts_and_continuous_rotation_keeps_corner_clearance(self):
        config, k, _, grasp, _, _ = self.geometry()
        grasp = test_above.T([.02, -.2, .04])
        goal = test_above.T([.1, .1, .5], [0., 0., -np.pi/2])
        above = goal.copy(); above[1, 3] += .015
        actual = above.copy()
        actual[:3, :3] = Rotation.from_euler("z", -np.pi/2-np.radians(5.)).as_matrix()
        # A realistic rotation about the held midpoint, not the CAD end origin.
        midpoint = np.asarray(config["rail_midpoint_cad_m"])
        actual[:3, 3] += (above[:3, :3]-actual[:3, :3]) @ midpoint
        report = plan_preplace_alignment(k, config, self.joints(k, actual, grasp), grasp, goal, above)
        self.assertGreater(report["lift_m"], .020)
        self.assertLess(report["lift_m"], .022)
        self.assertGreater(report["position_error_m"], .020)
        self.assertLess(report["midpoint_position_error_m"], 1e-12)
        self.assertGreater(report["rotation_chord_margin_m"], 0.)
        self.assertGreaterEqual(report["alignment_min_corner_clearance_bound_m"], .015-1e-12)
        rails = np.asarray(report["T_R_O_waypoints"])
        corners = np.asarray(report["mesh_bound_corners_cad_m"])
        goal_height = (corners @ goal[:3, :3].T+goal[:3, 3])[:, 1]
        for index, phase in enumerate(report["phase_by_step"]):
            start, end = rails[index:index+2]
            if phase == "align":
                relative = Rotation.from_matrix(start[:3, :3].T @ end[:3, :3]).as_rotvec()
                for u in np.linspace(0., 1., 17):
                    rotation = start[:3, :3] @ Rotation.from_rotvec(relative*u).as_matrix()
                    translation = (1-u)*start[:3, 3]+u*end[:3, 3]
                    clearance = (corners @ rotation.T+translation)[:, 1]-goal_height
                    self.assertGreaterEqual(float(clearance.min()), .015-1e-12)
            else:
                np.testing.assert_allclose(start[:3, :3], end[:3, :3], atol=1e-12)
                np.testing.assert_allclose(start[[0, 2], 3], end[[0, 2], 3], atol=1e-12)

    def test_horizontal_correction_needs_no_lift_and_has_bounded_endpoints(self):
        config, k, _, grasp, goal, above = self.geometry()
        actual = above.copy(); actual[0, 3] += .017
        report = plan_preplace_alignment(k, config, self.joints(k, actual, grasp), grasp, goal, above)
        self.assertEqual(report["lift_m"], 0.)
        self.assertEqual(set(report["phase_by_step"]), {"align"})
        self.assertLessEqual(report["max_translation_step_m"], config["preplace_alignment"]["max_step_m"]+1e-12)
        self.assertLessEqual(report["max_rotation_step_rad"], config["preplace_alignment"]["max_rotation_step_rad"]+1e-12)
        self.assertLessEqual(report["max_joint_step_rad"], config["preplace_alignment"]["max_joint_step_rad"])

    def test_yaw_about_base_up_has_zero_vertical_arc_margin(self):
        config, k, _, grasp, goal, above = self.geometry()
        actual = above.copy()
        actual[:3, :3] = Rotation.from_euler("y", np.radians(3.)).as_matrix() @ above[:3, :3]
        report = plan_preplace_alignment(k, config, self.joints(k, actual, grasp), grasp, goal, above)
        self.assertEqual(report["lift_m"], 0.)
        self.assertLess(report["rotation_chord_margin_m"], 1e-15)

    def test_translation_rotation_lift_and_invalid_above_limits_fail_before_ik(self):
        for cause in ("translation", "rotation", "lift", "below", "above_lateral", "above_rotation"):
            config, k, q, grasp, goal, above = self.geometry()
            actual = matrix(k.fk("right", q)) @ grasp
            if cause == "translation": actual[0, 3] += .1
            elif cause == "rotation": actual[:3, :3] = Rotation.from_euler("y", .3).as_matrix() @ above[:3, :3]
            elif cause == "lift": config["preplace_alignment"]["max_lift_m"] = .00001
            elif cause == "below": above[1, 3] = goal[1, 3]-.01
            elif cause == "above_lateral": above[0, 3] += .01
            else: above[:3, :3] = Rotation.from_euler("x", .01).as_matrix() @ goal[:3, :3]
            q = self.joints(k, actual, grasp)
            with self.subTest(cause=cause), patch.object(k, "ik") as ik, self.assertRaises(ValueError):
                plan_preplace_alignment(k, config, q, grasp, goal, above)
            ik.assert_not_called()

    def test_ik_failure_and_branch_jump_abort_the_whole_preplan(self):
        for cause in ("failure", "branch"):
            config, k, q, grasp, goal, above = self.geometry()
            original = k.ik
            def altered(*args):
                if cause == "failure": raise RuntimeError("unreachable")
                result = list(original(*args)); result[-1] = .12
                return result
            with self.subTest(cause=cause), patch.object(k, "ik", side_effect=altered), self.assertRaises(ValueError):
                plan_preplace_alignment(k, config, q, grasp, goal, above)

    def test_total_step_budget_rejects_before_ik(self):
        config, k, q, grasp, goal, above = self.geometry()
        config["preplace_alignment"].update(max_step_m=1e-7, position_tolerance_m=1e-8)
        with patch.object(k, "ik") as ik, self.assertRaisesRegex(ValueError, "planning budget"):
            plan_preplace_alignment(k, config, q, grasp, goal, above)
        ik.assert_not_called()

    def test_planning_timeout_includes_last_ik_call(self):
        config, k, _, grasp, goal, above = self.geometry()
        actual = above.copy(); actual[0, 3] += .001
        q = self.joints(k, actual, grasp)
        with patch("rail_receive.preplace_alignment.time.monotonic", side_effect=[0., 1., 61.]), self.assertRaises(TimeoutError):
            plan_preplace_alignment(k, config, q, grasp, goal, above)

    def test_invalid_mesh_has_no_ik_or_hardware_dependency(self):
        config, k, q, grasp, goal, above = self.geometry()
        with patch("trimesh.load", return_value=type("Mesh", (), {"bounds": np.zeros((2, 3))})()), patch.object(k, "ik") as ik:
            with self.assertRaisesRegex(ValueError, "mesh"):
                plan_preplace_alignment(k, config, q, grasp, goal, above)
            ik.assert_not_called()

    def test_settings_validation_rejects_malformed_and_inconsistent_values(self):
        invalid = [dict(max_correction_m=x) for x in (True, None, "0.04", 0., -.1, np.nan, np.inf)]
        invalid += [dict(enabled=1), dict(samples=True), dict(samples=1), dict(max_corrections=0),
            dict(velocity_ratio=1.1), dict(acceleration_ratio=101), dict(max_step_m=.006),
            dict(max_rotation_step_rad=.1), dict(max_joint_step_rad=.11), dict(max_lift_m=.11),
            dict(position_tolerance_m=.001), dict(rotation_tolerance_rad=.01),
            dict(path_tolerance_m=.0001), dict(path_rotation_tolerance_rad=.001),
            dict(settle_s=20.), dict(total_timeout_s=10.)]
        validate_settings(dict(DEFAULT_PREPLACE_ALIGNMENT))
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_settings(dict(DEFAULT_PREPLACE_ALIGNMENT, **change))


if __name__ == "__main__":
    unittest.main()
