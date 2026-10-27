"""Core offline regression coverage; all actuator and camera owners are mocked."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import support as test_above
from support import T, ToyKinematics
from rail_common.core import (
    difference,
    matrix,
    pose,
)
from rail_common.pose_alignment import RailPoseAlignment
from rail_receive.above import above_targets, plan_above
from rail_receive.logic import load_config
from rail_receive.position_spiral import (
    PositionSpiral,
    plan_position_spiral,
    MAX_STEPS as SPIRAL_MAX_STEPS,
)
from rail_receive.preplace_alignment import (
    plan_preplace_alignment,
    DEFAULT_PREPLACE_ALIGNMENT,
    MAX_STEPS as PREPLACE_MAX_STEPS,
)


def config():
    return load_config(HERE / "receive.yaml")


class AbovePlanningTests(unittest.TestCase):

    def test_above_uses_right_base_up_with_rotated_camera_and_cad(self):
        X = T([0.4, -0.2, 0.1], [0.3, -0.8, 0.4])
        goal = T([0.2, 0.1, 0.8], [-0.5, 0.3, 1.0])
        placed, above = above_targets(X, goal, 0.05)
        np.testing.assert_allclose(placed, X @ goal)
        np.testing.assert_allclose(above[:3, 3] - placed[:3, 3], [0, 0.05, 0], atol=1e-12)
        np.testing.assert_equal(above[:3, :3], placed[:3, :3])
        for height in (0, -0.1, float("nan"), True):
            with self.assertRaises(ValueError):
                above_targets(X, goal, height)

    def test_right_plan_preserves_grasp_and_ends_above_goal_with_joint_limits(self):
        c, k = (load_config(HERE / "receive.yaml"), ToyKinematics())
        grasp = T([0.02, -0.2, 0.04], [0.1, -0.2, 0.3])
        target = T([0.12, 0.05, 0.6], [0.2, 0.1, -0.1])
        q = np.array([0.2, -0.02, 0.5, 0.0, 0.0, 0.0, 0.0])
        path, report = plan_above(k, c, q, grasp, target)
        actual = matrix(k.fk("right", path.points[-1])) @ grasp
        np.testing.assert_allclose(actual, target, atol=1e-09)
        self.assertEqual(len(path.points), 2)
        self.assertEqual(report["control_mode"], "position")
        self.assertEqual(report["intermediate_targets"], 0)
        self.assertEqual(report["interpolation_owner"], "controller")
        self.assertEqual(report["side"], "right")
        self.assertFalse(report["collision_checked"])
        self.assertEqual(report["velocity_ratio"], 3)


class SpiralPlanningTests(unittest.TestCase):

    def geometry(self, start_offset=None):
        config, kinematics = (load_config(HERE / "receive.yaml"), test_above.ToyKinematics())
        grasp = test_above.T([0.02, -0.2, 0.04], [0.1, -0.2, 0.3])
        goal = test_above.T([0.12, 0.05, 0.6], [0.2, 0.1, -0.1])
        above = goal.copy()
        above[1, 3] += 0.05
        actual = above.copy()
        if start_offset is not None:
            actual[:3, 3] += start_offset
        joints = np.asarray(
            kinematics.ik("right", pose(actual @ np.linalg.inv(grasp), "right"), np.zeros(7))
        )
        return (config, kinematics, joints, grasp, goal, above)

    def advance(self, insertion, hardware, state, now):
        with patch(
            "rail_receive.position_spiral.time.monotonic", return_value=now
        ), redirect_stdout(io.StringIO()):
            return insertion.advance(hardware, state)

    def test_geometry_is_right_base_spiral_with_bounded_steps_and_exact_goal(self):
        for offset in (None, [0.002, 0.0, 0.0], [0.001, 0.001, -0.001]):
            with self.subTest(offset=offset):
                config, k, q, grasp, goal, above = self.geometry(offset)
                self.assertEqual(config["position_spiral"]["radius_m"], 0.003)
                report = plan_position_spiral(k, config, q, grasp, goal, above)
                initial = matrix(k.fk("right", q)) @ grasp
                targets = np.asarray(report["rail_targets"])
                path = np.concatenate([initial[None, :3, 3], targets[:, :3, 3]])
                self.assertTrue(
                    np.all(
                        np.linalg.norm(np.diff(path, axis=0), axis=1)
                        <= config["position_spiral"]["max_step_m"] + 1e-12
                    )
                )
                self.assertTrue(np.all(np.diff(path[:, 1]) < 0))
                self.assertTrue(np.all(targets[:, 1, 3] >= goal[1, 3]))
                np.testing.assert_array_equal(targets[-1], goal)
                for i, (rail, joints) in enumerate(
                    zip(targets, report["joint_waypoints_rad"][1:]), 1
                ):
                    np.testing.assert_array_equal(rail[:3, :3], goal[:3, :3])
                    np.testing.assert_allclose(
                        matrix(k.fk("right", joints)) @ grasp, rail, atol=1e-10
                    )
                    u = i / report["steps"]
                    center = (1 - u) * initial[:3, 3] + u * goal[:3, 3]
                    radius = config["position_spiral"]["radius_m"] * np.sin(np.pi * u)
                    theta = 2 * np.pi * config["position_spiral"]["turns"] * u
                    np.testing.assert_allclose(
                        (rail[:3, 3] - center)[[0, 2]],
                        radius * np.array([np.cos(theta), np.sin(theta)]),
                        atol=1e-12,
                    )
                self.assertLessEqual(report["steps"], SPIRAL_MAX_STEPS)
                self.assertFalse(report["force_sensor_used"])
                self.assertFalse(report["force_controlled"])

    def test_all_ik_is_preplanned_and_joint_branch_jumps_fail_before_motion(self):
        config, k, q, grasp, goal, above = self.geometry()
        original = k.ik

        def jump(*args):
            result = list(original(*args))
            result[-1] = 0.1
            return result

        with tempfile.TemporaryDirectory() as directory, patch.object(k, "ik", side_effect=jump):
            insertion = PositionSpiral(config, k, directory, q, grasp, goal, above)
            self.assertTrue(insertion.done.wait(3.0))
            hardware = Mock()
            with self.assertRaisesRegex(ValueError, "joint branch"):
                self.advance(
                    insertion,
                    hardware,
                    dict(joints_rad=q, T_B_F=matrix(k.fk("right", q))),
                    insertion.started,
                )
            hardware.move_to_position.assert_not_called()
            self.assertFalse((Path(directory) / "position_spiral_plan.json").exists())

    def test_wrong_start_geometry_descent_and_rotation_are_rejected(self):
        for cause in ("below_goal", "descent_limit", "lateral", "rotation", "displaced"):
            config, k, q, grasp, goal, above = self.geometry()
            if cause == "below_goal":
                goal[1, 3] = above[1, 3] + 0.001
            elif cause == "descent_limit":
                config["position_spiral"]["max_descent_m"] = 0.01
            elif cause == "lateral":
                goal[0, 3] += 0.001
            elif cause == "rotation":
                goal[:3, :3] = test_above.T(rpy=[0.2, 0.0, 0.0])[:3, :3]
            elif cause == "displaced":
                q[0] += 0.01
            with self.subTest(cause=cause), self.assertRaises(ValueError):
                plan_position_spiral(k, config, q, grasp, goal, above)


class PreplaceAlignmentTests(unittest.TestCase):

    def geometry(self):
        config, k = (load_config(HERE / "receive.yaml"), test_above.ToyKinematics())
        config["preplace_alignment"] = dict(DEFAULT_PREPLACE_ALIGNMENT)
        camera = test_above.T([0.2, -0.1, 0.05], [0.1, -0.15, 0.2])
        goal = camera @ test_above.T([0.12, 0.05, 0.6], [0.2, 0.1, -0.1])
        grasp = test_above.T([0.02, -0.2, 0.04], [0.1, -0.2, 0.3])
        above = goal.copy()
        above[1, 3] += 0.015
        actual = above.copy()
        actual[:3, 3] += [0.012, -0.004, 0.008]
        actual[:3, :3] = Rotation.from_rotvec([0.02, -0.01, 0.03]).as_matrix() @ actual[:3, :3]
        q = self.joints(k, actual, grasp)
        return (config, k, q, grasp, goal, above)

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
        self.assertEqual(len(rails), report["steps"] + 1)
        self.assertEqual(len(report["phase_by_step"]), report["steps"])
        self.assertLessEqual(report["steps"], PREPLACE_MAX_STEPS)
        for rail, flange, joints in zip(rails, flanges, report["joint_waypoints_rad"]):
            np.testing.assert_allclose(matrix(k.fk("right", joints)), flange, atol=1e-10)
            np.testing.assert_allclose(flange @ grasp, rail, atol=1e-10)
        for unchanged, expected in zip((grasp, goal, above, q), originals):
            np.testing.assert_array_equal(unchanged, expected)
        for flag in (
            "collision_checked",
            "force_sensor_used",
            "force_controlled",
            "assembly_verified",
            "automatic_release",
        ):
            self.assertIs(report[flag], False)
        json.dumps(report, allow_nan=False)

    def test_long_rail_tilt_lifts_and_continuous_rotation_keeps_corner_clearance(self):
        config, k, _, grasp, _, _ = self.geometry()
        grasp = test_above.T([0.02, -0.2, 0.04])
        goal = test_above.T([0.1, 0.1, 0.5], [0.0, 0.0, -np.pi / 2])
        above = goal.copy()
        above[1, 3] += 0.015
        actual = above.copy()
        actual[:3, :3] = Rotation.from_euler("z", -np.pi / 2 - np.radians(5.0)).as_matrix()
        midpoint = np.asarray(config["rail_midpoint_cad_m"])
        actual[:3, 3] += (above[:3, :3] - actual[:3, :3]) @ midpoint
        report = plan_preplace_alignment(
            k, config, self.joints(k, actual, grasp), grasp, goal, above
        )
        self.assertGreater(report["lift_m"], 0.02)
        self.assertLess(report["lift_m"], 0.022)
        self.assertGreater(report["position_error_m"], 0.02)
        self.assertLess(report["midpoint_position_error_m"], 1e-12)
        self.assertGreater(report["rotation_chord_margin_m"], 0.0)
        self.assertGreaterEqual(report["alignment_min_corner_clearance_bound_m"], 0.015 - 1e-12)
        rails = np.asarray(report["T_R_O_waypoints"])
        corners = np.asarray(report["mesh_bound_corners_cad_m"])
        goal_height = (corners @ goal[:3, :3].T + goal[:3, 3])[:, 1]
        for index, phase in enumerate(report["phase_by_step"]):
            start, end = rails[index : index + 2]
            if phase == "align":
                relative = Rotation.from_matrix(start[:3, :3].T @ end[:3, :3]).as_rotvec()
                for u in np.linspace(0.0, 1.0, 17):
                    rotation = start[:3, :3] @ Rotation.from_rotvec(relative * u).as_matrix()
                    translation = (1 - u) * start[:3, 3] + u * end[:3, 3]
                    clearance = (corners @ rotation.T + translation)[:, 1] - goal_height
                    self.assertGreaterEqual(float(clearance.min()), 0.015 - 1e-12)
            else:
                np.testing.assert_allclose(start[:3, :3], end[:3, :3], atol=1e-12)
                np.testing.assert_allclose(start[[0, 2], 3], end[[0, 2], 3], atol=1e-12)

    def test_large_total_residual_is_corrected_in_bounded_segments(self):
        config, k, _, grasp, goal, above = self.geometry()
        actual = above.copy()
        actual[:3, 3] += [-0.04855, 0.01075, 0.00082]
        actual[:3, :3] = Rotation.from_euler("y", 11.0, degrees=True).as_matrix() @ above[:3, :3]
        report = plan_preplace_alignment(
            k, config, self.joints(k, actual, grasp), grasp, goal, above
        )
        self.assertAlmostEqual(report["position_error_m"] * 1000, 49.74, delta=0.01)
        self.assertAlmostEqual(np.degrees(report["rotation_error_rad"]), 11.0)
        np.testing.assert_allclose(report["T_R_O_waypoints"][-1], above, atol=1e-10)
        self.assertLessEqual(report["max_translation_step_m"], 0.002 + 1e-12)
        self.assertLessEqual(report["max_rotation_step_rad"], np.radians(0.5) + 1e-12)
        self.assertLessEqual(report["max_joint_step_rad"], 0.05)

    def test_lift_and_invalid_above_limits_fail_before_ik(self):
        for cause in ("lift", "below", "above_lateral", "above_rotation"):
            config, k, q, grasp, goal, above = self.geometry()
            actual = matrix(k.fk("right", q)) @ grasp
            if cause == "lift":
                config["preplace_alignment"]["max_lift_m"] = 1e-05
            elif cause == "below":
                above[1, 3] = goal[1, 3] - 0.01
            elif cause == "above_lateral":
                above[0, 3] += 0.01
            else:
                above[:3, :3] = Rotation.from_euler("x", 0.01).as_matrix() @ goal[:3, :3]
            q = self.joints(k, actual, grasp)
            with self.subTest(cause=cause), patch.object(k, "ik") as ik, self.assertRaises(
                ValueError
            ):
                plan_preplace_alignment(k, config, q, grasp, goal, above)
            ik.assert_not_called()

    def test_ik_failure_and_branch_jump_abort_the_whole_preplan(self):
        for cause in ("failure", "branch"):
            config, k, q, grasp, goal, above = self.geometry()
            original = k.ik

            def altered(*args):
                if cause == "failure":
                    raise RuntimeError("unreachable")
                result = list(original(*args))
                result[-1] = 0.12
                return result

            with self.subTest(cause=cause), patch.object(
                k, "ik", side_effect=altered
            ), self.assertRaises(ValueError):
                plan_preplace_alignment(k, config, q, grasp, goal, above)


def degrees_transform(xyz=(0, 0, 0), degrees=(0, 0, 0)):
    return T(xyz, np.radians(degrees))


class AlignmentTests(unittest.TestCase):

    def setUp(self):
        self.config = load_config(HERE / "receive.yaml")
        self.alignment = RailPoseAlignment.from_config(self.config)
        self.flip = self.alignment.candidates[1]

    def test_flipped_endpoint_and_rotation_recover_same_pose_and_center(self):
        reference = degrees_transform([0.2, -0.1, 0.7], [19, -32, 64])
        actual = reference @ degrees_transform([0.1, -0.2, 0.03], [11, -17, 23])
        raw = actual @ self.flip
        aligned, info = self.alignment.align(raw, reference)
        np.testing.assert_allclose(aligned, actual, atol=1e-12)
        self.assertEqual(info["hypothesis"], "cad_z_half_turn")
        self.assertGreater(difference(raw, actual)[0], 0.479)
        center = np.r_[self.alignment.center, 1.0]
        np.testing.assert_allclose(raw @ center, aligned @ center, atol=1e-12)
        rotation_only = raw.copy()
        rotation_only[:3, :3] = aligned[:3, :3]
        self.assertGreater(difference(rotation_only, actual)[0], 0.479)

    def test_excess_on_any_axis_is_rejected_without_clamping(self):
        for angles in ([45.01, 0, 0], [0, -45.01, 0], [0, 0, 45.01], [0, 90, 0]):
            with self.subTest(angles=angles):
                raw = degrees_transform(degrees=angles)
                with self.assertRaisesRegex(ValueError, "per-axis"):
                    self.alignment.align(raw, np.eye(4))
                candidate, report = self.alignment.align(raw, np.eye(4), enforce_limit=False)
                self.assertFalse(report["within_limit"])
                self.assertGreaterEqual(max((abs(v) for v in report["relative_xyz_deg"])), 45)
