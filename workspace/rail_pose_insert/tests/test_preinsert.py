from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1]), str(HERE.parents[2])]

from rail_preinsert.core import (difference, interpolate_joints, load_calibration, load_config,
                                digest, matrix, mean_pose, targets, transform, validate_scene, write_json)
from rail_preinsert.geometry import RobotGeometry
from rail_preinsert.hardware import Hardware
from rail_preinsert.planner import JointPath, Planner
from rail_preinsert.workflow import check_plan_inputs, hand_references, observation_poses, verify_preinsert
from diffusion_policy.third_party.bimanual_teleop.types import Pose, Submission


def T(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    value = np.eye(4)
    value[:3, 3] = xyz
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    return value


class TransformTests(unittest.TestCase):
    def test_noncommuting_grasp_chain_and_conjugated_motion_agree(self):
        X, initial, goal, flange = T([.3, -.7, .1], [.2, -.5, .8]), T([.1, .2, .8], [.4, .6, -.3]), T([.2, .3, .9], [-.1, .4, .7]), T([.5, -.1, .2], [.3, -.8, .2])
        result = targets(X, goal, initial, flange, [0, 0, 1])
        np.testing.assert_allclose(flange @ result["T_F_O"], X @ initial)
        np.testing.assert_allclose(result["T_B_F_preinsert"] @ result["T_F_O"], result["T_B_O_preinsert"])
        delta_C = np.linalg.inv(X) @ result["T_B_O_preinsert"] @ np.linalg.inv(initial)
        np.testing.assert_allclose(X @ delta_C @ np.linalg.inv(X) @ flange, result["T_B_F_preinsert"])

    def test_preinsert_offset_uses_target_axis_not_base_z(self):
        result = targets(np.eye(4), T(rpy=[np.pi/2, 0, 0]), np.eye(4), np.eye(4), [0, 0, 1])
        np.testing.assert_allclose(result["T_B_O_preinsert"][:3, 3], [0, -.02, 0], atol=1e-12)

    def test_pose_rejects_reflection_nonfinite_and_bad_homogeneous_row(self):
        for bad in [np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan), np.zeros((4, 4))]:
            with self.assertRaises(ValueError):
                transform(bad)

    def test_rotation_averaging_handles_quaternion_sign_and_angle_wrap(self):
        poses = [T(rpy=[0, 0, x]) for x in [np.pi-.005, -np.pi+.005]]
        mean = mean_pose(poses)
        self.assertLess(difference(mean, T(rpy=[0, 0, np.pi]))[1], 1e-6)
        with self.assertRaises(ValueError):
            mean_pose([np.eye(4), T(rpy=[0, 0, np.pi])])

    def test_state_alignment_interpolates_but_never_extrapolates_or_crosses_gap(self):
        history = [(1_000_000_000, [0.]*7), (1_020_000_000, [1.]*7)]
        np.testing.assert_allclose(interpolate_joints(history, 1_010_000_000), [.5]*7)
        for stamp in [999_000_000, 1_030_000_000]:
            with self.assertRaises(ValueError):
                interpolate_joints(history, stamp)
        with self.assertRaises(ValueError):
            interpolate_joints([(0, [0.]*7), (1_000_000_000, [1.]*7)], 500_000_000)

    def test_synthetic_calibration_rejected_by_live_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            write_json(path, {"test_data": True, "parent_frame": "tianji_left_base", "child_frame": "head_color_optical",
                              "unit": "m", "camera_serial": "fake", "T_B_C": np.eye(4).tolist()})
            with self.assertRaises(ValueError):
                load_calibration(path)
            _, result = load_calibration(path, allow_test_data=True)
            np.testing.assert_equal(result, np.eye(4))

    def test_actual_candidate_joint_order_and_limits_are_valid(self):
        values = hand_references(load_config(HERE / "config.yaml"))
        self.assertEqual(values["open"].shape, (20,))
        self.assertGreater(np.linalg.norm(values["open"] - values["grasp"]), .5)


class ToyKinematics:
    def __init__(self):
        self.model = NS(arm=lambda side: NS(lower_rad=(-1.,)*7, upper_rad=(1.,)*7), digest="test-model")

    def fk(self, side, q):
        return Pose("tianji_left_base", "tianji_left_flange", tuple(q[:3]), (0., 0., 0., 1.))

    def ik(self, side, target, seed):
        return (*target.position_m, 0., 0., 0., 0.)


class PlanningTests(unittest.TestCase):
    def test_quintic_trajectory_respects_velocity_acceleration_and_endpoint(self):
        points = np.array([[0.]*7, [.4, -.2, .1, 0, 0, 0, 0], [.2, .1, .1, 0, 0, 0, 0]])
        path = JointPath(points, .12, .25)
        stamps = np.linspace(0, path.duration, 10001)
        q = np.array([path.at(t) for t in stamps])
        velocity = np.gradient(q, stamps, axis=0)
        acceleration = np.gradient(velocity, stamps, axis=0)
        self.assertLessEqual(abs(velocity).max(), .12001)
        self.assertLessEqual(abs(acceleration).max(), .2501)
        np.testing.assert_equal(path.at(0), points[0])
        np.testing.assert_equal(path.at(path.duration + 1), points[-1])

    def test_nonfinite_trajectory_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            JointPath([np.zeros(7), np.ones(7)], float("nan"), .25)

    def test_rrt_routes_around_obstacle_and_checks_segment_interiors(self):
        world = NS(last_failure=None)
        world.check = lambda q: not (.1 < q[0] < .3 and abs(q[1]) < .15)
        config = load_config(HERE / "config.yaml")
        config.update(edge_resolution_rad=.02, planning_timeout_s=5, rrt_seed=7)
        planner = Planner(config, ToyKinematics(), world)
        start, goal = np.zeros(7), T([.4, 0, 0])
        self.assertFalse(planner.edge(start, np.r_[.4, np.zeros(6)]))
        path = planner.plan(start, goal)
        self.assertGreater(len(path.points), 2)
        for a, b in zip(path.points, path.points[1:]):
            for u in np.linspace(0, 1, 200):
                self.assertTrue(world.check(a*(1-u) + b*u))

    def test_collision_at_goal_is_rejected(self):
        world = NS(last_failure={"parts": ["rail", "pin"]}, check=lambda q: q[0] < .2)
        planner = Planner(load_config(HERE / "config.yaml"), ToyKinematics(), world)
        with self.assertRaises(ValueError):
            planner.plan(np.zeros(7), T([.4, 0, 0]))

    def test_native_base_registration_matches_official_fk_away_from_zero(self):
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics
        k = TianjiKinematics(); geometry = RobotGeometry(load_config(HERE / "config.yaml"), k)
        q = np.zeros(54); q[:7] = np.radians([18, -50, -28, -40, 25, 8, 3])
        predicted = geometry.poses(q)["TCP_Link_L"]
        error = difference(predicted, matrix(k.fk("left", tuple(q[:7]))))
        self.assertLess(error[0], .0001)
        self.assertLess(error[1], .0002)


class RuntimeTests(unittest.TestCase):
    def hardware(self, motion=True):
        config = load_config(HERE / "config.yaml")
        config["state_max_age_s"] = 1.
        driver = Mock(); driver.submit.side_effect = lambda c: Submission(c.command_id, True)
        left = Mock(); left.device_id = "fake-hand"
        left.health.return_value = NS(ready=True)
        left.submit.side_effect = lambda c: Submission(c.command_id, True)
        hardware = Hardware(config, ToyKinematics(), Mock(), motion=motion, devices=(Mock(), driver, {"left": left, "right": Mock()}))
        hardware.arm_enabled = hardware.hand_enabled = True
        hardware.arm_target = np.zeros(7)
        hardware.hand_target = np.full(20, .2)
        def snapshot():
            full = np.r_[hardware.arm_target, np.zeros(7), np.full(40, .2)]
            return {"full_joints": full, "timestamp_ns": time.monotonic_ns(), "T_B_F": matrix(hardware.kinematics.fk("left", tuple(full[:7])))}
        hardware.snapshot = snapshot
        return hardware, driver, left

    def test_read_only_cannot_enable_motion(self):
        hardware, driver, hand = self.hardware(motion=False)
        with self.assertRaises(RuntimeError):
            hardware.engage()
        driver.engage.assert_not_called(); hand.engage.assert_not_called()

    def test_pause_keeps_grasp_and_stops_arm_without_reopening_hand(self):
        hardware, driver, hand = self.hardware()
        thread = threading.Thread(target=hardware._hand_loop)
        thread.start()
        try:
            time.sleep(.03)
            hardware.pause("operator")
            count = hand.submit.call_count
            time.sleep(.04)
            self.assertGreater(hand.submit.call_count, count)
            np.testing.assert_equal(hand.submit.call_args.args[0].payload.position_rad, np.full(20, .2))
            driver.request_hold.assert_called_once()
            hand.disable.assert_not_called()
        finally:
            hardware.stop.set(); thread.join(timeout=1)

    def test_transport_endpoint_continues_hold_and_does_not_start_insertion(self):
        hardware, driver, _ = self.hardware()
        path = JointPath([np.zeros(7), np.full(7, .002)], .12, .25)
        reference = hardware.snapshot()["full_joints"]
        hardware.start_path(path, reference)
        thread = threading.Thread(target=hardware._arm_loop); thread.start()
        try:
            deadline = time.monotonic() + 1
            while not hardware.trajectory_done and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(hardware.trajectory_done)
            count = driver.submit.call_count
            time.sleep(.025)
            self.assertGreater(driver.submit.call_count, count)
            np.testing.assert_allclose(driver.submit.call_args.args[0].payload.targets["left"], path.points[-1])
            self.assertFalse(hardware.paused, hardware.error)
        finally:
            hardware.stop.set(); thread.join(timeout=1)

    def test_cannot_execute_plan_after_right_arm_or_hand_moves(self):
        hardware, _, _ = self.hardware()
        reference = hardware.snapshot()["full_joints"]
        reference[7] = .1
        with self.assertRaises(ValueError):
            hardware.start_path(JointPath([np.zeros(7), np.full(7, .002)], .12, .25), reference)

    def test_cleanup_attempts_all_devices_even_if_one_close_fails(self):
        hardware, driver, left = self.hardware()
        left.close.side_effect = RuntimeError("left hand disable unconfirmed")
        with self.assertRaises(RuntimeError):
            hardware.close()
        hardware.hands["right"].close.assert_called_once()
        driver.close.assert_called_once(); hardware.session.close.assert_called_once()


class EndpointTests(unittest.TestCase):
    def test_preinsert_visual_verification_rejects_low_or_tilted_rail(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(HERE / "config.yaml")
            config["calibration"] = str(Path(directory) / "calibration.json")
            config["visual_position_tolerance_m"] = .1
            config["visual_rotation_tolerance_rad"] = .2
            write_json(config["calibration"], {"test_data": True, "parent_frame": "tianji_left_base",
                "child_frame": "head_color_optical", "camera_serial": "test", "unit": "m", "T_B_C": np.eye(4).tolist()})
            goal, pre = T([.5, .3, .8]), T([.5, .3, .82])
            report = {"targets": {"T_B_O_goal": goal.tolist(), "T_B_O_preinsert": pre.tolist(), "outward_axis_B": [0, 0, 1]}}
            observation = {"test_data": True, "camera_serial": "test", "unit": "m", "frame": "T_C_CAD",
                "mesh_sha256": digest(config["mesh"]), "records": [{"valid": True, "timestamp_ns": 1, "T_C_CAD": pre.tolist()}]}
            result = verify_preinsert(config, report, observation, allow_test_data=True)
            self.assertEqual(result["status"], "preinsert_reached")
            self.assertFalse(result["insertion_executed"])
            low = pre.copy(); low[2, 3] -= .006
            observation["records"][0]["T_C_CAD"] = low.tolist()
            self.assertEqual(verify_preinsert(config, report, observation, allow_test_data=True)["status"], "preinsert_unverified")
            # Positive X rotation tips the far CAD-Y end down for a negative angle.
            tilted = T(pre[:3, 3], [-.05, 0, 0])
            observation["records"][0]["T_C_CAD"] = tilted.tolist()
            self.assertEqual(verify_preinsert(config, report, observation, allow_test_data=True)["status"], "preinsert_unverified")

    def test_replay_pose_cannot_be_used_as_live_observation(self):
        config = load_config(HERE / "config.yaml")
        observation = {"replay_only": True, "test_data": False}
        with self.assertRaisesRegex(ValueError, "fresh live"):
            observation_poses(observation, {"camera_serial": "test"}, config["mesh"])

    def test_unfilled_scene_template_is_rejected_before_device_start(self):
        with self.assertRaises(ValueError):
            validate_scene(HERE / "calibrations/scene.example.json")

    def test_changed_geometry_invalidates_saved_execution_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"joint_speed_rad_s": .12, "joint_acceleration_rad_s2": .25}
            report = {"test_data": False, "end_stage": "preinsert", "joint_speed_rad_s": .12, "joint_acceleration_rad_s2": .25}
            for key in ("calibration", "target", "scene", "mesh", "urdf", "manifest"):
                path = Path(directory) / key; path.write_text("original")
                config[key] = str(path); report[f"{key}_sha256"] = digest(path)
            check_plan_inputs(config, report)
            Path(config["scene"]).write_text("changed")
            with self.assertRaises(ValueError):
                check_plan_inputs(config, report)


if __name__ == "__main__":
    unittest.main()
