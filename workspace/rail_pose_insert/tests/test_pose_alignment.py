"""End-for-end ambiguity, offset CAD origin, and per-axis task bounds; no I/O to devices."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace as NS

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

from rail_preinsert.core import difference, mean_pose
from rail_preinsert.pose_alignment import RailPoseAlignment, relative_rpy_deg
from rail_preinsert.workflow import observation_poses, check_plan_inputs
from rail_receive.above import plan_above
from rail_receive.logic import load_config
from receive_perception_worker import pose_reference
import receive_perception_worker as worker
import test_above as above_tests


def T(xyz=(0, 0, 0), degrees=(0, 0, 0)):
    result = np.eye(4)
    result[:3, 3] = xyz
    result[:3, :3] = Rotation.from_euler("xyz", degrees, degrees=True).as_matrix()
    return result


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(HERE / "receive.yaml")
        self.alignment = RailPoseAlignment.from_config(self.config)
        self.flip = self.alignment.candidates[1]

    def test_flipped_endpoint_and_rotation_recover_same_pose_and_center(self):
        reference = T([.2, -.1, .7], [19, -32, 64])
        actual = reference @ T([.1, -.2, .03], [11, -17, 23])
        raw = actual @ self.flip
        aligned, info = self.alignment.align(raw, reference)
        np.testing.assert_allclose(aligned, actual, atol=1e-12)
        self.assertEqual(info["hypothesis"], "cad_z_half_turn")
        self.assertGreater(difference(raw, actual)[0], .479)
        center = np.r_[self.alignment.center, 1.]
        np.testing.assert_allclose(raw @ center, aligned @ center, atol=1e-12)
        rotation_only = raw.copy(); rotation_only[:3, :3] = aligned[:3, :3]
        self.assertGreater(difference(rotation_only, actual)[0], .479)

    def test_each_axis_40_degrees_is_allowed_even_when_total_exceeds_45(self):
        actual = T([.1, .2, .7], [40, 40, 40])
        aligned, info = self.alignment.align(actual @ self.flip, np.eye(4))
        np.testing.assert_allclose(aligned, actual, atol=1e-12)
        np.testing.assert_allclose(info["relative_xyz_deg"], [40, 40, 40], atol=1e-12)
        self.assertGreater(info["rotation_error_deg"], 45)
        self.assertTrue(info["within_limit"])
        self.assertFalse(info["within_typical_range"])
        self.alignment.check_motion(actual, np.eye(4))

    def test_bounds_are_target_relative_and_invariant_to_camera_or_base_frame(self):
        reference, relative = T([.3, .2, .8], [58, -34, 27]), T(degrees=[-29, 12, 24])
        parent = T([-.2, .5, .1], [-64, 38, 99])
        aligned, report = self.alignment.align(parent @ reference @ relative @ self.flip, parent @ reference)
        np.testing.assert_allclose(relative_rpy_deg(parent @ reference, aligned), [-29, 12, 24], atol=1e-12)
        self.assertTrue(report["within_typical_range"])

    def test_excess_on_any_axis_is_rejected_without_clamping(self):
        for angles in ([45.01, 0, 0], [0, -45.01, 0], [0, 0, 45.01], [0, 90, 0]):
            with self.subTest(angles=angles):
                raw = T(degrees=angles)
                with self.assertRaisesRegex(ValueError, "per-axis"):
                    self.alignment.align(raw, np.eye(4))
                candidate, report = self.alignment.align(raw, np.eye(4), enforce_limit=False)
                self.assertFalse(report["within_limit"])
                self.assertGreaterEqual(max(abs(v) for v in report["relative_xyz_deg"]), 45)

    def test_roll_over_is_not_misidentified_as_end_for_end(self):
        for angles in ([180, 0, 0], [0, 180, 0]):
            with self.assertRaises(ValueError):
                self.alignment.align(T(degrees=angles), np.eye(4))

    def test_raw_records_are_retained_and_repeated_alignment_is_idempotent(self):
        actual = T([.1, -.2, .8], [15, -21, 19])
        raw = {"timestamp_ns": 123, "T_C_CAD": (actual @ self.flip).tolist()}
        original = copy.deepcopy(raw)
        aligned = self.alignment.align_record(raw, np.eye(4))
        repeated = self.alignment.align_record(aligned, np.eye(4))
        self.assertEqual(raw, original)
        self.assertEqual(repeated, aligned)
        np.testing.assert_allclose(aligned["T_C_CAD"], actual, atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(aligned["quaternion_xyzw"]).as_matrix(), actual[:3, :3])

    def test_stationary_average_aligns_branches_but_still_rejects_real_translation_drift(self):
        actual = T([.1, .2, .8], [10, -12, 14])
        observation = {"unit": "m", "frame": "T_C_CAD", "camera_serial": "test",
            "mesh_sha256": self.config["pose_alignment"]["mesh_sha256"], "records": [
                {"valid": True, "timestamp_ns": i+1, "T_C_CAD": pose.tolist()}
                for i, pose in enumerate([actual, actual @ self.flip, actual])]}
        records, poses = observation_poses(observation, {"camera_serial": "test"}, self.config["mesh"],
            allow_test_data=True, alignment=self.alignment, reference=np.eye(4))
        np.testing.assert_allclose(mean_pose(poses), actual, atol=1e-12)
        self.assertEqual(records[1]["pose_alignment"]["hypothesis"], "cad_z_half_turn")
        observation["records"][1]["T_C_CAD"][0][3] += .03
        with self.assertRaisesRegex(ValueError, "Stationary"):
            observation_poses(observation, {"camera_serial": "test"}, self.config["mesh"],
                allow_test_data=True, alignment=self.alignment, reference=np.eye(4))

    def test_wrong_mesh_and_invalid_angle_policy_fail(self):
        with self.assertRaisesRegex(ValueError, "different CAD"):
            RailPoseAlignment(self.config["pose_alignment"], "wrong")
        for value in (True, float("nan"), 0, 46):
            settings = dict(self.config["pose_alignment"], max_axis_rotation_deg=value)
            with self.assertRaises(ValueError):
                RailPoseAlignment(settings, settings["mesh_sha256"])

    def test_preview_uses_existing_goal_and_falls_back_only_when_no_matching_reference(self):
        target = json.loads((HERE / "targets/rail_goal.json").read_text())
        reference, error = pose_reference(self.config, target["camera_serial"])
        self.assertIsNone(error)
        np.testing.assert_allclose(reference, target["T_C_O_goal"])
        reference, error = pose_reference(self.config, "different_camera")
        self.assertIsNone(reference)
        self.assertIsNotNone(error)

    def test_old_plan_cannot_bypass_new_alignment_policy(self):
        report = {"schema_version": 2, "collision_checked": False, "end_stage": "preinsert"}
        with self.assertRaisesRegex(ValueError, "alignment policy"):
            check_plan_inputs(self.config, report)

    def run_worker(self, actual, with_reference=True):
        from dataclasses import make_dataclass
        Frame = make_dataclass("Frame", ["rgb", "K", "timestamp_ns", "mask", "metadata"])
        frame = Frame(np.zeros((10, 10, 3), np.uint8), np.eye(3), time.monotonic_ns(), None,
                      {"timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic"})
        raw = actual @ self.flip
        camera, detector, engine = Mock(), Mock(), Mock()
        camera.frame.return_value = frame
        engine.infer.return_value = raw, {"valid": True, "T_C_CAD": raw.tolist()}
        config = dict(self.config, pose_preview={"T_C_O_goal": np.eye(4).tolist()})
        if not with_reference:
            config.pop("pose_preview")  # Stored goal has a different camera than this mock.
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "request.json").write_text(json.dumps({"id": "first", "command": "initialize"}))
            (directory / "context.json").write_text(json.dumps({"enabled": True, "epoch": 0, "mode": "held"}))
            def detect(frame):
                (directory / "request.json").write_text(json.dumps({"id": "exit", "command": "shutdown"}))
                return np.ones((10, 10), bool), {"valid": True}
            detector.detect.side_effect = detect
            args = NS(serve=directory, camera_serial="test", mesh=Path(config["mesh"]),
                      foundationpose_root=Path(config["foundationpose_root"]))
            with patch.object(worker, "Camera", return_value=camera), \
                    patch("rail_receive.sam_prompt.SamPromptDetector", return_value=detector), \
                    patch.object(worker, "pose_engine", return_value=engine), \
                    patch("rail_receive.visualization.save_preview") as preview, \
                    patch("rail_receive.visualization.save_mask_snapshot"), \
                    patch.object(worker, "save_frame", return_value=(None, 1)):
                worker.serve(args, config)
            latest = json.loads((directory / "latest.json").read_text())
            return latest, preview, engine

    def test_worker_publishes_and_draws_the_same_aligned_pose(self):
        actual = T([.1, .2, .8], [10, 20, 30])
        latest, preview, engine = self.run_worker(actual)
        np.testing.assert_allclose(latest["T_C_CAD"], actual, atol=1e-12)
        np.testing.assert_allclose(latest["T_C_CAD_raw"], actual @ self.flip, atol=1e-12)
        np.testing.assert_allclose(preview.call_args.args[3], actual, atol=1e-12)
        self.assertIsNone(preview.call_args.args[4])  # No raw/goal/above overlays.
        np.testing.assert_allclose(engine.infer.return_value[0], actual @ self.flip, atol=1e-12)

    def test_worker_does_not_draw_rejected_or_unreferenced_poses(self):
        for actual, with_reference in ((T(degrees=[46, 0, 0]), True), (T(), False)):
            with self.subTest(with_reference=with_reference):
                latest, preview, _ = self.run_worker(actual, with_reference)
                self.assertFalse(latest["valid"])
                self.assertNotIn("pose_snapshot", latest)
                for call in preview.call_args_list:
                    self.assertIsNone(call.args[3])


class TransferAlignmentTests(unittest.TestCase):
    def fixture(self, directory):
        return above_tests.AboveTests().fixture(directory)

    def feed(self, transfer, hw, actual):
        stamp = time.monotonic_ns()
        hw.history = [(stamp-1_000_000, np.zeros(7)), (stamp+1_000_000, np.zeros(7))]
        record = above_tests.AboveTests().record(transfer, stamp,
            np.linalg.inv(transfer.geometry.T_B_C) @ actual)
        return transfer.advance(hw, {"joints_rad": np.zeros(7), "T_B_F": np.eye(4)}, record)

    def test_alternating_flips_plan_only_the_small_physical_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hw = self.fixture(directory)
            transfer.begin(); transfer.needs_reinitialize = False
            actual = transfer.goal @ T([.03, .02, -.04], [12, -8, 17])
            self.feed(transfer, hw, actual)
            self.feed(transfer, hw, actual @ transfer.alignment.candidates[1])
            self.assertTrue(transfer.planning_done.wait(timeout=2))
            self.assertIsNone(transfer.planning_error)
            np.testing.assert_allclose(transfer.report["T_F_O"], actual, atol=1e-12)
            np.testing.assert_allclose(transfer.report["T_R_O_initial"], actual, atol=1e-12)
            self.assertEqual(transfer.report["pose_alignment"][1]["hypothesis"], "cad_z_half_turn")
            held = json.loads((Path(directory) / "held_observation.json").read_text())
            self.assertIn("T_C_CAD_raw", held["records"][1])
            hw.move_to_position.assert_not_called()

    def test_out_of_range_observation_never_reaches_planner_or_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hw = self.fixture(directory)
            transfer.begin(); transfer.needs_reinitialize = False
            with patch("rail_receive.above.plan_above") as planner:
                with self.assertRaisesRegex(ValueError, "per-axis"):
                    self.feed(transfer, hw, transfer.goal @ T(degrees=[46, 0, 0]))
                planner.assert_not_called()
            hw.move_to_position.assert_not_called()

    def test_verification_aligns_fresh_flipped_pose_without_hiding_position_error(self):
        for offset, expected in ((0, "above_target"), (.02, "above_target_unverified")):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as directory:
                transfer, hw = self.fixture(directory)
                transfer.status = "verifying_above"
                transfer.report = {"T_R_F_above": np.eye(4)}
                transfer.not_before_ns = time.monotonic_ns()-100_000_000
                actual = transfer.above.copy(); actual[0, 3] += offset
                self.feed(transfer, hw, actual @ transfer.alignment.candidates[1])
                self.feed(transfer, hw, actual)
                self.assertEqual(transfer.status, expected)
                result = json.loads((Path(directory) / "result.json").read_text())
                self.assertAlmostEqual(result["position_error_m"], offset)
                self.assertLess(result["rotation_error_rad"], 1e-10)

    def test_final_planning_guard_catches_arm_motion_after_observation(self):
        config, kin = load_config(HERE / "receive.yaml"), above_tests.ToyKinematics()
        q = np.zeros(7); q[3] = np.radians(46)
        with self.assertRaisesRegex(ValueError, "per-axis"):
            plan_above(kin, config, q, np.eye(4), np.eye(4))


if __name__ == "__main__":
    unittest.main()
