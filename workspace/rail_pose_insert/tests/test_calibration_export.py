"""Verify the calibration boundary against real URDF/SDK frames without devices."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

from rail_preinsert.core import digest, load_calibration, load_config, matrix, write_json
from tools.export_head_calibration import export
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics


class CalibrationExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "external.json"
        self.output = Path(self.directory.name) / "head.json"
        self.config = load_config(HERE / "config.yaml")

    def test_export_maps_camera_points_into_sdk_base_at_nonzero_joint_poses(self):
        from rail_preinsert.geometry import RobotGeometry
        kinematics = TianjiKinematics()
        geometry = RobotGeometry(self.config, kinematics)
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler("xyz", [.3, -.2, .6]).as_matrix()
        camera[:3, 3] = [.7, -.4, 1.2]
        base = geometry.robot.base_pose("left")
        report = {
            "schema": "external_camera_result_v1", "passed": True,
            "quality": {"passed": True}, "side": "left", "reference_frame": "project_world",
            "camera": {"serial": "unit-test-only"}, "urdf_sha256": digest(self.config["urdf"]),
            "wrist_calibration_sha256": "unit-test-only", "T_world_robot_base": base.tolist(),
            "T_world_camera_opencv": camera.tolist(),
            "T_base_camera_opencv": (np.linalg.inv(base) @ camera).tolist(),
        }
        write_json(self.source, report)
        with patch("rail_preinsert.geometry.RobotGeometry", return_value=geometry):
            export(self.config, self.source, self.output, "unit-test-only", kinematics)
        _, X = load_calibration(self.output)
        # A point rigid to the flange must agree through camera/world and SDK FK.
        q = np.zeros(54)
        q[:7] = np.radians([30, -60, -34, -52, 30, 12, 4])
        p_F = [.02, -.03, .05, 1.]
        p_C = np.linalg.inv(camera) @ geometry.robot.flange("left", q, world=True) @ p_F
        # URDF decimal geometry and native DH FK agree within the planner's
        # established 0.1 mm registration tolerance (not bitwise equality).
        np.testing.assert_allclose(X @ p_C, matrix(kinematics.fk("left", tuple(q[:7]))) @ p_F,
                                   atol=1e-4, rtol=0)
        # Directly renaming the generic base matrix would lose the first DH offset.
        self.assertGreater(np.linalg.norm(X[:3, 3] - np.asarray(report["T_base_camera_opencv"])[:3, 3]), .1)
        with self.assertRaises(FileExistsError):
            export(self.config, self.source, self.output, "unit-test-only", kinematics)

    def test_failed_synthetic_and_other_camera_results_never_create_live_file(self):
        for update in ({"passed": False}, {"synthetic": True}, {"test_data": True},
                       {"camera": {"serial": "other-camera"}}, {"quality": {"passed": False}}):
            report = {"schema": "external_camera_result_v1", "passed": True,
                      "quality": {"passed": True}, "camera": {"serial": "unit-test-only"}}
            report.update(update)
            write_json(self.source, report)
            with self.assertRaises(ValueError):
                export(self.config, self.source, self.output, "unit-test-only", SimpleNamespace())
            self.assertFalse(self.output.exists())

    def eye_to_hand_report(self, kinematics):
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler("xyz", [.3, -.2, .6]).as_matrix()
        camera[:3, 3] = [.19, -.225, .013]
        return {
            "schema": "external_eye_to_hand_result_v2", "method": "eye_to_hand",
            "passed": True, "synthetic": False, "quality": {"passed": True},
            "side": "left", "reference_frame": "tianji_left_base",
            "camera": {"serial": "unit-test-only", "output_rotation_deg": 0},
            "robot_model_sha256": kinematics.model.digest,
            "T_base_camera_opencv": camera.tolist(),
        }

    def test_eye_to_hand_keeps_native_sdk_transform_without_urdf_offset(self):
        kinematics = TianjiKinematics()
        report = self.eye_to_hand_report(kinematics)
        write_json(self.source, report)
        with patch("rail_preinsert.geometry.RobotGeometry", side_effect=AssertionError("Already in SDK base")):
            payload = export(self.config, self.source, self.output, "unit-test-only", kinematics)
        _, X = load_calibration(self.output)
        np.testing.assert_array_equal(X, report["T_base_camera_opencv"])
        self.assertEqual(payload["source_result_sha256"], digest(self.source))
        self.assertEqual(payload["sdk_model_sha256"], kinematics.model.digest)
        with self.assertRaises(FileExistsError):
            export(self.config, self.source, self.output, "unit-test-only", kinematics)

    def test_eye_to_hand_rejects_wrong_base_model_camera_or_unmeasured_results(self):
        kinematics = TianjiKinematics()
        for update in (
                {"robot_model_sha256": "different-model"},
                {"side": "right", "reference_frame": "tianji_right_base"},
                {"reference_frame": "project_world"}, {"method": "other"},
                {"camera": {"serial": "unit-test-only", "output_rotation_deg": 180}},
                {"camera": {"serial": "other-camera", "output_rotation_deg": 0}},
                {"passed": False}, {"quality": {"passed": False}},
                {"synthetic": True}, {"test_data": True},
                {"T_base_camera_opencv": np.zeros((4, 4)).tolist()}):
            with self.subTest(update=update):
                report = self.eye_to_hand_report(kinematics)
                report.update(update)
                write_json(self.source, report)
                with self.assertRaises(ValueError):
                    export(self.config, self.source, self.output, "unit-test-only", kinematics)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
