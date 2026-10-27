"""Core offline regression coverage; all actuator and camera owners are mocked."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import receive
from rail_common.core import (
    digest,
    matrix,
    write_json,
    read_json,
    load_calibration,
    load_target,
)
from rail_receive.logic import load_config


def config():
    return load_config(HERE / "receive.yaml")


from tools.export_head_calibration import export
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics


class EntrypointTests(unittest.TestCase):
    def test_cli_rejects_removed_modes_and_invalid_drawing_before_devices(self):
        with patch.object(receive, "run") as run, redirect_stdout(io.StringIO()):
            for flags in (
                ["--pose-only"],
                ["--move-above"],
                ["--grasp-mode", "manual"],
                ["--draw-bboxes", "--no-view"],
            ):
                with self.subTest(flags=flags), self.assertRaises(SystemExit):
                    receive.main(flags)
            run.assert_not_called()

    def test_essential_motion_flags_reach_the_workflow(self):
        with patch.object(receive, "run") as run:
            self.assertEqual(
                receive.main(["--already-held", "--draw-bboxes", "--above-height", "0.02"]), 0
            )
            config, args = run.call_args.args
            self.assertTrue(args.already_held and args.draw_bboxes)
            self.assertEqual(config["move_above"]["height_m"], 0.02)

    def test_read_only_never_engages_an_actuator_and_always_closes(self):
        hardware = Mock()
        hardware.snapshot.return_value = {"joints_rad": np.zeros(7)}
        with patch.object(receive, "RightHardware", return_value=hardware) as factory, patch.object(
            receive, "kinematics"
        ), redirect_stdout(io.StringIO()):
            receive.read_state(config())
        self.assertFalse(factory.call_args.kwargs["motion"])
        hardware.engage.assert_not_called()
        hardware.begin_hand.assert_not_called()
        hardware.close.assert_called_once()

    def test_runtime_is_temporary_with_or_without_retained_results(self):
        from rail_common.runtime import Log

        for retain in (False, True):
            with self.subTest(retain=retain), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "output" if retain else None
                args = NS(output=output)
                runtime_paths = []

                def fake_run(config, args, runtime):
                    runtime.mkdir(parents=True)
                    (runtime / "worker.log").write_text("temporary diagnostics")
                    runtime_paths.append(runtime)
                    log = Log(args.output)
                    log.emit("stage", stage="returned_initial")
                    log.finish("returned_initial", exit_requested=True, errors=[])
                    log.close()

                with patch.object(receive, "_run", side_effect=fake_run):
                    receive.run(config(), args)
                self.assertFalse(runtime_paths[0].exists())
                if retain:
                    self.assertEqual(
                        {p.name for p in output.iterdir()}, {"events.jsonl", "result.json"}
                    )
                    self.assertEqual(
                        read_json(output / "result.json")["last_stage"], "returned_initial"
                    )
                else:
                    self.assertFalse(args.output.exists())

    def test_teaching_keeps_target_schema_without_robot_access(self):
        import teach_target

        with tempfile.TemporaryDirectory() as directory:
            c = config()
            c["calibration"] = str(Path(directory) / "calibration.json")
            write_json(
                c["calibration"],
                dict(
                    parent_frame="tianji_left_base",
                    child_frame="head_color_optical",
                    unit="m",
                    camera_serial="test",
                    T_B_C=np.eye(4).tolist(),
                ),
            )
            observation = dict(
                camera_serial="test",
                unit="m",
                frame="T_C_CAD",
                mesh_sha256=digest(c["mesh"]),
                records=[
                    dict(
                        valid=True,
                        timestamp_ns=i + 1,
                        T_C_CAD=np.eye(4).tolist(),
                        timestamp_source="sdk_global_timestamp_mapped_to_host_monotonic",
                    )
                    for i in range(3)
                ],
            )
            output = Path(directory) / "goal.json"
            with patch("rail_common.perception.Perception") as perception, patch.object(
                receive, "RightHardware"
            ) as hardware, redirect_stdout(io.StringIO()):
                perception.return_value.measure.return_value = observation
                teach_target.teach(c, NS(output=output))
                hardware.assert_not_called()
                payload, target = load_target(output, {"camera_serial": "test"}, c["mesh"])
                self.assertEqual(payload["schema_version"], 1)
                self.assertEqual(payload["calibration_sha256_at_teach"], digest(c["calibration"]))
                np.testing.assert_allclose(target, np.eye(4))
                with self.assertRaises(FileExistsError):
                    teach_target.teach(c, NS(output=output))

    def test_startup_failure_keeps_error_result_and_cleans_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            args = NS(output=output, draw_bboxes=False, no_view=False, already_held=False)
            with patch.object(receive, "preflight", return_value={"ready": True}), patch.object(
                receive, "kinematics"
            ), patch.object(
                receive, "ReceiveGeometry", side_effect=ValueError("invalid calibration")
            ), patch.object(
                receive, "RightHardware"
            ) as hardware, patch(
                "rail_receive.visualization.PreviewWindow"
            ) as preview:
                with self.assertRaisesRegex(ValueError, "invalid calibration"):
                    receive.run(config(), args)
                hardware.assert_not_called()
                preview.return_value.close.assert_called_once()
            self.assertEqual({p.name for p in output.iterdir()}, {"events.jsonl", "result.json"})
            result = read_json(output / "result.json")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "invalid calibration")


class CalibrationExportTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "external.json"
        self.output = Path(self.directory.name) / "head.json"
        self.config = load_config(HERE / "receive.yaml")

    def test_export_maps_camera_points_into_sdk_base_at_nonzero_joint_poses(self):
        from rail_common.geometry import RobotGeometry

        kinematics = TianjiKinematics()
        geometry = RobotGeometry(self.config, kinematics)
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.6]).as_matrix()
        camera[:3, 3] = [0.7, -0.4, 1.2]
        base = geometry.robot.base_pose("left")
        report = {
            "schema": "external_camera_result_v1",
            "passed": True,
            "quality": {"passed": True},
            "side": "left",
            "reference_frame": "project_world",
            "camera": {"serial": "unit-test-only"},
            "urdf_sha256": digest(self.config["urdf"]),
            "wrist_calibration_sha256": "unit-test-only",
            "T_world_robot_base": base.tolist(),
            "T_world_camera_opencv": camera.tolist(),
            "T_base_camera_opencv": (np.linalg.inv(base) @ camera).tolist(),
        }
        write_json(self.source, report)
        with patch("rail_common.geometry.RobotGeometry", return_value=geometry):
            export(self.config, self.source, self.output, "unit-test-only", kinematics)
        _, X = load_calibration(self.output)
        q = np.zeros(54)
        q[:7] = np.radians([30, -60, -34, -52, 30, 12, 4])
        p_F = [0.02, -0.03, 0.05, 1.0]
        p_C = np.linalg.inv(camera) @ geometry.robot.flange("left", q, world=True) @ p_F
        np.testing.assert_allclose(
            X @ p_C, matrix(kinematics.fk("left", tuple(q[:7]))) @ p_F, atol=0.0001, rtol=0
        )
        self.assertGreater(
            np.linalg.norm(X[:3, 3] - np.asarray(report["T_base_camera_opencv"])[:3, 3]), 0.1
        )
        with self.assertRaises(FileExistsError):
            export(self.config, self.source, self.output, "unit-test-only", kinematics)

    def eye_to_hand_report(self, kinematics):
        camera = np.eye(4)
        camera[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.6]).as_matrix()
        camera[:3, 3] = [0.19, -0.225, 0.013]
        return {
            "schema": "external_eye_to_hand_result_v2",
            "method": "eye_to_hand",
            "passed": True,
            "synthetic": False,
            "quality": {"passed": True},
            "side": "left",
            "reference_frame": "tianji_left_base",
            "camera": {"serial": "unit-test-only", "output_rotation_deg": 0},
            "robot_model_sha256": kinematics.model.digest,
            "T_base_camera_opencv": camera.tolist(),
        }

    def test_eye_to_hand_keeps_native_sdk_transform_without_urdf_offset(self):
        kinematics = TianjiKinematics()
        report = self.eye_to_hand_report(kinematics)
        write_json(self.source, report)
        with patch(
            "rail_common.geometry.RobotGeometry", side_effect=AssertionError("Already in SDK base")
        ):
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
            {"reference_frame": "project_world"},
            {"method": "other"},
            {"camera": {"serial": "unit-test-only", "output_rotation_deg": 180}},
            {"camera": {"serial": "other-camera", "output_rotation_deg": 0}},
            {"passed": False},
            {"quality": {"passed": False}},
            {"synthetic": True},
            {"test_data": True},
            {"T_base_camera_opencv": np.zeros((4, 4)).tolist()},
        ):
            with self.subTest(update=update):
                report = self.eye_to_hand_report(kinematics)
                report.update(update)
                write_json(self.source, report)
                with self.assertRaises(ValueError):
                    export(self.config, self.source, self.output, "unit-test-only", kinematics)
                self.assertFalse(self.output.exists())
