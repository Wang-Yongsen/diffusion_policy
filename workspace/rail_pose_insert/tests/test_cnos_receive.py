"""Legacy CNOS diagnostics and prompt-SAM postgrasp stage separation."""
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1]), str(HERE.parents[1] / "third_party/foundationpose/workspace/src")]
import receive_perception_worker as worker
from rail_preinsert.core import read_json, write_json
from rail_receive.logic import load_config


class WorkerStageTests(unittest.TestCase):
    def run_stage(self, mode):
        config = load_config(HERE / "receive.yaml")
        # The mocked camera/poses use this synthetic target frame.
        config["pose_preview"] = {"T_C_O_goal": np.eye(4).tolist()}
        frame = NS(rgb=np.zeros((480, 640, 3), np.uint8), K=np.eye(3),
                   timestamp_ns=time.monotonic_ns(), mask=None,
                   metadata={"timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic"})
        # dataclasses.replace needs the same shape as the actual camera frame.
        from dataclasses import make_dataclass
        Frame = make_dataclass("Frame", [(k, object) for k in vars(frame)])
        frame = Frame(**vars(frame))
        camera, detector, engine = Mock(), Mock(), Mock()
        camera.frame.return_value = frame
        engine.infer.return_value = np.eye(4), {"valid": True}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_json(directory / "request.json", {"id": "first", "command": "initialize"})
            write_json(directory / "context.json", {"enabled": True, "epoch": 0, "mode": mode})
            def detect(frame):
                write_json(directory / "request.json", {"id": "exit", "command": "shutdown"})
                return np.ones((480, 640), bool), {"valid": True, "detector": "cnos" if mode == "receive" else "sam_prompt", "detected_in_roi": True}
            detector.detect.side_effect = detect
            def create_engine(*args):
                detector.close.assert_called_once()
                return engine
            args = NS(serve=directory, camera_serial="test", mesh=Path(config["mesh"]),
                      foundationpose_root=Path(config["foundationpose_root"]))
            with patch.object(worker, "Camera", return_value=camera), \
                    patch("rail_receive.cnos.CnosDetector", return_value=detector) as cnos_factory, \
                    patch("rail_receive.sam_prompt.SamPromptDetector", return_value=detector) as sam_factory, \
                    patch.object(worker, "pose_engine", side_effect=create_engine) as pose_factory, \
                    patch("rail_receive.visualization.save_preview"), \
                    patch("rail_receive.visualization.save_mask_snapshot"), \
                    patch.object(worker, "save_frame", return_value=(None, 1)):
                worker.serve(args, config)
            if mode == "receive":
                pose_factory.assert_not_called()
                engine.infer.assert_not_called()
                cnos_factory.assert_called_once_with(config)
                sam_factory.assert_not_called()
            else:
                pose_factory.assert_called_once()
                self.assertTrue(engine.infer.call_args.kwargs["initialize"])
                self.assertTrue(engine.infer.call_args.args[0].mask.all())
                cnos_factory.assert_not_called()
                sam_factory.assert_called_once_with(config, camera_serial="test")
            self.assertEqual(camera.close.call_count, 1 if mode == "receive" else 2)

    def test_receive_uses_only_cnos(self):
        self.run_stage("receive")

    def test_held_mode_releases_sam_and_passes_its_mask_to_foundationpose(self):
        self.run_stage("held")

    def test_grasp_transition_redetects_new_image_and_passes_only_new_mask(self):
        from rail_pose.data import Frame
        config = load_config(HERE / "receive.yaml")
        # The mocked camera/poses use this synthetic target frame.
        config["pose_preview"] = {"T_C_O_goal": np.eye(4).tolist()}
        stamp = time.monotonic_ns()
        def frame(label, value, offset):
            return Frame(label, np.full((480, 640, 3), value, np.uint8),
                         np.full((480, 640), .7, np.float32), np.eye(3), stamp+offset,
                         metadata={"timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic"})
        before, after = frame("before_grasp", 20, 0), frame("after_grasp", 80, 1_000_000)
        old_mask, new_mask = np.zeros((480, 640), bool), np.zeros((480, 640), bool)
        old_mask[150:180, 490:550] = True
        new_mask[180:240, 510:590] = True
        rgb_camera, rgbd_camera, cnos, sam, engine = Mock(), Mock(), Mock(), Mock(), Mock()
        rgb_camera.frame.return_value, rgbd_camera.frame.return_value = before, after
        engine.infer.return_value = np.eye(4), {"valid": True}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_json(directory / "request.json", {"id": "before", "command": "initialize"})
            write_json(directory / "context.json", {"enabled": True, "epoch": 0, "mode": "receive"})
            def detect(current):
                if current.frame_id == "before_grasp":
                    pose_factory.assert_not_called()
                    # Simulate the controller completing its hand transition.
                    write_json(directory / "context.json", {"enabled": True, "epoch": 0, "mode": "held"})
                    write_json(directory / "request.json", {"id": "after", "command": "initialize"})
                    return old_mask, {"valid": True, "detector": "cnos", "detected_in_roi": True}
                self.assertIs(current, after)
                write_json(directory / "request.json", {"id": "exit", "command": "shutdown"})
                return new_mask, {"valid": True, "detector": "sam_prompt", "output_instances": 1}
            cnos.detect.side_effect = detect
            sam.detect.side_effect = detect
            args = NS(serve=directory, camera_serial="test", mesh=Path(config["mesh"]),
                      foundationpose_root=Path(config["foundationpose_root"]))
            with patch.object(worker, "Camera", side_effect=[rgb_camera, rgbd_camera]) as camera_factory, \
                    patch("rail_receive.cnos.CnosDetector", return_value=cnos) as cnos_factory, \
                    patch("rail_receive.sam_prompt.SamPromptDetector", return_value=sam) as sam_factory, \
                    patch.object(worker, "pose_engine", return_value=engine) as pose_factory, \
                    patch("rail_receive.visualization.save_preview"), \
                    patch("rail_receive.visualization.save_mask_snapshot"), \
                    patch.object(worker, "save_frame", return_value=(None, 1)):
                worker.serve(args, config)
            self.assertIs(cnos.detect.call_args.args[0], before)
            self.assertIs(sam.detect.call_args.args[0], after)
            initialized = engine.infer.call_args.args[0]
            self.assertEqual(initialized.frame_id, "after_grasp")
            np.testing.assert_array_equal(initialized.rgb, after.rgb)
            np.testing.assert_array_equal(initialized.mask, new_mask)
            self.assertFalse(np.array_equal(initialized.mask, old_mask))
            self.assertTrue(engine.infer.call_args.kwargs["initialize"])
            pose_factory.assert_called_once()
            cnos_factory.assert_called_once_with(config)
            sam_factory.assert_called_once_with(config, camera_serial="test")
            self.assertTrue(camera_factory.call_args_list[0].kwargs["rgb_only"])
            self.assertFalse(camera_factory.call_args_list[1].kwargs)
            latest = read_json(directory / "latest.json")
            self.assertEqual(latest["generation"], "after")
            self.assertEqual(latest["timestamp_ns"], after.timestamp_ns)


class OncePoseTests(unittest.TestCase):
    def run_once(self, *, rgbd_failure=False, sam_failure=False):
        from rail_pose.data import Frame
        config = load_config(HERE / "receive.yaml")
        # The mocked camera/poses use this synthetic target frame.
        config["pose_preview"] = {"T_C_O_goal": np.eye(4).tolist()}
        stamp = time.monotonic_ns()
        def frame(label, offset):
            return Frame(label, np.zeros((480, 640, 3), np.uint8),
                         np.full((480, 640), .7, np.float32), np.eye(3), stamp+offset,
                         metadata={"timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic"})
        initial, measured = frame("sam_after_grasp", 0), frame("fresh_pose", 1_000_000)
        rgb, rgbd, idle_camera, detector, engine = Mock(), Mock(), Mock(), Mock(), Mock()
        rgb.frame.return_value = initial
        rgbd.frame.side_effect = [initial, measured]
        detector.detect.return_value = np.ones((480, 640), bool), {"valid": True}
        engine.infer.return_value = np.eye(4), {"valid": True, "T_C_CAD": np.eye(4).tolist()}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_json(directory / "request.json", {"id": "once", "command": "initialize"})
            write_json(directory / "context.json", {"enabled": True, "epoch": 0, "mode": "held_once"})
            calls = []
            def camera_frame(**kwargs):
                if not kwargs.get("rgb_only"):
                    return initial if rgbd.frame.call_count <= 2 else measured
                calls.append(1)
                if len(calls) >= 3:
                    write_json(directory / "request.json", {"id": "exit", "command": "shutdown"})
                return measured
            idle_camera.frame.side_effect = camera_frame
            rgbd.frame.side_effect = camera_frame
            args = NS(serve=directory, camera_serial="test", mesh=Path(config["mesh"]),
                      foundationpose_root=Path(config["foundationpose_root"]))
            cameras = [rgb, RuntimeError("RGB-D unavailable") if rgbd_failure else rgbd, idle_camera]
            with patch.object(worker, "Camera", side_effect=cameras) as make_camera, \
                    patch("rail_receive.cnos.CnosDetector") as cnos_factory, \
                    patch("rail_receive.sam_prompt.SamPromptDetector", return_value=detector,
                          side_effect=FileNotFoundError("sam_prompt.json") if sam_failure else None) as detector_factory, \
                    patch.object(worker, "pose_engine", return_value=engine), \
                    patch("rail_receive.visualization.save_preview") as show, \
                    patch("rail_receive.visualization.save_mask_snapshot") as save_mask, \
                    patch("rail_receive.visualization.save_pose_snapshot") as save_snapshot:
                worker.serve(args, config)
            self.assertEqual(len(calls), 3)  # The camera keeps refreshing after inference stops.
            cnos_factory.assert_not_called()
            result = read_json(directory / "latest.json")
            if rgbd_failure or sam_failure:
                save_snapshot.assert_not_called()
                save_mask.assert_not_called()
                engine.infer.assert_not_called()
                detector.detect.assert_not_called()
                self.assertEqual(result["reason"], "held_pose_unavailable")
                if sam_failure:
                    self.assertEqual(result["phase"], "loading_sam")
                    self.assertIn("sam_prompt.json", result["detail"])
            else:
                detector.detect.assert_called_once()
                detector_factory.assert_called_once_with(config, camera_serial="test")
                save_mask.assert_called_once()
                self.assertIs(save_mask.call_args.args[1], initial)
                np.testing.assert_array_equal(save_mask.call_args.args[2], detector.detect.return_value[0])
                self.assertTrue((directory / "once" / "sam_detection.json").is_file())
                self.assertEqual(engine.infer.call_count, 2)  # Initialize, then one fresh frame.
                self.assertTrue(engine.infer.call_args_list[0].kwargs["initialize"])
                self.assertFalse(engine.infer.call_args_list[1].kwargs["initialize"])
                self.assertEqual(result["timestamp_ns"], measured.timestamp_ns)
                self.assertEqual(result["pose_snapshot"], "once/pose.png")
                save_snapshot.assert_called_once()
                self.assertIs(save_snapshot.call_args.args[1], measured)
                np.testing.assert_array_equal(save_snapshot.call_args.args[3], np.eye(4))
                self.assertIsNone(show.call_args.args[2])  # Camera-only preview, no old CAD overlay.
            if rgbd_failure or sam_failure:
                idle_camera.close.assert_called_once()
            else:
                self.assertEqual(make_camera.call_count, 2)  # Keep the working RGB-D stream for T.
                rgbd.close.assert_called_once()

    def test_pose_runs_once_then_only_camera_refreshes(self):
        self.run_once()

    def test_rgbd_failure_returns_to_camera_only_without_closing_window(self):
        self.run_once(rgbd_failure=True)

    def test_missing_prompt_returns_to_camera_only_so_operator_can_save_and_retry(self):
        self.run_once(sam_failure=True)

    def test_reinitialize_segments_again_and_never_reuses_previous_mask(self):
        self.run_reinitialize("held_once")

    def test_continuous_held_reinitialize_resegments_and_registers_instead_of_tracking_old_pose(self):
        self.run_reinitialize("held")

    def run_reinitialize(self, mode):
        from rail_pose.data import Frame
        config = load_config(HERE / "receive.yaml")
        # The mocked camera/poses use this synthetic target frame.
        config["pose_preview"] = {"T_C_O_goal": np.eye(4).tolist()}
        stamp = time.monotonic_ns()
        frames = [Frame(label, np.full((480, 640, 3), value, np.uint8),
                        np.full((480, 640), .7, np.float32), np.eye(3), stamp+index*1_000_000,
                        metadata={"timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic"})
                  for index, (label, value) in enumerate((("first_mask", 20), ("first_track", 30),
                                                         ("second_mask", 80), ("second_track", 90)))]
        masks = [np.zeros((480, 640), bool), np.zeros((480, 640), bool)]
        masks[0][210:230, 250:350] = True
        masks[1][190:215, 410:520] = True
        detectors, engines = [Mock(), Mock()], [Mock(), Mock()]
        for detector, mask in zip(detectors, masks):
            detector.detect.return_value = mask, {"valid": True, "detector": "sam_prompt", "output_instances": 1}
        for engine in engines:
            engine.initialized = True
        rgb, rgbd = Mock(), Mock()
        rgb.frame.return_value = frames[0]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_json(directory / "request.json", {"id": "first", "command": "initialize"})
            write_json(directory / "context.json", {"enabled": True, "epoch": 0, "mode": mode})
            # Each round reads RGB-D before loading SAM, refreshes it after
            # loading, and finally tracks one fresh image after registration.
            inference_frames = iter([frames[0], frames[0], frames[1], frames[2], frames[2], frames[3]])
            idle_reads = []
            for index, engine in enumerate(engines):
                def infer(frame, *, initialize, index=index):
                    if mode == "held" and not initialize:
                        # Simulate T while an initialized continuous tracker is
                        # still running, rather than after one-shot cleanup.
                        command = ({"id": "second", "command": "initialize"} if index == 0
                                   else {"id": "exit", "command": "shutdown"})
                        write_json(directory / "request.json", command)
                    return np.eye(4), {"valid": True, "T_C_CAD": np.eye(4).tolist()}
                engine.infer.side_effect = infer
            def camera_frame(**kwargs):
                if not kwargs.get("rgb_only"):
                    return next(inference_frames)
                idle_reads.append(1)
                command = ({"id": "second", "command": "initialize"} if len(idle_reads) == 1
                           else {"id": "exit", "command": "shutdown"})
                write_json(directory / "request.json", command)
                return frames[1 if len(idle_reads) == 1 else 3]
            rgbd.frame.side_effect = camera_frame
            args = NS(serve=directory, camera_serial="test", mesh=Path(config["mesh"]),
                      foundationpose_root=Path(config["foundationpose_root"]))
            def make_engine(*args):
                index = pose_factory.call_count-1
                detectors[index].close.assert_called_once()
                return engines[index]
            with patch.object(worker, "Camera", side_effect=[rgb, rgbd]) as camera_factory, \
                    patch("rail_receive.cnos.CnosDetector") as cnos_factory, \
                    patch("rail_receive.sam_prompt.SamPromptDetector", side_effect=detectors) as sam_factory, \
                    patch.object(worker, "pose_engine", side_effect=make_engine) as pose_factory, \
                    patch.object(worker, "save_frame", return_value=(None, 1)) as save_frame, \
                    patch("rail_receive.visualization.save_preview"), \
                    patch("rail_receive.visualization.save_mask_snapshot") as save_mask, \
                    patch("rail_receive.visualization.save_pose_snapshot") as save_snapshot:
                worker.serve(args, config)
            cnos_factory.assert_not_called()
            self.assertEqual(sam_factory.call_count, 2)
            self.assertEqual(pose_factory.call_count, 2)
            self.assertEqual(camera_factory.call_count, 2)  # T reuses the working RGB-D stream.
            self.assertEqual(len(idle_reads), 2 if mode == "held_once" else 0)
            self.assertEqual(save_mask.call_count, 2)
            self.assertEqual(save_snapshot.call_count, 2 if mode == "held_once" else 0)
            self.assertEqual(save_frame.call_count, 0 if mode == "held_once" else 4)
            for index, (detector, engine) in enumerate(zip(detectors, engines)):
                detector.detect.assert_called_once_with(frames[index*2])
                self.assertEqual(engine.infer.call_count, 2)
                initialization, tracking = engine.infer.call_args_list
                self.assertTrue(initialization.kwargs["initialize"])
                self.assertIs(initialization.args[0].mask, masks[index])
                self.assertEqual(initialization.args[0].frame_id, frames[index*2].frame_id)
                self.assertFalse(tracking.kwargs["initialize"])
                self.assertIsNone(tracking.args[0].mask)
            result = read_json(directory / "latest.json")
            self.assertEqual(result["generation"], "second")
            if mode == "held_once":
                self.assertEqual(result["pose_snapshot"], "second/pose.png")
            else:
                self.assertNotIn("pose_snapshot", result)
            self.assertEqual(result["timestamp_ns"], frames[3].timestamp_ns)
            self.assertTrue((directory / "first" / "sam_detection.json").is_file())
            self.assertTrue((directory / "second" / "sam_detection.json").is_file())

    def test_before_grasp_camera_runs_without_loading_any_model(self):
        config = load_config(HERE / "receive.yaml")
        # The mocked camera/poses use this synthetic target frame.
        config["pose_preview"] = {"T_C_O_goal": np.eye(4).tolist()}
        camera = Mock()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            write_json(directory / "request.json", {"id": "preview", "command": "initialize"})
            def frame(**kwargs):
                if camera.frame.call_count >= 3:
                    write_json(directory / "request.json", {"id": "exit", "command": "shutdown"})
                return Mock()
            camera.frame.side_effect = frame
            args = NS(serve=directory, camera_serial="test", mesh=Path(config["mesh"]),
                      foundationpose_root=Path(config["foundationpose_root"]))
            with patch.object(worker, "Camera", return_value=camera), \
                    patch("rail_receive.cnos.CnosDetector") as detector_factory, \
                    patch("rail_receive.sam_prompt.SamPromptDetector") as sam_factory, \
                    patch.object(worker, "pose_engine") as pose_factory, \
                    patch("rail_receive.visualization.save_preview"):
                worker.serve(args, config)
            detector_factory.assert_not_called()
            sam_factory.assert_not_called()
            pose_factory.assert_not_called()
            self.assertEqual(camera.frame.call_count, 3)


class CameraModeTests(unittest.TestCase):
    def camera(self, rgb_only):
        camera = worker.Camera.__new__(worker.Camera)
        camera.rgb_only, camera.offset_ns, camera.remap, camera.K = rgb_only, 0, None, np.eye(3)
        camera.rs = NS(timestamp_domain=NS(global_time=7))
        camera.pipeline = Mock()
        color = camera.pipeline.wait_for_frames.return_value.get_color_frame.return_value
        color.get_frame_timestamp_domain.return_value = 7
        color.get_timestamp.return_value = 3000.
        color.get_frame_number.return_value = 1
        color.get_data.return_value = np.zeros((480, 640, 3), np.uint8)
        return camera

    def test_receiving_reads_color_without_requesting_a_depth_frame(self):
        camera = self.camera(True)
        frames = camera.pipeline.wait_for_frames.return_value
        frames.get_depth_frame.side_effect = AssertionError("CNOS does not need depth")
        with patch("perception_worker.time.monotonic_ns", return_value=3_000_000_000):
            frame = camera.frame()
        self.assertTrue(frame.metadata["rgb_only"])
        self.assertEqual(frame.rgb.shape, (480, 640, 3))
        self.assertFalse(frame.depth_m.any())
        frames.get_depth_frame.assert_not_called()

    def test_pose_mode_retains_rgb_depth_timestamp_check(self):
        camera = self.camera(False)
        depth = camera.pipeline.wait_for_frames.return_value.get_depth_frame.return_value
        depth.get_frame_timestamp_domain.return_value = 7
        depth.get_timestamp.return_value = 3010.
        with patch("perception_worker.time.monotonic", side_effect=[0., 0., 2.]), \
                patch("perception_worker.time.monotonic_ns", return_value=3_000_000_000):
            with self.assertRaisesRegex(ValueError, "separation exceeds"):
                camera.frame()

    def test_pose_mode_discards_unsynchronized_frames_before_registering_depth(self):
        camera = self.camera(False)
        depth = camera.pipeline.wait_for_frames.return_value.get_depth_frame.return_value
        depth.get_frame_timestamp_domain.return_value = 7
        depth.get_timestamp.side_effect = [3010., 3000.]
        depth.get_data.return_value = np.ones((480, 640), np.uint16)*700
        camera.scale, camera.depth_K, camera.R, camera.t = .001, np.eye(3), np.eye(3), np.zeros(3)
        with patch("perception_worker.time.monotonic_ns", return_value=3_000_000_000), \
                patch("rail_pose.episode.register_depth", return_value=np.full((480, 640), .7)) as register:
            frame = camera.frame()
        self.assertEqual(camera.pipeline.wait_for_frames.call_count, 2)
        register.assert_called_once()
        self.assertEqual(frame.metadata["depth_minus_rgb_ms"], 0.)
        self.assertTrue(np.allclose(frame.depth_m, .7))

    def test_rgbd_pipeline_can_refresh_color_preview_without_reading_depth(self):
        camera = self.camera(False)
        frames = camera.pipeline.wait_for_frames.return_value
        frames.get_depth_frame.side_effect = AssertionError("Idle preview must skip depth registration")
        with patch("perception_worker.time.monotonic_ns", return_value=3_000_000_000):
            frame = camera.frame(rgb_only=True)
        self.assertTrue(frame.metadata["rgb_only"])
        frames.get_depth_frame.assert_not_called()

    def test_startup_allows_ten_seconds_for_first_frame_only(self):
        import pyrealsense2 as rs
        pipeline = Mock()
        profile = pipeline.start.return_value
        profile.get_device.return_value.query_sensors.return_value = []
        profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.return_value = NS(
            fx=600, fy=600, ppx=320, ppy=240, coeffs=[0, 0, 0, 0, 0])
        with patch.object(rs, "pipeline", return_value=pipeline):
            camera = worker.Camera("test", rgb_only=True)
            camera.close()
        waits = pipeline.wait_for_frames.call_args_list
        self.assertEqual(waits[0].args, (10000,))
        self.assertEqual([call.args for call in waits[1:]], [(2000,)]*14)


if __name__ == "__main__":
    unittest.main()
