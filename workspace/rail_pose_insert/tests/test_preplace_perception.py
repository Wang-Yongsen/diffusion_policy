"""Pre-placement registration gets a fresh mask at the transported rail location."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1]),
               str(HERE.parents[1] / "third_party/foundationpose/workspace/src")]
import receive_perception_worker as worker
from rail_preinsert.core import read_json, write_json
from rail_receive.logic import load_config
from rail_receive.perception import ReceivePerception
from rail_receive import sam_prompt
from rail_receive.sam_prompt import SamPromptDetector, projected_prompt


class ProjectedPromptTests(unittest.TestCase):
    def setUp(self):
        self.bounds = np.array([[-.05, -.2, -.005], [.05, .2, .005]])
        self.pose = np.eye(4)
        self.pose[:3, 3] = [.1, 0, 1]
        self.K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])

    def project(self, **kwargs):
        values = dict(bounds=self.bounds, predicted_pose=self.pose, K=self.K,
                      image_size=[640, 480], camera_serial="main")
        values.update(kwargs)
        return projected_prompt(**values)

    def test_projection_includes_cad_extent_and_physical_margin_without_old_points(self):
        prompt = self.project()
        # Closest depth is .995 m; a 30 mm margin contributes 18.09 px.
        self.assertEqual(prompt["box_xyxy"], [331., 101., 429., 379.])
        self.assertEqual(prompt["point_coords"], [])
        self.assertEqual(prompt["point_labels"], [])
        self.assertEqual(prompt["camera_serial"], "main")

    def test_prediction_translation_and_rotation_change_the_search_region(self):
        original = np.asarray(self.project()["box_xyxy"])
        shifted = self.pose.copy(); shifted[0, 3] += .1
        moved = np.asarray(self.project(predicted_pose=shifted)["box_xyxy"])
        self.assertGreater(moved[0]-original[0], 59)
        self.assertGreater(moved[2]-original[2], 59)
        rotated = self.pose.copy()
        rotated[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        box = self.project(predicted_pose=rotated)["box_xyxy"]
        self.assertGreater(box[2]-box[0], box[3]-box[1])

    def test_minimum_eight_pixel_margin_and_edge_clipping(self):
        distant = self.pose.copy(); distant[2, 3] = 10
        self.assertEqual(self.project(predicted_pose=distant)["box_xyxy"], [314., 219., 338., 261.])
        clipped = self.pose.copy(); clipped[0, 3] = .5
        box = self.project(predicted_pose=clipped)["box_xyxy"]
        self.assertEqual(box[2], 640)
        self.assertLess(box[0], box[2])

    def test_invalid_pose_projection_and_fully_offscreen_rail_are_rejected(self):
        outside = self.pose.copy(); outside[0, 3] = 4
        behind = self.pose.copy(); behind[2, 3] = -1
        nonrigid = self.pose.copy(); nonrigid[0, 0] = 2
        cases = [dict(predicted_pose=outside), dict(predicted_pose=behind), dict(predicted_pose=nonrigid),
                 dict(K=np.zeros((3, 3))), dict(margin_m=float("nan")), dict(min_margin_px=0),
                 dict(bounds=np.zeros((2, 3)))]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.project(**values)

    def test_temporary_prompt_bypasses_saved_file_and_uses_same_validation(self):
        prompt = self.project()
        model, predictor = Mock(), Mock()
        model.to.return_value = model
        fake_torch = NS(cuda=NS(is_available=lambda: True, empty_cache=Mock()),
                        device=lambda *_: nullcontext(), inference_mode=lambda: nullcontext())
        fake_sam = NS(SamPredictor=Mock(return_value=predictor),
                      sam_model_registry={"vit_h": Mock(return_value=model)})
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pth"; checkpoint.touch()
            missing = Path(temporary) / "never-created.json"
            config = {"sam_prompt": str(missing), "receive_image_size": [640, 480]}
            with patch.dict(sys.modules, {"torch": fake_torch, "segment_anything": fake_sam}), \
                    patch.object(sam_prompt, "CHECKPOINT", checkpoint):
                detector = SamPromptDetector(config, camera_serial="main", prompt=prompt)
                try:
                    prompt["box_xyxy"][0] = 0
                    self.assertEqual(detector.prompt["box_xyxy"][0], 331.)
                    detector.set_prompt(self.project())
                    with self.assertRaisesRegex(ValueError, "camera_serial"):
                        detector.set_prompt(dict(self.project(), camera_serial="other"))
                    self.assertFalse(missing.exists())
                finally:
                    detector.close()


class ReestimateBridgeTests(unittest.TestCase):
    def bridge(self, directory):
        bridge = ReceivePerception.__new__(ReceivePerception)
        bridge.directory, bridge.generation = directory, "old"
        bridge.ready = Mock(return_value=True)
        return bridge

    def test_initialize_returns_new_generation_and_rejects_old_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bridge = self.bridge(directory)
            pose = np.eye(4); pose[:3, 3] = [.1, -.2, .9]
            generation = bridge.initialize(predicted_pose=pose)
            self.assertEqual(generation, bridge.generation)
            self.assertNotEqual(generation, "old")
            request = read_json(directory / "request.json")
            self.assertEqual(request["id"], generation)
            np.testing.assert_allclose(request["T_C_O_predicted"], pose)
            write_json(directory / "latest.json", {"valid": True, "generation": "old"})
            self.assertIsNone(bridge.latest())
            write_json(directory / "latest.json", {"valid": True, "generation": generation})
            self.assertTrue(bridge.latest()["valid"])
            reset_generation = bridge.initialize()
            self.assertEqual(read_json(directory / "request.json"),
                             {"id": reset_generation, "command": "initialize"})

    def test_bad_prediction_does_not_change_generation_or_publish_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bridge = self.bridge(directory)
            for pose in (np.zeros((4, 4)), np.eye(3), np.full((4, 4), np.nan)):
                with self.assertRaises(ValueError):
                    bridge.initialize(predicted_pose=pose)
                self.assertEqual(bridge.generation, "old")
                self.assertFalse((directory / "request.json").exists())


class PreplaceWorkerTests(unittest.TestCase):
    def test_reestimate_resegments_latest_frame_without_body_prior_or_prediction_as_observation(self):
        from rail_pose.data import Frame
        config = load_config(HERE / "receive.yaml")
        config["pose_preview"] = {"T_C_O_goal": np.eye(4).tolist()}
        original_prior = deepcopy(config["body_orientation_prior"])
        predicted = np.eye(4); predicted[:3, 3] = [.1, 0, 1]
        measured = predicted.copy(); measured[0, 3] = .07
        bounds = np.array([[-.05, -.2, -.005], [.05, .2, .005]])
        stamp = time.monotonic_ns()
        K = np.array([[600., 0, 320], [0, 600., 240], [0, 0, 1]])
        frames = []
        for i in range(6):
            frame_K = K.copy(); frame_K[0, 0] += i
            frames.append(Frame(f"frame_{i}", np.full((480, 640, 3), i, np.uint8),
                                np.ones((480, 640), np.float32), frame_K, stamp+i*1_000_000,
                                metadata={"timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic"}))
        masks = [np.zeros((480, 640), bool) for _ in range(2)]
        masks[0][100:120, 200:250] = True
        masks[1][250:280, 350:400] = True
        detectors, engines = [Mock(), Mock()], [Mock(), Mock()]
        for detector, mask in zip(detectors, masks):
            detector.detect.return_value = mask, {"valid": True, "detector": "sam_prompt"}
        for engine in engines:
            engine.initialized = True
        rgb, rgbd = Mock(), Mock()
        rgb.frame.return_value = frames[0]
        rgbd.frame.side_effect = frames
        engine_configs = []
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_json(directory / "request.json", {"id": "old", "command": "initialize"})
            write_json(directory / "context.json", {"enabled": True, "epoch": 42, "mode": "held"})
            for index, engine in enumerate(engines):
                def infer(frame, *, initialize, index=index):
                    if not initialize:
                        request = ({"id": "new", "command": "initialize", "T_C_O_predicted": predicted.tolist()}
                                   if index == 0 else {"id": "exit", "command": "shutdown"})
                        write_json(directory / "request.json", request)
                    pose = np.eye(4) if index == 0 else measured
                    return pose, {"valid": True, "mode": "register" if initialize else "track", "T_C_CAD": pose.tolist()}
                engine.infer.side_effect = infer
            def make_engine(args, directory, settings):
                index = len(engine_configs)
                detectors[index].close.assert_called_once()
                engine_configs.append(deepcopy(settings))
                return engines[index]
            args = NS(serve=directory, camera_serial="test", mesh=Path(config["mesh"]),
                      foundationpose_root=Path(config["foundationpose_root"]))
            fake_torch = NS(cuda=NS(empty_cache=Mock()))
            with patch.dict(sys.modules, {"torch": fake_torch}), \
                    patch.object(worker, "Camera", side_effect=[rgb, rgbd]), \
                    patch("rail_receive.sam_prompt.SamPromptDetector", side_effect=detectors) as sam_factory, \
                    patch("trimesh.load", return_value=NS(bounds=bounds)), \
                    patch.object(worker, "pose_engine", side_effect=make_engine), \
                    patch("rail_receive.visualization.save_preview"), \
                    patch("rail_receive.visualization.save_mask_snapshot"), \
                    patch.object(worker, "save_frame", return_value=(None, 1)):
                worker.serve(args, config)
            self.assertEqual(engine_configs[0]["body_orientation_prior"], original_prior)
            self.assertFalse(engine_configs[1]["body_orientation_prior"]["enabled"])
            self.assertEqual(config["body_orientation_prior"], original_prior)
            self.assertNotIn("prompt", sam_factory.call_args_list[0].kwargs)
            self.assertEqual(sam_factory.call_args_list[1].kwargs["prompt"],
                             projected_prompt(bounds, predicted, frames[3].K, [640, 480], "test"))
            detectors[1].set_prompt.assert_called_once_with(
                projected_prompt(bounds, predicted, frames[4].K, [640, 480], "test"))
            detectors[1].detect.assert_called_once_with(frames[4])
            registration = engines[1].infer.call_args_list[0]
            self.assertTrue(registration.kwargs["initialize"])
            self.assertEqual(registration.args[0].frame_id, frames[4].frame_id)
            self.assertIs(registration.args[0].mask, masks[1])
            latest = read_json(directory / "latest.json")
            self.assertEqual(latest["generation"], "new")
            self.assertEqual(latest["timestamp_ns"], frames[5].timestamp_ns)
            self.assertEqual(latest["context_epoch"], 42)
            self.assertEqual(latest["pose_source"], "preplace_registration")
            self.assertTrue(latest["reestimate_request"])
            self.assertFalse(latest["body_orientation_prior_enabled"])
            np.testing.assert_allclose(latest["T_C_CAD"], measured)
            self.assertFalse(np.allclose(latest["T_C_CAD"], predicted))
            saved_mask = read_json(directory / "new/sam_detection.json")
            self.assertEqual(saved_mask["prompt_source"], "projected_preplace_pose")
            self.assertTrue((directory / "old/pose_records.jsonl").is_file())
            self.assertTrue((directory / "new/pose_records.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
