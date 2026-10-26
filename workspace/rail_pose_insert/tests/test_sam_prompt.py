"""SAM prompt coordinates and the one-instance mask contract, without devices."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from rail_receive import sam_prompt
from rail_receive.sam_prompt import SamPromptDetector, load_prompt


class SamPromptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "prompt.json"
        self.prompt = {"image_size": [640, 480], "camera_serial": "main-camera",
                       "box_xyxy": [100, 150, 550, 300],
                       "point_coords": [[150, 200], [500, 200], [320, 230]],
                       "point_labels": [1, 1, 0]}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.prompt))

    def detector(self, masks=None):
        # The fake predictor follows the installed official NumPy interface.
        if masks is None:
            masks = np.zeros((1, 480, 640), bool)
            masks[0, 180:210, 120:180] = True
            masks[0, 180:210, 480:530] = True
            masks[0, 50:55, 50:55] = True  # Outside the user box: must be removed.
        predictor, model = Mock(), Mock()
        model.to.return_value = model
        predictor.predict.return_value = masks, np.array([.83]), np.zeros((1, 256, 256))
        registry = {"vit_h": Mock(return_value=model)}
        fake_torch = NS(cuda=NS(is_available=lambda: True, empty_cache=Mock()),
                        device=Mock(side_effect=lambda *_: nullcontext()),
                        inference_mode=lambda: nullcontext())
        segment_anything = NS(SamPredictor=Mock(return_value=predictor), sam_model_registry=registry)
        with patch.dict(sys.modules, {"torch": fake_torch, "segment_anything": segment_anything}), \
                patch.object(sam_prompt, "CHECKPOINT", self.path):
            detector = SamPromptDetector({"sam_prompt": str(self.path), "receive_image_size": [640, 480]},
                                         camera_serial="main-camera")
        self.addCleanup(detector.close)
        return detector, predictor, model, registry, fake_torch

    def test_loads_original_coordinates_and_optional_box_only_points(self):
        self.assertEqual(load_prompt(self.path, [640, 480], "main-camera"), self.prompt)
        del self.prompt["point_coords"], self.prompt["point_labels"]
        self.save()
        prompt = load_prompt(self.path, [640, 480])
        self.assertEqual(prompt["point_coords"], [])
        self.assertEqual(prompt["point_labels"], [])

    def test_rejects_invalid_boxes_points_sizes_labels_and_wrong_camera(self):
        cases = [("box_xyxy", [550, 150, 100, 300]), ("box_xyxy", [0, 0, 641, 480]),
                 ("box_xyxy", [0, 0, float("nan"), 300]), ("box_xyxy", [0, True, 550, 300]),
                 ("point_coords", [[99, 200], [500, 200], [320, 230]]),
                 ("point_coords", [[150, 200], [550, 200], [320, 230]]),
                 ("point_labels", [1, 1, 2]), ("point_labels", [1, 1, True]),
                 ("point_labels", [1]), ("image_size", [1280, 720]),
                 ("image_size", [640., 480]), ("camera_serial", "another-camera")]
        original = self.prompt.copy()
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.prompt = dict(original, **{key: value})
                self.save()
                with self.assertRaises(ValueError):
                    load_prompt(self.path, [640, 480], "main-camera")

    def test_requires_absolute_prompt_path(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            load_prompt("prompt.json", [640, 480])

    def test_official_predictor_receives_box_and_points_without_cropping_or_ranking(self):
        detector, predictor, model, registry, torch = self.detector()
        rgb = np.zeros((480, 640, 3), np.uint8)
        mask, record = detector.detect(NS(rgb=rgb))
        registry["vit_h"].assert_called_once_with(checkpoint=str(self.path))
        torch.device.assert_called_once_with("cpu")
        model.to.assert_called_once_with("cuda")
        model.eval.assert_called_once()
        predictor.set_image.assert_called_once()
        np.testing.assert_array_equal(predictor.set_image.call_args.args[0], rgb)
        self.assertEqual(predictor.set_image.call_args.kwargs, {"image_format": "RGB"})
        kwargs = predictor.predict.call_args.kwargs
        np.testing.assert_array_equal(kwargs["box"], self.prompt["box_xyxy"])
        np.testing.assert_array_equal(kwargs["point_coords"], self.prompt["point_coords"])
        np.testing.assert_array_equal(kwargs["point_labels"], self.prompt["point_labels"])
        self.assertFalse(kwargs["multimask_output"])
        self.assertEqual(mask.shape, (480, 640))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(int(mask.sum()), 3300)
        self.assertTrue(mask[190, 150] and mask[190, 500])  # Both disconnected sides retained.
        self.assertFalse(mask[52, 52])
        self.assertEqual(record["output_instances"], 1)
        self.assertEqual(record["prompt_box_xyxy"], [100, 150, 550, 300])
        self.assertEqual(record["bbox_xyxy"], [120, 180, 530, 210])
        self.assertEqual(record["quality_score_type"], "SAM_predicted_IoU")
        self.assertNotIn("score", record)  # A SAM quality prediction is not a CAD matching score.
        self.assertNotIn("confidence_threshold", record)
        detector.close()
        self.assertIsNone(detector.predictor)
        self.assertIsNone(detector.model)

    def test_box_only_does_not_invent_a_foreground_point_on_the_hand(self):
        self.prompt.update(point_coords=[], point_labels=[])
        self.save()
        detector, predictor, *_ = self.detector()
        detector.detect(NS(rgb=np.zeros((480, 640, 3), np.uint8)))
        self.assertIsNone(predictor.predict.call_args.kwargs["point_coords"])
        self.assertIsNone(predictor.predict.call_args.kwargs["point_labels"])

    def test_empty_mask_is_not_published_as_a_valid_instance(self):
        detector, *_ = self.detector(np.zeros((1, 480, 640), bool))
        mask, record = detector.detect(NS(rgb=np.zeros((480, 640, 3), np.uint8)))
        self.assertIsNone(mask)
        self.assertFalse(record["valid"])
        self.assertEqual(record["output_instances"], 0)
        self.assertEqual(record["reason"], "sam_prompt_empty_mask")
        self.assertNotIn("bbox_xyxy", record)

    def test_image_size_mismatch_does_not_infer_or_rescale_saved_prompt(self):
        detector, predictor, *_ = self.detector()
        mask, record = detector.detect(NS(rgb=np.zeros((240, 320, 3), np.uint8)))
        self.assertIsNone(mask)
        self.assertFalse(record["valid"])
        predictor.predict.assert_not_called()

    def test_unexpected_multiple_masks_are_not_silently_ranked(self):
        detector, *_ = self.detector(np.zeros((3, 480, 640), bool))
        with self.assertRaisesRegex(RuntimeError, "exactly one mask"):
            detector.detect(NS(rgb=np.zeros((480, 640, 3), np.uint8)))


if __name__ == "__main__":
    unittest.main()
