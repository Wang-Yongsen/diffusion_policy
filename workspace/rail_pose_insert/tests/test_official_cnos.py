"""Exercise upstream CNOS filtering/matching/NMS with deterministic proposals."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / 'third_party/cnos')]
from rail_receive.cnos import CnosDetector, official_config
from rail_receive.logic import load_config


class OfficialCnosTests(unittest.TestCase):
    def test_upstream_defaults_keep_full_models_and_processing_parameters(self):
        model = official_config().model
        self.assertEqual(model.segmentor_model.sam.model_type, 'vit_h')
        self.assertEqual(model.segmentor_model.points_per_batch, 64)
        self.assertEqual(model.segmentor_model.stability_score_thresh, .97)
        self.assertEqual(model.descriptor_model.model_name, 'dinov2_vitl14')
        self.assertEqual(model.descriptor_model.chunk_size, 16)
        self.assertEqual(model.matching_config.aggregation_function, 'avg_5')
        self.assertEqual(model.matching_config.confidence_thresh, .15)
        self.assertEqual(model.matching_config.max_num_instances, 100)
        self.assertEqual(model.post_processing_config.mask_post_processing.min_box_size, .05)
        self.assertEqual(model.post_processing_config.mask_post_processing.min_mask_size, 3e-4)
        self.assertEqual(model.post_processing_config.nms_thresh, .25)
        self.assertEqual(model.onboarding_config.rendering_type, 'pyrender')

    def run_pipeline(self, *, matching=True):
        import torch
        from hydra.utils import instantiate
        from src.model.detector import CNOS
        cfg = official_config().model
        calls = []
        class Segmentor:
            def generate_masks(self, rgb):
                masks = torch.zeros((5, 480, 640))
                boxes = [[1, 1, 6, 6], [100, 150, 200, 250], [105, 155, 205, 255],
                         [300, 150, 400, 250], [450, 150, 550, 250]]
                for mask, (x1, y1, x2, y2) in zip(masks, boxes):
                    mask[y1:y2, x1:x2] = 1
                return {'masks': masks, 'boxes': torch.tensor(boxes)}
        class Descriptor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
            def rgb_normalize(self, rgb):
                return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()/255
            def forward(self, rgb, detections):
                calls.append(len(detections))
                features = [[1., 0.], [.99, .1], [0., 1.], [.8, .6]] if matching else [[0., 1.]]*4
                return torch.tensor(features)
        with tempfile.TemporaryDirectory() as directory:
            model = CNOS(segmentor_model=Segmentor(), descriptor_model=Descriptor(),
                onboarding_config=cfg.onboarding_config,
                matching_config=instantiate(cfg.matching_config),
                post_processing_config=cfg.post_processing_config, log_interval=5,
                log_dir=directory, save_mask=True).eval()
            model.dataset_name, model.name_prediction_file = 'rail', 'official'
            model.ref_data = {'descriptors': torch.tensor([[[1., 0.]]*42])}
            (Path(directory)/'predictions/rail/official').mkdir(parents=True)
            adapter = CnosDetector.__new__(CnosDetector)
            adapter.torch, adapter.config = torch, load_config(HERE/'receive.yaml')
            adapter.model, adapter.settings = model, cfg
            adapter.runtime, adapter.roi, adapter.index = NS(name=directory), [0, 0, 640, 480], 1
            with patch.object(model, 'test_step', wraps=model.test_step) as official_step:
                mask, record = adapter.detect(NS(rgb=np.zeros((480, 640, 3), np.uint8)))
            official_step.assert_called_once()
            self.assertEqual(calls, [4])  # Upstream removed the tiny proposal before description.
            if matching:
                self.assertEqual(record['instance_count'], 2)  # Duplicate removed by upstream NMS.
                self.assertEqual(record['output_instances'], 1)
                self.assertEqual(record['bbox_xyxy'], [100, 150, 200, 250])
                self.assertEqual(mask.sum(), 10000)
                self.assertTrue(mask[180, 130])
                self.assertEqual(len(record['instances']), 2)
            else:
                self.assertFalse(record['valid'])
                self.assertEqual(record['instance_count'], 0)
                self.assertIsNone(mask)

    def test_real_official_test_step_filters_matches_and_nms_before_single_output(self):
        self.run_pipeline()

    def test_no_matched_proposal_returns_no_mask(self):
        self.run_pipeline(matching=False)


if __name__ == '__main__':
    unittest.main()
