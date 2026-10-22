"""Training-only RGB augmentation, temporal consistency and clean evaluation."""

import tempfile
import unittest
from pathlib import Path

import hydra
import numpy as np
import torch

from diffusion_policy.common.image_preprocessing import preprocess_observation
from tests.rail_fixtures import rail_config, write_dataset


class RailImageAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "data.zarr"
        root = write_dataset(self.path, action_space="joint")
        root.attrs["target_fps"] = 30
        root["data"].array("timestamp", np.arange(45, dtype=np.float64) / 30)
        self.cfg = rail_config(
            "train_diffusion_unet_tianji_rail_320_crop_aug_workspace",
            [
                "task.action_space=joint",
                f"task.dataset.dataset_path={self.path}",
                "task.dataset.val_ratio=0.5",
            ],
        )

    def clean_dataset(self):
        return hydra.utils.instantiate(self.cfg.task.dataset, image_augmentation=None)

    def test_each_transform_changes_rgb_and_shares_parameters_across_history(self):
        baseline = self.clean_dataset()[0]
        for selected, name in enumerate(("color_jitter", "blur", "grayscale")):
            with self.subTest(transform=name):
                cfg = self.cfg.copy()
                # Force each operation separately to exercise all three without
                # depending on a rare grayscale or blur draw in this small test.
                for index, operation in enumerate(cfg.task.dataset.image_augmentation.transforms):
                    operation.p = float(index == selected)
                torch.manual_seed(42)
                dataset = hydra.utils.instantiate(cfg.task.dataset)
                sample = dataset[0]
                for key in dataset.rgb_keys:
                    images = sample["obs"][key]
                    self.assertEqual(images.shape, (2, 3, 240, 320))
                    self.assertEqual(images.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(images).all())
                    self.assertGreaterEqual(images.min().item(), 0.0)
                    self.assertLessEqual(images.max().item(), 1.0)
                    self.assertFalse(torch.equal(images, baseline["obs"][key]))
                    # Fixture frames are identical. Independent per-frame draws
                    # would introduce artificial motion or lighting flicker.
                    torch.testing.assert_close(images[0], images[1], rtol=0, atol=0)
                    if name == "grayscale":
                        torch.testing.assert_close(images[:, 0], images[:, 1], rtol=0, atol=0)
                        torch.testing.assert_close(images[:, 1], images[:, 2], rtol=0, atol=0)
                for key in dataset.lowdim_keys:
                    torch.testing.assert_close(
                        sample["obs"][key], baseline["obs"][key], rtol=0, atol=0
                    )
                torch.testing.assert_close(sample["action"], baseline["action"], rtol=0, atol=0)

    def test_default_pipeline_is_reproducible_and_validation_matches_live_preprocessing(self):
        dataset = hydra.utils.instantiate(self.cfg.task.dataset)
        torch.manual_seed(42)
        first = dataset[0]
        torch.manual_seed(42)
        repeated = dataset[0]
        torch.manual_seed(43)
        other = dataset[0]
        self.assertTrue(
            any(not torch.equal(first["obs"][key], other["obs"][key]) for key in dataset.rgb_keys)
        )
        for key in dataset.rgb_keys:
            torch.testing.assert_close(first["obs"][key], repeated["obs"][key], rtol=0, atol=0)

        validation = dataset.get_validation_dataset()
        baseline = self.clean_dataset().get_validation_dataset()
        self.assertIsNotNone(dataset.image_augmentation)
        self.assertIsNone(validation.image_augmentation)
        rng_state = torch.get_rng_state()
        sample = validation[0]
        repeated = validation[0]
        torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)
        start = int(validation.indices[0, 0])
        raw = {
            key: validation.data[key][start : start + validation.n_obs_steps]
            for key in self.cfg.task.shape_meta.obs
        }
        live = preprocess_observation(raw, self.cfg.task.shape_meta, self.cfg.task.dataset)
        clean = baseline[0]
        for key in sample["obs"]:
            torch.testing.assert_close(sample["obs"][key], clean["obs"][key], rtol=0, atol=0)
            torch.testing.assert_close(sample["obs"][key], repeated["obs"][key], rtol=0, atol=0)
            torch.testing.assert_close(
                sample["obs"][key], torch.from_numpy(live[key]), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
