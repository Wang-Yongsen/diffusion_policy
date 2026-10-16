"""Offline checks for source leakage, image processing and metric semantics."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import zarr

from diffusion_policy.dataset.tianji_rail_image_dataset import TianjiRailImageDataset
from diffusion_policy.env_runner.tianji_rail_offline_runner import TianjiRailOfflineRunner, action_errors
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def write_dataset(path, *, legacy=False):
    root = zarr.open_group(str(path), mode='w')
    root.attrs.update(schema_version=1 if legacy else 2, action_space='eef')
    data, meta = root.create_group('data'), root.create_group('meta')
    # First demonstration has a 20-frame segment and a 5-frame short segment.
    meta.array('episode_ends', np.asarray([20, 25, 45] if legacy else [25, 45], np.int64))
    if not legacy:
        meta.array('segment_ends', np.asarray([20, 25, 45], np.int64))
    meta.attrs['segments'] = [{'source_episode': s} for s in ('a', 'a', 'b')]
    timeline = np.arange(45, dtype=np.float32) / 100
    data.array('action', np.repeat(timeline[:, None], 52, axis=1))
    data.array('robot_eef_pose', np.repeat(timeline[:, None], 12, axis=1))
    data.array('hand_joint', np.repeat(timeline[:, None], 40, axis=1))
    # Distinct channels and a spatial pattern expose transpose/resize errors.
    images = np.zeros((45, 12, 16, 3), np.uint8)
    images[..., 0], images[..., 1], images[..., 2] = 255, np.arange(16)[None, None, :], 50
    for key in ('camera_0', 'camera_1', 'camera_2'):
        data.array(key, images, chunks=(1, 12, 16, 3))
    return root


def make_dataset(path, **options):
    shape = options.pop('image_shape', (3, 6, 8))
    obs = {key: {'shape': shape, 'type': 'rgb'} for key in ('camera_0', 'camera_1', 'camera_2')}
    obs.update(robot_eef_pose={'shape': [12]}, hand_joint={'shape': [40]})
    return TianjiRailImageDataset(path, {'obs': obs, 'action': {'shape': [52]}}, image_shape=shape, **options)


class IndexedPolicy(BaseImagePolicy):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))

    def predict_action(self, obs):
        # Last observation corresponds to execution label at n_obs_steps-1.
        start = obs['robot_eef_pose'][:, -1, 0]
        values = start[:, None] + torch.arange(8, device=start.device)[None, :] / 100
        return {'action': values[:, :, None].expand(-1, -1, 52)}


class TianjiRailTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'data.zarr'
        self.root = write_dataset(self.path)

    def test_source_split_short_segments_and_training_only_normalizer(self):
        dataset = make_dataset(self.path, val_ratio=.5)
        val = dataset.get_validation_dataset()
        train_sources = {dataset.sources[s] for _, s in dataset.indices}
        val_sources = {val.sources[s] for _, s in val.indices}
        self.assertFalse(train_sources & val_sources)
        self.assertEqual(len(dataset), 5)
        self.assertEqual(len(val), 5)
        self.assertNotIn(1, dataset.indices[:, 1])
        self.assertNotIn(1, val.indices[:, 1])
        stats = dataset.get_normalizer().get_input_stats()['action']
        expected = dataset.get_all_actions().numpy()
        np.testing.assert_allclose(stats['max'].numpy(), expected.max(axis=0))
        np.testing.assert_allclose(stats['min'].numpy(), expected.min(axis=0))

    def test_lazy_image_resize_switch_channels_and_no_padding(self):
        dataset = make_dataset(self.path)
        sample = dataset[0]
        self.assertIsInstance(dataset.data['camera_0'], zarr.Array)
        self.assertEqual(sample['obs']['camera_0'].shape, (2, 3, 6, 8))
        self.assertTrue((sample['obs']['camera_0'][:, 0] == 1).all())
        self.assertEqual(sample['action'].shape, (16, 52))
        native = make_dataset(self.path, image_shape=(3, 12, 16), resize_enabled=False)
        nearest = make_dataset(self.path, resize_interpolation='nearest')
        self.assertEqual(native[0]['obs']['camera_0'].shape, (2, 3, 12, 16))
        self.assertFalse(torch.equal(nearest[0]['obs']['camera_0'], sample['obs']['camera_0']))
        with self.assertRaisesRegex(ValueError, 'resolution'):
            make_dataset(self.path, resize_enabled=False)
        # Five-frame short segment contributes no windows, and remaining windows never cross it.
        self.assertEqual(len(dataset), 10)
        for start, segment in dataset.indices:
            self.assertLessEqual(start + 16, dataset.segment_ends[segment])

    def test_legacy_source_metadata_prevents_leakage(self):
        path = Path(self.temp.name) / 'legacy.zarr'
        write_dataset(path, legacy=True)
        dataset = make_dataset(path, val_ratio=.5)
        self.assertEqual(dataset.n_source_episodes, 2)
        self.assertEqual(dataset.train_mask[0], dataset.train_mask[1])

    def test_sampling_frequency_checks_metadata_and_segment_timestamps(self):
        # A new source/segment can restart its time axis; only adjacent values
        # within a segment must be 100 ms apart.
        times = np.r_[np.arange(20), np.arange(5), np.arange(20)] / 10
        self.root['data'].array('timestamp', times)
        self.root.attrs['target_fps'] = 10
        dataset = make_dataset(self.path, sample_hz=10)
        self.assertEqual(dataset.sample_hz, 10)
        self.assertEqual(len(dataset), 10)
        self.root.attrs['target_fps'] = 30
        with self.assertRaisesRegex(ValueError, 'target_fps=30'):
            make_dataset(self.path, sample_hz=10)
        self.root.attrs['target_fps'] = 10
        self.root['data/timestamp'][1] += .01
        with self.assertRaisesRegex(ValueError, 'timestamp intervals'):
            make_dataset(self.path, sample_hz=10)
        del self.root.attrs['target_fps']
        with self.assertRaisesRegex(ValueError, 'target_fps=None'):
            make_dataset(self.path, sample_hz=10)

    def test_boundary_corruption_and_impossible_holdout_rejected(self):
        self.root['meta/segment_ends'][:] = [20, 24, 45]
        with self.assertRaisesRegex(ValueError, 'preserve'):
            make_dataset(self.path)
        self.root['meta/segment_ends'][:] = [20, 25, 45]
        self.root['meta'].attrs['segments'] = [{'source_episode': 'a'}] * 3
        with self.assertRaisesRegex(ValueError, 'two independent'):
            make_dataset(self.path, val_ratio=.1)

    def test_metrics_units_and_rotation_wrap(self):
        target = np.zeros((2, 8, 52))
        prediction = target.copy()
        prediction[..., 0] = .003
        prediction[..., 1] = .004
        prediction[..., 5] = 2*np.pi
        prediction[..., 11] = np.pi/2
        prediction[..., 12:32] = np.pi/180
        errors = action_errors(prediction, target)
        np.testing.assert_allclose(errors['left_position_mm'], 5)
        np.testing.assert_allclose(errors['left_rotation_deg'], 0, atol=1e-12)
        np.testing.assert_allclose(errors['right_rotation_deg'], 90)
        np.testing.assert_allclose(errors['left_hand_joint_mae_deg'], 1)

    def test_runner_execution_alignment_report_and_empty_validation(self):
        policy = IndexedPolicy()
        runner = TianjiRailOfflineRunner(self.temp.name, make_dataset(self.path, val_ratio=.5), window_stride=1)
        result = runner.run(policy)
        self.assertFalse(result['offline/skipped'])
        self.assertEqual(result['offline/windows'], 5)
        self.assertLess(result['offline/left_position_mm'], .001)
        self.assertTrue(policy.training)
        runner = TianjiRailOfflineRunner(self.temp.name, make_dataset(self.path))
        self.assertTrue(runner.run(policy)['offline/skipped'])

    def test_fixed_crop_eval_and_disabled_policy_crop_contract(self):
        from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
        crop = CropRandomizer((3, 12, 16), 10, 14)
        images = torch.arange(3*12*16).reshape(1, 3, 12, 16).float()
        crop.eval()
        torch.testing.assert_close(crop(images), images[:, :, 1:11, 1:15])

    def test_policy_reports_invalid_crop_dimensions_before_encoder_build(self):
        from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
        from diffusers import DDIMScheduler
        meta = make_dataset(self.path).shape_meta
        for shape in ([6, 8], [0, 4], [3.5, 4]):
            with self.subTest(crop=shape), self.assertRaisesRegex(ValueError, 'crop_shape'):
                DiffusionTransformerHybridImagePolicy(meta, DDIMScheduler(), 16, 8, 2, crop_shape=shape)


if __name__ == '__main__':
    unittest.main()
