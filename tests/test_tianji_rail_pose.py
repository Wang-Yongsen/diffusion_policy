"""DP rotation conventions across dataset, normalization and device boundaries."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from diffusion_policy.common.image_preprocessing import preprocess_observation
from diffusion_policy.common.tianji_rail_pose import (
    EEF_ACTION_ROTATION_INDICES,
    EEF_OBS_ROTATION_INDICES,
    eef_action_to_6d,
    eef_action_to_rotvec,
    eef_observation_to_quaternion,
    policy_actions_to_controller,
    rotation_6d_to_matrix,
)
from diffusion_policy.env_runner.tianji_rail_offline_runner import action_errors
from diffusion_policy.real_world.tianji.checkpoint import load_policy
from diffusion_policy.real_world.tianji.replay import _predict
from diffusion_policy.real_world.tianji.runtime import split_action
from tests.rail_fixtures import make_dataset, rail_config, write_dataset


class RailPoseTests(unittest.TestCase):
    def test_rotation_conventions_layout_and_roundtrip(self):
        raw = np.arange(2 * 3 * 52, dtype=np.float64).reshape(2, 3, 52) / 100
        raw[..., 3:6] = [0.2, -0.4, 0.7]
        raw[..., 9:12] = [np.pi - 1e-5, 0, 0]
        poses = raw[..., :12]
        observed = eef_observation_to_quaternion(poses)
        self.assertEqual(observed.shape, (2, 3, 14))
        np.testing.assert_array_equal(observed[..., :3], raw[..., :3])
        np.testing.assert_array_equal(observed[..., 7:10], raw[..., 6:9])
        for offset, source in ((3, 3), (10, 9)):
            quaternion = observed[..., offset:offset + 4].reshape(-1, 4)
            np.testing.assert_allclose(np.linalg.norm(quaternion, axis=-1), 1)
            expected = Rotation.from_rotvec(raw[..., source:source + 3].reshape(-1, 3))
            np.testing.assert_allclose(Rotation.from_quat(quaternion).as_matrix(), expected.as_matrix())
            self.assertTrue((quaternion[:, 3] >= 0).all())
        encoded = eef_action_to_6d(raw)
        self.assertEqual(encoded.shape, (2, 3, 58))
        np.testing.assert_array_equal(encoded[..., 18:], raw[..., 12:])
        for offset, source in ((3, 3), (12, 9)):
            matrix = Rotation.from_rotvec(raw[..., source:source + 3].reshape(-1, 3)).as_matrix()
            np.testing.assert_allclose(encoded[..., offset:offset + 6].reshape(-1, 6),
                                       matrix[:, :2, :].reshape(-1, 6))
        decoded = eef_action_to_rotvec(encoded)
        np.testing.assert_allclose(decoded, raw, atol=1e-12)

    def test_6d_projection_and_degenerate_prediction_rejection(self):
        # Rows need not have unit length or be orthogonal after denoising.
        matrix = rotation_6d_to_matrix([2, 0, 0, 3, 4, 0])
        np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(matrix), 1)
        for invalid in (np.zeros(6), [1, 0, 0, 2, 0, 0], [np.nan, 0, 0, 0, 1, 0]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                rotation_6d_to_matrix(invalid)

    def test_dataset_and_live_preprocessing_share_quaternions_and_identity_rotation_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.zarr"
            root = write_dataset(path)
            dataset = make_dataset(path, val_ratio=0.5)
            sample = dataset[0]
            start = int(dataset.indices[0, 0])
            raw = {key: root['data'][key][start:start + 2] for key in dataset.shape_meta['obs']}
            live = preprocess_observation(raw, dataset.shape_meta, {'resize_enabled': True})
            for key, value in sample['obs'].items():
                np.testing.assert_array_equal(live[key], value.numpy())
            normalized = dataset.get_normalizer()
            for key, indices in (('robot_eef_pose', EEF_OBS_ROTATION_INDICES),
                                 ('action', EEF_ACTION_ROTATION_INDICES)):
                params = normalized[key].params_dict
                np.testing.assert_array_equal(params['scale'][list(indices)].numpy(), 1)
                np.testing.assert_array_equal(params['offset'][list(indices)].numpy(), 0)
                values = dataset._training_values(key)
                transformed = normalized[key].normalize(values).numpy()
                np.testing.assert_array_equal(transformed[:, indices], values[:, indices])
                remaining = np.setdiff1d(np.arange(values.shape[-1]), indices)
                np.testing.assert_allclose(transformed[:, remaining].min(axis=0), -1, atol=1e-6)
                np.testing.assert_allclose(transformed[:, remaining].max(axis=0), 1, atol=1e-6)
                np.testing.assert_allclose(normalized[key].unnormalize(transformed), values, atol=1e-6)
            # Reading/training does not rewrite the existing recording schema.
            self.assertEqual(root['data/robot_eef_pose'].shape[-1], 12)
            self.assertEqual(root['data/action'].shape[-1], 52)
            np.testing.assert_allclose(eef_action_to_rotvec(sample['action'].numpy()),
                                       root['data/action'][start:start + 16], atol=1e-7)

    def test_replay_and_metrics_decode_before_controller(self):
        raw = np.zeros((8, 52), np.float64)
        raw[:, :3] = [0.1, 0.2, 0.3]
        raw[:, 3:6] = Rotation.from_euler('z', 90, degrees=True).as_rotvec()
        raw[:, 12:32], raw[:, 32:] = 0.1, 0.2
        encoded = eef_action_to_6d(raw)
        inference = SimpleNamespace(predict_observation=lambda obs: {
            'action': torch.from_numpy(encoded[None])
        })
        decoded = _predict(inference, {})
        np.testing.assert_allclose(decoded, raw, atol=1e-12)
        arms, hands = split_action(decoded[0])
        np.testing.assert_allclose(arms['left'].position_m, raw[0, :3])
        np.testing.assert_allclose(Rotation.from_quat(arms['left'].orientation_xyzw).as_rotvec(),
                                   raw[0, 3:6])
        np.testing.assert_allclose(hands['right'].position_rad, raw[0, 32:])
        target = np.zeros_like(raw)
        expected = action_errors(raw, target)
        for key, values in action_errors(encoded, eef_action_to_6d(target)).items():
            np.testing.assert_allclose(values, expected[key], atol=1e-10)
        joint = np.arange(54, dtype=np.float32)
        np.testing.assert_array_equal(policy_actions_to_controller(joint, 'joint'), joint)

    def test_old_eef_checkpoint_is_rejected_before_model_build(self):
        cfg = rail_config('train_diffusion_unet_tianji_rail_workspace')
        cfg.task.shape_meta.action.shape = [52]
        cfg.task.shape_meta.obs.robot_eef_pose.shape = [12]
        with patch('torch.load', return_value={'cfg': cfg}), patch('hydra.utils.instantiate') as build:
            with self.assertRaisesRegex(ValueError, 'Legacy 52-D EEF checkpoint.*retrain'):
                load_policy('legacy.ckpt', 'cpu')
            build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
