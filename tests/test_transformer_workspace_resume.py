"""Run the actual loop with a tiny policy to check accumulation and resume."""
import tempfile
import unittest
from unittest.mock import patch, Mock

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import TrainDiffusionTransformerHybridWorkspace


class TinyDataset(BaseImageDataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {'obs': {'state': torch.zeros(2, 1)}, 'action': torch.zeros(16, 52)}

    def get_normalizer(self):
        return None


class TinyPolicy(BaseImagePolicy):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def set_normalizer(self, normalizer):
        pass

    def get_optimizer(self, **kwargs):
        return torch.optim.SGD(self.parameters(), lr=.1)

    def compute_loss(self, batch):
        return (self.weight - 2).square()

    def predict_action(self, obs):
        return {'action_pred': self.weight.expand(len(obs['state']), 16, 52)}


class TinyRunner(BaseImageRunner):
    def run(self, policy):
        return {'offline/skipped': True}


class WorkspaceResumeTests(unittest.TestCase):
    def test_tail_accumulation_checkpoint_and_resume_do_not_repeat_epochs(self):
        config_dir = Path(__file__).resolve().parents[1] / 'diffusion_policy/config'
        OmegaConf.register_new_resolver('eval', eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name='train_diffusion_transformer_tianji_rail_workspace', overrides=[
                'training.device=cpu', 'training.use_ema=false', 'training.num_epochs=2',
                'training.gradient_accumulate_every=2', 'training.max_train_steps=3',
                'training.lr_scheduler=constant', 'training.lr_warmup_steps=0',
                'training.checkpoint_every=1', 'training.sample_every=1',
                'dataloader.batch_size=1', 'dataloader.pin_memory=false'])
            OmegaConf.resolve(cfg)

        def instantiate(config, **kwargs):
            target = config['_target_']
            if '.policy.' in target:
                return TinyPolicy()
            if '.dataset.' in target:
                return TinyDataset()
            if '.env_runner.' in target:
                return TinyRunner(kwargs['output_dir'])
            raise AssertionError(target)

        with tempfile.TemporaryDirectory() as output:
            with patch('hydra.utils.instantiate', side_effect=instantiate), \
                 patch('wandb.init', return_value=Mock()), patch('wandb.config', Mock()):
                workspace = TrainDiffusionTransformerHybridWorkspace(cfg, output_dir=output)
                workspace.run()
                # 2 updates per epoch: one two-batch group and one correctly scaled tail.
                self.assertAlmostEqual(workspace.model.weight.item(), 2*(1-.8**4), places=6)
                self.assertEqual((workspace.epoch, workspace.global_step), (2, 6))
                cfg.training.resume = True
                restored = TrainDiffusionTransformerHybridWorkspace(cfg, output_dir=output)
                restored.run()
                self.assertEqual((restored.epoch, restored.global_step), (2, 6))
                torch.testing.assert_close(restored.model.weight, workspace.model.weight)


if __name__ == '__main__':
    unittest.main()
