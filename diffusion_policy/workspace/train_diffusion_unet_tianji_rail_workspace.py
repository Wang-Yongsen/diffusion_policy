"""Rail UNet optimizer specialization; keep this class path for checkpoints."""

import hydra

from diffusion_policy.workspace.train_image_workspace import TrainImageWorkspace


class TrainDiffusionUnetTianjiRailWorkspace(TrainImageWorkspace):
    def _create_optimizer(self, cfg):
        return hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
