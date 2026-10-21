"""UNet rail training with the shared EMA validation and resume loop."""

import hydra

from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import (
    TrainDiffusionTransformerHybridWorkspace,
)


class TrainDiffusionUnetTianjiRailWorkspace(TrainDiffusionTransformerHybridWorkspace):
    # The training loop uses the common image-policy interface. Only optimizer
    # construction differs: UNet uses AdamW over the whole policy, whereas the
    # Transformer groups its encoder and denoiser parameters separately.
    def _create_optimizer(self, cfg):
        return hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
