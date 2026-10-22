"""Transformer optimizer specialization; keep this class path for checkpoints."""

import hydra

from diffusion_policy.common.paths import CONFIG_DIR
from diffusion_policy.workspace.train_image_workspace import TrainImageWorkspace


class TrainDiffusionTransformerHybridWorkspace(TrainImageWorkspace):
    def _create_optimizer(self, cfg):
        return self.model.get_optimizer(**cfg.optimizer)


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIR),
    config_name="train_diffusion_transformer_hybrid_workspace",
)
def main(cfg):
    TrainDiffusionTransformerHybridWorkspace(cfg).run()


if __name__ == "__main__":
    main()
