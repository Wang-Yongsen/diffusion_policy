"""Rail UNet composition, training, evaluation and deployment reload, without hardware."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from diffusion_policy.common.policy_evaluation import evaluation_context
from diffusion_policy.real_world.tianji_rail_env import preprocess_observation
from tests.test_tianji_rail import write_dataset


CONFIG_DIR = Path(__file__).resolve().parents[1] / "diffusion_policy/config"


def rail_config(name, overrides=()):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=name, overrides=list(overrides))
        OmegaConf.resolve(cfg)
        return cfg


class RailUnetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_formal_and_smoke_configs_select_rail_task_and_both_action_spaces(self):
        for suffix in ("workspace", "smoke_workspace"):
            for space, dim in (("eef", 52), ("joint", 54)):
                with self.subTest(config=suffix, action_space=space):
                    cfg = rail_config(f"train_diffusion_unet_tianji_rail_{suffix}",
                                      [f"task.action_space={space}"])
                    self.assertEqual(cfg.task.name, "tianji_rail_image")
                    self.assertTrue(cfg.policy._target_.endswith("DiffusionUnetHybridImagePolicy"))
                    self.assertEqual(list(cfg.shape_meta.action.shape), [dim])
                    self.assertEqual((cfg.task.sample_hz, cfg.n_obs_steps, cfg.horizon,
                                      cfg.n_action_steps, cfg.policy.num_inference_steps), (30, 2, 16, 8, 8))
                    self.assertIsNone(cfg.policy.crop_shape)
                    self.assertNotIn("n_layer", cfg.policy)
                    self.assertFalse(cfg.training.resume)
                    self.assertEqual(cfg.logging.mode, "disabled")
                    self.assertEqual(cfg.val_dataloader.num_workers, cfg.dataloader.num_workers)
                    self.assertEqual(cfg.checkpoint.topk.monitor_key, "val_loss")

    def test_real_unet_training_ema_checkpoint_and_deployment_prediction(self):
        import eval_tianji_rail as entry

        # Smaller spatial/UNet dimensions make this a CPU integration test; all
        # encoders, losses, EMA, optimizer, checkpoint and inference paths are real.
        for space, dim, obs_steps in (("eef", 52, 1), ("joint", 54, 2)):
            with self.subTest(action_space=space), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data_path, output = root / "data.zarr", root / "run"
                data = write_dataset(data_path, action_space=space)
                data.attrs["target_fps"] = 30
                data["data"].array("timestamp", np.arange(45, dtype=np.float64) / 30)
                output.mkdir()
                cfg = rail_config("train_diffusion_unet_tianji_rail_smoke_workspace", [
                    f"task.action_space={space}", f"task.dataset.dataset_path={data_path}",
                    "task.dataset.val_ratio=0.5", "task.image_shape=[3,32,32]",
                    "task.dataset.resize_enabled=true", f"n_obs_steps={obs_steps}",
                    "policy.down_dims=[16,32,64]", "policy.diffusion_step_embed_dim=32",
                    "policy.num_inference_steps=2", "training.device=cpu",
                    "training.num_epochs=1", "training.max_train_steps=1",
                    "training.max_val_steps=1", "dataloader.pin_memory=false",
                    "checkpoint.topk.k=0",
                ])
                workspace_class = hydra.utils.get_class(cfg._target_)
                with patch("wandb.init", return_value=Mock()), patch("wandb.config", Mock()):
                    workspace = workspace_class(cfg, output_dir=str(output))
                    self.assertIsInstance(workspace.optimizer, torch.optim.AdamW)
                    workspace.run()
                self.assertEqual((workspace.epoch, workspace.global_step), (1, 1))
                log = json.loads((output / "logs.json.txt").read_text().splitlines()[-1])
                self.assertTrue(log["val_loss_uses_ema"])
                for key in ("train_loss", "val_loss", "train_action_mse_error"):
                    self.assertTrue(np.isfinite(log[key]), key)
                report = json.loads((output / "offline_report.json").read_text())
                self.assertEqual(report["status"], "complete")

                dataset = hydra.utils.instantiate(cfg.task.dataset)
                sample = dataset[0]
                obs = {key: value.unsqueeze(0) for key, value in sample["obs"].items()}
                with evaluation_context(workspace.ema_model, seed=42):
                    expected = workspace.ema_model.predict_action(obs)
                restored, saved_cfg, state_key = entry.load_policy(output / "checkpoints/latest.ckpt", "cpu")
                self.assertEqual(state_key, "ema_model")
                self.assertEqual(entry.validate_policy_frequency(saved_cfg, 30), 30)
                start = int(dataset.indices[0, 0])
                raw = {key: dataset.data[key][start:start + obs_steps] for key in saved_cfg.task.shape_meta.obs}
                prepared = preprocess_observation(raw, saved_cfg.task.shape_meta, saved_cfg.task.dataset)
                live_obs = {key: torch.from_numpy(value).unsqueeze(0) for key, value in prepared.items()}
                with evaluation_context(restored, seed=42):
                    actual = restored.predict_action(live_obs)
                self.assertEqual(actual["action"].shape, (1, 8, dim))
                self.assertEqual(actual["action_pred"].shape, (1, 16, dim))
                torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(actual["action"]).all())


if __name__ == "__main__":
    unittest.main()
