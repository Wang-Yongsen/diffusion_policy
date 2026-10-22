"""Rail policy composition, training, deployment reload and process startup."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import hydra
import numpy as np
import torch

from diffusion_policy.common.image_preprocessing import preprocess_observation
from diffusion_policy.common.policy_evaluation import evaluation_context
from tests.rail_fixtures import rail_config, write_dataset


class RailTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_formal_and_smoke_configs_select_rail_task_and_both_action_spaces(self):
        for suffix in ("workspace", "smoke_workspace"):
            for space, dim in (("eef", 58), ("joint", 54)):
                with self.subTest(config=suffix, action_space=space):
                    cfg = rail_config(
                        f"train_diffusion_unet_tianji_rail_{suffix}", [f"task.action_space={space}"]
                    )
                    self.assertEqual(cfg.task.name, "tianji_rail_image")
                    self.assertTrue(cfg.policy._target_.endswith("DiffusionUnetHybridImagePolicy"))
                    self.assertEqual(list(cfg.shape_meta.action.shape), [dim])
                    self.assertEqual(
                        (
                            cfg.task.sample_hz,
                            cfg.n_obs_steps,
                            cfg.horizon,
                            cfg.n_action_steps,
                            cfg.policy.num_inference_steps,
                        ),
                        (30, 2, 16, 8, 8),
                    )
                    self.assertIsNone(cfg.policy.crop_shape)
                    self.assertNotIn("n_layer", cfg.policy)
                    self.assertFalse(cfg.training.resume)
                    self.assertEqual(cfg.logging.mode, "disabled")
                    self.assertEqual(cfg.val_dataloader.num_workers, cfg.dataloader.num_workers)
                    self.assertEqual(cfg.checkpoint.topk.monitor_key, "val_loss")

    def test_real_training_ema_checkpoint_and_deployment_prediction(self):
        from diffusion_policy.real_world.tianji import evaluation as entry

        # Smaller spatial/UNet dimensions make this a CPU integration test; all
        # encoders, losses, EMA, optimizer, checkpoint and inference paths are real.
        cases = (
            ("train_diffusion_unet_tianji_rail_smoke_workspace", "eef", 58, 2, "null", 30),
            ("train_diffusion_unet_tianji_rail_smoke_workspace", "joint", 54, 2, "null", 30),
            (
                "train_diffusion_unet_tianji_rail_320_crop_aug_smoke_workspace",
                "joint",
                54,
                2,
                "[28,28]",
                30,
            ),
            ("train_diffusion_transformer_tianji_rail_smoke_workspace", "eef", 58, 1, "null", 10),
            ("train_diffusion_transformer_tianji_rail_smoke_workspace", "joint", 54, 2, "null", 10),
        )
        for name, space, dim, obs_steps, crop, hz in cases:
            with (
                self.subTest(config=name, action_space=space),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                data_path, output = root / "data.zarr", root / "run"
                data = write_dataset(data_path, action_space=space)
                data.attrs["target_fps"] = hz
                data["data"].array("timestamp", np.arange(45, dtype=np.float64) / hz)
                output.mkdir()
                model_options = (
                    ["policy.down_dims=[16,32,64]", "policy.diffusion_step_embed_dim=32"]
                    if "unet" in name
                    else ["policy.n_layer=1", "policy.n_emb=16", "policy.n_head=4"]
                )
                cfg = rail_config(
                    name,
                    [
                        f"task.action_space={space}",
                        f"task.dataset.dataset_path={data_path}",
                        "task.dataset.val_ratio=0.5",
                        "task.image_shape=[3,32,32]",
                        "task.dataset.resize_enabled=true",
                        f"n_obs_steps={obs_steps}",
                        *model_options,
                        f"task.sample_hz={hz}",
                        f"policy.crop_shape={crop}",
                        "policy.num_inference_steps=2",
                        "training.device=cpu",
                        "training.num_epochs=1",
                        "training.max_train_steps=1",
                        "training.max_val_steps=1",
                        "dataloader.pin_memory=false",
                        "checkpoint.topk.k=0",
                    ],
                )
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

                dataset = hydra.utils.instantiate(cfg.task.dataset).get_validation_dataset()
                sample = dataset[0]
                obs = {key: value.unsqueeze(0) for key, value in sample["obs"].items()}
                with evaluation_context(workspace.ema_model, seed=42):
                    expected = workspace.ema_model.predict_action(obs)
                restored, saved_cfg, state_key = entry.load_policy(
                    output / "checkpoints/latest.ckpt", "cpu"
                )
                self.assertEqual(state_key, "ema_model")
                self.assertEqual(entry.validate_policy_frequency(saved_cfg, hz), hz)
                start = int(dataset.indices[0, 0])
                raw = {
                    key: dataset.data[key][start : start + obs_steps]
                    for key in saved_cfg.task.shape_meta.obs
                }
                prepared = preprocess_observation(
                    raw, saved_cfg.task.shape_meta, saved_cfg.task.dataset
                )
                live_obs = {
                    key: torch.from_numpy(value).unsqueeze(0) for key, value in prepared.items()
                }
                with evaluation_context(restored, seed=42):
                    actual = restored.predict_action(live_obs)
                self.assertEqual(actual["action"].shape, (1, 8, dim))
                self.assertEqual(actual["action_pred"].shape, (1, 16, dim))
                torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
                self.assertTrue(torch.isfinite(actual["action"]).all())
                if "aug" in name or (space == "eef" and "unet" in name):
                    self.check_external_process(
                        root, output / "checkpoints/latest.ckpt", raw, actual["action"]
                    )

    def check_external_process(self, root, checkpoint, observation, expected):
        import subprocess
        import sys
        import textwrap

        outside = root / "outside"
        outside.mkdir()
        inputs = outside / "inputs.npz"
        np.savez(inputs, **observation, expected=expected.cpu().numpy())
        script = outside / "check_inference.py"
        script.write_text(
            textwrap.dedent("""\
            import importlib.abc
            import json
            from pathlib import Path
            import sys
            import numpy as np
            import torch

            class RejectLegacyImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname in {"eval_tianji_rail", "home_tianji_rail"}:
                        raise ImportError("Package code must not import a root script")

            sys.meta_path.insert(0, RejectLegacyImports())

            if __name__ == "__main__":
                from diffusion_policy.real_world.tianji.inference import PolicyInferenceProcess
                from diffusion_policy.cli.tianji.eval import main as check_model
                from diffusion_policy.cli.tianji.probe import main as probe
                torch.set_num_threads(1)
                checkpoint, inputs, dataset = sys.argv[1:]
                with np.load(inputs) as saved:
                    obs = {key: saved[key] for key in saved.files if key != "expected"}
                    expected = saved["expected"]
                client = PolicyInferenceProcess(checkpoint, "cpu")
                try:
                    client.reseed(42)
                    np.testing.assert_array_equal(client.predict_observation(obs)["action"].numpy(), expected)
                finally:
                    client.close()
                assert not client._process.is_alive()
                output = Path(inputs).parent
                assert check_model(["--checkpoint", checkpoint, "--device", "cpu", "--check-only",
                                    "--output", str(output / "preflight")]) == 0
                assert probe(["--checkpoint", checkpoint, "--train-zarr", dataset, "--val-zarr", dataset,
                              "--output", str(output / "probe"), "--chunks", "1"]) == 0
                report = json.loads((output / "probe/obs_gap_results.json").read_text())
                assert all(field["mae"] == 0 for field in report["observation_difference"].values())
            """)
        )
        result = subprocess.run(
            [sys.executable, str(script), str(checkpoint), str(inputs), str(root / "data.zarr")],
            cwd=outside,
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
