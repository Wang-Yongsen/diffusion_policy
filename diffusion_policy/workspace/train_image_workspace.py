"""Shared image-policy training, validation and checkpoint lifecycle."""

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.policy_evaluation import (
    compute_validation_loss,
    evaluation_context,
)
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass
class TrainingRun:
    train_loader: DataLoader
    val_loader: DataLoader
    train_batches: int
    scheduler: object
    ema: object
    runner: BaseImageRunner
    log: object
    checkpoints: TopKCheckpointManager
    device: torch.device
    sampling_batch: object = None


class TrainImageWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        random.seed(cfg.training.seed)
        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None
        self.optimizer = self._create_optimizer(cfg)
        self.global_step = 0
        self.epoch = 0

    def _create_optimizer(self, cfg):
        raise NotImplementedError

    def run(self):
        cfg = self._training_config()
        self._resume(cfg)
        run = self._prepare_training(cfg)
        try:
            with JsonLogger(Path(self.output_dir) / "logs.json.txt") as json_log:
                while self.epoch < cfg.training.num_epochs:
                    metrics = self._train_epoch(cfg, run, json_log)
                    metrics.update(self._evaluate_epoch(cfg, run))
                    self._save_epoch(cfg, run, metrics)
                    run.log.log(metrics, step=self.global_step - 1)
                    json_log.log(metrics)
        finally:
            self._wait_for_checkpoint()
            run.log.finish()

    def _training_config(self):
        cfg = copy.deepcopy(self.cfg)
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            for key in ("rollout_every", "checkpoint_every", "val_every", "sample_every"):
                cfg.training[key] = 1
        if cfg.training.gradient_accumulate_every < 1:
            raise ValueError("gradient_accumulate_every must be positive")
        return cfg

    def _resume(self, cfg):
        path = self.get_checkpoint_path()
        if cfg.training.resume and path.is_file():
            print(f"Resuming from checkpoint {path}")
            self.load_checkpoint(path=path)

    def _prepare_training(self, cfg):
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError("Training requires a BaseImageDataset")
        train_loader = DataLoader(dataset, **cfg.dataloader)
        validation = dataset.get_validation_dataset()
        val_loader = DataLoader(validation, **cfg.val_dataloader)
        if len(validation) == 0 and cfg.checkpoint.topk.monitor_key == "val_loss":
            raise ValueError("Cannot monitor val_loss with an empty validation dataset")
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        train_batches = len(train_loader)
        if cfg.training.max_train_steps is not None:
            train_batches = min(train_batches, cfg.training.max_train_steps)
        if train_batches < 1:
            raise ValueError("Training requires at least one batch")
        updates_per_epoch = math.ceil(train_batches / cfg.training.gradient_accumulate_every)
        scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=updates_per_epoch * cfg.training.num_epochs,
            last_epoch=self.epoch * updates_per_epoch - 1,
        )
        ema = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            ema.optimization_step = self.epoch * updates_per_epoch
        runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
        if not isinstance(runner, BaseImageRunner):
            raise TypeError("Training requires a BaseImageRunner")
        log = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})
        checkpoints = TopKCheckpointManager(
            save_dir=str(Path(self.output_dir) / "checkpoints"),
            **cfg.checkpoint.topk,
        )
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        return TrainingRun(
            train_loader=train_loader,
            val_loader=val_loader,
            train_batches=train_batches,
            scheduler=scheduler,
            ema=ema,
            runner=runner,
            log=log,
            checkpoints=checkpoints,
            device=device,
        )

    def _train_epoch(self, cfg, run, json_log):
        losses = []
        accumulation = cfg.training.gradient_accumulate_every
        self.optimizer.zero_grad()
        with self._progress(run.train_loader, cfg, "Training") as batches:
            for index, batch in enumerate(batches):
                batch = dict_apply(batch, lambda value: value.to(run.device, non_blocking=True))
                if run.sampling_batch is None:
                    run.sampling_batch = batch
                loss = self.model.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                group_start = (index // accumulation) * accumulation
                # Scale the final incomplete accumulation group by its actual size.
                (loss / min(accumulation, run.train_batches - group_start)).backward()
                last_batch = index + 1 == run.train_batches
                if (index + 1) % accumulation == 0 or last_batch:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    run.scheduler.step()
                    if run.ema is not None:
                        run.ema.step(self.model)
                value = loss.item()
                batches.set_postfix(loss=value, refresh=False)
                losses.append(value)
                metrics = {
                    "train_loss": value,
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                    "lr": run.scheduler.get_last_lr()[0],
                }
                if last_batch:
                    break
                run.log.log(metrics, step=self.global_step)
                json_log.log(metrics)
                self.global_step += 1
        metrics["train_loss"] = np.mean(losses)
        return metrics

    def _evaluate_epoch(self, cfg, run):
        policy = self.ema_model if cfg.training.use_ema else self.model
        metrics = {}
        if self.epoch % cfg.training.rollout_every == 0:
            with evaluation_context(policy, seed=cfg.training.seed):
                metrics.update(run.runner.run(policy))
        if self.epoch % cfg.training.val_every == 0:
            with self._progress(run.val_loader, cfg, "Validation") as batches:
                loss = compute_validation_loss(
                    policy,
                    batches,
                    run.device,
                    max_batches=cfg.training.max_val_steps,
                    seed=cfg.training.seed,
                )
            if loss is not None:
                metrics.update(
                    val_loss=loss,
                    val_loss_uses_ema=cfg.training.use_ema,
                    val_loss_seed=cfg.training.seed,
                )
        if self.epoch % cfg.training.sample_every == 0:
            with evaluation_context(policy, seed=cfg.training.seed):
                batch = dict_apply(
                    run.sampling_batch, lambda value: value.to(run.device, non_blocking=True)
                )
                prediction = policy.predict_action(batch["obs"])["action_pred"]
                metrics["train_action_mse_error"] = torch.nn.functional.mse_loss(
                    prediction, batch["action"]
                ).item()
        return metrics

    def _save_epoch(self, cfg, run, metrics):
        completed_epoch = self.epoch
        # Checkpoints store the next epoch and the exact number of completed batches.
        self.global_step += 1
        self.epoch += 1
        if completed_epoch % cfg.training.checkpoint_every != 0:
            return
        self._wait_for_checkpoint()
        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint()
        if cfg.checkpoint.save_last_snapshot:
            self._wait_for_checkpoint()
            self.save_snapshot()
        values = {key.replace("/", "_"): value for key, value in metrics.items()}
        path = run.checkpoints.get_ckpt_path(values)
        if path is not None:
            self._wait_for_checkpoint()
            self.save_checkpoint(path=path)

    def _wait_for_checkpoint(self):
        if self._saving_thread is not None:
            self._saving_thread.join()

    def _progress(self, loader, cfg, label):
        return tqdm.tqdm(
            loader,
            desc=f"{label} epoch {self.epoch}",
            leave=False,
            mininterval=cfg.training.tqdm_interval_sec,
        )
