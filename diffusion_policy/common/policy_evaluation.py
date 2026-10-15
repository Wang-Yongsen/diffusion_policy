"""Evaluation mode and reproducible randomness without changing training state."""
from contextlib import contextmanager
import random

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply


@contextmanager
def evaluation_context(policy, seed=None):
    """Disable dropout/gradients and restore module modes and RNG states on exit."""
    modes = [(module, module.training) for module in policy.modules()]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    device = policy.device
    devices = []
    if device.type == 'cuda':
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.default_generator.manual_seed(seed)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(seed)
            policy.eval()
            with torch.no_grad():
                yield policy
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training


def compute_validation_loss(policy, dataloader, device, max_batches=None, seed=None):
    """Sample-weighted diffusion noise MSE; return None for an empty holdout."""
    total_loss, total_samples = 0.0, 0
    with evaluation_context(policy, seed=seed):
        for batch_idx, batch in enumerate(dataloader):
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            loss = policy.compute_loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite validation loss')
            count = len(batch['action'])
            total_loss += loss.item() * count
            total_samples += count
            if max_batches is not None and batch_idx + 1 >= max_batches:
                break
    return total_loss / total_samples if total_samples else None
