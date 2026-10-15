"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.policy_evaluation import evaluation_context

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--dataset_path', type=click.Path(exists=True), default=None,
              help='Override the dataset location saved in the checkpoint after moving it.')
@click.option('--seed', type=int, default=None,
              help='Evaluation random seed; defaults to the checkpoint training seed.')
@click.option('--clip-sample/--no-clip-sample', default=None,
              help='Override DDIM sample clipping for this evaluation only.')
def main(checkpoint, output_dir, device, dataset_path, seed, clip_sample):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    if dataset_path is not None:
        cfg.task.dataset.dataset_path = str(pathlib.Path(dataset_path).resolve())
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    scheduler = getattr(policy, 'noise_scheduler', None)
    if clip_sample is not None:
        if scheduler is None or not hasattr(scheduler.config, 'clip_sample'):
            raise click.ClickException('This policy does not support sample clipping overrides.')
        scheduler.register_to_config(clip_sample=clip_sample)
    if seed is None:
        seed = cfg.training.seed
    
    device = torch.device(device)
    policy.to(device)
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    with evaluation_context(policy, seed=seed):
        runner_log = env_runner.run(policy)
    settings = dict(
        checkpoint=str(pathlib.Path(checkpoint).resolve()),
        dataset_path=str(cfg.task.dataset.dataset_path),
        model='ema' if cfg.training.use_ema else 'model', seed=seed,
        clip_sample=bool(scheduler.config.clip_sample) if scheduler is not None and hasattr(scheduler.config, 'clip_sample') else None,
        num_inference_steps=getattr(policy, 'num_inference_steps', None))
    pathlib.Path(output_dir, 'eval_settings.json').write_text(
        json.dumps(settings, indent=2), encoding='utf-8')
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
