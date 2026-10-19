"""Offline absolute-action errors; no physical success-rate estimate."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Subset

from diffusion_policy.dataset.tianji_rail_image_dataset import ACTION_DIMS
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


def action_errors(prediction, target, action_space="eef"):
    """One error per prediction step, using SO(3) distance for EEF rotations."""
    dim = ACTION_DIMS[action_space]
    if prediction.shape != target.shape or prediction.shape[-1] != dim:
        raise ValueError(f"Predictions and targets must have identical (...,{dim}) shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite action prediction or target")
    result = {}
    if action_space == "eef":
        for side, offset in (("left", 0), ("right", 6)):
            result[f"{side}_position_mm"] = np.linalg.norm(prediction[..., offset:offset+3] - target[..., offset:offset+3], axis=-1) * 1000
            a = Rotation.from_rotvec(prediction[..., offset+3:offset+6].reshape(-1, 3))
            b = Rotation.from_rotvec(target[..., offset+3:offset+6].reshape(-1, 3))
            result[f"{side}_rotation_deg"] = np.rad2deg((a.inv() * b).magnitude()).reshape(target.shape[:-1])
        hand_offsets = (("left", 12), ("right", 32))
    else:
        for side, offset in (("left", 0), ("right", 7)):
            result[f"{side}_arm_joint_mae_deg"] = np.rad2deg(np.abs(prediction[..., offset:offset+7] - target[..., offset:offset+7])).mean(axis=-1)
        hand_offsets = (("left", 14), ("right", 34))
    for side, offset in hand_offsets:
        result[f"{side}_hand_joint_mae_deg"] = np.rad2deg(np.abs(prediction[..., offset:offset+20] - target[..., offset:offset+20])).mean(axis=-1)
    return result


class TianjiRailOfflineRunner(BaseImageRunner):
    def __init__(self, output_dir, dataset, n_action_steps=8, batch_size=8,
                 window_stride=8, max_windows=None):
        super().__init__(output_dir)
        if (type(n_action_steps) is not int or n_action_steps < 1
                or dataset.n_obs_steps - 1 + n_action_steps > dataset.horizon):
            raise ValueError("n_action_steps must fit inside the Dataset action horizon")
        if type(batch_size) is not int or batch_size < 1 or type(window_stride) is not int or window_stride < 1:
            raise ValueError("batch_size and window_stride must be positive integers")
        if max_windows is not None and (type(max_windows) is not int or max_windows < 1):
            raise ValueError("max_windows must be null or a positive integer")
        self.dataset = dataset.get_validation_dataset()
        self.n_action_steps, self.batch_size = n_action_steps, batch_size
        # Stride restarts in each segment so unrelated boundaries do not shift sampling.
        self.selected = [i for i, (start, segment) in enumerate(self.dataset.indices)
                         if (start - (0 if segment == 0 else dataset.segment_ends[segment-1])) % window_stride == 0]
        if max_windows is not None:
            self.selected = self.selected[:max_windows]

    def run(self, policy):
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if not self.selected:
            report = {"status": "skipped", "reason": "No independent validation windows", "windows": 0, "prediction_steps": 0, "sources": []}
            (output / "offline_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return {"offline/skipped": True, "offline/reason": report["reason"], "offline/windows": 0,
                    "offline/prediction_steps": 0}
        loader = DataLoader(Subset(self.dataset, self.selected), batch_size=self.batch_size, shuffle=False, num_workers=0)
        by_source, window_reports, cursor = {}, [], 0
        was_training = policy.training
        policy.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    policy.reset()
                    obs = {key: value.to(policy.device) for key, value in batch["obs"].items()}
                    prediction = policy.predict_action(obs)["action"].cpu().numpy()
                    first = self.dataset.n_obs_steps - 1
                    target = batch["action"][:, first:first+self.n_action_steps].numpy()
                    errors = action_errors(prediction, target, self.dataset.action_space)
                    for row in range(len(target)):
                        index = self.selected[cursor]
                        start, segment = self.dataset.indices[index]
                        source = self.dataset.sources[segment]
                        collected = by_source.setdefault(source, {key: [] for key in errors})
                        for key, values in errors.items():
                            collected[key].append(values[row])
                        window_reports.append(dict(source_episode=source, start_frame=int(start),
                                                   errors={key: float(value[row].mean()) for key, value in errors.items()}))
                        cursor += 1
        finally:
            policy.train(was_training)
        metrics = {key: float(np.concatenate([np.concatenate(fields[key]) for fields in by_source.values()]).mean()) for key in errors}
        demos = {source: {"windows": len(next(iter(fields.values()))),
                          "errors": {key: float(np.concatenate(values).mean()) for key, values in fields.items()}}
                 for source, fields in by_source.items()}
        report = dict(status="complete", windows=len(self.selected), prediction_steps=len(self.selected)*self.n_action_steps,
                      sources=list(by_source), errors=metrics, per_episode=demos, per_window=window_reports)
        (output / "offline_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        return {"offline/skipped": False, "offline/windows": len(self.selected),
                "offline/prediction_steps": report["prediction_steps"],
                **{f"offline/{key}": value for key, value in metrics.items()}}
