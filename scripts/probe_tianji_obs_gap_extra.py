"""Follow-up probes: X offset, sampling noise, validation rollout, frozen images."""

import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.probe_tianji_obs_gap import Episode, Predictor, chunk_report
from eval_tianji_rail import load_policy


def bias_x(obs, millimetres):
    shifted = {key: value.copy() for key, value in obs.items()}
    shifted["robot_eef_pose"][:, 0] = shifted["robot_eef_pose"][:, 0] + millimetres / 1000.0
    return shifted


def row(action, obs, **fields):
    report = chunk_report(action, obs["robot_eef_pose"][-1], obs["robot_eef_pose"][-1])
    report["z_path_mm"] = np.round(action[:, 2] * 1000, 2).tolist()
    report.update(fields)
    return report


def main():
    out = Path("data/input_diagnosis/live_obs_gap_20261017/extra.json")
    train = Episode("data/replay_obs/rail_place3_episode_000000.zarr", "train")
    val = Episode("data/replay_obs/rail_place3_episode_000034.zarr", "val")
    with threadpool_limits(limits=2, user_api="blas"):
        torch.set_num_threads(2)
        policy, cfg, _ = load_policy(
            "data/outputs/2026.10.16/23.09.56_tianji_rail/checkpoints/epoch=0050.ckpt", "cpu")
        predict = Predictor(policy, cfg)
        seeds = []
        for seed in (42, 43, 44, 45, 46):
            for episode, start in ((train, 0), (val, 0), (train, 30), (val, 30)):
                obs = episode.window(start)
                action = predict(obs, seed)
                item = row(action, obs, experiment="seed", seed=seed, episode=episode.name,
                           start=start, extend_mm=None)
                item["extend_mm"] = item["extend_from_input_mm"]
                seeds.append(item)
                print(f"seed {seed} {episode.name} @{start} extend {item['extend_mm']:+.1f} "
                      f"last {item['last_left_xyz_mm']}", flush=True)
        offsets = []
        base = train.window(0)
        for millimetres in (0, -30, -60, -80):
            obs = bias_x(base, millimetres)
            action = predict(obs, 42)
            item = row(action, obs, experiment="x_bias", bias_mm=millimetres)
            offsets.append(item)
            print(f"x {millimetres:+d} last {item['last_left_xyz_mm']} extend {item['extend_from_input_mm']:+.1f} "
                  f"path {item['z_path_mm']}", flush=True)
        frozen = []
        obs = train.window(0)
        images = {key: obs[key].copy() for key in ("camera_0", "camera_1", "camera_2")}
        for step in range(12):
            action = predict(obs, 42)
            item = row(action, obs, experiment="frozen_start_images", step=step)
            frozen.append(item)
            print(f"frozen {step} Z {item['last_left_xyz_mm'][2]:.1f} "
                  f"extend {item['extend_from_input_mm']:+.1f}", flush=True)
            obs["robot_eef_pose"][:] = action[-1, :12]
            obs["hand_joint"][:] = action[-1, 12:]
            for key, image in images.items():
                obs[key] = image
        validation = []
        for step in range(10):
            obs = val.window(step * 8)
            action = predict(obs, 42)
            item = row(action, obs, experiment="val_teacher", step=step,
                       demo_z_mm=round(float(obs["robot_eef_pose"][-1, 2] * 1000), 3))
            validation.append(item)
            print(f"val teacher {step} Z {item['last_left_xyz_mm'][2]:.1f} "
                  f"extend {item['extend_from_input_mm']:+.1f}", flush=True)
    out.write_text(json.dumps({"seeds": seeds, "x_bias": offsets, "frozen_images": frozen,
                               "val_teacher": validation}, indent=2), encoding="utf-8")
    print("wrote", out, "calls", predict.calls, flush=True)


if __name__ == "__main__":
    main()
