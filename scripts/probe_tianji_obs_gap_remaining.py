"""Offline pieces still open after the first observation-gap probe.

No robot motion. Covers deployment-versus-dataset preprocessing, one-camera
swaps, state-field swaps, frame order, and one subprocess prediction.
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

from diffusion_policy.real_world.policy_inference_process import PolicyInferenceProcess
from diffusion_policy.real_world.tianji_rail_env import preprocess_observation
from eval_tianji_rail import load_policy
from scripts.probe_tianji_obs_gap import CAMERAS, Episode, Predictor, chunk_report


REFERENCE_DIR = Path("/tmp/tianji-complete-diag/tianji-rail-complete-diagnosis-20261017/reference")
CKPT = Path("data/outputs/2026.10.16/23.09.56_tianji_rail/checkpoints/epoch=0050.ckpt")
OUT = Path("data/input_diagnosis/live_obs_gap_20261017/remaining.json")


def dataset_style(obs):
    """Same arithmetic as TianjiRailImageDataset when resize is disabled."""
    prepared = {}
    for key in CAMERAS:
        images = np.asarray(obs[key])
        prepared[key] = np.ascontiguousarray(np.moveaxis(images, -1, 1), dtype=np.float32) / 255.0
    prepared["robot_eef_pose"] = np.asarray(obs["robot_eef_pose"], dtype=np.float32)
    prepared["hand_joint"] = np.asarray(obs["hand_joint"], dtype=np.float32)
    return prepared


def max_abs(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def copied(obs):
    return {key: np.array(value, copy=True) for key, value in obs.items()}


def replace_cameras(state_obs, image_obs, names):
    mixed = copied(state_obs)
    for name in names:
        mixed[name] = np.array(image_obs[name], copy=True)
    return mixed


def replace_fields(base, donor, pose_slice=None, hand_slice=None, frame=None):
    mixed = copied(base)
    frames = slice(None) if frame is None else frame
    if pose_slice is not None:
        mixed["robot_eef_pose"][frames, pose_slice] = donor["robot_eef_pose"][frames, pose_slice]
    if hand_slice is not None:
        mixed["hand_joint"][frames, hand_slice] = donor["hand_joint"][frames, hand_slice]
    return mixed


def reversed_frames(obs):
    mixed = copied(obs)
    for value in mixed.values():
        value[:] = value[::-1]
    return mixed


def so3_deg(first, second):
    delta = Rotation.from_rotvec(second) * Rotation.from_rotvec(first).inv()
    return round(float(np.rad2deg(delta.magnitude())), 3)


def field_distances(base, donor):
    a = base["robot_eef_pose"][-1]
    b = donor["robot_eef_pose"][-1]
    return {
        "left_xyz_mm": round(float(np.linalg.norm(a[:3] - b[:3]) * 1000), 3),
        "right_xyz_mm": round(float(np.linalg.norm(a[6:9] - b[6:9]) * 1000), 3),
        "left_rotation_deg": so3_deg(a[3:6], b[3:6]),
        "right_rotation_deg": so3_deg(a[9:12], b[9:12]),
        "left_hand_mae_deg": round(float(np.rad2deg(np.mean(np.abs(base["hand_joint"][-1, :20] - donor["hand_joint"][-1, :20])))), 3),
        "right_hand_mae_deg": round(float(np.rad2deg(np.mean(np.abs(base["hand_joint"][-1, 20:] - donor["hand_joint"][-1, 20:])))), 3),
        "left_z_mm": [round(float(a[2] * 1000), 3), round(float(b[2] * 1000), 3)],
    }


def report_action(predictor, obs, seed=42, **fields):
    action = predictor(obs, seed)
    item = chunk_report(action, obs["robot_eef_pose"][-1], obs["robot_eef_pose"][-1])
    item["seed"] = seed
    item.update(fields)
    print(f"{fields.get('experiment')} {fields.get('condition', '')} {fields.get('episode', '')} "
          f"extend {item['extend_from_input_mm']:+.1f} last {item['last_left_xyz_mm']}", flush=True)
    return item


def preprocess_check(cfg):
    rows = []
    for path in sorted(REFERENCE_DIR.glob("episode_*.npz")):
        raw = np.load(path)
        obs = {key: raw[key] for key in (*CAMERAS, "robot_eef_pose", "hand_joint")}
        deployed = preprocess_observation(obs, cfg.task.shape_meta, cfg.task.dataset)
        manual = dataset_style(obs)
        gaps = {key: max_abs(deployed[key], manual[key]) for key in deployed}
        rows.append({"file": path.name, "max_abs": gaps, "worst": max(gaps.values())})
        print(f"preprocess {path.name} worst {max(gaps.values())}", flush=True)
    return rows


def subprocess_check(obs):
    process = PolicyInferenceProcess(CKPT, "cpu", startup_timeout_s=180, request_timeout_s=60)
    try:
        action = process.predict_observation(obs)["action"][0].numpy()
    finally:
        process.close()
    return action


def main():
    train = Episode("data/replay_obs/rail_place3_episode_000000.zarr", "train")
    val = Episode("data/replay_obs/rail_place3_episode_000034.zarr", "val")
    third = Episode("data/replay_obs/rail_place3_episode_000003.zarr", "split_unknown")
    start = train.window(0)
    returning = train.window(train.return_start)
    other = val.window(0)
    distances = {
        "train_start_vs_return": field_distances(start, returning),
        "train_start_vs_val_start": field_distances(start, other),
    }
    print("distances", json.dumps(distances), flush=True)

    with threadpool_limits(limits=2, user_api="blas"):
        torch.set_num_threads(2)
        policy, cfg, state_key = load_policy(CKPT, "cpu")
        preprocess_rows = preprocess_check(cfg)
        predictor = Predictor(policy, cfg)
        first = predictor(start, 42)
        second = predictor(start, 42)
        repeat_gap = max_abs(first, second)

        cameras = []
        for episode, donor, donor_name in (
            (train, other, "val_start"),
            (train, returning, "return"),
            (val, start, "train_start"),
        ):
            base = episode.window(0)
            for name in CAMERAS:
                mixed = replace_cameras(base, donor, (name,))
                cameras.append(report_action(
                    predictor, mixed, experiment="one_camera", condition=f"{name}_from_{donor_name}",
                    episode=episode.name))
            cameras.append(report_action(
                predictor, replace_cameras(base, donor, CAMERAS), experiment="one_camera",
                condition=f"all_from_{donor_name}", episode=episode.name))

        components = []
        swaps = (
            ("left_eef", dict(pose_slice=slice(0, 6))),
            ("right_eef", dict(pose_slice=slice(6, 12))),
            ("left_xyz", dict(pose_slice=slice(0, 3))),
            ("left_rotation", dict(pose_slice=slice(3, 6))),
            ("left_hand", dict(hand_slice=slice(0, 20))),
            ("right_hand", dict(hand_slice=slice(20, 40))),
            ("latest_state_only", dict(pose_slice=slice(0, 12), hand_slice=slice(0, 40), frame=1)),
            ("older_state_only", dict(pose_slice=slice(0, 12), hand_slice=slice(0, 40), frame=0)),
        )
        for episode, donor in ((train, returning), (val, val.window(val.return_start))):
            base = episode.window(0)
            for name, slices in swaps:
                before = {key: value.copy() for key, value in donor.items()}
                mixed = replace_fields(base, donor, **slices)
                for key in before:
                    if not np.array_equal(before[key], donor[key]):
                        raise RuntimeError(f"{name} mutated the donor")
                components.append(report_action(
                    predictor, mixed, experiment="state_field", condition=name, episode=episode.name))

        order = []
        for episode, window_start in ((train, 0), (train, 90), (val, 0)):
            base = episode.window(window_start)
            order.append(report_action(
                predictor, reversed_frames(base), experiment="frame_order", condition="reversed",
                episode=episode.name, start=window_start))
            images_only = copied(base)
            for key in CAMERAS:
                images_only[key][0] = images_only[key][1]
            order.append(report_action(
                predictor, images_only, experiment="frame_order", condition="images_repeat_latest",
                episode=episode.name, start=window_start))

        seeds = []
        for seed in (42, 43, 44, 45, 46):
            mixed = replace_fields(start, returning, pose_slice=slice(0, 12), hand_slice=slice(0, 40))
            seeds.append(report_action(
                predictor, mixed, seed=seed, experiment="state_swap_seed", condition="all_state_from_return",
                episode=train.name))

        third_rows = []
        for window_start in (0, 30, 90):
            obs = third.window(window_start)
            third_rows.append(report_action(
                predictor, obs, experiment="third_episode", condition="open_loop",
                episode=third.name, start=window_start,
                expert_extend_mm=round(float((third.expert(window_start)[-1, 2] - obs["robot_eef_pose"][-1, 2]) * 1000), 3)))

        direct = predictor(start, 42)
        print("starting subprocess", flush=True)
        child = subprocess_check(start)
        subprocess_gap = {
            "max_abs": max_abs(direct, child),
            "last_left_xyz_direct_mm": np.round(direct[-1, :3] * 1000, 3).tolist(),
            "last_left_xyz_subprocess_mm": np.round(child[-1, :3] * 1000, 3).tolist(),
            "note": "Subprocess runs check_inference before the request and does not reseed, and it limits BLAS threads to 1. Action differences are therefore not an input-conversion difference.",
        }
        print("subprocess gap", subprocess_gap, flush=True)

    payload = {
        "state_dict": state_key,
        "resize_enabled": bool(cfg.task.dataset.get("resize_enabled", True)),
        "distances": distances,
        "preprocess": preprocess_rows,
        "repeat_same_seed_max_abs": repeat_gap,
        "one_camera": cameras,
        "state_field": components,
        "frame_order": order,
        "state_swap_seeds": seeds,
        "third_episode": third_rows,
        "subprocess": subprocess_gap,
        "forward_calls_in_parent": predictor.calls,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
