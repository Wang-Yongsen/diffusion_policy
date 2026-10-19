"""区分“示教观测回放成功”和“真机观测停滞”。

不连接设备。每条预测在调用前重设同一个 CPU 种子，所以条件之间共用同一份初始噪声。
外伸量是 8 步动作的最后一步左臂 Z，减去这份观测自己最新的左臂 Z。
真机失败时这个外伸只有几毫米；示教起点上大约是两厘米。
"""

import argparse
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits

from diffusion_policy.real_world.tianji_rail_env import preprocess_observation
from eval_tianji_rail import load_policy


CAMERAS = ("camera_0", "camera_1", "camera_2")
REFERENCE_LAST_LEFT_XYZ_MM = np.array([612.4520897865295, 117.76337027549744, 251.01473927497864])
FAILURE_EXTEND_MM = 4.0


def chunk_report(action, current_pose, fixed_pose, expert=None):
    """Summarize one 8-step absolute action. Positions are metres in, millimetres out."""
    action = np.asarray(action, dtype=np.float64)
    current_pose = np.asarray(current_pose, dtype=np.float64)
    fixed_pose = np.asarray(fixed_pose, dtype=np.float64)
    if action.shape != (8, 52) or current_pose.shape != (12,) or fixed_pose.shape != (12,):
        raise ValueError("Expected one chunk, one current pose and one fixed pose")
    left = action[:, 0:3]
    step = np.diff(left[:, 2])
    flips = int(np.sum(step[1:] * step[:-1] < 0))
    delta = (left[-1] - current_pose[:3]) * 1000
    report = {
        "last_left_xyz_mm": (left[-1] * 1000).round(3).tolist(),
        "input_left_xyz_mm": (current_pose[:3] * 1000).round(3).tolist(),
        "left_delta_xyz_mm": delta.round(3).tolist(),
        "extend_from_input_mm": round(float(delta[2]), 3),
        "extend_from_fixed_mm": round(float((left[-1, 2] - fixed_pose[2]) * 1000), 3),
        "path_z_mm": round(float((left[-1, 2] - left[0, 2]) * 1000), 3),
        "z_sign_flips": flips,
    }
    if expert is not None:
        expert = np.asarray(expert, dtype=np.float64)
        if expert.shape != action.shape:
            raise ValueError("Expert chunk shape mismatch")
        report["left_pos_err_mm"] = round(float(np.linalg.norm(left - expert[:, 0:3], axis=-1).mean() * 1000), 3)
        report["expert_extend_from_input_mm"] = round(float((expert[-1, 2] - current_pose[2]) * 1000), 3)
    return report


def _check_reports():
    action = np.zeros((8, 52))
    action[:, 2] = np.linspace(0.24, 0.26, 8)
    current = np.zeros(12)
    current[2] = 0.23
    fixed = current.copy()
    fixed[2] = 0.20
    report = chunk_report(action, current, fixed)
    if report["extend_from_input_mm"] != 30.0 or report["extend_from_fixed_mm"] != 60.0:
        raise AssertionError(report)
    if report["z_sign_flips"] != 0:
        raise AssertionError(report)
    aliased = np.zeros((2, 3))
    swapped = aliased.copy()
    swapped[:, 0] = 1
    if aliased[0, 0] != 0:
        raise AssertionError("copy failed")


class Episode:
    def __init__(self, path, split):
        self.path = str(path)
        self.split = split
        self.root = zarr.open_group(self.path, mode="r")
        data = self.root["data"]
        self.eef = np.asarray(data["robot_eef_pose"][:], dtype=np.float32)
        self.hand = np.asarray(data["hand_joint"][:], dtype=np.float32)
        self.action = np.asarray(data["action"][:], dtype=np.float32)
        self.n = int(self.eef.shape[0])
        self.name = self.root["meta"].attrs["segments"][0]["source_episode"]
        self.return_start = return_window(self.eef)

    def window(self, start):
        stop = start + 2
        if type(start) is not int or start < 0 or stop > self.n:
            raise ValueError(f"{self.name}: window {start} is outside 0..{self.n}")
        obs = {key: np.asarray(self.root["data"][key][start:stop]) for key in CAMERAS}
        obs["robot_eef_pose"] = self.eef[start:stop].copy()
        obs["hand_joint"] = self.hand[start:stop].copy()
        return obs

    def expert(self, start):
        chunk = self.action[start + 1:start + 9]
        if len(chunk) != 8:
            return None
        return chunk.copy()


def return_window(eef):
    """First window after the outward peak whose latest Z is back near the start Z."""
    z = eef[:, 2]
    peak = int(np.argmax(z))
    latest = np.flatnonzero((np.arange(len(z)) > peak + 60) & (np.abs(z - z[1]) < 0.025))
    if not len(latest):
        latest = np.flatnonzero((np.arange(len(z)) > peak + 60) & (z < z[1] + 0.04))
    if not len(latest):
        raise ValueError("No return phase near the start height")
    return int(latest[0]) - 1


def pixel_mae(left, right):
    return {
        key: round(float(np.mean(np.abs(left[key].astype(np.int16) - right[key].astype(np.int16)))), 3)
        for key in CAMERAS
    }


def pose_distance(left, right):
    """Latest-frame left-arm position error and a rough rotation vector gap."""
    a, b = left["robot_eef_pose"][-1], right["robot_eef_pose"][-1]
    return {
        "left_xyz_mm": round(float(np.linalg.norm(a[:3] - b[:3]) * 1000), 3),
        "left_rotvec_deg": round(float(np.rad2deg(np.linalg.norm(a[3:6] - b[3:6]))), 3),
        "left_z_mm": [round(float(a[2] * 1000), 3), round(float(b[2] * 1000), 3)],
    }


class Predictor:
    def __init__(self, policy, cfg):
        self.policy = policy
        self.cfg = cfg
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, obs, seed=42):
        torch.manual_seed(seed)
        prepared = preprocess_observation(obs, self.cfg.task.shape_meta, self.cfg.task.dataset)
        tensors = {key: torch.from_numpy(value).unsqueeze(0).to(self.policy.device) for key, value in prepared.items()}
        self.policy.reset()
        started = time.perf_counter()
        with torch.inference_mode():
            action = self.policy.predict_action(tensors)["action"][0].detach().cpu().numpy()
        self.seconds += time.perf_counter() - started
        self.calls += 1
        return action


def evaluate(predictor, obs, fixed_pose, expert=None):
    action = predictor(obs)
    report = chunk_report(action, obs["robot_eef_pose"][-1], fixed_pose, expert)
    report["calls_so_far"] = predictor.calls
    report["mean_forward_s"] = round(predictor.seconds / predictor.calls, 3)
    return action, report


def shifted(episode, start, image_shift=0, state_shift=0):
    images = episode.window(start + image_shift)
    state = episode.window(start + state_shift)
    obs = {key: images[key] for key in CAMERAS}
    obs["robot_eef_pose"] = state["robot_eef_pose"]
    obs["hand_joint"] = state["hand_joint"]
    return obs


def with_images(state_obs, image_obs):
    obs = {key: image_obs[key].copy() for key in CAMERAS}
    obs["robot_eef_pose"] = state_obs["robot_eef_pose"].copy()
    obs["hand_joint"] = state_obs["hand_joint"].copy()
    return obs


def with_z_bias(obs, millimetres):
    biased = {key: value.copy() for key, value in obs.items()}
    biased["robot_eef_pose"][:, 2] = biased["robot_eef_pose"][:, 2] + millimetres / 1000.0
    return biased


def with_left_xyz(obs, source):
    mixed = {key: value.copy() for key, value in obs.items()}
    mixed["robot_eef_pose"][:, 0:3] = source["robot_eef_pose"][:, 0:3]
    return mixed


def repeat_latest(obs):
    repeated = {key: value.copy() for key, value in obs.items()}
    for key, value in repeated.items():
        value[0] = value[1]
    return repeated


def nearest_window(episode, xyz):
    distance = np.linalg.norm(episode.eef[:, 0:3] - np.asarray(xyz, dtype=np.float32), axis=1)
    distance[0] = np.inf
    latest = int(np.argmin(distance))
    return latest - 1, float(distance[latest] * 1000)


def run_open_loop(predictor, episodes, starts_for):
    rows = []
    for episode in episodes:
        for start in starts_for(episode):
            obs = episode.window(start)
            _, report = evaluate(predictor, obs, obs["robot_eef_pose"][-1], episode.expert(start))
            report.update(experiment="open_loop", episode=episode.name, split=episode.split,
                          start=start, time_s=round(start / 30, 3),
                          input_left_z_mm=round(float(obs["robot_eef_pose"][-1, 2] * 1000), 3))
            rows.append(report)
            print(f"open {episode.name} t={start/30:.2f}s extend {report['extend_from_input_mm']:+.1f} mm "
                  f"expert {report['expert_extend_from_input_mm']:+.1f} err {report['left_pos_err_mm']:.1f}",
                  flush=True)
    return rows


def run_perturbations(predictor, train, val):
    rows = []
    pairs = {train.name: val, val.name: train}
    for episode in (train, val):
        other = pairs[episode.name]
        contexts = {
            0: {
                "return": episode.window(episode.return_start),
                "mid": episode.window(150),
                "other": other.window(0),
            },
            90: {},
        }
        for start, extra in contexts.items():
            base = episode.window(start)
            fixed = base["robot_eef_pose"][-1].copy()
            variants = [("baseline", base)]
            for millimetres in (-20, -10, 10, 20, 40):
                variants.append((f"z_bias_{millimetres:+d}mm", with_z_bias(base, millimetres)))
            variants.append(("repeat_latest", repeat_latest(base)))
            if start == 0:
                variants.extend((
                    ("images_from_return", with_images(base, extra["return"])),
                    ("images_from_mid_5s", with_images(base, extra["mid"])),
                    ("images_from_other_start", with_images(base, extra["other"])),
                    ("state_from_return", with_images(extra["return"], base)),
                    ("state_from_other_start", with_images(extra["other"], base)),
                    ("left_xyz_from_return", with_left_xyz(base, extra["return"])),
                ))
            else:
                for lag in (4, 8):
                    variants.append((f"images_lag_{lag}", shifted(episode, start, image_shift=-lag)))
                    variants.append((f"state_lag_{lag}", shifted(episode, start, state_shift=-lag)))
            for name, obs in variants:
                _, report = evaluate(predictor, obs, fixed, episode.expert(start) if name == "baseline" else None)
                report.update(experiment="perturbation", condition=name, episode=episode.name,
                              split=episode.split, start=start,
                              input_left_z_mm=round(float(obs["robot_eef_pose"][-1, 2] * 1000), 3))
                rows.append(report)
                print(f"perturb {episode.name} @{start} {name}: extend {report['extend_from_input_mm']:+.1f} "
                      f"vs fixed {report['extend_from_fixed_mm']:+.1f} flips {report['z_sign_flips']}",
                      flush=True)
    return rows


def run_closed_loop(predictor, episode, chunks):
    rows = []
    modes = ("teacher", "clock_image_fed_state", "nearest_demo", "half_track_nearest")
    for mode in modes:
        obs = episode.window(0)
        for step in range(chunks):
            clock = step * 8
            if mode == "teacher":
                obs = episode.window(clock)
            elif mode == "clock_image_fed_state" and step:
                fresh = episode.window(clock)
                for key in CAMERAS:
                    obs[key] = fresh[key]
            action, report = evaluate(predictor, obs, obs["robot_eef_pose"][-1])
            demo = episode.window(min(clock, episode.n - 2))
            report.update(experiment="closed_loop", mode=mode, episode=episode.name, step=step,
                          clock_start=clock,
                          demo_left_z_mm=round(float(demo["robot_eef_pose"][-1, 2] * 1000), 3))
            command = action[-1]
            if mode == "clock_image_fed_state":
                obs["robot_eef_pose"][:] = command[:12]
                obs["hand_joint"][:] = command[12:]
            elif mode in ("nearest_demo", "half_track_nearest"):
                current = obs["robot_eef_pose"][-1, 0:3]
                target = command[0:3]
                query = target if mode == "nearest_demo" else current + 0.5 * (target - current)
                matched, error_mm = nearest_window(episode, query)
                obs = episode.window(matched)
                report["matched_start"] = matched
                report["match_error_mm"] = round(error_mm, 3)
            rows.append(report)
            print(f"loop {mode} step {step}: Z {report['last_left_xyz_mm'][2]:.1f} "
                  f"extend {report['extend_from_input_mm']:+.1f} demoZ {report['demo_left_z_mm']:.1f}",
                  flush=True)
    return rows


def normalizer_span(policy):
    stats = policy.normalizer["robot_eef_pose"].get_input_stats()
    low = stats["min"].detach().cpu().numpy()
    high = stats["max"].detach().cpu().numpy()
    return {"left_z_mm": [round(float(low[2] * 1000), 3), round(float(high[2] * 1000), 3)]}


def write_figure(path, perturbations, loops):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font).exists():
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    wanted = [
        "baseline", "z_bias_-20mm", "z_bias_-10mm", "z_bias_+10mm", "z_bias_+20mm", "z_bias_+40mm",
        "images_from_return", "images_from_mid_5s", "images_from_other_start",
        "state_from_return", "state_from_other_start", "left_xyz_from_return", "repeat_latest",
    ]
    labels = {
        "baseline": "起点原样",
        "z_bias_-20mm": "左Z −20 mm",
        "z_bias_-10mm": "左Z −10 mm",
        "z_bias_+10mm": "左Z +10 mm",
        "z_bias_+20mm": "左Z +20 mm",
        "z_bias_+40mm": "左Z +40 mm",
        "images_from_return": "图像换成回程",
        "images_from_mid_5s": "图像换成第5秒",
        "images_from_other_start": "图像换成另一条起点",
        "state_from_return": "状态换成回程",
        "state_from_other_start": "状态换成另一条起点",
        "left_xyz_from_return": "只换左臂位置",
        "repeat_latest": "两帧重复最新",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.4), dpi=120)
    ax = axes[0]
    y = np.arange(len(wanted))
    for offset, episode, color in ((-0.18, "episode_000000", "#3b6d9a"), (0.18, "episode_000034", "#c47b2b")):
        values = []
        for name in wanted:
            match = [row for row in perturbations if row["condition"] == name and row["episode"] == episode and row["start"] == 0]
            values.append(match[0]["extend_from_input_mm"] if match else np.nan)
        ax.barh(y + offset, values, height=0.34, color=color, label="训练 000000" if episode.endswith("000000") else "验证 000034")
    ax.axvline(FAILURE_EXTEND_MM, color="#9a3b3b", ls="--", lw=1, label="失败记录第一块约 +4 mm")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[name] for name in wanted])
    ax.set_xlabel("相对这份观测自身的左臂 Z 外伸（mm）")
    ax.set_title("起点观测被替换后，第一步还往外伸多少")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", color="#dddddd")

    ax = axes[1]
    colors = {
        "teacher": "#3b6d9a",
        "clock_image_fed_state": "#5b8f6a",
        "nearest_demo": "#c47b2b",
        "half_track_nearest": "#7a5b8f",
    }
    names = {
        "teacher": "示教观测逐块",
        "clock_image_fed_state": "示教图像 + 预测状态回灌",
        "nearest_demo": "预测位姿对到最近示教帧",
        "half_track_nearest": "只跟上半步再对示教帧",
    }
    for mode, color in colors.items():
        series = [row for row in loops if row["mode"] == mode]
        ax.plot([row["step"] * 8 / 30 for row in series], [row["last_left_xyz_mm"][2] for row in series],
                marker="o", ms=3.5, color=color, label=names[mode])
    if loops:
        demo = [row for row in loops if row["mode"] == "teacher"]
        ax.plot([row["step"] * 8 / 30 for row in demo], [row["demo_left_z_mm"] for row in demo],
                color="#888888", ls=":", label="示教实测 Z")
    ax.axhspan(225, 295, color="#9a3b3b", alpha=0.12, label="失败记录目标 Z 区间")
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("左臂 Z（mm，向外为正）")
    ax.set_title("训练片段前 2.7 秒的离线闭环")
    ax.legend(fontsize=8)
    ax.grid(color="#eeeeee")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    _check_reports()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--train-zarr", type=Path, required=True)
    parser.add_argument("--val-zarr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    train, val = Episode(args.train_zarr, "train"), Episode(args.val_zarr, "val")
    reference = np.load(args.reference_npz)
    zarr_obs = train.window(0)
    reference_gap = {
        key: float(np.max(np.abs(zarr_obs[key].astype(np.int32) - reference[key].astype(np.int32))))
        for key in CAMERAS
    }
    reference_gap["robot_eef_pose"] = float(np.max(np.abs(zarr_obs["robot_eef_pose"] - reference["robot_eef_pose"])))
    reference_gap["hand_joint"] = float(np.max(np.abs(zarr_obs["hand_joint"] - reference["hand_joint"])))
    appearance = {
        "train_start_vs_return": {**pixel_mae(train.window(0), train.window(train.return_start)),
                                  **pose_distance(train.window(0), train.window(train.return_start))},
        "train_start_vs_val_start": {**pixel_mae(train.window(0), val.window(0)),
                                     **pose_distance(train.window(0), val.window(0))},
        "return_windows": {train.name: train.return_start, val.name: val.return_start},
    }
    print("reference array gap", reference_gap, flush=True)
    print("return windows", appearance["return_windows"], flush=True)

    with threadpool_limits(limits=2, user_api="blas"):
        torch.set_num_threads(2)
        policy, cfg, state_key = load_policy(args.checkpoint, "cpu")
        predictor = Predictor(policy, cfg)
        reference_obs = {key: reference[key] for key in (*CAMERAS, "robot_eef_pose", "hand_joint")}
        _, reference_report = evaluate(predictor, reference_obs, reference_obs["robot_eef_pose"][-1])
        reference_report["experiment"] = "reference_npz"
        reference_report["delta_from_server_mm"] = (
            np.asarray(reference_report["last_left_xyz_mm"]) - REFERENCE_LAST_LEFT_XYZ_MM
        ).round(3).tolist()
        print("reference delta mm", reference_report["delta_from_server_mm"], flush=True)
        open_loop = run_open_loop(
            predictor, (train, val),
            lambda episode: (0, 30, 90, 150, episode.return_start),
        )
        perturbations = run_perturbations(predictor, train, val)
        loops = run_closed_loop(predictor, train, args.chunks)

    payload = {
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "state_dict": state_key,
        "device": "cpu",
        "torch_threads": 2,
        "seed_before_each_call": 42,
        "normalizer_left_z": normalizer_span(policy),
        "reference_array_gap": reference_gap,
        "appearance": appearance,
        "failure_signature": {
            "source": "motion-sync-028 first chunk",
            "extend_from_likely_input_mm": FAILURE_EXTEND_MM,
            "target_z_band_mm": [225, 295],
        },
        "reference": reference_report,
        "open_loop": open_loop,
        "perturbations": perturbations,
        "closed_loop": loops,
        "forward_calls": predictor.calls,
        "mean_forward_s": round(predictor.seconds / max(predictor.calls, 1), 3),
    }
    (args.output / "obs_gap_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_figure(args.output / "obs_gap.png", perturbations, loops)
    print(f"wrote {args.output} in {predictor.calls} forwards", flush=True)


if __name__ == "__main__":
    main()
