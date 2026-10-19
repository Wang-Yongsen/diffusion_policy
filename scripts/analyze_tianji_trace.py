"""Offline comparison of a short Tianji trace. This does not move the robot."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_policy.real_world.decision_trace import (
    SIDES, alignment_series, assess_trace, compare_action_pair, hand_tracking_deg,
    inference_motion_mm, load_trace, replay_follow_mm, request_to_command_mm, write_alignment_plot,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path, help="含 trace_meta.json 的 trial 目录")
    parser.add_argument("--checkpoint", type=Path, help="有示范/真实成对观测时，用这份权重做配对推理")
    parser.add_argument("--device", help="配对推理设备；缺省时用会话里的 device。CPU 与 CUDA 不能当成同一噪声")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    args = parser.parse_args(argv)
    loaded = load_trace(args.trial)
    fk, fk_error = _load_fk()
    series = alignment_series(loaded["commands"], loaded["feedback"], fk)
    measured = _measured_xyz(loaded["feedback"], fk)
    motion = inference_motion_mm(loaded["feedback"]["received_ns"], measured,
                                 loaded["decisions"]["anchor_ns"], loaded["decisions"]["returned_ns"])
    hands = _hand_series(loaded["hands"], series["origin_ns"])
    gaps = {"request_to_command_fk_mm": request_to_command_mm(loaded["commands"]),
            "inference_motion_mm": _finite_summary(motion),
            "fk_error": fk_error}
    if loaded["meta"].get("capture_real") and "real_robot_eef_pose" in loaded["decisions"]:
        gaps["replay_follow_mm"] = _finite_summary(replay_follow_mm(
            loaded["decisions"]["robot_eef_pose"], loaded["decisions"]["real_robot_eef_pose"],
            loaded["decisions"]["real_present"]))
    report = {"assessment": assess_trace(loaded["meta"], gaps), "gaps": gaps,
              "schema": loaded["meta"].get("schema"), "trace_valid": loaded["meta"].get("valid"),
              "trace_invalid_reason": loaded["meta"].get("invalid_reason"),
              "dropped": loaded["meta"].get("dropped")}
    if args.checkpoint is not None:
        report["paired"] = _paired(args, loaded)
    write_alignment_plot(args.trial, series, hands, motion)
    (args.trial / "trace_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _print_report(report)
    return 0


def _load_fk():
    try:
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics
        kinematics = TianjiKinematics()
    except Exception as error:
        return None, str(error)

    def fk(side, joints):
        pose = kinematics.fk(side, tuple(float(value) for value in joints))
        return np.asarray(pose.position_m, dtype=np.float64)

    return fk, None


def _measured_xyz(feedback, fk):
    count = len(feedback["received_ns"])
    points = np.full((count, 3), np.nan)
    if fk is None:
        return points
    for index in range(count):
        if not feedback["q_present"][index, 0]:
            continue
        point = fk("left", feedback["q"][index, 0])
        if point is not None:
            points[index] = point
    return points


def _hand_series(hands, origin_ns):
    result = {}
    for index, side in enumerate(SIDES):
        commands = hands["side"] == index
        feedback = hands["feedback_side"] == index
        errors = hand_tracking_deg(hands["started_ns"][commands], hands["q"][commands], hands["present"][commands],
                                   hands["feedback_received_ns"][feedback], hands["feedback_q"][feedback],
                                   hands["feedback_present"][feedback])
        result[side] = {"time_s": (hands["started_ns"][commands] - origin_ns) / 1e9, "error_deg": errors}
    return result


def _paired(args, loaded):
    decisions = loaded["decisions"]
    if "real_camera_0" not in decisions:
        return {"skipped": "这条记录没有单独保存真实观测"}
    device = args.device or _session_device(args.trial)
    if device is None:
        return {"skipped": "没有 --device，会话里也没有 device"}
    import torch
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return {"skipped": f"{device} 不可用。不改到 CPU，因为那不是同一套噪声"}
    from eval_tianji_rail import load_policy
    policy, cfg, _ = load_policy(args.checkpoint, device)
    usable = [index for index in range(len(decisions["present"]))
              if decisions["present"][index] and decisions["real_present"][index]]
    if not usable:
        return {"skipped": "没有同时具备示范输入和真实输入的决策"}
    demo_actions = np.empty((len(args.seeds), len(usable), decisions["actions"].shape[1], 52))
    real_actions = np.empty_like(demo_actions)
    for seed_index, seed in enumerate(args.seeds):
        for item_index, decision in enumerate(usable):
            _reset_rng(seed, device)
            demo_actions[seed_index, item_index] = _predict(policy, cfg, _observation(decisions, decision, real=False))
            _reset_rng(seed, device)
            real_actions[seed_index, item_index] = _predict(policy, cfg, _observation(decisions, decision, real=True))
    np.savez(args.trial / "paired_actions.npz", seeds=np.asarray(args.seeds), decision_index=np.asarray(usable),
             demo_actions=demo_actions, real_actions=real_actions, device=str(device))
    per_decision = [compare_action_pair(demo_actions[:, index], real_actions[:, index]) for index in range(len(usable))]
    return {"device": str(device), "seeds": list(args.seeds), "decisions": usable,
            "noise": "每个种子都在这份设备上重置后分别预测；CPU 种子和 CUDA 种子不是同一噪声",
            "comparisons": per_decision,
            "executed_replay": _replay_saved_noise(policy, cfg, decisions, loaded["meta"], usable, device)}


def _replay_saved_noise(policy, cfg, decisions, meta, usable, device):
    if "rng_cpu" not in decisions:
        return {"skipped": "记录里没有采样前的 RNG"}
    recorded = []
    for index in usable:
        item = meta["decisions"][index] or {}
        recorded.append(item.get("rng_device"))
        if item.get("rng_device") != str(device):
            return {"skipped": f"决策 {index} 的噪声来自 {item.get('rng_device')}，当前设备是 {device}"}
        if str(device).startswith("cuda") and not item.get("rng_cuda_present"):
            return {"skipped": "CUDA 推理没有采样前的 CUDA RNG，不能用 CPU 状态充当同一噪声"}
    import torch
    reproduced = []
    for index in usable:
        torch.set_rng_state(torch.from_numpy(np.array(decisions["rng_cpu"][index], copy=True)))
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(torch.from_numpy(np.array(decisions["rng_cuda"][index], copy=True)), device)
        predicted = _predict(policy, cfg, _observation(decisions, index, real=False))
        rows = int(decisions["action_rows"][index])
        saved = decisions["actions"][index, :rows]
        gap = np.max(np.abs(predicted[:rows] - saved)) if rows else None
        reproduced.append({"decision": int(index), "max_abs_error": None if gap is None else float(gap)})
    return {"recorded_device": recorded, "demo_input": reproduced}


def _predict(policy, cfg, obs):
    import torch
    from diffusion_policy.real_world.tianji_rail_env import preprocess_observation
    prepared = preprocess_observation(obs, cfg.task.shape_meta, cfg.task.dataset)
    tensors = {key: torch.from_numpy(value).unsqueeze(0).to(policy.device) for key, value in prepared.items()}
    with torch.inference_mode():
        return policy.predict_action(tensors)["action"][0].detach().cpu().numpy()


def _reset_rng(seed, device):
    import torch
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))


def _observation(decisions, index, *, real):
    prefix = "real_" if real else ""
    obs = {camera: np.asarray(decisions[f"{prefix}{camera}"][index]) for camera in ("camera_0", "camera_1", "camera_2")}
    obs["robot_eef_pose"] = np.asarray(decisions[f"{prefix}robot_eef_pose"][index])
    obs["hand_joint"] = np.asarray(decisions[f"{prefix}hand_joint"][index])
    return obs


def _session_device(trial):
    for path in (trial.parent / "session.json", trial / "session.json"):
        if path.is_file():
            return json.loads(path.read_text()).get("device")
    return None


def _finite_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return {"count": int(len(finite)), "missing": int(np.size(values) - len(finite)),
            "max": float(finite.max()), "median": float(np.median(finite))}


def _print_report(report):
    assessment = report["assessment"]
    print(assessment["reason"])
    gaps = report["gaps"]["request_to_command_fk_mm"]
    for side, item in gaps.items():
        if item is None:
            print(f"{side} 臂：没有同时具备请求位姿和命令正解的采样")
        else:
            print(f"{side} 臂请求位姿到命令正解：最大 {item['max']:.1f} mm，中位 {item['median']:.1f} mm")
    if "replay_follow_mm" in report["gaps"]:
        follow = report["gaps"]["replay_follow_mm"]
        if follow is None:
            print("没有可比较的真实观测。")
        else:
            print(f"真实观测相对该决策所用示范状态，左臂位置差最大 {follow['max']:.1f} mm")
    if report.get("paired", {}).get("skipped"):
        print(report["paired"]["skipped"])
    print("图：alignment_left.png，alignment_hands.png，inference_motion.png")


if __name__ == "__main__":
    raise SystemExit(main())
