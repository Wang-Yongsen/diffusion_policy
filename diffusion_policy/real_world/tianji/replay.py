"""用一条已转换示教的观测驱动策略，可选把预测动作下发到真机。

策略看不到现场画面。每个执行块用示教里对应时刻的三路图像、末端位姿和手关节，
执行完 8 步后再取下一段示教观测。这是开环核对：动作像不像示教，只说明策略
在这些观测上复现了标签，不代替现场闭环成功率。

真机模式固定同步执行。若当前姿态离第一个预测目标较远，先以不超过 0.15 m/s
直线插值过去；路径不避障。距离超过 0.8 m 或转角超过 90° 时拒绝启动。
"""

import json
import select
import sys
import time
from pathlib import Path

import numpy as np
import torch

from diffusion_policy.common.file_util import file_sha256 as _checkpoint_hash
from diffusion_policy.common.image_preprocessing import preprocess_observation
from diffusion_policy.common.paths import PROJECT_ROOT
from diffusion_policy.common.tianji_rail_pose import policy_actions_to_controller
from diffusion_policy.env_runner.tianji_rail_offline_runner import action_errors
from diffusion_policy.env_runner.tianji_rail_real_runner import (
    TrialRecorder,
    write_json,
)
from diffusion_policy.real_world.tianji.checkpoint import load_policy
from diffusion_policy.real_world.tianji.config import load_deployment_config
from diffusion_policy.real_world.tianji.env import TianjiRailEnv
from diffusion_policy.real_world.tianji.inference import PolicyInferenceProcess
from diffusion_policy.real_world.tianji.observation import ObservationNotReady
from diffusion_policy.real_world.tianji.recorded_episode import (
    APPROACH_MAX_POSITION_M,
    APPROACH_MAX_ROTATION_RAD,
    approach_actions,
    approach_within_limits,
    open_recorded_segment,
    playback_starts,
    pose_delta,
)
from diffusion_policy.real_world.tianji.runtime import validate_actions
from diffusion_policy.real_world.tianji.trace import (
    SessionTrace,
    runtime_files,
    write_provenance,
)

ERROR_LABELS = (
    ("left_position_mm", "左臂位置 mm"),
    ("right_position_mm", "右臂位置 mm"),
    ("left_rotation_deg", "左臂旋转 °"),
    ("right_rotation_deg", "右臂旋转 °"),
    ("left_hand_joint_mae_deg", "左手关节 °"),
    ("right_hand_joint_mae_deg", "右手关节 °"),
)


def run(args):
    if args.trace and not args.motion:
        raise ValueError("--trace 要和 --motion 一起用")
    if args.trace and (
        args.max_windows is None
        or args.max_windows > args.trace_capacity
        or args.trace_capacity < 1
    ):
        raise ValueError("--trace 时必须指定不超过 --trace-capacity 的 --max-windows")
    if args.max_windows is not None and args.max_windows < 1:
        raise ValueError("--max-windows 须为正整数")
    segment = open_recorded_segment(args.dataset, episode=args.episode, segment=args.segment)
    config = load_deployment_config(args.config)
    device = args.device or config["device"]
    print(
        f"示教片段 {segment.info['index']}（{segment.source}，条目 {segment.info['episode_index']}）："
        f"{segment.n_frames} 帧，约 {segment.n_frames / 30:.1f} s"
    )
    if not args.motion:
        return run_offline(args, segment, device)
    if not sys.stdin.isatty():
        print("真机回放要在交互终端里输入 s 才会动。请在你自己的终端运行这条命令。")
        return 2
    config["inference_mode"] = "sync"
    output = args.output or Path(config["output_dir"]) / time.strftime("replay-obs-%Y%m%d-%H%M%S")
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    return run_motion(args, segment, config, device, output)


def run_offline(args, segment, device):
    policy, cfg, _ = load_policy(args.checkpoint, device)
    n_obs, n_action = int(policy.n_obs_steps), int(policy.n_action_steps)
    policy.reset()

    def predict(obs):
        prepared = preprocess_observation(obs, cfg.task.shape_meta, cfg.task.dataset)
        tensors = {
            key: torch.from_numpy(value).unsqueeze(0).to(policy.device)
            for key, value in prepared.items()
        }
        with torch.inference_mode():
            actions = policy.predict_action(tensors)["action"][0].detach().cpu().numpy()
        return policy_actions_to_controller(actions, cfg.task.dataset.get("action_space", "eef"))

    collected = []
    starts = _starts(segment, n_obs, n_action, args.max_windows)
    print(f"离线回放 {len(starts)} 个执行块，不下发动作。")
    for number, start in enumerate(starts, start=1):
        actions = predict(segment.observation(start, n_obs))
        errors = _label_errors(segment, start, n_obs, n_action, actions)
        if errors is not None:
            collected.append(errors)
        if number == 1 or number == len(starts) or number % 10 == 0:
            print(f"  块 {number}/{len(starts)}  {_format_errors(errors)}")
    summary = _mean_errors(collected)
    print("相对示教标签：" + (_format_mean(summary) if summary else "没有完整标签可比较"))
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"输出目录已存在：{args.output}")
        args.output.mkdir(parents=True)
        write_json(
            args.output / "offline_replay.json",
            {
                "source": segment.source,
                "segment": segment.info,
                "windows": len(starts),
                "errors": summary,
            },
        )
    return 0


def run_motion(args, segment, config, device, output):
    inference = PolicyInferenceProcess(args.checkpoint, device, capture_rng=args.trace)
    print(f"checkpoint sha256 {_checkpoint_hash(args.checkpoint)}")
    env = None
    recorder = None
    trace = None
    outcome, reason = "playback_complete", None
    try:
        n_obs = int(inference.preflight["n_obs_steps"])
        n_action = int(inference.preflight["n_action_steps"])
        inference.reset()
        env = TianjiRailEnv(config, n_obs, motion=True)
        if args.trace:
            trace = SessionTrace(decision_capacity=args.trace_capacity)
            env.enable_trace(trace)
            files = runtime_files()
            files["eval_tianji_rail_recorded_obs"] = Path(__file__).resolve()
            write_provenance(output, files, PROJECT_ROOT)
        env.start()
        print("正在等待现场状态，比较示教起点；此时不下发动作。")
        current = _wait_live_pose(env)
        starts = _starts(segment, n_obs, n_action, args.max_windows)
        if trace is not None:
            probe, _, _ = env.get_obs()
            trace.allocate_decisions(probe, n_action, capture_real=True)
            print(
                f"完整记录缓冲：{trace.decision_capacity} 次决策，图像约 {trace.image_mebibytes:.0f} MiB。"
                "缓冲满会把本段标为无效，不会停住控制。"
            )
            print("每个执行块会多读一次真实观测，只保存，不送进模型，也不额外推理。")
        first_obs = segment.observation(starts[0], n_obs)
        first = _predict(inference, first_obs)
        delta = pose_delta(current, first[0])
        print("当前姿态到第一个预测目标：" + _format_delta(delta))
        write_json(
            output / "start_delta.json",
            {
                "live_pose": current,
                "first_action": first[0],
                "delta": delta,
                "segment": segment.info,
                "checkpoint": str(args.checkpoint),
            },
        )
        _reject_unexecutable_chunk(first, config)
        if not approach_within_limits(current, first[0]):
            outcome, reason = "refused", "Start pose is too far from the first predicted target"
            print(
                f"距离超过 {APPROACH_MAX_POSITION_M:.1f} m 或转角超过 {np.rad2deg(APPROACH_MAX_ROTATION_RAD):.0f}°，"
                "不自动插值。请先把双臂手动移近这条示教的起始姿态，再重新运行。"
            )
            return 2
        print("s 开始；p 暂停；y 记成功；n 记失败；q 退出。")
        print("确认周围清空后再按 s。接近阶段不超过 0.15 m/s，直线插值不避障。")
        print("随后按示教观测逐块执行；每块之间会短暂停住等下一次推理。速度上限来自所选部署配置。")
        if trace is not None:
            print(
                "这段只覆盖起步到接近导轨。不要递送导轨，也不要扶机器人；若必须介入，用 p 记下时间。"
            )
        if not _wait_for_start():
            outcome, reason = "takeover", "Operator declined"
            return 0
        recorder = TrialRecorder(
            output / "trial_0001",
            {
                "mode": "motion",
                "source": segment.source,
                "segment": segment.info,
                "checkpoint": str(args.checkpoint.resolve()),
                "started_monotonic_ns": time.monotonic_ns(),
            },
            config["record_queue_capacity"],
        )
        env.recorder = recorder
        if trace is not None:
            trace.begin_trial()
        env.engage()
        reference = env.runtime.reference_action
        if not approach_within_limits(reference, first[0]):
            raise RuntimeError("接合后的实测姿态离第一个预测目标太远")
        if not args.skip_approach and _needs_approach(reference, first[0]):
            path = approach_actions(reference, first[0])
            print(f"先慢速接近，约 {len(path) / 30:.1f} s。")
            command = _submit_path(env, path, n_action)
            if command:
                outcome, reason = _operator_outcome(command)
                env.pause(reason)
                return 0
        print(f"开始按示教观测执行，共 {len(starts)} 块。")
        if args.seed is not None:
            inference.reseed(args.seed)
        collected = []
        for number, start in enumerate(starts, start=1):
            demo = first_obs if number == 1 else segment.observation(start, n_obs)
            live = _capture_replay_observation(env) if trace is not None else None
            actions = _predict(inference, demo)
            prediction_id = (
                None
                if trace is None
                else _store_replay_decision(trace, env, inference, demo, actions, start, live=live)
            )
            errors = _label_errors(segment, start, n_obs, n_action, actions)
            if errors is not None:
                collected.append(errors)
            env.emit(
                "recorded_window",
                window_start=start,
                actions=actions,
                errors=None
                if errors is None
                else {key: float(value.mean()) for key, value in errors.items()},
            )
            print(f"  块 {number}/{len(starts)}  {_format_errors(errors)}")
            command = _submit_path(env, actions, n_action, prediction_id=prediction_id)
            if command:
                outcome, reason = _operator_outcome(command)
                env.pause(reason)
                break
        else:
            print("示教观测已回放完。相对标签：" + _format_mean(_mean_errors(collected)))
        write_json(output / "label_errors.json", _mean_errors(collected))
        return 0
    except Exception as error:
        outcome, reason = "device_abort", str(error)
        print(f"回放中止：{error}")
        return 1
    finally:
        if env is not None:
            try:
                env.pause(reason or outcome)
            except Exception:
                pass
            env.recorder = None
        if recorder is not None:
            recorder.close(outcome, reason)
            if trace is not None:
                _flush_replay_trace(trace, recorder.path)
        try:
            if env is not None:
                env.close()
        finally:
            inference.close()


def _starts(segment, n_obs, n_action, max_windows):
    starts = playback_starts(segment.n_frames, n_obs, n_action)
    if max_windows is not None:
        starts = starts[:max_windows]
    if not starts:
        raise ValueError("这条片段比观测窗口短，不能回放")
    return starts


def _predict(inference, obs):
    actions = inference.predict_observation(obs)["action"][0].detach().cpu().numpy()
    return policy_actions_to_controller(actions, "eef")


def _label_errors(segment, start, n_obs, n_action, actions):
    label = segment.label_actions(start, n_obs, n_action)
    if label is None:
        return None
    return action_errors(actions, label)


def _reject_unexecutable_chunk(actions, config):
    times = np.arange(len(actions)) * round(1e9 / config["policy_hz"])
    validate_actions(np.asarray(actions, dtype=np.float64), times, actions[0], config)


def _wait_live_pose(env, timeout_s=20):
    deadline, last = time.monotonic() + timeout_s, "no observation yet"
    while time.monotonic() < deadline:
        try:
            obs, _, _ = env.get_obs()
            return np.concatenate([obs["robot_eef_pose"][-1], obs["hand_joint"][-1]]).astype(
                np.float64
            )
        except ObservationNotReady as error:
            last = str(error)
            time.sleep(0.05)
    raise TimeoutError(f"现场观测未就绪：{last}")


def _wait_for_start():
    while True:
        command = _read_command()
        if command == "s":
            return True
        if command in ("q", "n", "p"):
            return False
        time.sleep(0.02)


def _capture_replay_observation(env):
    started = time.monotonic_ns()
    try:
        obs, anchor, alignment = env.get_obs()
        return {
            "obs": obs,
            "anchor": anchor,
            "alignment": alignment,
            "missing": None,
            "started": started,
            "returned": time.monotonic_ns(),
        }
    except ObservationNotReady as error:
        return {
            "obs": None,
            "anchor": None,
            "alignment": None,
            "missing": str(error),
            "started": started,
            "returned": time.monotonic_ns(),
        }


def _store_replay_decision(trace, env, inference, demo, actions, window_start, *, live=None):
    """Save the demonstration input that will be executed, plus the live observation beside it."""
    if live is None:
        raise ValueError(
            "Replay trace must receive the live observation captured before prediction"
        )
    real_obs, real_anchor, real_alignment = live["obs"], live["anchor"], live["alignment"]
    real_missing = live["missing"]
    now = time.monotonic_ns()
    timing = inference.last_timing or {}
    return trace.store_decision(
        obs=demo,
        alignment={"window_start": int(window_start), "source": "demonstration"},
        anchor_ns=real_anchor if real_anchor is not None else now,
        actions=actions,
        started_ns=timing.get("request_started_ns", now),
        observed_ns=timing.get("received_ns", now),
        preprocessed_ns=timing.get("preprocessed_ns", now),
        inputs_ready_ns=timing.get("inputs_ready_ns", now),
        returned_ns=timing.get("response_received_ns", now),
        rng=getattr(inference, "last_rng", None),
        real_obs=real_obs,
        real_alignment=real_alignment,
        real_anchor_ns=real_anchor,
        window_start=window_start,
        real_missing=real_missing,
        control_input="demonstration",
        live_capture_started_ns=live["started"],
        live_capture_returned_ns=live["returned"],
    )


def _flush_replay_trace(trace, path):
    try:
        meta = trace.flush(path)
    except Exception as error:
        (Path(path) / "trace_flush_error.txt").write_text(str(error))
        return
    trial_path = Path(path) / "trial.json"
    if trial_path.is_file() and meta is not None:
        trial = json.loads(trial_path.read_text())
        trial["trace_valid"] = meta["valid"]
        trial["trace_invalid_reason"] = meta["invalid_reason"]
        trial["trace_dropped"] = meta["dropped"]
        write_json(trial_path, trial)
        print(
            "完整记录有效。"
            if meta["valid"]
            else f"完整记录无效：{meta['invalid_reason']}。不能据此判定根因。"
        )


def _submit_path(env, actions, n_action, prediction_id=None):
    actions = np.asarray(actions, dtype=np.float64)
    for offset in range(0, len(actions), n_action):
        env.exec_actions(
            actions[offset : offset + n_action],
            time.monotonic_ns(),
            prediction_id=prediction_id if offset == 0 else None,
        )
        while env.chunk_completed_ns() is None or env.runtime.active:
            command = _read_command()
            if command in ("p", "q", "y", "n"):
                return command
            fault = env.runtime.fault or env.error
            if fault:
                raise RuntimeError(fault)
            time.sleep(0.002)
    return None


def _needs_approach(current, goal):
    delta = pose_delta(current, goal)
    return (
        max(delta["left_position_mm"], delta["right_position_mm"]) > 15
        or max(delta["left_rotation_deg"], delta["right_rotation_deg"]) > 3
        or delta["hand_max_deg"] > 3
    )


def _read_command():
    if not sys.stdin.isatty() or not select.select([sys.stdin], [], [], 0)[0]:
        return None
    line = sys.stdin.readline()
    if line == "":
        return "q"
    return line.strip().lower()


def _operator_outcome(command):
    return {"y": "success", "n": "failure", "p": "takeover", "q": "takeover"}[
        command
    ], f"Operator {command}"


def _mean_errors(collected):
    if not collected:
        return {}
    return {
        key: float(np.concatenate([item[key] for item in collected]).mean()) for key in collected[0]
    }


def _format_errors(errors):
    if errors is None:
        return "片段末尾，没有完整标签"
    return "  ".join(f"{label} {errors[key].mean():.1f}" for key, label in ERROR_LABELS)


def _format_mean(summary):
    return "  ".join(f"{label} {summary[key]:.1f}" for key, label in ERROR_LABELS if key in summary)


def _format_delta(delta):
    return (
        f"左 {delta['left_position_mm']:.0f} mm / {delta['left_rotation_deg']:.0f}°，"
        f"右 {delta['right_position_mm']:.0f} mm / {delta['right_rotation_deg']:.0f}°，"
        f"手关节最大 {delta['hand_max_deg']:.0f}°"
    )
