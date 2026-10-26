"""Run one slot from a Tianji rail batch manifest.

``offline-preflight`` checks the frozen factors with a fake clock and does not
connect to hardware. ``hardware`` and ``home`` require an interactive terminal
and an explicit manual start; they never retry a slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Root scripts such as home_tianji_rail.py are not installed packages. The
# inference child also imports eval_tianji_rail from this directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_pythonpath = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
if str(_PROJECT_ROOT) not in _pythonpath:
    os.environ["PYTHONPATH"] = os.pathsep.join([str(_PROJECT_ROOT), *_pythonpath])

import numpy as np

from diffusion_policy.real_world.tianji.batch_factors import (
    StabilityGate,
    run_fake_preflight,
    visual_is_deployable,
)
from diffusion_policy.real_world.tianji.recorded_episode import observation_sources
from diffusion_policy.real_world.tianji_rail_env import PROJECT_ROOT, load_deployment_config


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "data/outputs/2026.10.16/23.09.56_tianji_rail/checkpoints/epoch=0050.ckpt"
)
DEFAULT_CONFIG = PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval.yaml"
# The batch checkpoint was trained at 30 Hz; these values match the former loose config.
BATCH_OVERRIDES = ("policy_hz=30", "replanning_steps=2")
DEFAULT_DATASET = PROJECT_ROOT / "data/replay_obs/rail_place3_episode_000000.zarr"
FREEZE_FILES = (
    "visual_candidate.json",
    "stability_gate.json",
    "initialization.json",
    "runtime_expected.json",
    "integration_status.json",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_batch(manifest):
    plan = read_json(manifest)
    root = (Path(manifest).resolve().parent / plan.get("freeze_root", ".")).resolve()
    frozen = {name: read_json(root / name) for name in FREEZE_FILES}
    if any(item.get("frozen") is not True for item in frozen.values()):
        raise ValueError("V/S、初始化和运行时哈希必须在运动前冻结")
    return plan, root, frozen


def slot_item(plan, slot_id):
    matches = [item for item in plan["slots"] if item["slot_id"] == slot_id]
    if len(matches) != 1:
        raise ValueError(f"未知 slot：{slot_id}")
    item = matches[0]
    slot_observation_mode(item)
    return item


def slot_observation_mode(item):
    if item["kind"] == "mixed_observation":
        mode = item["observation_mode"]
        if mode == "demonstration":
            raise ValueError("混合观测 slot 不能使用完整示教输入")
        observation_sources(mode)
        if item.get("visual_candidate") or item.get("stable_sampling"):
            raise ValueError("混合观测对比固定为原始图像和普通同步采样")
        return mode
    return "demonstration" if item["kind"] == "demonstration_calibration" else "real"


def slot_observation_sources(item):
    mode = slot_observation_mode(item)
    return (
        {"images": "live", "robot_eef_pose": "live", "hand_joint": "live"}
        if mode == "real"
        else observation_sources(mode)
    )


def deployment_config(path, plan):
    config = load_deployment_config(path, BATCH_OVERRIDES)
    # The batch manifest controls the replay scope; null removes its wall-time cutoff.
    config["inference_mode"] = "sync"
    limit_s = plan["limits"]["active_duration_s"]
    config["trial_timeout_s"] = None if limit_s is None else float(limit_s)
    return config


def provenance_files():
    from diffusion_policy.real_world.tianji.trace import runtime_files

    files = runtime_files()
    files["batch_factors"] = (
        PROJECT_ROOT / "diffusion_policy/real_world/tianji/batch_factors.py"
    ).resolve()
    files["run_tianji_final_batch_slot"] = Path(__file__).resolve()
    files["eval_tianji_rail"] = (PROJECT_ROOT / "eval_tianji_rail.py").resolve()
    files["eval_tianji_rail_recorded_obs"] = (
        PROJECT_ROOT / "eval_tianji_rail_recorded_obs.py"
    ).resolve()
    files["recorded_episode"] = (
        PROJECT_ROOT / "diffusion_policy/real_world/recorded_episode.py"
    ).resolve()
    return files


def active_bounds(trial_dir, fallback_start):
    events_path = trial_dir / "events.jsonl"
    if events_path.is_file():
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        schedules = [
            event
            for event in events
            if event.get("event") == "schedule" and event.get("times_ns")
        ]
        pauses = [event for event in events if event.get("event") == "pause"]
        if schedules and pauses:
            start = min(int(event["times_ns"][0]) for event in schedules)
            ends = [
                int(event["monotonic_ns"])
                for event in pauses
                if int(event["monotonic_ns"]) >= start
            ]
            if ends:
                return start, min(ends)
    start = int(fallback_start)
    return start, start


def write_slot(output, item, plan, frozen, root, trial, init_report):
    from diffusion_policy.real_world.tianji.trace import write_provenance

    write_provenance(output, provenance_files(), PROJECT_ROOT)
    outcome, manual = classify_outcome(trial, init_report)
    trial_dir = output / "trial_0001"
    started = trial.get("started_monotonic_ns", time.monotonic_ns())
    active_started, active_finished = active_bounds(trial_dir, started)
    payload = {
        "slot_id": item["slot_id"],
        "condition": item["condition"],
        "repeat": item["repeat"],
        "seed": item["seed"],
        "outcome": outcome,
        "observation_mode": slot_observation_mode(item),
        "observation_sources": slot_observation_sources(item),
        "execution_scope": plan.get("execution_scope", "bounded"),
        "planned_windows": trial.get("planned_windows"),
        "completed_windows": trial.get("completed_windows"),
        "full_segment_completed": trial.get("full_segment_completed", False),
        "active_started_ns": active_started,
        "active_finished_ns": active_finished,
        "initialization_sha256": sha256(root / "initialization.json"),
        "initialization_mode": "current_pose",
        "initialization_check_required": False,
        "initial_state_comparable": (
            None if init_report is None else bool(init_report.get("comparable"))
        ),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "visual_candidate_sha256": sha256(root / "visual_candidate.json"),
        "stability_gate_sha256": sha256(root / "stability_gate.json"),
        "manual_intervention": manual,
        "visual_enabled": bool(item.get("visual_candidate", False)),
        "stable_sampling_enabled": bool(item.get("stable_sampling", False)),
        "trial_outcome": trial.get("outcome"),
        "trial_reason": trial.get("reason"),
        "task_success": outcome == "success" and trial.get("outcome") == "success",
    }
    write_json(output / "slot.json", payload)
    if init_report is not None:
        write_json(output / "initialization_check.json", init_report)
    return payload


def classify_outcome(trial, init_report):
    raw = trial.get("outcome")
    if init_report is not None and not init_report.get("comparable"):
        return "invalid_comparison", False
    if raw in ("takeover", "invalid_comparison") or trial.get("trace_valid") is not True:
        return "invalid_comparison", raw == "takeover"
    if raw == "success":
        return "success", False
    return "valid_failure", False


def offline_preflight(args, plan, root, frozen, item):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "slot.json").exists():
        raise FileExistsError(f"{output} 已有终态，预检不能覆盖")
    report = run_fake_preflight(frozen["initialization.json"]["target"])
    shadow = root / "shadow_preflight" / "preflight.json"
    report["shadow_preflight"] = shadow.is_file()
    report["checkpoint_sha256_ok"] = sha256(args.checkpoint) == plan["checkpoint_sha256"]
    report["dataset_present"] = Path(args.dataset).exists()
    report["inference_mode"] = "sync"
    report["limits_config"] = str(Path(args.config).resolve())
    report["visual_enabled"] = bool(item.get("visual_candidate", False))
    report["stable_sampling_enabled"] = bool(item.get("stable_sampling", False))
    report["observation_mode"] = slot_observation_mode(item)
    report["observation_sources"] = slot_observation_sources(item)
    report["initialization_mode"] = "current_pose"
    report["initialization_check_required"] = False
    report["execution_scope"] = plan.get("execution_scope", "bounded")
    report["active_duration_s"] = plan["limits"]["active_duration_s"]
    report["max_decisions"] = plan["limits"]["max_decisions"]
    if report["execution_scope"] == "full_segment" and report["dataset_present"]:
        from diffusion_policy.real_world.tianji.recorded_episode import (
            open_recorded_segment,
            playback_starts,
        )

        segment = open_recorded_segment(args.dataset, episode="0")
        n_obs, n_action = int(plan["model"]["n_obs_steps"]), int(
            plan["model"]["n_action_steps"]
        )
        starts = playback_starts(segment.n_frames, n_obs, n_action)
        report.update(
            dataset_frames=segment.n_frames,
            planned_windows=len(starts),
            nominal_duration_s=segment.n_frames / float(plan["model"]["policy_hz"]),
            last_window_actions=min(n_action, segment.n_frames - starts[-1] - n_obs + 1),
            trace_image_storage="memmap",
        )
    blocked = report["visual_enabled"] and not visual_is_deployable(
        frozen["visual_candidate.json"]
    )
    report["hardware_allowed"] = (
        (not blocked)
        and report["shadow_preflight"]
        and report["checkpoint_sha256_ok"]
        and report["dataset_present"]
    )
    report["blocked_reason"] = (
        None if not blocked else frozen["visual_candidate.json"].get("reason")
    )
    report["slot_id"] = item["slot_id"]
    write_json(output / "preflight.json", report)
    print(
        json.dumps(
            {
                "slot": item["slot_id"],
                "hardware_started": False,
                "hardware_allowed": report["hardware_allowed"],
                "blocked_reason": report["blocked_reason"],
            },
            ensure_ascii=False,
        )
    )
    checks_ok = (
        report["shadow_preflight"]
        and report["checkpoint_sha256_ok"]
        and report["dataset_present"]
    )
    return 0 if checks_ok else 1


def refuse_visual(output, item, spec):
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "not_started.json",
        {
            "slot_id": item["slot_id"],
            "hardware_started": False,
            "reason": spec.get("reason"),
            "condition": item["condition"],
        },
    )
    print(spec.get("reason"))
    print(f"{item['slot_id']} 使用视觉修正，但预检没有可冻结的 V。不连接设备。")
    return 3


def run_hardware(args, plan, root, frozen, item):
    if not args.require_manual_start or not args.no_auto_retry:
        raise ValueError("真机 slot 必须同时带 --require-manual-start 和 --no-auto-retry")
    if not sys.stdin.isatty():
        print("真机要在交互终端里输入 s 才会动。请在你自己的终端运行这条命令。")
        return 2
    output = Path(args.output)
    if (output / "slot.json").exists() or (output / "trial_0001").exists():
        raise FileExistsError(f"{output} 已有试验记录，不会自动重跑")
    if sha256(args.checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("权重哈希和冻结清单不一致")
    spec = frozen["visual_candidate.json"]
    if item.get("visual_candidate") and not visual_is_deployable(spec):
        return refuse_visual(output, item, spec)
    output.mkdir(parents=True, exist_ok=True)
    config = deployment_config(args.config, plan)
    if item["kind"] in ("demonstration_calibration", "mixed_observation"):
        return run_calibration(args, plan, root, frozen, item, output, config)
    return run_closed_loop(args, plan, root, frozen, item, output, config)


def run_calibration(args, plan, root, frozen, item, output, config):
    from types import SimpleNamespace
    from diffusion_policy.real_world.tianji.recorded_episode import open_recorded_segment
    from diffusion_policy.real_world.tianji.replay import run_motion

    segment = open_recorded_segment(args.dataset, episode="0")
    full_segment = plan.get("execution_scope") == "full_segment"
    max_windows = plan["limits"]["max_decisions"]
    limit_s = plan["limits"]["active_duration_s"]
    namespace = SimpleNamespace(
        checkpoint=Path(args.checkpoint),
        batch_seed=int(item["seed"]),
        skip_approach=True,
        observation_mode=slot_observation_mode(item),
        max_windows=None if max_windows is None else int(max_windows),
        trace=True,
        trace_capacity=None if full_segment else int(max_windows),
        full_segment=full_segment,
        active_limit_s=None if limit_s is None else float(limit_s),
    )
    code = run_motion(namespace, segment, config, args.device or config["device"], output)
    trial_path = output / "trial_0001" / "trial.json"
    if not trial_path.is_file():
        print("本段没有形成试验记录，不占用终态。")
        return code
    trial = read_json(trial_path)
    write_slot(output, item, plan, frozen, root, trial, None)
    return code


def run_closed_loop(args, plan, root, frozen, item, output, config):
    from diffusion_policy.env_runner.tianji_rail_real_runner import TianjiRailRealRunner
    from diffusion_policy.real_world.tianji.trace import SessionTrace, write_provenance
    from diffusion_policy.real_world.tianji.inference import PolicyInferenceProcess
    from diffusion_policy.real_world.tianji.env import TianjiRailEnv

    device = args.device or config["device"]
    inference = PolicyInferenceProcess(args.checkpoint, device, capture_rng=True)
    env = None
    try:
        n_obs = int(inference.preflight["n_obs_steps"])
        if n_obs != 2 or int(inference.preflight["n_action_steps"]) < 2:
            raise ValueError(
                "Synchronous inference requires checkpoint n_obs_steps=2 and n_action_steps >= 2"
            )
        env = TianjiRailEnv(config, n_obs, motion=True)
        trace = SessionTrace(decision_capacity=int(plan["limits"]["max_decisions"]))
        env.enable_trace(trace)
        write_provenance(output, provenance_files(), PROJECT_ROOT)
        env.start()
        print("从当前姿态开始；按 s 接合并执行策略。")
        metadata = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": plan["checkpoint_sha256"],
            "prediction_seed": int(item["seed"]),
            "initialization_mode": "current_pose",
            "initialization_check_required": False,
            "deployment": config,
            "slot": item,
            "limits_config": str(Path(args.config).resolve()),
        }
        runner = TianjiRailRealRunner(env, inference, None, metadata, output, trace=trace)
        runner.batch_hooks = argparse.Namespace(
            stable=bool(item.get("stable_sampling")),
            max_decisions=int(plan["limits"]["max_decisions"]),
            gate=StabilityGate(frozen["stability_gate.json"]["parameters"]),
            check_init=None,
            init_report=None,
        )
        runner.run()
    finally:
        if env is not None:
            env.close()
        inference.close()
    trial_path = output / "trial_0001" / "trial.json"
    if not trial_path.is_file():
        print("本段没有形成试验记录，不占用终态。")
        return 0
    write_slot(output, item, plan, frozen, root, read_json(trial_path), None)
    return 0


def run_home(args, frozen):
    if not args.require_manual_start:
        raise ValueError("回位必须带 --require-manual-start")
    if not sys.stdin.isatty():
        print("回位要在交互终端里确认后才会动。请在你自己的终端运行这条命令。")
        return 2
    from diffusion_policy.real_world.tianji import home

    target = frozen["initialization.json"]["target"]
    arm_settings = home.load_arm_config(home.ARM_CONFIG)
    hand_config = home.load_hand_config(home.HAND_CONFIG)
    print("手动回到冻结起点。")
    for side in ("left", "right"):
        print(f"{side} 臂目标 °：" + home._format_deg(target["arm_q"][side]))
        print(f"{side} 手目标 °：" + home._format_deg(target["hand"][side]))
    with home.MotionTerminal() as terminal:
        if not home.confirm_motion(terminal):
            return 0
        home.move_arms(
            arm_settings, "both", terminal, inspect=False, joint_targets=target["arm_q"]
        )
        home.move_hands(
            hand_config,
            ("left", "right"),
            terminal,
            duration_s=3.0,
            tolerance_rad=np.deg2rad(3.0),
            settle_timeout_s=2.0,
            inspect=False,
            goals=target["hand"],
        )
    print("回位完成。退出后灵巧手会松开；实验从启动时的当前姿态开始。")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--slot", help="清单中的 slot ID；home 模式不需要")
    parser.add_argument(
        "--mode", choices=("offline-preflight", "hardware", "home"), default="offline-preflight"
    )
    parser.add_argument("--require-manual-start", action="store_true")
    parser.add_argument("--no-auto-retry", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    plan, root, frozen = load_batch(args.manifest)
    if args.mode == "home":
        return run_home(args, frozen)
    if not args.slot:
        parser.error("除 home 外必须指定 --slot")
    item = slot_item(plan, args.slot)
    if args.output is None:
        parser.error("需要 --output")
    if args.mode == "offline-preflight":
        return offline_preflight(args, plan, root, frozen, item)
    return run_hardware(args, plan, root, frozen, item)


if __name__ == "__main__":
    raise SystemExit(main())
