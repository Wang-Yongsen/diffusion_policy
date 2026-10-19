"""Run one slot of the fixed 13-slot Tianji rail batch.

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
from scipy.spatial.transform import Rotation

from diffusion_policy.real_world.batch_factors import (
    StabilityGate, compare_initialization, run_fake_preflight, visual_is_deployable,
)
from diffusion_policy.real_world.hand_settle import HandSettleCancelled, settle_and_hold
from diffusion_policy.real_world.tianji_rail_env import PROJECT_ROOT, load_deployment_config


DEFAULT_CHECKPOINT = PROJECT_ROOT / "data/outputs/2026.10.16/23.09.56_tianji_rail/checkpoints/epoch=0050.ckpt"
DEFAULT_CONFIG = PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval_loose.yaml"
DEFAULT_DATASET = PROJECT_ROOT / "data/replay_obs/rail_place3_episode_000000.zarr"
FREEZE_FILES = ("visual_candidate.json", "stability_gate.json", "initialization.json",
                "runtime_expected.json", "integration_status.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_batch(manifest):
    plan = read_json(manifest)
    root = Path(manifest).resolve().parent
    frozen = {name: read_json(root / name) for name in FREEZE_FILES}
    if any(item.get("frozen") is not True for item in frozen.values()):
        raise ValueError("V/S、初始化和运行时哈希必须在运动前冻结")
    return plan, root, frozen


def slot_item(plan, slot_id):
    matches = [item for item in plan["slots"] if item["slot_id"] == slot_id]
    if len(matches) != 1:
        raise ValueError(f"未知 slot：{slot_id}")
    return matches[0]


def deployment_config(path, plan):
    config = load_deployment_config(path)
    # The loose file supplies the jump, speed and workspace limits. The batch
    # itself stays on the original synchronous retiming and the 10 s / 24 decision cap.
    config["inference_mode"] = "sync"
    config["trial_timeout_s"] = float(plan["limits"]["active_duration_s"])
    return config


def measure_robot(env):
    sample = env.driver.get_latest()
    if sample is None or not sample.header.valid:
        raise RuntimeError("没有机械臂反馈")
    measured = {"arm_q": {}, "eef": {}, "hand": {}}
    for side in ("left", "right"):
        joints = np.asarray(sample.payload.arms[side].joints.position_rad, dtype=np.float64)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise RuntimeError(f"{side} 臂关节无效")
        pose = env.kinematics.fk(side, tuple(float(value) for value in joints))
        measured["arm_q"][side] = joints
        measured["eef"][side] = np.r_[pose.position_m, Rotation.from_quat(pose.orientation_xyzw).as_rotvec()]
        hand = env.hands[side].get_latest()
        if hand is None or not hand.header.valid:
            raise RuntimeError(f"{side} 手没有反馈")
        fingers = np.asarray(hand.payload.position_rad, dtype=np.float64)
        if fingers.shape != (20,) or not np.isfinite(fingers).all():
            raise RuntimeError(f"{side} 手关节无效")
        measured["hand"][side] = fingers
    return measured


def make_init_check(target, tolerances):
    def check(env):
        return compare_initialization(measure_robot(env), target, tolerances)
    return check


def provenance_files():
    from diffusion_policy.real_world.decision_trace import runtime_files
    files = runtime_files()
    files["batch_factors"] = (PROJECT_ROOT / "diffusion_policy/real_world/batch_factors.py").resolve()
    files["run_tianji_final_batch_slot"] = Path(__file__).resolve()
    files["eval_tianji_rail"] = (PROJECT_ROOT / "eval_tianji_rail.py").resolve()
    files["eval_tianji_rail_recorded_obs"] = (PROJECT_ROOT / "eval_tianji_rail_recorded_obs.py").resolve()
    return files


def active_bounds(trial_dir, fallback_start):
    events_path = trial_dir / "events.jsonl"
    if events_path.is_file():
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        schedules = [event for event in events if event.get("event") == "schedule" and event.get("times_ns")]
        pauses = [event for event in events if event.get("event") == "pause"]
        if schedules and pauses:
            start = min(int(event["times_ns"][0]) for event in schedules)
            ends = [int(event["monotonic_ns"]) for event in pauses if int(event["monotonic_ns"]) >= start]
            if ends:
                return start, min(ends)
    start = int(fallback_start)
    return start, start


def write_slot(output, item, plan, frozen, root, trial, init_report):
    from diffusion_policy.real_world.decision_trace import write_provenance
    write_provenance(output, provenance_files(), PROJECT_ROOT)
    outcome, manual = classify_outcome(trial, init_report)
    trial_dir = output / "trial_0001"
    started = trial.get("started_monotonic_ns", time.monotonic_ns())
    active_started, active_finished = active_bounds(trial_dir, started)
    payload = {
        "slot_id": item["slot_id"], "condition": item["condition"], "repeat": item["repeat"],
        "seed": item["seed"], "outcome": outcome,
        "active_started_ns": active_started, "active_finished_ns": active_finished,
        "initialization_sha256": sha256(root / "initialization.json"),
        "initial_state_comparable": bool(init_report and init_report.get("comparable")),
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "visual_candidate_sha256": sha256(root / "visual_candidate.json"),
        "stability_gate_sha256": sha256(root / "stability_gate.json"),
        "manual_intervention": manual,
        "visual_enabled": bool(item.get("visual_candidate", False)),
        "stable_sampling_enabled": bool(item.get("stable_sampling", False)),
        "trial_outcome": trial.get("outcome"), "trial_reason": trial.get("reason"),
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
    blocked = report["visual_enabled"] and not visual_is_deployable(frozen["visual_candidate.json"])
    report["hardware_allowed"] = (not blocked) and report["shadow_preflight"] and report["checkpoint_sha256_ok"] and report["dataset_present"]
    report["blocked_reason"] = None if not blocked else frozen["visual_candidate.json"].get("reason")
    report["slot_id"] = item["slot_id"]
    write_json(output / "preflight.json", report)
    print(json.dumps({"slot": item["slot_id"], "hardware_started": False,
                      "hardware_allowed": report["hardware_allowed"],
                      "blocked_reason": report["blocked_reason"]}, ensure_ascii=False))
    checks_ok = report["shadow_preflight"] and report["checkpoint_sha256_ok"] and report["dataset_present"]
    return 0 if checks_ok else 1


def refuse_visual(output, item, spec):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "not_started.json", {
        "slot_id": item["slot_id"], "hardware_started": False,
        "reason": spec.get("reason"), "condition": item["condition"]})
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
    target = frozen["initialization.json"]["target"]
    tolerances = frozen["initialization.json"]["measured_thresholds"]
    check = make_init_check(target, tolerances)
    if item["kind"] == "demonstration_calibration":
        return run_calibration(args, plan, root, frozen, item, output, config, check, target, tolerances)
    return run_closed_loop(args, plan, root, frozen, item, output, config, check, target, tolerances)


def confirm_hand_settle(terminal):
    import home_tianji_rail as home
    while True:
        keys = terminal.read(0.)
        if home._stopped(keys):
            print("已取消摆手。")
            return False
        if keys is None:
            break
    print("核对前会先把双手摆回冻结起点并保持。这段时间只有手会动，胳膊不动。")
    print("确认周围清空后按回车开始摆手；摆手过程中按 Q 取消。")
    while True:
        keys = terminal.read(.1)
        if home._stopped(keys):
            print("已取消摆手。")
            return False
        if keys and any(key in ("\n", "\r") for key in keys):
            return True


def prepare_slot_hands(env, target, tolerances):
    import home_tianji_rail as home
    goals = {side: tuple(float(q) for q in target["hand"][side]) for side in ("left", "right")}
    tolerance_rad = np.deg2rad(float(tolerances["max_hand_joint_error_deg"]))
    with home.MotionTerminal() as terminal:
        if not confirm_hand_settle(terminal):
            raise HandSettleCancelled()
        settle_and_hold(env, goals, tolerance_rad, terminal.read)
    print("按 s 才会让机械臂开始。")


def run_calibration(args, plan, root, frozen, item, output, config, check, target, tolerances):
    from types import SimpleNamespace
    from diffusion_policy.real_world.recorded_episode import open_recorded_segment
    from eval_tianji_rail_recorded_obs import run_motion

    segment = open_recorded_segment(args.dataset, episode="0")
    namespace = SimpleNamespace(
        checkpoint=Path(args.checkpoint), batch_seed=int(item["seed"]), skip_approach=True,
        max_windows=int(plan["limits"]["max_decisions"]), trace=True,
        trace_capacity=int(plan["limits"]["max_decisions"]),
        active_limit_s=float(plan["limits"]["active_duration_s"]), initialization_check=check,
        prepare_hands=lambda env: prepare_slot_hands(env, target, tolerances))
    code = run_motion(namespace, segment, config, args.device or config["device"], output)
    trial_path = output / "trial_0001" / "trial.json"
    if not trial_path.is_file():
        print("本段没有形成试验记录，不占用终态。")
        return code
    trial = read_json(trial_path)
    init_report = read_json(output / "initialization_check.json") if (output / "initialization_check.json").is_file() else None
    write_slot(output, item, plan, frozen, root, trial, init_report)
    return code


def run_closed_loop(args, plan, root, frozen, item, output, config, check, target, tolerances):
    from diffusion_policy.env_runner.tianji_rail_real_runner import TianjiRailRealRunner
    from diffusion_policy.real_world.decision_trace import SessionTrace, write_provenance
    from diffusion_policy.real_world.policy_inference_process import PolicyInferenceProcess
    from diffusion_policy.real_world.tianji_rail_env import TianjiRailEnv

    device = args.device or config["device"]
    inference = PolicyInferenceProcess(args.checkpoint, device, capture_rng=True)
    env = None
    try:
        n_obs = int(inference.preflight["n_obs_steps"])
        env = TianjiRailEnv(config, n_obs, motion=True)
        trace = SessionTrace(decision_capacity=int(plan["limits"]["max_decisions"]))
        env.enable_trace(trace)
        write_provenance(output, provenance_files(), PROJECT_ROOT)
        env.start()
        try:
            prepare_slot_hands(env, target, tolerances)
        except HandSettleCancelled:
            print("已取消摆手，未核对，也未接合。")
            return 0
        preview = check(env)
        print("冻结起点核对：" + ("通过" if preview["comparable"] else "未通过"))
        print(preview["deltas"])
        if not preview["comparable"]:
            print("按 s 会把本段记为无效比较，并且不会接合。按 q 退出后先回位。")
        metadata = {"checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_sha256": plan["checkpoint_sha256"], "batch_seed": int(item["seed"]),
                    "deployment": config, "slot": item, "limits_config": str(Path(args.config).resolve())}
        runner = TianjiRailRealRunner(env, inference, None, metadata, output, trace=trace)
        runner.batch_hooks = argparse.Namespace(
            stable=bool(item.get("stable_sampling")), max_decisions=int(plan["limits"]["max_decisions"]),
            gate=StabilityGate(frozen["stability_gate.json"]["parameters"]), check_init=check, init_report=None)
        runner.run()
        init_report = runner.batch_hooks.init_report or preview
    finally:
        if env is not None:
            env.close()
        inference.close()
    trial_path = output / "trial_0001" / "trial.json"
    if not trial_path.is_file():
        print("本段没有形成试验记录，不占用终态。")
        return 0
    write_slot(output, item, plan, frozen, root, read_json(trial_path), init_report)
    return 0


def run_home(args, frozen):
    if not args.require_manual_start:
        raise ValueError("回位必须带 --require-manual-start")
    if not sys.stdin.isatty():
        print("回位要在交互终端里确认后才会动。请在你自己的终端运行这条命令。")
        return 2
    import home_tianji_rail as home
    target = frozen["initialization.json"]["target"]
    arm_settings = home.load_arm_config(home.ARM_CONFIG)
    hand_config = home.load_hand_config(home.HAND_CONFIG)
    print("本批次所有 slot 共用这一组起点。回位不读取示范图像，也不下发策略动作。")
    for side in ("left", "right"):
        print(f"{side} 臂目标 °：" + home._format_deg(target["arm_q"][side]))
        print(f"{side} 手目标 °：" + home._format_deg(target["hand"][side]))
    with home.MotionTerminal() as terminal:
        if not home.confirm_motion(terminal):
            return 0
        home.move_arms(arm_settings, "both", terminal, inspect=False, joint_targets=target["arm_q"])
        home.move_hands(hand_config, ("left", "right"), terminal, duration_s=3.,
                        tolerance_rad=np.deg2rad(3.), settle_timeout_s=2., inspect=False,
                        goals=target["hand"])
    print("回位完成。退出后灵巧手会松开；每个 slot 核对前会重新摆手并保持，再按冻结容差核对。")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--slot", help="P00–P12；home 模式不需要")
    parser.add_argument("--mode", choices=("offline-preflight", "hardware", "home"), default="offline-preflight")
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
