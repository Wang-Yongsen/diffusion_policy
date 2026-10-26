"""Rail experiment: teach target, plan offline, or move only to pre-insertion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from queue import Queue, Empty
import sys
import subprocess
import signal
import threading
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

from rail_preinsert.core import (digest, load_calibration, load_config, load_target, mean_pose,
                                read_json, write_json)


class Log:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = Queue()
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            with (self.directory / "events.jsonl").open("w") as stream:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
        except Exception as error:
            self.error = str(error)

    def emit(self, event, **values):
        if self.error:
            raise RuntimeError(f"Experiment log failed: {self.error}")
        self.queue.put({"event": event, "host_monotonic_ns": time.monotonic_ns(), **values})

    def close(self):
        self.queue.put(None); self.thread.join(timeout=3)


def kinematics():
    from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics
    return TianjiKinematics()


def check(config):
    from rail_preinsert.workflow import hand_references
    problems = []
    for name in ("arm_config", "hand_config", "mesh", "urdf", "manifest", "hand_reference"):
        if not Path(config[name]).is_file():
            problems.append(f"Missing {name}: {config[name]}")
    try:
        hand_references(config)
    except Exception as error:
        problems.append(f"Hand reference: {error}")
    calibration = None
    try:
        calibration, _ = load_calibration(config["calibration"])
    except Exception as error:
        problems.append(f"Calibration: {error}")
    if not Path(config["target"]).is_file():
        problems.append(f"Missing taught target: {config['target']}")
    elif calibration is not None:
        try:
            load_target(config["target"], calibration, config["mesh"])
        except Exception as error:
            problems.append(f"Target: {error}")
    print(json.dumps({"end_stage": "preinsert", "ready_for_live_configuration": not problems,
                      "problems": problems}, ensure_ascii=False, indent=2))
    return 1 if problems else 0


def teach(config, args):
    from rail_preinsert.perception import Perception
    from rail_preinsert.workflow import observation_poses
    from rail_preinsert.pose_alignment import RailPoseAlignment
    calibration, _ = load_calibration(config["calibration"])
    if args.observation:
        observation = read_json(args.observation)
    else:
        with tempfile.TemporaryDirectory(prefix="rail-teach-") as directory:
            with_perception = Perception(config, Path(directory) / "perception", calibration["camera_serial"])
            try:
                print("导轨正确就位后，在相机窗口标注可见导轨区域。", flush=True)
                observation = with_perception.measure(config["perception_samples"])
            finally:
                with_perception.close()
    records, poses = observation_poses(observation, calibration, config["mesh"],
                                alignment=RailPoseAlignment.from_config(config))
    target = {"schema_version": 1, "unit": "m", "frame": "T_C_CAD",
              "camera_serial": calibration["camera_serial"], "mesh_sha256": digest(config["mesh"]),
              "T_C_O_goal": mean_pose(poses).tolist(), "test_data": False,
              "pose_alignment": [r.get("pose_alignment") for r in records],
              "observation": observation, "calibration_sha256_at_teach": digest(config["calibration"])}
    if Path(args.output).exists():
        raise FileExistsError("Target already exists; choose a new output rather than overwriting")
    write_json(args.output, target)
    print(f"目标位姿已保存：{Path(args.output).resolve()}")


def offline_plan(config, args):
    from rail_preinsert.workflow import prepare_plan
    state, observation = read_json(args.state), read_json(args.observation)
    _, report, _ = prepare_plan(config, kinematics(), state, observation, allow_test_data=args.allow_test_data)
    if Path(args.output).exists():
        raise FileExistsError("Plan output already exists")
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "end_stage": report["end_stage"],
                      "duration_s": report["duration_s"], "output": str(Path(args.output).resolve())}, ensure_ascii=False))


def plan_process(config, directory, hardware, cancelled):
    from rail_preinsert.planner import JointPath
    import yaml
    effective = dict(config); effective.pop("config_path", None)
    config_path = directory / "effective.yaml"
    config_path.write_text(yaml.safe_dump(effective, allow_unicode=True))
    plan_path = directory / f"plan_{time.monotonic_ns()}.json"
    command = [sys.executable, str(HERE / "run.py"), "--config", str(config_path), "plan",
               "--state", str(directory / "initial_state.json"),
               "--observation", str(directory / "initial_observation.json"), "--output", str(plan_path)]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (directory / "planner.log").open("a") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + config["planning_timeout_s"] + 60
            while process.poll() is None:
                hardware.raise_if_fault()
                if cancelled() or time.monotonic() > deadline:
                    raise RuntimeError("Planning cancelled or timed out")
                time.sleep(.03)
            if process.returncode:
                raise RuntimeError(f"Planning failed; inspect {directory / 'planner.log'}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=2)
    report = read_json(plan_path)
    path = JointPath(report["joint_waypoints_rad"], report["joint_speed_rad_s"], report["joint_acceleration_rad_s2"])
    return path, report


def live(config, args):
    from diffusion_policy.real_world.tianji.terminal import MotionTerminal
    from rail_preinsert.hardware import Hardware
    from rail_preinsert.perception import Perception
    from rail_preinsert.workflow import hand_references, check_plan_inputs, verify_preinsert
    # Validate all physical inputs before opening or enabling a device.
    calibration, _ = load_calibration(config["calibration"])
    load_target(config["target"], calibration, config["mesh"])
    references = hand_references(config)
    log = Log(args.output)
    hardware, perception = None, None
    end = threading.Event(); commands = Queue()
    stage = "starting"
    old_sigint = None
    write_json(log.directory / "config.json", config)
    try:
        hardware = Hardware(config, kinematics(), log.emit, motion=args.motion).start()
        if not args.motion:
            value = hardware.snapshot()
            value = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}
            write_json(log.directory / "read_only_state.json", value)
            print("只读设备反馈：" + json.dumps(value, ensure_ascii=False))
            return
        with MotionTerminal() as terminal:
            old_sigint = signal.signal(signal.SIGINT, lambda *_: hardware.pause("Operator Ctrl+C: retain grasp"))
            def keyboard():
                while not end.is_set():
                    keys = terminal.read(.1)
                    if keys == "":
                        hardware.pause("Terminal closed"); end.set(); return
                    for key in keys or "":
                        if key.lower() in ("q", " ", "\x03"):
                            hardware.pause("Operator pause: retain hand target")
                        elif key.lower() == "x":
                            hardware.pause("Operator exit"); end.set()
                        else:
                            allowed = {"waiting_open": "o", "open": "g", "grasped": "m", "planned": "me",
                                       "preinsert_reached": "mv", "preinsert_unverified": "mv"}
                            if key.lower() in allowed.get(stage, ""):
                                commands.put(key.lower())
            watcher = threading.Thread(target=keyboard, daemon=True); watcher.start()
            print("正在初始化头部 RGB-D 与 FoundationPose……", flush=True)
            perception = Perception(config, log.directory / "perception", calibration["camera_serial"],
                                    cancelled=lambda: end.is_set() or hardware.paused)
            stage = "waiting_open"
            print("O 张手；G 闭合；人撤手后 M 测量并规划；E 执行；V 复核。")
            print("Q/空格暂停机械臂并保持手；支撑好导轨后 X 退出（退出会停用灵巧手）。", flush=True)
            trajectory = report = None
            while not end.is_set():
                if hardware.paused and stage != "paused":
                    stage = "paused"
                    write_json(log.directory / "result.json", {"status": stage, "reason": hardware.error,
                                                               "end_stage": "preinsert", "insertion_executed": False})
                    print(f"已暂停：{hardware.error}。手保持原目标，支撑导轨后按 X 退出。", flush=True)
                try:
                    key = commands.get(timeout=.1)
                except Empty:
                    continue
                try:
                    if stage == "paused":
                        continue
                    if key == "o" and stage == "waiting_open":
                        stage = "opening"
                        hardware.move_hand(references["open"]); hardware.wait_hand()
                        stage = "open"
                        print("手已张开。放入导轨后按 G 闭合。", flush=True)
                    elif key == "g" and stage == "open":
                        stage = "closing"
                        hardware.move_hand(references["grasp"]); hardware.wait_hand()
                        stage = "grasped"
                        print("闭合保持。人撤手、导轨稳定后按 M 测量并规划。", flush=True)
                    elif key == "m" and stage in ("grasped", "planned", "preinsert_reached", "preinsert_unverified"):
                        stage = "measuring"
                        observation = perception.measure(config["perception_samples"])
                        hardware.raise_if_fault()
                        state = hardware.stationary_history(observation)
                        state["model_sha256"] = hardware.kinematics.model.digest
                        write_json(log.directory / "initial_observation.json", observation)
                        write_json(log.directory / "initial_state.json", state)
                        trajectory, report = plan_process(config, log.directory, hardware, end.is_set)
                        hardware.raise_if_fault()
                        write_json(log.directory / "plan.json", report)
                        stage = "planned"
                        print(f"路径已保存到 plan.json，预计 {trajectory.duration:.1f} s，终点为预插入位姿。按 E 执行。", flush=True)
                    elif key == "e" and stage == "planned":
                        stage = "executing"
                        check_plan_inputs(config, report)
                        if report["model_sha256"] != hardware.kinematics.model.digest:
                            raise ValueError("Kinematic model changed after planning")
                        hardware.start_path(trajectory, report["initial_full_joints"])
                        hardware.wait_path(report["targets"]["T_B_F_preinsert"])
                        stage = "verifying"
                        print("机械臂已停稳并保持夹持，正在复测导轨……", flush=True)
                        observation = perception.measure(config["perception_samples"])
                        hardware.raise_if_fault()
                        hardware.stationary_history(observation)
                        write_json(log.directory / "final_observation.json", observation)
                        result = verify_preinsert(config, report, observation)
                        stage = result["status"]
                        write_json(log.directory / "result.json", result)
                        print(f"{stage}：保持夹持，实验停在预插入阶段。V 复测；M 重新规划；Q 暂停；X 退出。", flush=True)
                    elif key == "v" and stage in ("preinsert_reached", "preinsert_unverified"):
                        stage = "verifying"
                        observation = perception.measure(config["perception_samples"])
                        hardware.raise_if_fault(); hardware.stationary_history(observation)
                        result = verify_preinsert(config, report, observation)
                        stage = result["status"]
                        write_json(log.directory / "final_observation.json", observation)
                        write_json(log.directory / "result.json", result)
                        print(f"复核：{stage}，继续保持夹持。", flush=True)
                    else:
                        continue
                    log.emit("stage", stage=stage)
                except KeyboardInterrupt:
                    hardware.pause("Operator interrupt")
                except Exception as error:
                    hardware.pause(str(error))
            watcher.join(timeout=1)
    finally:
        end.set()
        cleanup_errors = []
        for resource in (perception, hardware):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    cleanup_errors.append(str(error))
        log.close()
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)
        if cleanup_errors:
            write_json(log.directory / "cleanup_errors.json", {"errors": cleanup_errors})
            raise RuntimeError("; ".join(cleanup_errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="不带参数直接启动左臂预插入；只读请运行 read_preinsert_state.py。")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="只检查本地输入，不连接设备")
    teach_parser = sub.add_parser("teach-target", help="导轨正确就位后记录目标")
    teach_parser.add_argument("--observation", type=Path)
    teach_parser.add_argument("--output", type=Path, default=HERE / "targets/rail_goal.json")
    plan_parser = sub.add_parser("plan", help="从已记录状态与位姿离线规划，不连接设备")
    plan_parser.add_argument("--state", type=Path, required=True)
    plan_parser.add_argument("--observation", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--allow-test-data", action="store_true", help="仅离线接受明确标记的测试数据")
    run_parser = sub.add_parser("run", help="默认只读设备；--motion 才允许键盘触发动作")
    run_parser.add_argument("--motion", action="store_true")
    run_parser.add_argument("--output", type=Path, help="可选：保留记录；默认临时运行，退出后清理")
    replay_parser = sub.add_parser("replay", help="检查关节限位并重放轨迹到 PNG，不连接设备")
    replay_parser.add_argument("--plan", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv or ["run", "--motion"])
    config = load_config(args.config)
    try:
        if args.command == "check":
            return check(config)
        if args.command == "teach-target":
            teach(config, args)
        elif args.command == "plan":
            offline_plan(config, args)
        elif args.command == "replay":
            from rail_preinsert.replay import replay
            print(json.dumps(replay(config, kinematics(), args.plan, args.output), ensure_ascii=False))
        else:
            if args.output is None:
                with tempfile.TemporaryDirectory(prefix="rail-preinsert-") as directory:
                    args.output = Path(directory) / "run"
                    live(config, args)
            else:
                live(config, args)
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"实验停止：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
