"""Receive a rail, save its mean held pose, then M moves above its taught goal."""
from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path
from queue import Empty, Full, Queue
import signal
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from rail_preinsert.core import HERE, read_json, write_json
from rail_preinsert.perception import python_command
from rail_receive.logic import JOG_KEYS, ReceiveGeometry, hand_targets, load_config
from rail_receive.hardware import RightHardware
from rail_receive.perception import ReceivePerception
from run import Log, kinematics


HELP = """右臂点动（右臂原生基座）：W/S 前/后，R/F 上/下，D/A 向外/向内；
I/K 绕 X 正/负转，J/L 绕 Y 正/负转，U/O 绕 Z 正/负转；+/- 调整步长。
右手张开并准备好后等 5 秒开始闭合；抓握完成后再等 8 秒，按保存的 bbox 运行 SAM 和 FoundationPose。
估姿启动前可在窗口按 E 画 bbox，Enter 保存；仅估姿模式下 E/T 可重估。G 可提前闭合。
抓后估姿完成会保存均值并等待；按 M 直接使用已保存均值规划移动（终端/窗口均可）。
上方到位后，按 N 启动所选插入流程；保持抓握，完成后等待人工确认（终端/窗口均可）。
位置螺旋模式默认到位后重新估姿并纠偏，再按 N 下放；其余模式需复核通过。
Q / 空格 / Ctrl+C 暂停机械臂和抓握推进，保持最后手目标；本次不自动恢复。
X 退出并停用右手：退出前先由人或夹具支撑导轨。"""

POSE_PHASES = {"opening_rgbd": "打开 RGB-D 相机（首帧最多等待 10 秒）",
    "acquiring_rgbd": "采集 RGB-D 帧", "loading_sam": "加载本地 SAM 框/点提示分割器",
    "segmenting_held_rail": "SAM 按保存的框/点分割单个导轨",
    "initializing_pose": "已取得新 mask，正在初始化 FoundationPose",
    "aligning_pose": "按目标方向和每轴小于 45° 的约束对齐位姿"}
def preflight(config, *, probe_gpu=True, move_above=False, insertion=False):
    problems = []
    insertion_report = None
    if insertion:
        insertion_mode = config["lowering"]["mode"]
        try:
            missing = []
            if insertion_mode == "force_spiral":
                from rail_receive.insertion import validate_settings
                missing = validate_settings(config["insertion"], require_ready=False)
            elif insertion_mode == "position_spiral":
                from rail_receive.position_spiral import validate_settings
                validate_settings(config["position_spiral"])
                if config["move_above"]["height_m"] > config["position_spiral"]["max_descent_m"]:
                    raise ValueError("Above-goal height exceeds position_spiral.max_descent_m")
            else:
                from rail_receive.lowering import validate_settings
                validate_settings(config["lowering"])
            insertion_report = {"ready": not missing, "missing": missing,
                                "mode": insertion_mode,
                                "force_sensor_used": insertion_mode == "force_spiral"}
            if missing:
                problems.append("Force insertion missing field parameters: " + ", ".join(missing))
        except (ValueError, TypeError, KeyError) as error:
            insertion_report = {"ready": False, "mode": insertion_mode, "reason": str(error)}
            problems.append("Insertion configuration: " + str(error))
    for name in ("numpy", "scipy", "quadprog", "yaml", "wuji_sdk"):
        if importlib.util.find_spec(name) is None:
            problems.append(f"Missing control dependency: {name}")
    try:
        hand_targets(config)
        for name in ("arm_config", "hand_config"):
            if not Path(config[name]).is_file():
                problems.append(f"Missing {name}: {config[name]}")
    except (ValueError, OSError, KeyError) as error:
        problems.append(str(error))
    if move_above:
        try:
            from rail_receive.above import AboveTransfer
            AboveTransfer(config, ReceiveGeometry(config, kinematics()), kinematics(), HERE)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            problems.append(f"Right-arm above-goal configuration: {error}")
    if config["grasp_mode"] == "auto":
        for name in ("torch", "torchvision", "segment_anything"):
            if importlib.util.find_spec(name) is None:
                problems.append(f"Missing SAM dependency: {name}")
        for path in (HERE.parents[1] / "third_party/cnos/pretrained/segment-anything/sam_vit_h_4b8939.pth",
                     Path(config["mesh"]), Path(config["calibration"])):
            if not path.is_file():
                problems.append(f"Missing SAM/pose input: {path}")
        try:
            from rail_receive.sam_prompt import load_prompt
            load_prompt(config["sam_prompt"], config["receive_image_size"],
                        read_json(config["calibration"])["camera_serial"])
        except (OSError, ValueError, KeyError) as error:
            problems.append(f"SAM 框/点未就绪：{error}；先运行 python workspace/rail_pose_insert/set_sam_prompt.py")
    if config["grasp_mode"] == "auto" or move_above:
        try:
            from rail_preinsert.body_orientation_prior import BodyOrientationPrior
            BodyOrientationPrior.from_config(config,
                camera_serial=read_json(config["calibration"])["camera_serial"])
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            problems.append(f"Whole-robot orientation prior: {error}")
    if move_above:
        try:
            ReceiveGeometry(config, kinematics())
            root = Path(config["foundationpose_root"])
            for stamp in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
                for name in ("config.yml", "model_best.pth"):
                    if not (root / "weights" / stamp / name).is_file():
                        problems.append(f"Missing FoundationPose weight: {stamp}/{name}")
            if probe_gpu:
                probe = (
                    "import sys; from pathlib import Path; root=Path(sys.argv[1]); "
                    "sys.path[:0]=[str(root),str(root/'workspace/src')]; "
                    "import torch, nvdiffrast.torch as dr, cv2, pyrealsense2; import estimater; "
                    "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                    "assert callable(getattr(estimater.mycpp, 'cluster_poses', None)), 'mycpp extension missing'; "
                    "assert estimater.wp is not None, 'Warp depth filtering missing'; "
                    "dr.RasterizeCudaContext()"
                )
                command = python_command(config) + ["-c", probe, str(root)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
                if result.returncode:
                    detail = (result.stderr or result.stdout).strip()
                    problems.append("FoundationPose environment: " + detail[-1500:])
        except (ValueError, RuntimeError, OSError, KeyError, subprocess.TimeoutExpired) as error:
            problems.append(str(error))
    report = {"ready": not problems, "side": "right", "grasp_mode": config["grasp_mode"], "problems": problems,
            "grasp_profile": "recorded right-hand posture; grasp with current holding parameters not verified",
            "collision_checked": False}
    if insertion_report is not None:
        report["insertion"] = insertion_report
    return report


def keyboard(terminal, hardware, stage, commands, end):
    while not end.is_set():
        keys = terminal.read(.1)
        if keys == "":
            hardware.pause("Terminal closed; retain hand target")
            return
        if not keys:
            continue
        keys = keys.lower()
        if "x" in keys:
            end.set()
            hardware.pause("Operator exit")
            return
        if any(k in keys for k in ("q", " ", "\x03")):
            hardware.pause("Operator pause; retain hand target")
            continue
        # Never turn pasted text/key repeats into a motion queue.
        key = keys[0]
        if len(keys) != 1 or key not in set(JOG_KEYS) | {"+", "=", "-", "t", "g", "m", "n"}:
            continue
        # M/N always reach the main loop, including paused/early states, so
        # the operator receives a reason rather than a silently ignored key.
        if key not in ("m", "n") and hardware.paused:
            continue
        if key not in ("m", "n") and stage[0] not in ("waiting_rail", "holding", "above_target", "above_target_unverified", "placement_reached") and not (key == "g" and stage[0] == "opening"):
            continue
        if key == "g":
            try:
                commands.get_nowait()  # G takes priority over a queued jog.
            except Empty:
                pass
        try:
            commands.put_nowait(key)
        except Full:
            pass


def start_with_preview(hardware, preview, path, config, end, *, move_initial=True):
    """Keep OpenCV on the main thread while blocking SDK setup runs separately."""
    errors = []
    def initialize():
        try:
            hardware.start()
            if not end.is_set() and not hardware.paused:
                hardware.engage()
                if move_initial and not end.is_set() and not hardware.paused:
                    print("右臂移向固定接料初始位姿：原初始化位姿向机器人左侧调整 15 cm；"
                          "不累积偏移，到位后张手。", flush=True)
                    hardware.move_to_initial_pose()
        except Exception as error:
            errors.append(error)
    setup = threading.Thread(target=initialize, name="receive-setup")
    setup.start()
    try:
        while setup.is_alive():
            key = preview.update(path, receiving={"config": config, "timed_grasp": True,
                "pause_reason": hardware.error if hardware.paused else None,
                "stage": "paused" if hardware.paused else (
                    "initial_position" if getattr(hardware, "initial_move_active", False) is True else "starting")})
            if key in ("q", " ", "\x1b", "x"):
                if key == "x":
                    end.set()
                hardware.pause("Operator exit during setup" if key == "x" else "Operator pause during setup")
            elif key == "m":
                print("M 未执行：设备正在初始化，还没有保存的位姿均值。", flush=True)
            setup.join(timeout=.02)
    except BaseException:
        hardware.pause("Receiver setup interrupted")
        raise
    finally:
        # Never close devices concurrently with the initialization owner.
        setup.join()
    if errors:
        raise errors[0]


def paused_result(hardware, transfer):
    """Keep insertion evidence when an operator or controller stops the arm."""
    result = {"status": "paused", "reason": hardware.error, "side": "right",
              "hand_action": "retain_last_target", "insertion_executed": False}
    if transfer is not None and transfer.insertion_started is True:
        snapshot = transfer.poll_insertion(hardware) or {}
        result = dict(snapshot, status="paused", reason=hardware.error,
            insertion_status=snapshot.get("status", transfer.status), side="right",
            hand_action="retain_last_target", insertion_started=True,
            automatic_release=False, assembly_verified=False,
            force_controlled=True, collision_checked=False)
    elif transfer is not None and transfer.lowering is not None:
        if transfer.config["lowering"]["mode"] == "position_spiral":
            return transfer.lowering.paused_summary(reason=hardware.error)
        result.update(lowering_started=hasattr(transfer.lowering, "step_started"),
                      lowering_completed_steps=transfer.lowering.index,
                      contact_verified=False, assembly_verified=False)
        result.pop("insertion_executed")
    elif transfer is not None and getattr(transfer, "preplace", None) is not None:
        result["preplace_alignment"] = transfer.preplace.stopped(reason=hardware.error)
    return result


def run(config, args):
    grasp_deadline = pose_deadline = None
    from diffusion_policy.real_world.tianji.terminal import MotionTerminal
    # Motion needs control dependencies. A vision problem must not disable G.
    report = preflight(dict(config, grasp_mode="manual"), probe_gpu=False)
    if not report["ready"] and args.motion:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    references = hand_targets(config)
    log = Log(args.output)
    hardware, perception, watcher, geometry = None, None, None, None
    transfer, preview = None, None
    move_above = bool(args.motion and getattr(args, "move_above", True))
    already_held = bool(getattr(args, "already_held", False))
    if already_held and not move_above:
        log.close()
        raise ValueError("--already-held requires --move-above")
    end, commands, stage = threading.Event(), Queue(maxsize=1), ["starting"]
    old_sigint, trigger_record, grasp_source = None, None, None
    vision_error = None
    held_pose, held_pose_status, held_since_ns = None, "idle", 0
    held_mask = None
    pose_phase = None
    samples = deque(maxlen=200)
    write_json(log.directory / "config.json", config)

    def disable_auto(error):
        nonlocal perception, vision_error
        vision_error = str(error)
        detail = ("已保存的位姿均值仍可用于 M 搬运；到位复核需要视觉"
                  if transfer is not None and transfer.cached_grasp is not None
                  else "定时闭合、G 和键盘点动不受影响")
        if (transfer is not None and transfer.cached_grasp is not None
                and config["lowering"]["mode"] == "position_spiral"):
            detail = ("预放置精对准需要新的视觉结果；未对准时不能启动 N"
                      if transfer.requires_preplace_alignment else
                      "已保存的位姿均值仍可用于 M 搬运；机械到位后按 N 螺旋插入")
        print(f"视觉不可用：{error}；{detail}。", flush=True)
        log.emit("perception_disabled", reason=str(error))
        if perception is not None:
            try:
                perception.close()
            except Exception as cleanup_error:
                log.emit("perception_cleanup_error", reason=str(cleanup_error))
        perception = None

    def grasp(source, record=None):
        nonlocal trigger_record, grasp_source
        hardware.begin_hand(references["grasp"])
        stage[0] = "closing"
        trigger_record, grasp_source = record, source
        samples.clear()
        if record is not None:
            write_json(log.directory / "receive_observation.json", record)
        log.emit("grasp_trigger", source=source)
        print("正在按默认手型闭合右手。", flush=True)

    try:
        if args.motion:
            from rail_receive.visualization import PreviewWindow
            preview = PreviewWindow(enabled=not getattr(args, "no_view", False))
            preview.update(log.directory / "perception" / "preview.png", receiving={
                "config": config, "stage": "starting", "timed_grasp": True})
            if not preview.enabled and not getattr(args, "no_view", False):
                print("未找到图形桌面环境（DISPLAY/WAYLAND_DISPLAY），预览窗口未启用。", flush=True)
        if move_above:
            from rail_receive.above import AboveTransfer
            geometry = ReceiveGeometry(config, kinematics())
            transfer = AboveTransfer(config, geometry, kinematics(), log.directory)
            config["pose_preview"] = transfer.preview_poses
            write_json(log.directory / "config.json", config)
        if args.motion:
            try:
                camera_serial = read_json(config["calibration"])["camera_serial"]
                perception = ReceivePerception(config, log.directory / "perception", camera_serial)
                print(f"抓前只显示相机；抓握完成后等待 {config['post_grasp_pose_delay_s']:g} 秒，SAM 框提示分割后估姿。"
                      "窗口 E 可设置/修改框，Enter 保存。", flush=True)
            except Exception as error:
                disable_auto(error)
        hardware = RightHardware(config, kinematics(), log.emit, motion=args.motion)
        if not args.motion:
            hardware.start()
            state = hardware.snapshot()
            feedback = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in state.items()}
            write_json(log.directory / "read_only_state.json", feedback)
            print("右臂、右手只读反馈：" + json.dumps(feedback, ensure_ascii=False))
            return
        with MotionTerminal() as terminal:
            print(HELP, flush=True)
            if move_above:
                start_text = ("估姿完成后保存均值并等待 M，按 M 规划并移到目标上方 "
                              if config["move_above"].get("start_mode", "manual") == "manual"
                              else "估姿通过后自动规划并移到目标上方 ")
                arrival_text = (("到位后重新运行 FoundationPose，自动纠偏位置和角度；完成后按 N 螺旋下插。"
                                 if config["preplace_alignment"]["enabled"] else
                                 "机械到位后保持抓握，按 N 螺旋下插。")
                    if config["lowering"]["mode"] == "position_spiral" else "到位复核后保持抓握。")
                print(f"{start_text}{config['move_above']['height_m']*1000:.0f} mm；"
                      f"{arrival_text}估姿、规划和移动期间禁止点动及编辑 bbox，Q/空格可暂停。", flush=True)
                print("搬运方式：SDK 位置控制，直接下发一个终点，不生成中间轨迹点；"
                      f"速度/加速度比例上限 {config['velocity_ratio']}% / {config['acceleration_ratio']}%，"
                      "按控制器反馈及实测末端到位判断。", flush=True)
                if config["lowering"]["mode"] == "force_spiral":
                    print("到上方后按 N：六维力基线 → 轻触 → X–Z 平面螺旋搜索 → 受力约束沿 −Y 下插；"
                          "完成后保持抓握等待人工确认，卡滞回退，暂停后不能重启。", flush=True)
                    if config["insertion_missing"]:
                        print("力控插入尚未就绪，N 会拒绝启动；缺少现场参数："
                              + ", ".join(config["insertion_missing"]) + "。接料和 M 搬运仍可使用。", flush=True)
                elif config["lowering"]["mode"] == "position_spiral":
                    spiral = config["position_spiral"]
                    print(f"到上方后按 N：位置控制螺旋下降，最大半径 {spiral['radius_m']*1000:g} mm，"
                          f"{spiral['turns']:g} 圈，单段不超过 {spiral['max_step_m']*1000:g} mm；"
                          "不采集六维力，到目标后保持抓握，等待人工确认。", flush=True)
                else:
                    print(f"到上方后按 N 下放：每段 {config['lowering']['step_m']*1000:g} mm，"
                          f"速度/加速度比例 {config['lowering']['velocity_ratio']}% / "
                          f"{config['lowering']['acceleration_ratio']}%，逐段实测到位后继续；到装配目标后保持抓握。", flush=True)
            else:
                print("仅估姿模式：抓后估姿一次并保持；T 或重新保存 bbox 可重估，不自动搬运。", flush=True)
            old_sigint = signal.signal(signal.SIGINT, lambda *_: hardware.pause("Operator Ctrl+C; retain hand target"))
            watcher = threading.Thread(target=keyboard, args=(terminal, hardware, stage, commands, end), daemon=True)
            watcher.start()
            try:
                start_with_preview(hardware, preview, log.directory / "perception" / "preview.png", config, end,
                                   move_initial=not already_held)
                if end.is_set() or hardware.paused:
                    pass
                elif already_held:
                    transfer.begin()
                    stage[0] = transfer.status
                    print("从当前实测右手手型接管，重新估计已抓持导轨位姿。", flush=True)
                else:
                    hardware.begin_hand(references["open"])
                    stage[0] = "opening"
                    print(f"正在张开右手；准备好后等待 {config['grasp_delay_s']:g} 秒再闭合，G 可提前闭合。", flush=True)
            except Exception as error:
                hardware.pause(str(error))
            last_jog_ns, paused_reported = 0, False
            while not end.is_set():
                try:
                    prompt_saved, preview_action = False, None
                    if preview is not None:
                        transferring = transfer is not None and transfer.status != "idle"
                        view_stage = stage[0]
                        if view_stage == "holding":
                            view_stage = {"waiting": "waiting_pose", "pending": "estimating_pose", "ready": "pose_done",
                                          "failed": "pose_error"}.get(held_pose_status, view_stage)
                        receiving = {"config": config, "stage": "paused" if hardware.paused else view_stage,
                                     "pause_reason": hardware.error if hardware.paused else None,
                                     "grasp_source": grasp_source, "vision_error": vision_error,
                                     "timed_grasp": True, "live_pose_preview": transferring,
                                     "prompt_editable": not transferring and not hardware.paused,
                                     "grasp_remaining_s": (max(0., grasp_deadline-time.monotonic())
                                                           if grasp_deadline is not None else config["grasp_delay_s"]),
                                     "pose_remaining_s": (max(0., pose_deadline-time.monotonic())
                                                          if pose_deadline is not None else 0.),
                                     "pose_record": held_pose, "mask_record": held_mask,
                                     "saved_pose_age_s": ((time.monotonic_ns()-transfer.pose_sampling["mean_saved_monotonic_ns"])/1e9
                                         if transfer is not None and transfer.pose_sampling else None),
                                     "pose_phase": pose_phase}
                        if transfer is not None:
                            receiving["visual_verification"] = transfer.visual_verification
                            if transfer.preplace is not None:
                                receiving["preplace_alignment"] = transfer.preplace.summary()
                        key = preview.update(log.directory / "perception" / "preview.png", receiving=receiving)
                        prompt_saved = key == "prompt_saved"
                        preview_action = key if key in ("m", "n") else None
                        if key in ("q", " ", "\x1b"):
                            hardware.pause("Operator preview pause; retain hand target")
                        elif key == "x":
                            end.set()
                            hardware.pause("Operator preview exit")
                            continue
                    try:
                        key = commands.get_nowait()
                    except Empty:
                        key = None
                    if preview_action:
                        key = preview_action
                    if hardware.paused:
                        if end.is_set():
                            break
                        if not paused_reported:
                            stage[0] = "paused"
                            print(f"已暂停：{hardware.error}。手保持最后目标；按 X 退出。", flush=True)
                            write_json(log.directory / "result.json", paused_result(hardware, transfer))
                            paused_reported = True
                        if key in ("m", "n"):
                            message = f"{key.upper()} 已收到，但机械臂因以下原因暂停，不能启动：{hardware.error}"
                            print(message, flush=True)
                            log.emit("manual_move_rejected" if key == "m" else "manual_lower_rejected",
                                     stage="paused", reason=hardware.error)
                        end.wait(.05)
                        continue
                    hardware.raise_if_fault()
                    state, now = hardware.snapshot(), time.monotonic()
                    samples.append((now, state["hand_position_rad"]))
                    force_move = bool(key == "m" and transfer is not None
                                      and transfer.status in ("estimating_held_pose", "waiting_move_key", "waiting_pose_override"))
                    if key == "n" and (transfer is None or transfer.status == "idle"):
                        print("N 未执行：尚未完成目标上方搬运。", flush=True)
                    if key == "m" and not force_move:
                        message = {"planning_above": "正在规划，请等待规划结果", "moving_above": "正在执行移动",
                            "verifying_above": "已到达机械臂目标，正在视觉复核", "above_target": "已经到达目标上方",
                            "above_target_unverified": "已到达机械臂目标，但视觉复核未通过"}.get(stage[0],
                            "还没有保存的位姿均值，请等待抓后估姿完成" if move_above else "当前为仅估姿模式，没有启用搬运")
                        print(f"M 已收到，未重复启动：{message}。", flush=True)
                        log.emit("manual_move_rejected", stage=stage[0], reason=message)
                    if prompt_saved:
                        if stage[0] == "holding":
                            # Use the same fresh-mask initialization path as T.
                            # The waiting branch below still preserves the 8 s delay.
                            key = "t"
                        else:
                            print("新 bbox 已保存，将按抓握后的既定时序进行分割和估姿。", flush=True)
                    if key == "g" and stage[0] in ("opening", "waiting_rail"):
                        grasp("keyboard")
                    elif key in JOG_KEYS or key in ("+", "=", "-"):
                        if stage[0] in ("waiting_rail", "holding", "above_target", "above_target_unverified", "placement_reached"):
                            target = hardware.jog(key)
                            if target is not None:
                                last_jog_ns = time.monotonic_ns()
                            print(f"点动步长：{config['jog_step_m']*hardware.jogger.scale*1000:.2f} mm / "
                                  f"{np.degrees(config['jog_step_rad'])*hardware.jogger.scale:.2f}°", flush=True)
                    elif key == "t" and held_pose_status == "waiting":
                        print(f"抓后等待中，剩余 {max(0., pose_deadline-now):.1f} 秒后自动识别。", flush=True)
                    elif key == "t" and stage[0] == "holding" and perception is not None:
                        try:
                            perception.initialize()
                            if not move_above:
                                held_pose, held_pose_status = None, "pending"
                                held_mask = None
                                pose_phase = None
                                held_since_ns, vision_error = time.monotonic_ns(), None
                                print("已请求重新检测并估计抓持位姿。", flush=True)
                        except ValueError as error:
                            print(str(error), flush=True)
                        except Exception as error:
                            disable_auto(error)
                    if (stage[0] == "waiting_rail" and config["grasp_mode"] == "auto"
                            and grasp_deadline is not None and now >= grasp_deadline):
                        grasp("timer")
                    if stage[0] in ("opening", "closing") and hardware.hand_transition_finished(
                            samples, now, require_target=stage[0] == "opening"):
                        if stage[0] == "opening":
                            stage[0] = "waiting_rail"
                            grasp_deadline = time.monotonic() + config["grasp_delay_s"]
                            print(f"接料准备完成，开始 {config['grasp_delay_s']:g} 秒抓握倒计时；可点动或按 G 提前闭合。", flush=True)
                        else:
                            stage[0] = "holding"
                            result = {"status": "holding", "side": "right", "grasp_source": grasp_source,
                                      "grasp_verified": False, "insertion_executed": False, "collision_checked": False,
                                      "hand_position_rad": state["hand_position_rad"].tolist(),
                                      "hand_target_rad": hardware.hand_target.tolist(),
                                      "T_B_F": state["T_B_F"].tolist(), "trigger_observation": trigger_record}
                            write_json(log.directory / "result.json", result)
                            write_json(log.directory / "held_state.json", result)
                            held_since_ns = time.monotonic_ns()
                            pose_deadline = time.monotonic() + config["post_grasp_pose_delay_s"]
                            held_pose_status = "waiting" if perception is not None else "failed"
                            print(f"抓握完成并保持，等待 {config['post_grasp_pose_delay_s']:g} 秒后开始 SAM 分割与估姿。", flush=True)
                        log.emit("stage", stage=stage[0])
                        samples.clear()
                    if held_pose_status == "waiting" and time.monotonic() >= pose_deadline:
                        held_pose_status = "pending"
                        if move_above:
                            transfer.begin()
                            stage[0] = transfer.status
                        print("抓后等待结束，开始 SAM 框提示分割并估计位姿。", flush=True)
                    transferring = transfer is not None and transfer.status != "idle"
                    if transferring and perception is None and transfer.cached_grasp is None:
                        raise RuntimeError("抓持后视觉不可用；保持抓握，未继续向目标移动")
                    if (transferring and perception is None and transfer.preplace is not None
                            and not transfer.preplace.converged):
                        raise RuntimeError("预放置重新估姿不可用；保持抓握，未启动下放")
                    estimating_once = not move_above and held_pose_status == "pending"
                    if stage[0] == "closing" and perception is not None:
                        try:
                            perception.publish_context(state, np.eye(4), enabled=False, epoch=0)
                        except Exception as error:
                            disable_auto(error)
                    record = None
                    if perception is not None and (transferring or estimating_once):
                        try:
                            palm = (np.linalg.inv(geometry.T_B_C) @ state["T_B_F"] @ geometry.T_F_G
                                    if transferring else np.eye(4))
                            perception.publish_context(state, palm, enabled=True,
                                epoch=last_jog_ns if transferring else 0,
                                mode="held" if transferring else "held_once")
                            if transferring and transfer.needs_reinitialize and perception.ready():
                                predicted = transfer.reestimate_predicted_pose
                                generation = (perception.initialize() if predicted is None
                                              else perception.initialize(predicted_pose=predicted))
                                transfer.perception_initialized(generation)
                                pose_phase = None
                            record = perception.latest()
                            if transferring and record and record.get("context_epoch") != last_jog_ns:
                                record = None
                            reason = record.get("reason") if record else None
                            if reason in POSE_PHASES and reason != pose_phase:
                                pose_phase = reason
                                print(f"抓后识别：{POSE_PHASES[reason]}。", flush=True)
                            if estimating_once:
                                if record and record.get("mask_snapshot"):
                                    held_mask = record
                                if record and record.get("reason") == "held_pose_unavailable":
                                    failed_phase = POSE_PHASES.get(record.get("phase"), "抓持后估姿")
                                    held_pose_status, vision_error = "failed", f"{failed_phase}：{record.get('detail', '失败')}"
                                    print(f"抓持后估姿不可用：{vision_error}；窗口与右臂点动继续，T 可重试。", flush=True)
                                elif (record and record.get("valid") and record.get("T_C_CAD") is not None
                                      and not record.get("replay_only") and record.get("detector") != "cnos"
                                      and held_since_ns < record.get("timestamp_ns", 0) <= time.monotonic_ns()):
                                    # This is a labelled snapshot, not a live servo observation.
                                    held_pose, held_pose_status = record, "ready"
                                    write_json(log.directory / "held_pose.json", record)
                                    log.emit("held_pose", observation=record)
                                    print("抓持后位姿已更新；窗口持续显示，停止推理。T 可重新估姿。", flush=True)
                        except Exception as error:
                            if transferring and transfer.cached_grasp is None:
                                raise
                            disable_auto(error)
                            if estimating_once:
                                held_pose_status = "failed"
                    # Coarse transport uses the initial frozen grasp; preplacement
                    # refinement requires a new registration and updates that grasp.
                    if transferring:
                        previous_stage = stage[0]
                        stage[0] = transfer.advance(hardware, state, record, force_move=force_move,
                                                    lower_requested=key == "n")
                        if key == "n" and transfer.manual_lower_message:
                            print(transfer.manual_lower_message, flush=True)
                            log.emit("manual_lower_request", message=transfer.manual_lower_message,
                                     stage=stage[0])
                        if force_move and transfer.manual_move_message:
                            print(transfer.manual_move_message, flush=True)
                            log.emit("manual_pose_move", message=transfer.manual_move_message,
                                     pose_sampling=transfer.pose_sampling)
                        if stage[0] != previous_stage:
                            message = {"planning_above": "正在使用已保存的位姿均值规划右臂路径",
                                "waiting_move_key": "位姿均值已保存，等待 M；按 M 直接使用该均值规划移动（终端/窗口均可）",
                                "waiting_pose_override": "位姿均值已保存，但稳定性未达标；按 M 直接使用该均值规划移动",
                                "moving_above": "右臂以位置控制直接移向装配目标上方",
                                "reestimating_preplace": "预放置已到位，重新 SAM 分割与 FoundationPose 注册，计算新的位姿偏差",
                                "planning_preplace": "正在根据新位姿规划位置和三轴角度纠偏，检查各段 IK 与导轨几何高度",
                                "aligning_preplace": "右臂低速分段纠偏，必要时先抬升再调整；完成后重新估姿",
                                "verifying_above": "右臂到位，正在用 FoundationPose 验证导轨位置",
                                "above_target": "导轨已到目标上方，保持抓握；按 N 启动配置的插入流程",
                                "above_target_unverified": "右臂已到位，导轨视觉误差超出容差；保持抓握，N 仅重新复核，通过后再按 N 插入",
                                "planning_lowering": "正在检查全部下放目标的 IK 与限位",
                                "lowering": "右臂开始缓慢下放，Q/空格可暂停；右手保持抓握",
                                "planning_position_spiral": "正在检查全部螺旋目标的 IK 与限位",
                                "position_spiral": "右臂以位置控制螺旋下降，不使用六维力；Q/空格可暂停",
                                "placement_reached": "已到记录的装配目标，保持抓握；未自动松手或确认接触装配",
                                "switching_to_insertion": "正在从实测位姿切换到插入阻抗控制",
                                "baseline": "保持悬空静止，采集六维力基线",
                                "approaching": "沿 −Y 缓慢接近，等待轻触",
                                "searching": "维持接触压力，在 X–Z 平面有界螺旋找孔",
                                "capture": "检测到入孔候选，冻结横向搜索并检查受力与进度",
                                "inserting": "在受力约束下缓慢下插，保持抓握",
                                "awaiting_confirmation": "已到插入终点并保持；请人工确认双孔落座，尚未确认装配成功",
                                "retracting": "插入未完成，沿 +Y 卸载回退，保持抓握",
                                "retracted": "已退回起始净空高度，保持抓握；本次不自动重试",
                                "fault": "插入故障已停止，保持抓握；请检查停止原因"}
                            if config["lowering"]["mode"] == "position_spiral":
                                message["above_target"] = ("预放置对准完成，抓持关系已更新；按 N 螺旋下插，保持抓握"
                                    if transfer.requires_preplace_alignment else
                                    "右臂已到目标上方；按 N 螺旋下插，保持抓握")
                            print(message.get(stage[0], stage[0]), flush=True)
                            log.emit("stage", stage=stage[0])
                except Exception as error:
                    hardware.pause(str(error))
                end.wait(.02)
    finally:
        exit_requested = end.is_set()
        end.set()
        if watcher is not None:
            watcher.join(timeout=1)
        errors = []
        if transfer is not None and transfer.insertion_started is True and hardware is not None:
            try:
                transfer.poll_insertion(hardware)
            except Exception as error:
                errors.append(f"Insertion log finalization: {error}")
        if (transfer is not None and transfer.lowering is not None and hardware is not None
                and config["lowering"]["mode"] == "position_spiral"
                and transfer.lowering.status != "awaiting_confirmation"):
            try:
                transfer.lowering.paused_summary(reason=hardware.error or "Receiver exit")
            except Exception as error:
                errors.append(f"Position spiral log finalization: {error}")
        if (transfer is not None and transfer.preplace is not None and not transfer.preplace.converged
                and transfer.preplace.status != "paused"):
            try:
                transfer.preplace.stopped(reason=(hardware.error if hardware is not None else None) or "Receiver exit")
            except Exception as error:
                errors.append(f"Preplacement alignment log finalization: {error}")
        for resource in (preview, perception, hardware):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    errors.append(str(error))
        log.close()
        if args.motion:
            write_json(log.directory / "exit_state.json", {"last_stage": stage[0], "exit_key_received": exit_requested,
                "hand_action": "disabled_on_exit" if not errors else "inspect_cleanup_errors",
                "cleanup_errors": errors})
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)
        if errors:
            raise RuntimeError("; ".join(errors))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="不带参数直接启动右手接料、估姿及目标上方搬运；只读请运行 read_state.py。")
    parser.add_argument("--config", type=Path, default=HERE / "receive.yaml")
    parser.add_argument("--grasp-mode", choices=("auto", "manual"))
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="检查右手接料配置与依赖，不连接设备")
    check.add_argument("--move-above", action="store_true", help="同时检查右臂目标上方配置")
    check.add_argument("--insertion", action="store_true", help="同时检查所选插入模式的参数，不连接设备")
    live = sub.add_parser("run", help="默认只读；--motion 启用接料、估姿及目标上方搬运")
    live.add_argument("--motion", action="store_true")
    mode = live.add_mutually_exclusive_group()
    mode.add_argument("--move-above", action="store_true", default=None,
                      help="抓持后保存位姿均值，默认等待 M 后移动至目标上方（启用动作时默认开启）")
    mode.add_argument("--pose-only", dest="move_above", action="store_false",
                      help="抓后只估姿并保持，不自动搬运")
    live.add_argument("--already-held", action="store_true", help="从已抓持状态估姿并搬运，不张手；不能与 --pose-only 同用")
    live.add_argument("--above-height", type=float, help="目标上方高度，单位 m，默认 0.015（1.5 cm）")
    live.add_argument("--no-view", action="store_true", help="仅保存位姿叠加图，不打开窗口")
    live.add_argument("--output", type=Path, help="可选：保留运行记录；默认使用临时目录，退出后自动清理")
    for command in (check, live):
        command.add_argument("--insertion-mode", choices=("force_spiral", "position_spiral", "position_steps"),
                             help="覆盖 lowering.mode；position_spiral 使用位置控制，不采集六维力")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv or ["run", "--motion"])
    if args.command == "run":
        if args.move_above is None:
            args.move_above = args.motion
        if args.move_above and not args.motion:
            parser.error("--move-above requires --motion")
    if getattr(args, "already_held", False) and not args.move_above:
        parser.error("--already-held requires --move-above")
    if getattr(args, "above_height", None) is not None and not args.move_above:
        parser.error("--above-height requires --move-above")
    try:
        config = load_config(args.config, insertion_mode=args.insertion_mode)
        if args.grasp_mode:
            config["grasp_mode"] = args.grasp_mode
        if getattr(args, "above_height", None) is not None:
            config["move_above"]["height_m"] = args.above_height
        if args.command == "check":
            report = preflight(config, move_above=args.move_above, insertion=args.insertion)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready"] else 1
        if args.output is None:
            with tempfile.TemporaryDirectory(prefix="rail-receive-") as directory:
                args.output = Path(directory) / "run"
                run(config, args)
        else:
            run(config, args)
        return 0
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        print(f"右手接料停止：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
