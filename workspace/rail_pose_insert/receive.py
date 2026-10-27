"""Right-arm rail insertion orchestration; launched by insert_spiral.py."""

from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
from pathlib import Path
from queue import Empty, Full, Queue
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np

from rail_common.core import HERE, read_json, write_json
from rail_common.perception import python_command
from rail_receive.logic import JOG_KEYS, ReceiveGeometry, hand_targets, load_config
from rail_receive.hardware import RightHardware
from rail_receive.perception import ReceivePerception
from rail_common.runtime import Log, kinematics

HELP = """右臂点动（右臂原生基座）：W/S 前/后，R/F 上/下，D/A 向外/向内；
I/K 绕 X 正/负转，J/L 绕 Y 正/负转，U/O 绕 Z 正/负转；+/- 调整步长。
右手张开并准备好后等 5 秒开始闭合；抓握完成后再等 8 秒，按保存的 bbox 运行 SAM 和 FoundationPose。
估姿启动前可在窗口按 E 画 bbox，Enter 保存；G 可提前闭合。
抓后估姿完成会保存均值并等待；按 M 直接使用已保存均值规划移动（终端/窗口均可）。
单轮纠偏到位后，按 N 启动位置螺旋下插；到位后自动松手、按配置抬升并回初始位置。
--draw-bboxes：初始抓持位置、导轨台上方各等一次；窗口 E 拖框、Enter 保存继续，或直接 Enter 复用。
Q / 空格 / Ctrl+C 暂停机械臂和抓握推进，保持最后手目标；本次不自动恢复。
X 退出并停用右手：退出前先由人或夹具支撑导轨。"""

POSE_PHASES = {
    "opening_rgbd": "打开 RGB-D 相机（首帧最多等待 10 秒）",
    "acquiring_rgbd": "采集 RGB-D 帧",
    "loading_sam": "加载本地 SAM 框/点提示分割器",
    "segmenting_held_rail": "SAM 按保存的框/点分割单个导轨",
    "initializing_pose": "已取得新 mask，正在初始化 FoundationPose",
    "aligning_pose": "按目标方向和每轴小于 45° 的约束对齐位姿",
}


def preflight(config, *, probe_gpu=True, control_only=False):
    problems = []
    insertion_report = None
    if not control_only:
        try:
            from rail_receive.position_spiral import validate_settings

            validate_settings(config["position_spiral"])
            if config["move_above"]["height_m"] > config["position_spiral"]["max_descent_m"]:
                raise ValueError("Above-goal height exceeds position_spiral.max_descent_m")
            insertion_report = {
                "ready": True,
                "mode": "position_spiral",
                "force_sensor_used": False,
            }
        except (ValueError, TypeError, KeyError) as error:
            insertion_report = {"ready": False, "mode": "position_spiral", "reason": str(error)}
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
    if not control_only:
        try:
            from rail_receive.above import AboveTransfer

            AboveTransfer(config, ReceiveGeometry(config, kinematics()), kinematics(), HERE)
        except (ValueError, RuntimeError, OSError, KeyError) as error:
            problems.append(f"Right-arm above-goal configuration: {error}")
    if not control_only:
        for name in ("torch", "torchvision", "segment_anything"):
            if importlib.util.find_spec(name) is None:
                problems.append(f"Missing SAM dependency: {name}")
        for path in (
            HERE.parents[1] / "third_party/cnos/pretrained/segment-anything/sam_vit_h_4b8939.pth",
            Path(config["mesh"]),
            Path(config["calibration"]),
        ):
            if not path.is_file():
                problems.append(f"Missing SAM/pose input: {path}")
        try:
            from rail_receive.sam_prompt import load_prompt, prompt_path, PROMPT_NAMES

            load_prompt(
                config["sam_prompt"],
                config["receive_image_size"],
                read_json(config["calibration"])["camera_serial"],
            )
        except (OSError, ValueError, KeyError) as error:
            problems.append(
                f"SAM 框/点未就绪：{error}；先运行 python workspace/rail_pose_insert/set_sam_prompt.py"
            )
        if not control_only:
            try:
                load_prompt(
                    prompt_path(config, "preplace"),
                    config["receive_image_size"],
                    read_json(config["calibration"])["camera_serial"],
                )
            except (OSError, ValueError, KeyError) as error:
                problems.append(
                    f"{PROMPT_NAMES['preplace']}未就绪：{error}；先运行 set_sam_prompt.py 设置两个框"
                )
    if not control_only:
        try:
            from rail_common.body_orientation_prior import BodyOrientationPrior

            BodyOrientationPrior.from_config(
                config, camera_serial=read_json(config["calibration"])["camera_serial"]
            )
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            problems.append(f"Whole-robot orientation prior: {error}")
    if not control_only:
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
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=45, check=False
                )
                if result.returncode:
                    detail = (result.stderr or result.stdout).strip()
                    problems.append("FoundationPose environment: " + detail[-1500:])
        except (ValueError, RuntimeError, OSError, KeyError, subprocess.TimeoutExpired) as error:
            problems.append(str(error))
    report = {
        "ready": not problems,
        "side": "right",
        "problems": problems,
        "grasp_profile": "recorded right-hand posture; grasp with current holding parameters not verified",
        "collision_checked": False,
    }
    if insertion_report is not None:
        report["insertion"] = insertion_report
    return report


def keyboard(terminal, hardware, stage, commands, end):
    while not end.is_set():
        keys = terminal.read(0.1)
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
        if len(keys) != 1 or key not in set(JOG_KEYS) | {"+", "=", "-", "g", "m", "n"}:
            continue
        # M/N always reach the main loop, including paused/early states, so
        # the operator receives a reason rather than a silently ignored key.
        if key not in ("m", "n") and hardware.paused:
            continue
        if (
            key not in ("m", "n")
            and stage[0] not in ("waiting_rail", "holding", "above_target")
            and not (key == "g" and stage[0] == "opening")
        ):
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
            if (
                move_initial
                and config.get("left_initial_pose")
                and not end.is_set()
                and not hardware.paused
            ):
                hardware.move_left_to_initial_pose()
            if not end.is_set() and not hardware.paused:
                if move_initial:
                    print(
                        "右臂移向固定接料初始位姿：原初始化位姿向机器人左侧调整 15 cm；"
                        "不累积偏移，位置回位停稳后切换为阻抗控制，再张手。",
                        flush=True,
                    )
                    hardware.move_to_initial_pose()
                if not end.is_set() and not hardware.paused:
                    hardware.engage()
        except Exception as error:
            errors.append(error)

    setup = threading.Thread(target=initialize, name="receive-setup")
    setup.start()
    try:
        while setup.is_alive():
            key = preview.update(
                path,
                receiving={
                    "config": config,
                    "pause_reason": hardware.error if hardware.paused else None,
                    "stage": (
                        "paused"
                        if hardware.paused
                        else (
                            "left_initial_position"
                            if getattr(hardware, "left_initial_move_active", False) is True
                            else (
                                "initial_position"
                                if getattr(hardware, "initial_move_active", False) is True
                                else "starting"
                            )
                        )
                    ),
                },
            )
            if key in ("q", " ", "\x1b", "x"):
                if key == "x":
                    end.set()
                hardware.pause(
                    "Operator exit during setup" if key == "x" else "Operator pause during setup"
                )
            elif key == "m":
                print("M 未执行：设备正在初始化，还没有保存的位姿均值。", flush=True)
            setup.join(timeout=0.02)
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
    result = {
        "status": "paused",
        "reason": hardware.error,
        "side": "right",
        "hand_action": "retain_last_target",
        "insertion_executed": False,
    }
    if transfer is not None and transfer.release_return is not None:
        return transfer.release_return.paused_summary(reason=hardware.error)
    if transfer is not None and transfer.spiral is not None:
        return transfer.spiral.paused_summary(reason=hardware.error)
    elif transfer is not None and getattr(transfer, "preplace", None) is not None:
        result["preplace_alignment"] = transfer.preplace.stopped(reason=hardware.error)
    return result


PROMPT_ACTIONS = {
    "prompt_saved": ("initial", False),
    "preplace_prompt_saved": ("preplace", False),
    "prompt_reuse_initial": ("initial", True),
    "prompt_reuse_preplace": ("preplace", True),
}


def accept_runtime_prompt(action, transfer, hardware, state):
    """Route a saved/reused slot without treating it as a movement key."""
    if action not in PROMPT_ACTIONS or transfer is None or transfer.waiting_prompt_stage is None:
        return False
    stage, reused = PROMPT_ACTIONS[action]
    if stage != transfer.waiting_prompt_stage:
        print("已保存另一阶段的 bbox；当前阶段仍等待绘框或复用确认。", flush=True)
        return False
    try:
        transfer.accept_prompt(stage, hardware, state, reused=reused)
    except (ValueError, OSError, KeyError) as error:
        print(f"bbox 未采用：{error}；请在窗口按 E 重新绘制本阶段框。", flush=True)
        return False
    print(
        f"{'初始' if stage == 'initial' else '导轨台上方'} bbox 已{'复用' if reused else '保存'}，开始新的分割与估姿。",
        flush=True,
    )
    return True


def _run(config, args, runtime):
    grasp_deadline = pose_deadline = None
    draw_bboxes = bool(getattr(args, "draw_bboxes", False))
    if draw_bboxes and getattr(args, "no_view", False):
        raise ValueError("--draw-bboxes requires preplace_alignment and a preview window")
    from diffusion_policy.real_world.tianji.terminal import MotionTerminal

    report = preflight(config, probe_gpu=False, control_only=True)
    if not report["ready"]:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    references = hand_targets(config)
    log = Log(args.output)
    hardware, perception, watcher, geometry = (None, None, None, None)
    transfer, preview = (None, None)
    already_held = bool(getattr(args, "already_held", False))
    end, commands, stage = (threading.Event(), Queue(maxsize=1), ["starting"])
    old_sigint, grasp_source = (None, None)
    vision_error = None
    held_pose_status = "idle"
    pose_phase = None
    samples = deque(maxlen=200)
    last_state = None

    def disable_auto(error):
        nonlocal perception, vision_error
        vision_error = str(error)
        detail = (
            "已保存的位姿均值仍可用于 M 搬运；到位后的重新估姿需要视觉"
            if transfer is not None and transfer.cached_grasp is not None
            else "定时闭合、G 和键盘点动不受影响"
        )
        if transfer is not None and transfer.preplace is not None and transfer.preplace.completed:
            detail = "单轮纠偏已经完成，保持位置；人工确认后按 N 下插"
        if transfer is not None and transfer.release_return is not None:
            detail = "已结束插入估姿；松手、抬升和回位使用实测机械臂与手指反馈"
        print(f"视觉不可用：{error}；{detail}。", flush=True)
        log.emit("perception_disabled", reason=str(error))
        if perception is not None:
            try:
                perception.close()
            except Exception as cleanup_error:
                log.emit("perception_cleanup_error", reason=str(cleanup_error))
        perception = None

    def grasp(source):
        nonlocal grasp_source
        hardware.begin_hand(references["grasp"])
        stage[0] = "closing"
        grasp_source = source
        samples.clear()
        log.emit("grasp_trigger", source=source)
        print("正在按默认手型闭合右手。", flush=True)

    try:
        from rail_receive.visualization import PreviewWindow

        preview = PreviewWindow(enabled=not getattr(args, "no_view", False))
        preview.update(
            runtime / "perception" / "preview.png",
            receiving={"config": config, "stage": "starting"},
        )
        if not preview.enabled and (not getattr(args, "no_view", False)):
            print("未找到图形桌面环境（DISPLAY/WAYLAND_DISPLAY），预览窗口未启用。", flush=True)
        from rail_receive.above import AboveTransfer

        geometry = ReceiveGeometry(config, kinematics())
        transfer = AboveTransfer(config, geometry, kinematics(), log.directory)
        transfer.draw_bboxes = draw_bboxes
        if transfer.prompt_required("initial") or transfer.prompt_required("preplace"):
            if preview is None or not preview.enabled:
                raise ValueError(
                    "绘制 bbox 需要图形预览窗口；请启用窗口后运行 --draw-bboxes，或先准备两份有效的 bbox"
                )
        config["pose_preview"] = transfer.preview_poses
        try:
            camera_serial = read_json(config["calibration"])["camera_serial"]
            perception = ReceivePerception(config, runtime / "perception", camera_serial)
            print(
                f"抓前只显示相机；抓握完成后等待 {config['post_grasp_pose_delay_s']:g} 秒，SAM 框提示分割后估姿。窗口 E 可设置/修改框，Enter 保存。",
                flush=True,
            )
        except Exception as error:
            disable_auto(error)
        hardware = RightHardware(config, kinematics(), log.emit, motion=True)
        with MotionTerminal() as terminal:
            print(HELP, flush=True)
            print(
                f"初始估姿完成后等待 M 搬到目标上方 {config['move_above']['height_m']*1000:g} mm；"
                "上方重新估姿并纠偏一轮后等待新的 N。",
                flush=True,
            )
            if draw_bboxes:
                print(
                    "本次分两阶段绘框：初始抓持处先等 E/Enter；M 到台上方停稳后再等 E/Enter。两次分别保存，之后不加 --draw-bboxes 即自动复用。",
                    flush=True,
                )
            print(
                f"搬运方式：SDK 位置控制，直接下发一个终点，不生成中间轨迹点；速度/加速度比例上限 {config['velocity_ratio']}% / {config['acceleration_ratio']}%，按控制器反馈及实测末端到位判断。",
                flush=True,
            )
            spiral = config["position_spiral"]
            print(
                f"到上方后按 N：位置控制螺旋下降，最大半径 {spiral['radius_m'] * 1000:g} mm，{spiral['turns']:g} 圈，单段不超过 {spiral['max_step_m'] * 1000:g} mm；到位后自动松手、抬高 {config['release_return']['lift_height_m'] * 100:g} cm 并回初始位置。",
                flush=True,
            )
            old_sigint = signal.signal(
                signal.SIGINT, lambda *_: hardware.pause("Operator Ctrl+C; retain hand target")
            )
            watcher = threading.Thread(
                target=keyboard, args=(terminal, hardware, stage, commands, end), daemon=True
            )
            watcher.start()
            try:
                start_with_preview(
                    hardware,
                    preview,
                    runtime / "perception" / "preview.png",
                    config,
                    end,
                    move_initial=not already_held,
                )
                if end.is_set() or hardware.paused:
                    pass
                elif already_held:
                    transfer.begin()
                    stage[0] = transfer.status
                    print("从当前实测右手手型接管，重新估计已抓持导轨位姿。", flush=True)
                else:
                    hardware.begin_hand(references["open"])
                    stage[0] = "opening"
                    print(
                        f"正在张开右手；准备好后等待 {config['grasp_delay_s']:g} 秒再闭合，G 可提前闭合。",
                        flush=True,
                    )
            except Exception as error:
                hardware.pause(str(error))
            last_jog_ns, paused_reported = (0, False)
            while not end.is_set():
                try:
                    prompt_saved, prompt_action, preview_action = (False, None, None)
                    if preview is not None:
                        transferring = transfer is not None and transfer.status != "idle"
                        waiting_prompt = (
                            transfer.waiting_prompt_stage if transfer is not None else None
                        )
                        view_stage = stage[0]
                        if view_stage == "holding":
                            view_stage = {
                                "waiting": "waiting_pose",
                                "pending": "estimating_pose",
                                "ready": "pose_done",
                                "failed": "pose_error",
                            }.get(held_pose_status, view_stage)
                        receiving = {
                            "config": config,
                            "stage": "paused" if hardware.paused else view_stage,
                            "pause_reason": hardware.error if hardware.paused else None,
                            "grasp_source": grasp_source,
                            "vision_error": vision_error,
                            "live_pose_preview": transferring and (not waiting_prompt),
                            "sam_prompt_stage": waiting_prompt
                            or (
                                "preplace"
                                if transfer is not None and transfer.preplace is not None
                                else "initial"
                            ),
                            "prompt_editable": (
                                bool(waiting_prompt) or (not draw_bboxes and (not transferring))
                            )
                            and (not hardware.paused),
                            "prompt_required": bool(waiting_prompt),
                            "prompt_stage_locked": bool(waiting_prompt),
                            "prompt_frame_not_before_ns": (
                                transfer.prompt_frame_not_before_ns if waiting_prompt else None
                            ),
                            "prompt_capture_not_before_ns": (
                                transfer.prompt_capture_not_before_ns if waiting_prompt else None
                            ),
                            "grasp_remaining_s": (
                                max(0.0, grasp_deadline - time.monotonic())
                                if grasp_deadline is not None
                                else config["grasp_delay_s"]
                            ),
                            "pose_remaining_s": (
                                max(0.0, pose_deadline - time.monotonic())
                                if pose_deadline is not None
                                else 0.0
                            ),
                            "saved_pose_age_s": (
                                (
                                    time.monotonic_ns()
                                    - transfer.pose_sampling["mean_saved_monotonic_ns"]
                                )
                                / 1000000000.0
                                if transfer is not None and transfer.pose_sampling
                                else None
                            ),
                            "pose_phase": pose_phase,
                        }
                        if transfer is not None:
                            if transfer.preplace is not None:
                                receiving["preplace_alignment"] = transfer.preplace.summary()
                        key = preview.update(
                            runtime / "perception" / "preview.png", receiving=receiving
                        )
                        prompt_saved = key == "prompt_saved"
                        prompt_action = key if key in PROMPT_ACTIONS else None
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
                            print(
                                f"已暂停：{hardware.error}。手保持最后目标；按 X 退出。", flush=True
                            )
                            write_json(
                                log.directory / "result.json", paused_result(hardware, transfer)
                            )
                            paused_reported = True
                        if key in ("m", "n"):
                            message = f"{key.upper()} 已收到，但机械臂因以下原因暂停，不能启动：{hardware.error}"
                            print(message, flush=True)
                            log.emit(
                                "manual_move_rejected" if key == "m" else "manual_lower_rejected",
                                stage="paused",
                                reason=hardware.error,
                            )
                        end.wait(0.05)
                        continue
                    hardware.raise_if_fault()
                    state, now = (hardware.snapshot(), time.monotonic())
                    last_state = state
                    samples.append((now, state["hand_position_rad"]))
                    if (
                        prompt_action
                        and transfer is not None
                        and (transfer.waiting_prompt_stage is not None)
                    ):
                        if accept_runtime_prompt(prompt_action, transfer, hardware, state):
                            stage[0] = transfer.status
                            pose_phase = None
                            log.emit("bbox_accepted", action=prompt_action, stage=stage[0])
                        prompt_saved = False
                        key = None
                    force_move = bool(
                        key == "m"
                        and transfer is not None
                        and (transfer.status in ("estimating_held_pose", "waiting_move_key"))
                    )
                    if key == "n" and (transfer is None or transfer.status == "idle"):
                        print("N 未执行：尚未完成目标上方搬运。", flush=True)
                    if key == "m" and (not force_move):
                        message = {
                            "planning_above": "正在规划，请等待规划结果",
                            "moving_above": "正在执行移动",
                            "waiting_initial_bbox": "等待初始 bbox，请在窗口 E 绘制或 Enter 复用",
                            "waiting_preplace_bbox": "等待导轨台上方 bbox，请在窗口 E 绘制或 Enter 复用",
                            "reestimating_preplace": "正在重新估姿",
                            "planning_preplace": "正在规划纠偏",
                            "aligning_preplace": "正在执行纠偏",
                            "above_target": "已经到达目标上方",
                        }.get(stage[0], "还没有保存的位姿均值，请等待抓后估姿完成")
                        print(f"M 已收到，未重复启动：{message}。", flush=True)
                        log.emit("manual_move_rejected", stage=stage[0], reason=message)
                    if prompt_saved:
                        print("新 bbox 已保存，将按抓握后的既定时序进行分割和估姿。", flush=True)
                    if key == "g" and stage[0] in ("opening", "waiting_rail"):
                        grasp("keyboard")
                    elif key in JOG_KEYS or key in ("+", "=", "-"):
                        if stage[0] in ("waiting_rail", "holding", "above_target"):
                            target = hardware.jog(key)
                            if target is not None:
                                last_jog_ns = time.monotonic_ns()
                            print(
                                f"点动步长：{config['jog_step_m'] * hardware.jogger.scale * 1000:.2f} mm / {np.degrees(config['jog_step_rad']) * hardware.jogger.scale:.2f}°",
                                flush=True,
                            )
                    if (
                        stage[0] == "waiting_rail"
                        and grasp_deadline is not None
                        and (now >= grasp_deadline)
                    ):
                        grasp("timer")
                    if stage[0] in ("opening", "closing") and hardware.hand_transition_finished(
                        samples, now, require_target=stage[0] == "opening"
                    ):
                        if stage[0] == "opening":
                            stage[0] = "waiting_rail"
                            grasp_deadline = time.monotonic() + config["grasp_delay_s"]
                            print(
                                f"接料准备完成，开始 {config['grasp_delay_s']:g} 秒抓握倒计时；可点动或按 G 提前闭合。",
                                flush=True,
                            )
                        else:
                            stage[0] = "holding"
                            result = {
                                "status": "holding",
                                "side": "right",
                                "grasp_source": grasp_source,
                                "grasp_verified": False,
                                "insertion_executed": False,
                                "collision_checked": False,
                                "hand_position_rad": state["hand_position_rad"].tolist(),
                                "hand_target_rad": hardware.hand_target.tolist(),
                                "T_B_F": state["T_B_F"].tolist(),
                            }
                            write_json(log.directory / "result.json", result)
                            pose_deadline = time.monotonic() + config["post_grasp_pose_delay_s"]
                            held_pose_status = "waiting" if perception is not None else "failed"
                            print(
                                f"抓握完成并保持，等待 {config['post_grasp_pose_delay_s']:g} 秒后开始 SAM 分割与估姿。",
                                flush=True,
                            )
                        log.emit("stage", stage=stage[0])
                        samples.clear()
                    if held_pose_status == "waiting" and time.monotonic() >= pose_deadline:
                        held_pose_status = "pending"
                        transfer.begin()
                        stage[0] = transfer.status
                        print("抓后等待结束，开始 SAM 框提示分割并估计位姿。", flush=True)
                    transferring = transfer is not None and transfer.status != "idle"
                    if transferring and perception is None and (transfer.cached_grasp is None):
                        raise RuntimeError("抓持后视觉不可用；保持抓握，未继续向目标移动")
                    if (
                        transferring
                        and perception is None
                        and (transfer.waiting_prompt_stage is not None)
                    ):
                        raise RuntimeError("等待绘框时相机不可用；保持抓握，未继续移动")
                    if (
                        transferring
                        and perception is None
                        and (transfer.preplace is not None)
                        and (not transfer.preplace.completed)
                    ):
                        raise RuntimeError("预放置重新估姿不可用；保持抓握，未启动下放")
                    if stage[0] == "closing" and perception is not None:
                        try:
                            perception.publish_context(state, enabled=False, epoch=0)
                        except Exception as error:
                            disable_auto(error)
                    record = None
                    if perception is not None and transferring:
                        try:
                            waiting_prompt = (
                                transferring and transfer.waiting_prompt_stage is not None
                            )
                            returning = transferring and transfer.release_return is not None
                            perception.publish_context(
                                state, enabled=not (waiting_prompt or returning), epoch=last_jog_ns
                            )
                            if transferring and transfer.needs_reinitialize and perception.ready():
                                predicted = transfer.reestimate_predicted_pose
                                generation = (
                                    perception.initialize()
                                    if predicted is None
                                    else perception.initialize(predicted_pose=predicted)
                                )
                                transfer.perception_initialized(generation)
                                pose_phase = None
                            record = None if waiting_prompt or returning else perception.latest()
                            if (
                                transferring
                                and record
                                and (record.get("context_epoch") != last_jog_ns)
                            ):
                                record = None
                            reason = record.get("reason") if record else None
                            if reason in POSE_PHASES and reason != pose_phase:
                                pose_phase = reason
                                print(f"抓后识别：{POSE_PHASES[reason]}。", flush=True)
                        except Exception as error:
                            if transferring and transfer.cached_grasp is None:
                                raise
                            disable_auto(error)
                    if transferring:
                        previous_stage = stage[0]
                        stage[0] = transfer.advance(
                            hardware,
                            state,
                            record,
                            force_move=force_move,
                            lower_requested=key == "n",
                        )
                        if key == "n" and transfer.manual_lower_message:
                            print(transfer.manual_lower_message, flush=True)
                            log.emit(
                                "manual_lower_request",
                                message=transfer.manual_lower_message,
                                stage=stage[0],
                            )
                        if force_move and transfer.manual_move_message:
                            print(transfer.manual_move_message, flush=True)
                            log.emit(
                                "manual_pose_move",
                                message=transfer.manual_move_message,
                                pose_sampling=transfer.pose_sampling,
                            )
                        if stage[0] != previous_stage:
                            message = {
                                "planning_above": "正在使用已保存的位姿均值规划右臂路径",
                                "waiting_initial_bbox": "等待初始 bbox：窗口 E 拖框、Enter 保存，或直接 Enter 复用",
                                "waiting_preplace_bbox": "导轨已停在台上方：窗口 E 拖第二个框、Enter 保存，或直接 Enter 复用",
                                "waiting_move_key": "位姿均值已保存，等待 M；按 M 直接使用该均值规划移动",
                                "moving_above": "右臂以位置控制直接移向装配目标上方",
                                "reestimating_preplace": "重新 SAM 分割与 FoundationPose 注册，计算位姿偏差",
                                "planning_preplace": "正在根据新位姿规划一轮纠偏，检查各段 IK 与导轨几何高度",
                                "aligning_preplace": "右臂低速执行一轮纠偏；完成后保持位置，等待人工按 N",
                                "above_target": "右臂已到目标上方；保持抓握，人工确认后按 N 螺旋下插",
                                "planning_position_spiral": "正在检查全部螺旋目标的 IK 与限位",
                                "position_spiral": "右臂以位置控制螺旋下降，Q/空格可暂停",
                                "insertion_reached": "已到插入终点，准备自动松手回位",
                                "planning_return": "插入到位，正在检查抬升与回位路径；完成规划后自动松手",
                                "releasing_hand": f"正在张开右手，张手指令发送完成后等待 {config['hand_settle_s']:g} 秒，再自动抬升",
                                "lifting_after_release": f"张手指令与等待已完成，保持张手目标，向上抬高 {config['release_return']['lift_height_m'] * 100:g} cm",
                                "returning_initial": "抬升已到位，右臂保持张手并返回固定初始位置",
                                "returned_initial": "右臂已回到初始位置并保持张手，本次流程完成",
                            }
                            if transfer.preplace is not None:
                                message["above_target"] = (
                                    "首次重估已在到位容差内，无需纠偏移动；保持抓握，人工确认后按 N 螺旋下插"
                                    if transfer.preplace.plans == 0
                                    else "单轮纠偏已到位，保持抓握；请人工确认后按 N 螺旋下插"
                                )
                            print(message.get(stage[0], stage[0]), flush=True)
                            log.emit("stage", stage=stage[0])
                except Exception as error:
                    hardware.pause(str(error))
                end.wait(0.02)
    except BaseException as error:
        if hardware is not None:
            hardware.pause(str(error))
        result = (
            paused_result(hardware, transfer)
            if hardware is not None
            else dict(status="failed", reason=str(error), side="right", assembly_verified=False)
        )
        write_json(log.directory / "result.json", result)
        log.emit("error", reason=str(error), stage=stage[0])
        raise
    finally:
        exit_requested = end.is_set()
        end.set()
        if watcher is not None:
            watcher.join(timeout=1)
        errors = []
        if (
            transfer is not None
            and transfer.release_return is not None
            and (transfer.release_return.status not in ("returned_initial", "paused"))
        ):
            try:
                transfer.release_return.paused_summary(
                    reason=(hardware.error if hardware is not None else None) or "Receiver exit"
                )
            except Exception as error:
                errors.append(f"Release/return log finalization: {error}")
        if (
            transfer is not None
            and transfer.spiral is not None
            and (hardware is not None)
            and (transfer.spiral.status != "insertion_reached")
        ):
            try:
                transfer.spiral.paused_summary(reason=hardware.error or "Receiver exit")
            except Exception as error:
                errors.append(f"Position spiral log finalization: {error}")
        if (
            transfer is not None
            and transfer.preplace is not None
            and (not transfer.preplace.completed)
            and (transfer.preplace.status != "paused")
        ):
            try:
                transfer.preplace.stopped(
                    reason=(hardware.error if hardware is not None else None) or "Receiver exit"
                )
            except Exception as error:
                errors.append(f"Preplacement alignment log finalization: {error}")
        for resource in (preview, perception, hardware):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    errors.append(str(error))
        try:
            log.finish(stage[0], exit_requested=exit_requested, errors=errors, feedback=last_state)
        finally:
            try:
                log.close()
            finally:
                if old_sigint is not None:
                    signal.signal(signal.SIGINT, old_sigint)
        if errors:
            raise RuntimeError("; ".join(errors))


def run(config, args):
    with tempfile.TemporaryDirectory(prefix="rail-insert-") as directory:
        if args.output is None:
            args.output = Path(directory) / "run"
        _run(config, args, Path(directory) / "runtime")


def read_state(config):
    """Connect for one feedback sample without engaging either device."""
    hardware = RightHardware(config, kinematics(), lambda *a, **k: None, motion=False)
    try:
        hardware.start()
        state = hardware.snapshot()
        print(
            json.dumps(
                {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in state.items()},
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        hardware.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="右臂接料、单轮视觉纠偏、螺旋插入与自动松手回位")
    parser.add_argument("--config", type=Path, default=HERE / "receive.yaml")
    parser.add_argument(
        "--already-held", action="store_true", help="已抓住导轨，跳过启动回位和张手"
    )
    parser.add_argument(
        "--draw-bboxes", action="store_true", help="在初始位置和导轨台上方分别等待绘框"
    )
    parser.add_argument("--no-view", action="store_true", help="关闭预览，要求两份 bbox 均已有效")
    parser.add_argument("--above-height", type=float, help="目标上方高度，单位 m，默认 0.015")
    parser.add_argument(
        "--output", type=Path, help="保留 events.jsonl 和 result.json；默认退出后清理"
    )
    args = parser.parse_args(argv)
    if args.draw_bboxes and args.no_view:
        parser.error("--draw-bboxes cannot be combined with --no-view")
    try:
        config = load_config(args.config)
        if args.above_height is not None:
            config["move_above"]["height_m"] = args.above_height
        run(config, args)
        return 0
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        print(f"插入流程停止：{error}")
        return 1
