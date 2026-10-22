"""双臂回到配置里的初始关节，双手回到 rail_place3 示范起点。

机械臂目标只读取 diffusion_policy/config/tianji_teleop.yaml 的 ready_pose，
与 home_tianji 使用的关节角相同，本脚本不另设一套手臂角度。
灵巧手目标是 41 条有效示范在动作开始前 0.2 秒的关节中位数，不是零位。

先回双臂，再同时回双手。到位后双手保持该姿态，按 Q 退出并松开。
"""

from __future__ import annotations

import math
import threading
import time
from copy import deepcopy
from dataclasses import replace

from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import (
    load_config as load_arm_config,
)
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import (
    TianjiDriver,
)
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import (
    JOINT_LIMITS_RAD,
    JOINT_NAMES,
    WujiHandDriver,
)
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import (
    load_config as load_hand_config,
)
from diffusion_policy.third_party.bimanual_teleop.types import (
    ControlProfile,
    DeviceCommand,
    JointTarget,
)

from .terminal import MotionTerminal

# 每侧 20 个关节角（度）：拇指、食指、中指、无名指、小指，每指 joint1..joint4。
# rail_place3 去掉 discarded/failed 后剩 41 条；每条取动作前 0.2 秒的中位数，再取跨轨迹中位数。
HAND_HOME_DEG = {
    "left": (
        3.47,
        11.81,
        17.38,
        25.56,
        5.16,
        5.34,
        5.33,
        43.16,
        15.76,
        6.52,
        4.48,
        41.03,
        16.38,
        5.32,
        1.22,
        33.07,
        5.01,
        6.24,
        39.95,
        1.40,
    ),
    "right": (
        18.31,
        6.23,
        13.64,
        8.65,
        1.33,
        3.81,
        32.16,
        21.80,
        16.93,
        0.56,
        14.99,
        18.37,
        13.08,
        -2.92,
        22.83,
        7.87,
        1.39,
        -0.18,
        37.67,
        2.82,
    ),
}
SIDES = ("left", "right")
CONTROL_HZ = 120
PERIOD_NS = round(1e9 / CONTROL_HZ)
COMMAND_TTL_NS = 50_000_000


class Cancelled(Exception):
    """操作员在回位过程中按下了 Q 或关闭了终端。"""


def positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} 必须是有限正数")
    return value


def selected_sides(side):
    if side == "both":
        return SIDES
    if side not in SIDES:
        raise ValueError("side 必须是 left、right 或 both")
    return (side,)


def ready_joint_targets(settings, side="both"):
    """读取配置中的 ready_pose，不改写其中的关节角。"""
    source = settings["ready_pose"]
    order = source.get("order", list(SIDES))
    if side == "both" and (
        not isinstance(order, list) or len(order) != 2 or set(order) != set(SIDES)
    ):
        raise ValueError("ready_pose.order 必须恰好包含 left 和 right 各一次")
    chosen = tuple(order) if side == "both" else selected_sides(side)
    targets = {}
    for arm in chosen:
        values = source["target_deg"][arm]
        if (
            not isinstance(values, list)
            or len(values) != 7
            or any(
                isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q)
                for q in values
            )
        ):
            raise ValueError(f"{arm} 需要 7 个有限关节角（度）")
        targets[arm] = tuple(math.radians(q) for q in values)
    return source, chosen, targets


def ready_profile(settings, source, order):
    profile = ControlProfile(**settings["profile"])
    parameters = deepcopy(profile.parameters)
    parameters["active_arms"] = list(order)
    parameters["arms"] = {arm: parameters["arms"][arm] for arm in order}
    for arm in order:
        for key in ("velocity_ratio", "acceleration_ratio"):
            if key in source:
                parameters["arms"][arm][key] = source[key]
    return replace(profile, profile_id=profile.profile_id + "-ready", parameters=parameters)


def hand_home_rad(side):
    degrees = HAND_HOME_DEG[side]
    if len(degrees) != len(JOINT_NAMES) or any(
        isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q)
        for q in degrees
    ):
        raise ValueError(f"{side} 灵巧手目标必须是 20 个有限角度")
    radians = tuple(math.radians(q) for q in degrees)
    for name, q, (low, high) in zip(JOINT_NAMES, radians, JOINT_LIMITS_RAD):
        if not low <= q <= high:
            raise ValueError(f"{side} {name} 目标 {math.degrees(q):.2f}° 超出关节限位")
    return radians


def blend_joints(start_rad, goal_rad, elapsed_s, duration_s):
    """从实测关节平滑过渡到目标；两端分别是起点和目标。"""
    positive(duration_s, "duration_s")
    if len(start_rad) != len(JOINT_NAMES) or len(goal_rad) != len(JOINT_NAMES):
        raise ValueError("灵巧手插值需要 20 个关节")
    if any(not math.isfinite(q) for q in (*start_rad, *goal_rad, elapsed_s)):
        raise ValueError("灵巧手插值包含非有限角度")
    alpha = max(0.0, min(1.0, elapsed_s / duration_s))
    alpha = alpha * alpha * (3 - 2 * alpha)
    return tuple((1 - alpha) * start + alpha * goal for start, goal in zip(start_rad, goal_rad))


def _stopped(keys):
    return keys == "" or (keys and any(key.lower() in ("q", "\x03", "\x04") for key in keys))


def confirm_motion(terminal):
    while True:
        keys = terminal.read(0.0)
        if _stopped(keys):
            print("已取消启动。")
            return False
        if keys is None:
            break
    print(
        "设备即将运动。请确认机械臂和灵巧手周围无人员、障碍物，"
        "运动范围内无碰撞风险，实体急停已释放，并做好随时急停的准备。"
    )
    print("确认安全后按回车键开始回位；运动中按 Q 或 Ctrl+C 取消。")
    while True:
        keys = terminal.read(0.1)
        if _stopped(keys):
            print("已取消启动。")
            return False
        if keys and any(key in ("\n", "\r") for key in keys):
            return True


def _await_arm_feedback(driver):
    deadline = time.monotonic() + 2
    while (sample := driver.get_latest()) is None:
        if time.monotonic() >= deadline:
            raise RuntimeError("没有收到天机反馈")
        time.sleep(0.01)
    return sample


def _format_deg(radians):
    return "[" + ", ".join(f"{math.degrees(q):.2f}" for q in radians) + "]"


def _announce_arm_targets(order, targets):
    for side in order:
        label = "左臂" if side == "left" else "右臂"
        print(f"{label}目标：{_format_deg(targets[side])}°")


def home_connected_arms(
    driver, settings, terminal, *, sides="both", joint_targets=None, announce=True
):
    """把已经连接的机械臂回到 ready_pose。不启动、也不关闭驱动。"""
    source, order, targets = ready_joint_targets(settings, sides)
    if joint_targets is not None:
        targets = {side: tuple(float(q) for q in joint_targets[side]) for side in order}
    if announce:
        _announce_arm_targets(order, targets)
    cancel = threading.Event()
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            if _stopped(terminal.read(0.05)):
                cancel.set()
                return

    watcher = threading.Thread(target=watch, name="arm-home-cancel", daemon=True)
    watcher.start()
    try:
        if getattr(driver, "engaged", False):
            driver.request_hold("return home")
        if cancel.is_set():
            raise Cancelled()
        profile = ready_profile(settings, source, order)
        print("正在清理机械臂故障……")
        driver.clear_errors(order, cancel=cancel)
        driver.configure(profile)
        for side in order:
            if cancel.is_set():
                raise Cancelled()
            label = "左臂" if side == "left" else "右臂"
            ratio = profile.parameters["arms"][side]["velocity_ratio"]
            print(f"{label}回位中 · 速度 {ratio}%")
            driver.move_joints(side, targets[side], cancel=cancel, timeout_s=60.0)
        print("机械臂已回到配置中的初始关节角。")
    except RuntimeError:
        if cancel.is_set():
            raise Cancelled() from None
        raise
    finally:
        stop.set()
        watcher.join(timeout=1)


def move_arms(settings, sides, terminal, *, inspect, joint_targets=None):
    source, order, targets = ready_joint_targets(settings, sides)
    if joint_targets is not None:
        targets = {side: tuple(float(q) for q in joint_targets[side]) for side in order}
    _announce_arm_targets(order, targets)
    if inspect:
        driver = TianjiDriver(settings["controller_ip"])
        try:
            driver.start()
            sample = _await_arm_feedback(driver)
            for side in order:
                arm = sample.payload.arms[side]
                label = "左臂" if side == "left" else "右臂"
                print(
                    f"{label}当前：{_format_deg(arm.joints.position_rad)}° · 状态 {arm.state} · 错误 {arm.error}"
                )
        finally:
            driver.close()
        return

    driver = TianjiDriver(settings["controller_ip"])
    try:
        driver.start()
        _await_arm_feedback(driver)
        home_connected_arms(
            driver, settings, terminal, sides=sides, joint_targets=joint_targets, announce=False
        )
    finally:
        driver.close()


def _fresh_hand(hand, now_ns):
    health = hand.health(check_latch=True)
    if not health.ready:
        raise RuntimeError(f"{hand.side} 灵巧手反馈异常：{health.detail}")
    sample = hand.get_latest()
    if (
        sample is None
        or not sample.header.valid
        or now_ns - sample.header.received_monotonic_ns > hand.timeout_ns
    ):
        raise RuntimeError(f"{hand.side} 灵巧手反馈无效或过期")
    return sample


def _max_error_rad(sample, goal):
    joints = sample.payload.position_rad
    if len(joints) != len(goal) or any(q is None or not math.isfinite(q) for q in joints):
        return None
    return max(abs(q - target) for q, target in zip(joints, goal))


def move_hands(
    config, sides, terminal, *, duration_s, tolerance_rad, settle_timeout_s, inspect, goals=None
):
    goals = (
        {side: hand_home_rad(side) for side in sides}
        if goals is None
        else {side: tuple(float(q) for q in goals[side]) for side in sides}
    )
    for side in sides:
        for name, q, (low, high) in zip(JOINT_NAMES, goals[side], JOINT_LIMITS_RAD):
            if not low <= q <= high:
                raise ValueError(f"{side} {name} 目标 {math.degrees(q):.2f}° 超出关节限位")
    for side in sides:
        label = "左手" if side == "left" else "右手"
        print(f"{label}目标：{_format_deg(goals[side])}°")
    profile = ControlProfile(config["profile_id"], config.get("mode", "mit"), config["parameters"])
    hands = [
        WujiHandDriver(
            side,
            config["devices"][side]["hand"],
            feedback_hz=config.get("feedback_hz", 200),
            timeout_s=config.get("hand_timeout_s", 0.5),
        )
        for side in sides
    ]
    opened = []
    try:
        for hand in hands:
            hand.start()
            opened.append(hand)
            deadline = time.monotonic() + 3
            while not hand.health(check_latch=True).ready and time.monotonic() < deadline:
                time.sleep(0.02)
            _fresh_hand(hand, time.monotonic_ns())
        if inspect:
            for hand in opened:
                sample = hand.get_latest()
                error = _max_error_rad(sample, goals[hand.side])
                label = "左手" if hand.side == "left" else "右手"
                print(
                    f"{label}当前：{_format_deg(sample.payload.position_rad)}° · "
                    f"距示范起点最大 {math.degrees(error):.1f}°"
                )
            return True

        for hand in opened:
            hand.configure(profile)
            hand.engage()
            if hand.last_target is None:
                raise RuntimeError(f"{hand.side} 灵巧手没有保留接合时的实测关节")
            start = tuple(hand.last_target.position_rad)
            if any(not low <= q <= high for q, (low, high) in zip(start, JOINT_LIMITS_RAD)):
                raise RuntimeError(f"{hand.side} 实测关节超出限位，拒绝移动")
            hand.start_rad = start
        label = "双手" if len(opened) == 2 else ("左手" if opened[0].side == "left" else "右手")
        print(f"{label}从当前关节平滑移动到示范起点，约 {duration_s:g} 秒；按 Q 中止。")
        start_ns = time.monotonic_ns()
        motion_end_ns = start_ns + round(duration_s * 1e9)
        settle_end_ns = motion_end_ns + round(settle_timeout_s * 1e9)
        deadline = start_ns
        count = 0
        arrived = False
        while True:
            now = time.monotonic_ns()
            if now < deadline:
                time.sleep((deadline - now) / 1e9)
                now = time.monotonic_ns()
            deadline += PERIOD_NS
            if deadline <= now:
                deadline = now + PERIOD_NS
            if _stopped(terminal.read(0)):
                if arrived:
                    return True
                raise Cancelled()
            for hand in opened:
                sample = _fresh_hand(hand, now)
                target = JointTarget(
                    JOINT_NAMES,
                    blend_joints(
                        hand.start_rad, goals[hand.side], (now - start_ns) / 1e9, duration_s
                    ),
                )
                count += 1
                command = DeviceCommand(
                    hand.device_id,
                    f"{hand.device_id}-home-{count}",
                    target,
                    (sample.header.ref,),
                    now,
                    now + COMMAND_TTL_NS,
                    profile.profile_id,
                )
                result = hand.submit(command)
                if not result.accepted:
                    raise RuntimeError(f"{hand.side} 灵巧手命令被拒绝：{result.reason}")
            if now < motion_end_ns or arrived:
                continue
            errors = [_max_error_rad(hand.get_latest(), goals[hand.side]) for hand in opened]
            if all(error is not None and error <= tolerance_rad for error in errors):
                arrived = True
                worst = max(errors)
                print(
                    f"{label}已到位，最大反馈误差 {math.degrees(worst):.1f}°；保持该姿态，按 Q 退出并松开。"
                )
            elif now >= settle_end_ns:
                detail = "、".join(
                    "无有效反馈" if error is None else f"{math.degrees(error):.1f}°"
                    for error in errors
                )
                raise RuntimeError(f"灵巧手未在容差内到位：{detail}")
    finally:
        problems = []
        for hand in reversed(opened):
            try:
                hand.close()
            except Exception as exc:
                problems.append(str(exc))
        if problems:
            raise RuntimeError("；".join(problems))


def run(args):
    parts = ("arms", "hands") if args.part == "both" else (args.part,)
    arm_settings = (
        load_arm_config(args.tianji_config, args.ip) if "arms" in parts or args.inspect else None
    )
    hand_config = load_hand_config(args.wuji_config) if "hands" in parts or args.inspect else None
    duration_s = positive(args.duration_s, "--duration-s")
    tolerance_rad = math.radians(positive(args.tolerance_deg, "--tolerance-deg"))
    settle_timeout_s = positive(args.settle_timeout_s, "--settle-timeout-s")
    # 连接设备前检查双手目标，避免连上以后才发现角度非法。
    for side in selected_sides(args.side):
        if "hands" in parts:
            hand_home_rad(side)

    def act(terminal):
        if "arms" in parts:
            move_arms(arm_settings, args.side, terminal, inspect=args.inspect)
        if "hands" in parts:
            move_hands(
                hand_config,
                selected_sides(args.side),
                terminal,
                duration_s=duration_s,
                tolerance_rad=tolerance_rad,
                settle_timeout_s=settle_timeout_s,
                inspect=args.inspect,
            )

    if args.inspect:
        act(None)
        return 0
    with MotionTerminal() as terminal:
        if not confirm_motion(terminal):
            return 0
        act(terminal)
    print("回位完成，灵巧手已松开。" if "hands" in parts else "回位完成。")
    return 0
