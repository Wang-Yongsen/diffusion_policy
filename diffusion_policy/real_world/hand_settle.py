"""Move both hands to the frozen start and keep commanding that pose.

The home script disables the hands when it exits. Fingers do not hold a pose
without motor torque, so each slot enables the hands already owned by the
environment, blends to the frozen joints, and streams that target until the
policy runtime takes over or the environment closes.
"""

from __future__ import annotations

import math
import threading
import time

from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_LIMITS_RAD, JOINT_NAMES
from diffusion_policy.third_party.bimanual_teleop.types import DeviceCommand, JointTarget
from home_tianji_rail import blend_joints


SIDES = ("left", "right")
COMMAND_TTL_NS = 50_000_000


class HandSettleCancelled(Exception):
    """操作员取消了核对前的摆手。"""


class HandHold:
    """Stream one joint target until stop. Does not disable the motors."""

    def __init__(self, hands, locks, goals, *, clock_ns=time.monotonic_ns, period_s=1 / 120, on_error=None):
        self.hands, self.locks, self.goals = hands, locks, goals
        self.clock_ns, self.period_s, self.on_error = clock_ns, period_s, on_error
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        self._count = 0

    def submit(self, now_ns):
        for side in SIDES:
            hand = self.hands[side]
            _fresh(hand, now_ns)
            self._count += 1
            target = JointTarget(JOINT_NAMES, self.goals[side])
            command = DeviceCommand(
                hand.device_id, f"{hand.device_id}-hold-{self._count}", target, (),
                now_ns, now_ns + COMMAND_TTL_NS, hand.profile.profile_id)
            with self.locks[side]:
                result = hand.submit(command)
            if not result.accepted:
                raise RuntimeError(f"{side} 灵巧手保持命令被拒绝：{result.reason}")

    def start(self):
        if self._thread is not None:
            raise RuntimeError("双手保持已经启动")
        self._thread = threading.Thread(target=self._run, name="hand-settle-hold", daemon=True)
        self._thread.start()

    @property
    def running(self):
        thread = self._thread
        return thread is not None and thread.is_alive() and self.error is None

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2)
            if thread.is_alive():
                raise RuntimeError("双手保持线程没有退出")

    def _run(self):
        while not self._stop.wait(self.period_s):
            try:
                self.submit(self.clock_ns())
            except Exception as error:
                self.error = error
                if self.on_error is not None:
                    self.on_error(error)
                return


def settle_and_hold(env, goals, tolerance_rad, read_key, *, duration_s=3., settle_timeout_s=2.,
                    clock_ns=None, sleep=None, goal_name="冻结起点"):
    """Enable env.hands, move to goals, and leave a hold running on env._hand_hold."""
    if tolerance_rad <= 0 or not math.isfinite(tolerance_rad):
        raise ValueError("手关节容差必须是有限正数")
    if env._hand_hold is not None:
        raise RuntimeError("双手已经在保持，不能再次摆手")
    goals = {side: _goal(side, goals[side]) for side in SIDES}
    clock_ns = time.monotonic_ns if clock_ns is None else clock_ns
    sleep = time.sleep if sleep is None else sleep
    config = getattr(env, "config", None)
    hz = 120 if not isinstance(config, dict) else int(config.get("hand_hz", 120))
    period_ns = round(1e9 / hz)
    locks = env.runtime._device_locks
    for side in SIDES:
        hand = env.hands[side]
        if hand.profile is None:
            hand.configure(env.hand_profile)
        if not hand.enabled:
            hand.engage()
        if hand.last_target is None:
            raise RuntimeError(f"{side} 灵巧手没有保留接合时的实测关节")
        start = tuple(float(q) for q in hand.last_target.position_rad)
        if any(not low <= q <= high for q, (low, high) in zip(start, JOINT_LIMITS_RAD)):
            raise RuntimeError(f"{side} 实测关节超出限位，拒绝移动")
        hand.settle_start = start

    worst = _max_error(env, goals, clock_ns())
    if worst is not None and worst <= tolerance_rad:
        print(f"双手离{goal_name}不超过 {math.degrees(worst):.1f}°，直接保持。")
    else:
        worst = _blend(env, goals, tolerance_rad, read_key, duration_s, settle_timeout_s,
                       clock_ns, sleep, period_ns, locks, goal_name)
    hold = HandHold(env.hands, locks, goals, clock_ns=clock_ns, period_s=period_ns / 1e9,
                    on_error=lambda error: setattr(env, "error", str(error)))
    env._hand_hold = hold
    hold.submit(clock_ns())
    hold.start()
    print(f"双手已到位，最大反馈误差 {math.degrees(worst):.1f}°；保持该姿态，不会松开。")
    return worst


def _blend(env, goals, tolerance_rad, read_key, duration_s, settle_timeout_s, clock_ns, sleep, period_ns, locks,
           goal_name):
    print(f"双手从当前关节平滑移动到{goal_name}，约 {duration_s:g} 秒；按 Q 中止。")
    start_ns = clock_ns()
    motion_end_ns = start_ns + round(duration_s * 1e9)
    settle_end_ns = motion_end_ns + round(settle_timeout_s * 1e9)
    deadline = start_ns
    count = 0
    while True:
        now = clock_ns()
        if now < deadline:
            sleep((deadline - now) / 1e9)
            now = clock_ns()
        deadline += period_ns
        if deadline <= now:
            deadline = now + period_ns
        if _stopped(read_key(0)):
            raise HandSettleCancelled()
        elapsed_s = (now - start_ns) / 1e9
        for side in SIDES:
            hand = env.hands[side]
            _fresh(hand, now)
            count += 1
            target = JointTarget(JOINT_NAMES, blend_joints(
                hand.settle_start, goals[side], elapsed_s, duration_s))
            command = DeviceCommand(
                hand.device_id, f"{hand.device_id}-settle-{count}", target, (),
                now, now + COMMAND_TTL_NS, hand.profile.profile_id)
            with locks[side]:
                result = hand.submit(command)
            if not result.accepted:
                raise RuntimeError(f"{side} 灵巧手命令被拒绝：{result.reason}")
        if now < motion_end_ns:
            continue
        worst = _max_error(env, goals, now)
        if worst is not None and worst <= tolerance_rad:
            return worst
        if now >= settle_end_ns:
            detail = "无有效反馈" if worst is None else f"{math.degrees(worst):.1f}°"
            raise RuntimeError(f"灵巧手未在容差内到位：{detail}")


def _goal(side, values):
    goal = tuple(float(q) for q in values)
    if len(goal) != len(JOINT_NAMES) or any(not math.isfinite(q) for q in goal):
        raise ValueError(f"{side} 冻结手目标必须是 20 个有限关节角")
    for name, q, (low, high) in zip(JOINT_NAMES, goal, JOINT_LIMITS_RAD):
        if not low <= q <= high:
            raise ValueError(f"{side} {name} 目标 {math.degrees(q):.2f}° 超出关节限位")
    return goal


def _fresh(hand, now_ns):
    health = hand.health(check_latch=True)
    if not health.ready:
        raise RuntimeError(f"{hand.side} 灵巧手反馈异常：{health.detail}")
    sample = hand.get_latest()
    if sample is None or not sample.header.valid or now_ns - sample.header.received_monotonic_ns > hand.timeout_ns:
        raise RuntimeError(f"{hand.side} 灵巧手反馈无效或过期")
    return sample


def _max_error(env, goals, now_ns):
    errors = []
    for side in SIDES:
        sample = _fresh(env.hands[side], now_ns)
        joints = sample.payload.position_rad
        if len(joints) != len(goals[side]) or any(q is None or not math.isfinite(q) for q in joints):
            return None
        errors.append(max(abs(q - target) for q, target in zip(joints, goals[side])))
    return max(errors)


def _stopped(keys):
    return keys == "" or (keys and any(key.lower() in ("q", "\x03", "\x04") for key in keys))
