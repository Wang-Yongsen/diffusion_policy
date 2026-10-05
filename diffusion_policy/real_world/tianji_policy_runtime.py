"""Absolute 52-D targets, monotonic scheduling and independent device loops."""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.third_party.bimanual_teleop.types import DeviceCommand, JointTarget, Pose, RobotTarget
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_LIMITS_RAD, JOINT_NAMES

SIDES = ("left", "right")


def split_action(action):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (52,) or not np.isfinite(action).all():
        raise ValueError("Require finite absolute 52-D action")
    poses, hands = {}, {}
    for i, side in enumerate(SIDES):
        xyz, rotvec = action[i * 6:i * 6 + 3], action[i * 6 + 3:i * 6 + 6]
        poses[side] = Pose(f"tianji_{side}_base", f"tianji_{side}_flange",
                           tuple(xyz), tuple(Rotation.from_rotvec(rotvec).as_quat()))
        hands[side] = JointTarget(JOINT_NAMES, tuple(action[12 + i * 20:32 + i * 20]))
    return poses, hands


def future_actions(actions, anchor_ns, now_ns, hz=30, margin_s=.01):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 52 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Require nonempty finite N x 52 predictions")
    stamps = int(anchor_ns) + np.rint(np.arange(len(actions)) * 1e9 / hz).astype(np.int64)
    keep = stamps > now_ns + round(margin_s * 1e9)
    if not keep.any():
        raise RuntimeError("Prediction has no future action coverage")
    return actions[keep].copy(), stamps[keep], int((~keep).sum())


def validate_actions(actions, times, reference, config):
    limits = np.tile(np.asarray(JOINT_LIMITS_RAD), (2, 1))
    if np.any(actions[:, 12:] < limits[:, 0]) or np.any(actions[:, 12:] > limits[:, 1]):
        raise ValueError("Hand target outside Hand2 joint limits")
    previous = np.vstack((reference, actions[:-1]))
    for i, side in enumerate(SIDES):
        xyz = actions[:, 6 * i:6 * i + 3]
        workspace = config["workspace"][side]
        if workspace is None:
            raise ValueError(f"Set measured workspace.{side}.min/max before motion")
        if np.any(xyz < workspace["min"]) or np.any(xyz > workspace["max"]):
            raise ValueError(f"{side} target outside configured workspace")
        distance = np.linalg.norm(xyz - previous[:, 6 * i:6 * i + 3], axis=1)
        current_r = Rotation.from_rotvec(actions[:, 6 * i + 3:6 * i + 6])
        previous_r = Rotation.from_rotvec(previous[:, 6 * i + 3:6 * i + 6])
        angle = (current_r * previous_r.inv()).magnitude()
        if np.any(distance > config["max_position_jump_m"]) or np.any(angle > config["max_rotation_jump_rad"]):
            raise ValueError(f"{side} arm target jumps too far")
        dt = np.diff(times) / 1e9
        if (np.any(distance[1:] > dt * config["max_position_speed_m_s"])
                or np.any(angle[1:] > dt * config["max_rotation_speed_rad_s"])):
            raise ValueError(f"{side} arm target changes too fast")
    hand_delta = np.abs(actions[:, 12:] - previous[:, 12:])
    if np.any(hand_delta > config["max_hand_jump_rad"]):
        raise ValueError("Hand target jumps too far")
    if np.any(hand_delta[1:] > np.diff(times)[:, None] / 1e9 * config["max_hand_speed_rad_s"]):
        raise ValueError("Hand target changes too fast")


class TianjiPolicyRuntime:
    def __init__(self, arm, hands, config, health, emit, *, clock_ns=time.monotonic_ns):
        self.arm, self.hands, self.config = arm, hands, config
        self.health, self.emit, self.clock_ns = health, emit, clock_ns
        self.active = False
        self.fault = None
        self._lock = threading.RLock()
        self._device_locks = {key: threading.RLock() for key in ("arm", *SIDES)}
        self._stop = threading.Event()
        self._threads = []
        self._actions = np.empty((0, 52))
        self._times = np.empty(0, dtype=np.int64)
        self._last_action = None
        self.arm_ready = False
        self._held_until_ns = None
        self._last_nominal_ns = None
        self._generation = 0

    def start(self):
        for kind, side in (("arm", None), ("hand", "left"), ("hand", "right")):
            thread = threading.Thread(target=self._loop, args=(kind, side), name=f"policy-{side or kind}", daemon=True)
            self._threads.append(thread)
            thread.start()

    def seed(self, action, *, arm_ready=False):
        with self._lock:
            self._last_action = np.asarray(action, dtype=float).copy()
            self.arm_ready = arm_ready
            self.fault = None
            self.active = False
            self._actions = np.empty((0, 52))
            self._times = np.empty(0, dtype=np.int64)
            self._held_until_ns = self._last_nominal_ns = None
            self._generation += 1

    def submit(self, actions, anchor_ns):
        with self._lock:
            if self.fault or self._stop.is_set():
                raise RuntimeError(self.fault or "Runtime closed")
            actions, times, dropped = future_actions(actions, anchor_ns, self.clock_ns(),
                self.config["policy_hz"], self.config["submit_margin_s"])
            validate_actions(actions, times, self._last_action, self.config)
            held_until = int(self._times[-1]) + round(1e9 / self.config["policy_hz"]) if self.active else None
            if held_until is not None and self.clock_ns() >= held_until:
                raise RuntimeError("Previous policy coverage already expired")
            self._generation += 1
            self._actions, self._times = actions, times
            self._held_until_ns = held_until
            self.active = True
            self.emit("schedule", generation=self._generation, times_ns=times,
                      actions=actions, dropped_steps=dropped)
            return dropped

    def pause(self, reason, *, fault=False):
        with self._lock:
            self.active = False
            self._actions = np.empty((0, 52))
            self._times = np.empty(0, dtype=np.int64)
            self._generation += 1
            self.arm_ready = False
            if fault:
                self.fault = self.fault or str(reason)
        errors = []
        for key, device in (("arm", self.arm), *self.hands.items()):
            with self._device_locks[key]:
                try:
                    device.request_hold(str(reason))
                except Exception as error:
                    errors.append(str(error))
        with self._lock:
            if errors:
                self.fault = self.fault or "; ".join(errors)
        try:
            self.emit("pause", reason=str(reason), fault=fault)
        except Exception as error:
            self.fault = self.fault or str(error)

    def _target(self, now):
        if now >= self._times[-1] + round(1e9 / self.config["policy_hz"]):
            raise RuntimeError("Policy target coverage expired")
        index = int(np.searchsorted(self._times, now, side="right") - 1)
        if index >= 0:
            self._last_action = self._actions[index].copy()
            self._last_nominal_ns = int(self._times[index])
        elif self._held_until_ns is not None and now >= self._held_until_ns:
            raise RuntimeError("Previous target expired before the next scheduled action")
        return self._last_action

    def tick(self, kind, side=None):
        """One control tick, also used by fake-clock tests."""
        if kind == "hand" and side is None:
            for selected in SIDES:
                self.tick(kind, selected)
            return
        with self._device_locks["arm" if kind == "arm" else side]:
            with self._lock:
                now = self.clock_ns()
                active, generation = self.active, self._generation
                if active:
                    problem = self.health()
                    if problem:
                        raise RuntimeError(problem)
                    action = self._target(now)
                    expiry = int(self._times[-1]) + round(1e9 / self.config["policy_hz"])
                else:
                    # Holding is a distinct mode with the last accepted hand target.
                    action, expiry = self._last_action, now + 50_000_000
                nominal = self._last_nominal_ns
            if action is None:
                return
            poses, targets = split_action(action)
            command_id = f"{generation}-{side or kind}-{now}"
            if kind == "arm":
                if not self.arm_ready or not self.arm.driver.engaged:
                    if active:
                        raise RuntimeError("Arm engagement was lost")
                    return
                with self._lock:
                    if generation != self._generation:
                        return
                result = self.arm.submit(RobotTarget(command_id, poses, (), now, expiry,
                                                     self.arm.driver.profile.profile_id))
                receipts = {"arms": result}
            else:
                hand = self.hands[side]
                if not hand.enabled:
                    if active:
                        raise RuntimeError(f"{side} Hand2 engagement was lost")
                    return
                if not hand.health(check_latch=True).ready:
                    hand.disable("Unhealthy Hand2 during policy/hold")
                    raise RuntimeError(f"{side} Hand2 feedback is unhealthy")
                target = targets[side] if active else hand.last_target
                if active:
                    previous = np.asarray(hand.last_target.position_rad)
                    limit = self.config["max_hand_speed_rad_s"] / self.config["hand_hz"]
                    bounded = previous + np.clip(np.asarray(target.position_rad) - previous, -limit, limit)
                    target = JointTarget(JOINT_NAMES, tuple(bounded))
                with self._lock:
                    if generation != self._generation:
                        return
                receipts = {side: hand.submit(DeviceCommand(hand.device_id, command_id,
                    target, (), now, expiry, hand.profile.profile_id))}
            self.emit("submission", kind=kind, side=side, command_id=command_id, planned_ns=nominal,
                      started_ns=now, returned_ns=self.clock_ns(), receipts=receipts,
                      mode="policy" if active else "hold", requested_action=action,
                      submitted_targets=poses if kind == "arm" else {side: hand.last_target})
            if any(not r.accepted for r in receipts.values()):
                raise RuntimeError(f"Partial/rejected {kind} submission: {receipts}")

    def _loop(self, kind, side=None):
        period = 1 / self.config[f"{kind}_hz"]
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                self.tick(kind, side)
            except Exception as error:
                try:
                    self.pause(error, fault=True)
                except Exception:
                    # A failed recorder must not prevent the driver stop request above.
                    pass
            deadline += period
            if deadline < time.monotonic():
                deadline = time.monotonic()
            self._stop.wait(max(0., deadline - time.monotonic()))

    def close(self):
        try:
            self.pause("Runtime closing")
        finally:
            self._stop.set()
            for thread in self._threads:
                thread.join(timeout=2)
