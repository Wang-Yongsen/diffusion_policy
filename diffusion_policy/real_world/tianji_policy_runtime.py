"""Absolute 52-D targets, monotonic scheduling and independent device loops."""

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.third_party.bimanual_teleop.types import DeviceCommand, JointTarget, Pose, RobotTarget
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_LIMITS_RAD, JOINT_NAMES

SIDES = ("left", "right")
INFERENCE_MODES = ("async", "sync")
# Equals the Tianji driver watchdog, which bounds every accepted target anyway;
# arm IK plus submission has been measured up to 56 ms under parent-process load.
HOLD_TTL_NS = 80_000_000


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


def _prediction_chunk(actions):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 52 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Require nonempty finite N x 52 predictions")
    return actions


def future_actions(actions, anchor_ns, now_ns, hz=30, margin_s=.01):
    actions = _prediction_chunk(actions)
    stamps = int(anchor_ns) + np.rint(np.arange(len(actions)) * 1e9 / hz).astype(np.int64)
    keep = stamps > now_ns + round(margin_s * 1e9)
    if not keep.any():
        raise RuntimeError("Prediction has no future action coverage "
            f"(observation age={(now_ns - anchor_ns) / 1e6:.1f} ms, "
            f"last action offset={(stamps[-1] - anchor_ns) / 1e6:.1f} ms, "
            f"submit margin={margin_s * 1000:.1f} ms)")
    return actions[keep].copy(), stamps[keep], int((~keep).sum())


def retimed_actions(actions, now_ns, hz=30, margin_s=.01):
    """Synchronous chunks start after submission; valid only if the robot held still during inference."""
    actions = _prediction_chunk(actions)
    stamps = int(now_ns) + round(margin_s * 1e9) + np.rint(np.arange(len(actions)) * 1e9 / hz).astype(np.int64)
    return actions.copy(), stamps, 0


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
        self.sync = config["inference_mode"] == "sync"
        # Sync only: engaged and holding the last target until the next prediction is submitted.
        self.sync_waiting = False
        self.chunk_completed_ns = None

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
            self.sync_waiting = self.sync and arm_ready
            self.chunk_completed_ns = self.clock_ns() if self.sync_waiting else None

    def submit(self, actions, anchor_ns):
        with self._lock:
            if self.fault or self._stop.is_set():
                raise RuntimeError(self.fault or "Runtime closed")
            if self.sync:
                if self.active:
                    raise RuntimeError("Synchronous chunk is still executing")
                actions, times, dropped = retimed_actions(actions, self.clock_ns(),
                    self.config["policy_hz"], self.config["submit_margin_s"])
            else:
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
            self.sync_waiting = False
            self.chunk_completed_ns = None
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
            self.sync_waiting = False
            self.chunk_completed_ns = None
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
        self.emit("pause", reason=str(reason), fault=fault)

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

    def _complete_chunk(self, now):
        self._last_action = self._actions[-1].copy()
        self._last_nominal_ns = int(self._times[-1])
        self._actions = np.empty((0, 52))
        self._times = np.empty(0, dtype=np.int64)
        self.active = False
        self.sync_waiting = True
        self.chunk_completed_ns = int(now)
        self.emit("chunk_complete", generation=self._generation, completed_ns=int(now))

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
                if active and self.sync and now >= self._times[-1] + round(1e9 / self.config["policy_hz"]):
                    self._complete_chunk(now)
                    active = False
                waiting = self.sync_waiting
                if active or waiting:
                    problem = self.health()
                    if problem:
                        raise RuntimeError(problem)
                if active:
                    action = self._target(now)
                    expiry = int(self._times[-1]) + round(1e9 / self.config["policy_hz"])
                    if self.sync:
                        # Sync holds the final step after the chunk, so no step must expire at coverage end.
                        expiry = max(expiry, now + HOLD_TTL_NS)
                else:
                    # Holding is a distinct mode with the last accepted hand target.
                    action, expiry = self._last_action, now + HOLD_TTL_NS
                nominal = self._last_nominal_ns
            if action is None:
                return
            poses, targets = split_action(action)
            command_id = f"{generation}-{side or kind}-{now}"
            if kind == "arm":
                if not self.arm_ready or not self.arm.driver.engaged:
                    if active or waiting:
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
                    if active or waiting:
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
            self.emit("submission", kind=kind, side=side, mode="policy" if active else "hold",
                      planned_ns=nominal, started_ns=now, returned_ns=self.clock_ns())
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
