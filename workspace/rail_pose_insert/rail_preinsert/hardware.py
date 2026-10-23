"""Device I/O stays alive while perception, annotation and planning are blocked."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import threading
import time

import numpy as np

from .core import difference, matrix, vector


class Hardware:
    def __init__(self, config, kinematics, emit, *, motion=False, devices=None):
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import load_config as arm_config
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import TianjiDriver
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import load_config as hand_config, sdk_user_name
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import WujiSdkSession, WujiHandDriver
        from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile
        self.config, self.kinematics, self.emit, self.motion = config, kinematics, emit, motion
        arms, hands = arm_config(config["arm_config"]), hand_config(config["hand_config"])
        parameters = deepcopy(arms["profile"]["parameters"])
        parameters["active_arms"] = ["left"]
        parameters["arms"] = {"left": parameters["arms"]["left"]}
        self.arm_profile = ControlProfile("rail-preinsert-left", "cartesian_impedance", parameters)
        self.hand_profile = ControlProfile("rail-preinsert-grasp", "mit", hands["parameters"])
        if devices is None:
            self.session = WujiSdkSession(user_name=sdk_user_name(hands))
            self.driver = TianjiDriver(arms["controller_ip"], record_force=True)
            self.hands = {side: WujiHandDriver(side, hands["devices"][side]["hand"], feedback_hz=200)
                          for side in ("left", "right")}
        else:
            self.session, self.driver, self.hands = devices
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.history = deque(maxlen=65000)  # >300 s, including manual mask annotation.
        self.advanced = {}
        self.latest = None
        self.error = None
        self.paused = False
        self.arm_enabled = False
        self.hand_enabled = False
        self.arm_target = None
        self.hand_target = None
        self.hand_blend = None
        self.trajectory = None
        self.trajectory_started = None
        self.trajectory_done = False
        self.execution_reference = None
        self.threads = []

    def start(self):
        try:
            self.session.open()
            self.driver.start()
            for hand in self.hands.values():
                hand.start()
            thread = threading.Thread(target=self._collect, daemon=True, name="rail-measured-state")
            self.threads.append(thread); thread.start()
            deadline = time.monotonic() + 10
            while self.latest is None:
                if self.error:
                    raise RuntimeError(self.error)
                if time.monotonic() > deadline:
                    raise TimeoutError("Waiting for both arm and hand measured states")
                time.sleep(.01)
            return self
        except BaseException:
            self.close(); raise

    def _collect(self):
        previous, last_log = None, 0.
        while not self.stop.is_set():
            try:
                sample = self.driver.get_latest()
                hands = {side: hand.get_latest() for side, hand in self.hands.items()}
                if sample is not None and sample.header.valid and all(
                        v is not None and v.header.valid for v in hands.values()):
                    now = time.monotonic_ns()
                    for side in ("left", "right"):
                        sequence = sample.payload.arms[side].source_sequence
                        old = self.advanced.get(side)
                        if old is None or sequence != old[0]:
                            self.advanced[side] = (sequence, sample.header.received_monotonic_ns)
                        if now - self.advanced[side][1] > self.config["state_max_age_s"] * 1e9:
                            raise RuntimeError(f"{side} arm feedback counter stalled")
                    stamps = [sample.header.received_monotonic_ns] + [v.header.received_monotonic_ns for v in hands.values()]
                    if any(now - s > self.config["state_max_age_s"] * 1e9 for s in stamps):
                        raise RuntimeError("Measured state is stale")
                    q = np.concatenate([sample.payload.arms[s].joints.position_rad for s in ("left", "right")]
                                       + [hands["left"].payload.position_rad, hands["right"].payload.position_rad])
                    vector(q, 54, "measured joint state")
                    value = {"timestamp_ns": self.advanced["left"][1], "full_joints": q,
                             "T_B_F": matrix(self.kinematics.fk("left", tuple(q[:7]))),
                             "hand_timestamp_ns": hands["left"].header.received_monotonic_ns}
                    with self.lock:
                        self.latest = value
                        if value["timestamp_ns"] != previous:
                            self.history.append((value["timestamp_ns"], q[:7].copy()))
                            previous = value["timestamp_ns"]
                    if self.arm_enabled:
                        health = self.driver.health()
                        if not health.ready:
                            raise RuntimeError(health.detail)
                    if time.monotonic() - last_log > .05:
                        self.emit("measured", timestamp_ns=int(value["timestamp_ns"]), full_joints=q.tolist(),
                                  T_B_F=value["T_B_F"].tolist(),
                                  wrench_sensor_axes={side: list(sample.payload.arms[side].wrench)
                                      if sample.payload.arms[side].wrench is not None else None
                                      for side in ("left", "right")})
                        last_log = time.monotonic()
            except Exception as error:
                self.pause(str(error))
            self.stop.wait(.01)

    def snapshot(self):
        with self.lock:
            if self.latest is None:
                raise RuntimeError("Measured state unavailable")
            value = {key: val.copy() if isinstance(val, np.ndarray) else val for key, val in self.latest.items()}
            if time.monotonic_ns() - value["timestamp_ns"] > self.config["state_max_age_s"] * 1e9:
                raise RuntimeError("Measured arm state is stale")
            if time.monotonic_ns() - value["hand_timestamp_ns"] > self.config["state_max_age_s"] * 1e9:
                raise RuntimeError("Measured hand state is stale")
            return value

    def engage(self):
        if not self.motion:
            raise RuntimeError("Read-only mode cannot engage devices")
        if self.paused:
            raise RuntimeError("Paused/faulted trial cannot automatically re-engage")
        self.snapshot()
        if not self.arm_enabled:
            self.driver.configure(self.arm_profile)
            self.driver.engage()
            self.arm_target = np.asarray(self.driver.engagement_sample.payload.arms["left"].joints.position_rad)
            self.arm_enabled = True
            thread = threading.Thread(target=self._arm_loop, daemon=True, name="rail-arm-200hz")
            self.threads.append(thread); thread.start()
        if not self.hand_enabled:
            self.hands["left"].configure(self.hand_profile)
            self.hands["left"].engage(cancelled=lambda: self.paused or self.stop.is_set())
            self.hand_target = np.asarray(self.hands["left"].last_target.position_rad)
            self.hand_enabled = True
            thread = threading.Thread(target=self._hand_loop, daemon=True, name="rail-hand-120hz")
            self.threads.append(thread); thread.start()

    def _arm_loop(self):
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import TianjiJointCommand
        from diffusion_policy.third_party.bimanual_teleop.types import DeviceCommand
        previous = time.monotonic()
        while not self.stop.is_set():
            started = time.monotonic()
            if self.paused:
                self.stop.wait(.005); continue
            try:
                if started - previous > .05:
                    raise RuntimeError("Arm control scheduling gap >50 ms")
                previous = started
                state = self.snapshot()
                with self.lock:
                    if self.trajectory is not None:
                        elapsed = started - self.trajectory_started
                        self.arm_target = self.trajectory.at(elapsed)
                        self.trajectory_done = elapsed >= self.trajectory.duration
                    q = vector(self.arm_target, 7, "arm command").copy()
                    reference = self.execution_reference
                arm = self.kinematics.model.arm("left")
                if (np.any(q < arm.lower_rad) or np.any(q > arm.upper_rad)
                        or 1.025*abs(q[5]) + abs(q[6]) >= np.radians(110.5)):
                    raise RuntimeError("Arm command outside joint/coupled wrist limits")
                if np.max(np.abs(state["full_joints"][:7] - q)) > self.config["arm_tracking_error_rad"]:
                    raise RuntimeError("Actual left arm departed from commanded trajectory")
                if reference is not None:
                    if np.max(np.abs(state["full_joints"][7:14] - reference[7:14])) > self.config["right_arm_drift_rad"]:
                        raise RuntimeError("Right arm moved from recorded planning state")
                    if np.max(np.abs(state["full_joints"][14:] - reference[14:])) > self.config["hand_shape_drift_rad"]:
                        raise RuntimeError("Measured hand shape departed from recorded planning state")
                now = time.monotonic_ns()
                command = DeviceCommand("tianji", f"rail-{now}", TianjiJointCommand({"left": tuple(q)}, {}),
                                        (), now, now + 50_000_000, self.arm_profile.profile_id)
                result = self.driver.submit(command)
                if not result.accepted:
                    raise RuntimeError(result.reason)
            except Exception as error:
                self.pause(str(error))
            self.stop.wait(max(0., .005 - (time.monotonic() - started)))

    def _hand_loop(self):
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_NAMES
        from diffusion_policy.third_party.bimanual_teleop.types import DeviceCommand, JointTarget
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                with self.lock:
                    if self.hand_blend is not None and not self.paused:
                        before, after, when, duration = self.hand_blend
                        u = np.clip((started - when)/duration, 0, 1)
                        alpha = 10*u**3 - 15*u**4 + 6*u**5
                        self.hand_target = before + alpha*(after - before)
                    target = self.hand_target.copy()
                health = self.hands["left"].health(check_latch=True)
                if not health.ready:
                    raise RuntimeError(health.detail)
                now = time.monotonic_ns()
                command = DeviceCommand(self.hands["left"].device_id, f"rail-hand-{now}",
                                        JointTarget(JOINT_NAMES, tuple(target)), (), now, now + 50_000_000,
                                        self.hand_profile.profile_id)
                result = self.hands["left"].submit(command)
                if not result.accepted:
                    raise RuntimeError(result.reason)
            except Exception as error:
                self.pause(str(error))
            self.stop.wait(max(0., 1/120 - (time.monotonic() - started)))

    def move_hand(self, target):
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_LIMITS_RAD
        target = vector(target, 20, "hand target")
        if any(not low <= q <= high for q, (low, high) in zip(target, JOINT_LIMITS_RAD)):
            raise ValueError("Recorded candidate exceeds Hand2 joint limits")
        self.engage()
        with self.lock:
            # The first target was seeded from measured joints by Hand2.engage,
            # including its <=0.5 degree correction at mechanical boundaries.
            current = self.hand_target.copy()
            self.hand_blend = (current.copy(), target.copy(), time.monotonic(), self.config["hand_transition_s"])

    def wait_hand(self):
        with self.lock:
            end = self.hand_blend[2] + self.hand_blend[3]
        previous, stable, deadline = None, None, end + 10
        while time.monotonic() < deadline:
            self.raise_if_fault()
            current = self.snapshot()["full_joints"][14:34]
            now = time.monotonic()
            if now >= end and previous is not None and np.max(np.abs(current - previous)) < .01:
                stable = stable or now
                if now - stable >= self.config["settle_s"]:
                    return
            else:
                stable = None
            previous = current
            self.stop.wait(.05)
        raise TimeoutError("Hand did not settle after closure")

    def start_path(self, trajectory, reference):
        self.raise_if_fault()
        if not self.motion or not self.arm_enabled or not self.hand_enabled:
            raise RuntimeError("Arm and hand must be engaged before transport")
        actual = self.snapshot()["full_joints"]
        if np.max(np.abs(actual - reference)) > .01:
            raise ValueError("Robot moved after planning; measure and plan again")
        with self.lock:
            self.execution_reference = np.asarray(reference).copy()
            self.trajectory, self.trajectory_started = trajectory, time.monotonic()
            self.trajectory_done = False

    def wait_path(self, target_flange):
        stable, deadline = None, time.monotonic() + self.trajectory.duration + 15
        while time.monotonic() < deadline:
            self.raise_if_fault()
            state = self.snapshot()
            distance, angle = difference(state["T_B_F"], target_flange)
            if (self.trajectory_done and distance <= self.config["preinsert_position_tolerance_m"]
                    and angle <= self.config["preinsert_rotation_tolerance_rad"]):
                stable = stable or time.monotonic()
                if time.monotonic() - stable >= self.config["settle_s"]:
                    return
            else:
                stable = None
            self.stop.wait(.02)
        raise TimeoutError("Measured flange did not settle at preinsert")

    def stationary_history(self, observation):
        stamps = [int(v["timestamp_ns"]) for v in observation["records"]]
        with self.lock:
            history = [(int(t), q.tolist()) for t, q in self.history]
        since = [q for t, q in history if t >= min(stamps)]
        if not since or np.ptp(np.asarray(since), axis=0).max() > .01:
            raise ValueError("Arm moved during image acquisition/inference; repeat stationary measurement")
        state = self.snapshot()
        return {"timestamp_ns": int(state["timestamp_ns"]), "full_joints": state["full_joints"].tolist(),
                "left_joint_history": history, "test_data": False}

    def raise_if_fault(self):
        if self.stop.is_set() or self.paused:
            raise RuntimeError(self.error or "Trial paused")

    def pause(self, reason):
        with self.lock:
            first = not self.paused
            self.paused, self.error = True, str(reason)
            self.hand_blend = None  # Freeze its last command, preserving grip.
            self.trajectory = None
        if first:
            try:
                self.emit("paused", reason=str(reason), hand_action="retain_last_target")
            finally:
                if self.arm_enabled:
                    self.driver.request_hold(str(reason))

    def close(self):
        self.stop.set()
        errors = []
        if self.arm_enabled:
            try:
                self.driver.request_hold("experiment exit")
            except Exception as error:
                errors.append(str(error))
        for thread in self.threads:
            if thread is not threading.current_thread():
                thread.join(timeout=2)
        for close in [*[hand.close for hand in self.hands.values()], self.driver.close, self.session.close]:
            try:
                close()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("Device cleanup: " + "; ".join(errors))
