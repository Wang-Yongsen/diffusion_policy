"""Right-hand owner with left-arm startup return and independent holding loops."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from rail_common.core import difference, matrix, vector
from .logic import KeyboardJog


def initial_receiving_pose(config, kinematics, *, side="right"):
    """Fixed target anchored to the original ready pose, independent of feedback."""
    if side == "right":
        reference, offset = config["initial_reference_joints_deg"], config["initial_offset_base_m"]
    elif side == "left":
        reference, offset = (
            config["left_initial_pose"][key] for key in ("reference_joints_deg", "offset_base_m")
        )
    else:
        raise ValueError("Initial receiving pose requires side left or right")
    target = matrix(kinematics.fk(side, tuple(np.radians(vector(reference, 7)))))
    target[:3, 3] += vector(offset, 3)
    return target


class RightHardware:
    def __init__(self, config, kinematics, emit, *, motion=False, devices=None, executor=None):
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import (
            load_config as arm_config,
        )
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import (
            load_config as hand_config,
            sdk_user_name,
        )
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import TianjiDriver
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import (
            WujiSdkSession,
            WujiHandDriver,
        )
        from diffusion_policy.third_party.bimanual_teleop.control.arm.cartesian import (
            TianjiCartesianExecutor,
        )
        from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile

        self.config, self.kinematics, self.emit, self.motion = config, kinematics, emit, motion
        # Position-only insertion must also start without an available force channel.
        arms, hands = arm_config(config["arm_config"]), hand_config(config["hand_config"])
        left_parameters = deepcopy(arms["profile"]["parameters"])
        left_parameters["active_arms"] = ["left"]
        left_parameters["arms"] = {"left": left_parameters["arms"]["left"]}
        for key in ("velocity_ratio", "acceleration_ratio"):
            left_parameters["arms"]["left"][key] = min(
                left_parameters["arms"]["left"][key], config[key]
            )
        self.left_initial_profile = ControlProfile(
            "rail-receive-left-initial", "cartesian_impedance", left_parameters
        )
        parameters = deepcopy(arms["profile"]["parameters"])
        parameters["active_arms"] = ["right"]
        parameters["arms"] = {"right": parameters["arms"]["right"]}
        for key in ("velocity_ratio", "acceleration_ratio"):
            parameters["arms"]["right"][key] = min(parameters["arms"]["right"][key], config[key])
        self.arm_profile = ControlProfile("rail-receive-right", "cartesian_impedance", parameters)
        self.position_ratios = tuple(
            parameters["arms"]["right"][k] for k in ("velocity_ratio", "acceleration_ratio")
        )
        self.hand_profile = ControlProfile(
            "rail-receive-right-hand", "mit", config["hand_parameters"]
        )
        if devices is None:
            self.session = WujiSdkSession(user_name=sdk_user_name(hands))
            self.driver = TianjiDriver(arms["controller_ip"], record_force=False)
            self.hand = WujiHandDriver(
                "right",
                hands["devices"]["right"]["hand"],
                feedback_hz=hands["feedback_hz"],
                timeout_s=hands["hand_timeout_s"],
            )
        else:
            self.session, self.driver, self.hand = devices
        self.executor = executor or TianjiCartesianExecutor(self.driver, kinematics)
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.paused = False
        self.error = None
        self.arm_enabled = self.hand_enabled = False
        self.hand_target = self.hand_blend = self.jogger = None
        self.hand_command_completed_at = None
        self.arm_mode = "cartesian_impedance"
        self.position_request = None
        self.position_active = False
        self.position_cancel = threading.Event()
        self.position_done = threading.Event()
        self.initial_move_active = False
        self.left_initial_move_active = False
        self.history = deque(maxlen=6000)
        self.threads = []

    def start(self):
        try:
            self.session.open()
            self.driver.start()
            self.hand.start()
            deadline = time.monotonic() + 10
            while True:
                try:
                    self.snapshot()
                    return self
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        sample, hand = self.driver.get_latest(), self.hand.get_latest()
        now = time.monotonic_ns()
        if sample is None or hand is None or not hand.header.valid:
            raise RuntimeError("Waiting for right-arm and right-hand feedback")
        health = self.driver.health(sides=("right",))
        if not health.ready:
            raise RuntimeError(health.detail)
        for stamp in (sample.header.received_monotonic_ns, hand.header.received_monotonic_ns):
            if not 0 <= now - stamp <= self.config["state_max_age_s"] * 1e9:
                raise RuntimeError("Right-arm/hand feedback is stale")
        q = vector(sample.payload.arms["right"].joints.position_rad, 7)
        hand_q = vector(hand.payload.position_rad, 20)
        stamp = sample.header.received_monotonic_ns
        with self.lock:
            if not self.history or stamp > self.history[-1][0]:
                self.history.append((stamp, q.copy()))
        return {
            "timestamp_ns": stamp,
            "joints_rad": q,
            "hand_position_rad": hand_q,
            "hand_current_a": hand.payload.motor_current_a,
            "T_B_F": matrix(self.kinematics.fk("right", tuple(q))),
            "arm_control_state": sample.payload.arms["right"].state,
            "arm_low_speed": sample.payload.arms["right"].low_speed,
            "arm_impedance_type": sample.payload.arms["right"].impedance_type,
            "right_sequence": sample.payload.arms["right"].source_sequence,
            "wrench": None,
        }

    def engage(self):
        if not self.motion or self.paused:
            raise RuntimeError("Read-only or paused receiver cannot engage")
        self.raise_if_fault()
        self.snapshot()
        # Construct and validate the constrained servo before starting the watchdog.
        self.executor.configure(self.arm_profile, cancel=self.position_cancel)
        self.raise_if_fault()
        # The executor and driver seed impedance from the same measured sample,
        # not the planned home endpoint or an old Cartesian command reference.
        self.executor.engage(cancel=self.position_cancel)
        self.arm_enabled = True
        if self.paused or self.stop.is_set():
            self.executor.request_hold("Receiving engagement cancelled")
            raise RuntimeError("Receiving engagement cancelled")
        reference = matrix(self.executor.engagement_poses["right"])
        self.jogger = KeyboardJog(self.config, reference)
        self.arm_mode = "cartesian_impedance"
        arm_thread = threading.Thread(target=self._arm_loop, daemon=True, name="receive-right-arm")
        self.threads.append(arm_thread)
        arm_thread.start()
        self.raise_if_fault()
        self.emit(
            "right_impedance_started",
            control_mode="cartesian_impedance",
            reference_T_B_F=reference.tolist(),
        )
        print("右臂已切换为笛卡尔阻抗控制，以切换时的实测位姿作为保持目标。", flush=True)
        self.hand.configure(self.hand_profile)
        self.hand.engage(cancelled=lambda: self.paused or self.stop.is_set())
        with self.lock:
            self.hand_target = np.asarray(self.hand.last_target.position_rad).copy()
            self.hand_enabled = True
        hand_thread = threading.Thread(
            target=self._hand_loop, daemon=True, name="receive-right-hand"
        )
        self.threads.append(hand_thread)
        hand_thread.start()

    def _arm_loop(self):
        from diffusion_policy.third_party.bimanual_teleop.types import Pose, RobotTarget

        while not self.stop.is_set():
            started = time.monotonic()
            if self.paused:
                self.stop.wait(0.005)
                continue
            try:
                self.snapshot()  # Enforce device health and feedback freshness.
                with self.lock:
                    position_mode = self.arm_mode == "position"
                    target = self.jogger.target.copy()
                if position_mode:
                    self._position_step()
                    self.stop.wait(0.01)
                    continue
                pose = Pose(
                    "tianji_right_base",
                    "tianji_right_flange",
                    tuple(target[:3, 3]),
                    tuple(Rotation.from_matrix(target[:3, :3]).as_quat()),
                )
                now = time.monotonic_ns()
                result = self.executor.submit(
                    RobotTarget(
                        f"receive-{now}",
                        {"right": pose},
                        (),
                        now,
                        now + 50_000_000,
                        self.arm_profile.profile_id,
                    ),
                    before_submit=self.raise_if_fault,
                )
                if not result.accepted:
                    raise RuntimeError(result.reason)
            except Exception as error:
                self.pause(str(error))
            self.stop.wait(max(0.0, 0.005 - (time.monotonic() - started)))

    def move_to_position(
        self,
        initial_joints,
        target_joints,
        *,
        timeout_s,
        velocity_ratio=None,
        acceleration_ratio=None,
    ):
        """Queue one position target for the sole arm owner; never stream IK points."""
        from rail_common.planner import Planner

        self.raise_if_fault()
        if not self.motion or not self.arm_enabled:
            raise RuntimeError("Right arm must be engaged for a position move")
        initial, target = vector(initial_joints, 7), vector(target_joints, 7)
        Planner(self.config["move_above"], self.kinematics, side="right").validate_path(
            [initial, target]
        )
        if isinstance(timeout_s, bool) or not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Position move timeout must be finite and positive")
        ratios = []
        for name, value in zip(
            ("velocity_ratio", "acceleration_ratio"), (velocity_ratio, acceleration_ratio)
        ):
            ceiling = self.arm_profile.parameters["arms"]["right"][name]
            value = ceiling if value is None else value
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"Position {name} must be an integer from 1 to {ceiling}")
            ratios.append(value)
        state = self.snapshot()
        if np.max(np.abs(state["joints_rad"] - initial)) > self.config["arm_tracking_error_rad"]:
            raise ValueError("Right arm moved since planning; estimate and plan again")
        endpoint = matrix(self.kinematics.fk("right", tuple(target)))
        with self.lock:
            self.raise_if_fault()
            if self.position_active:
                raise RuntimeError("A right-arm position move is already active")
            self.position_cancel.clear()
            self.position_done.clear()
            self.position_active = True
            self.position_request = (initial.copy(), target.copy(), timeout_s, tuple(ratios))
            self.jogger.target = endpoint
            self.arm_mode = "position"

    def _position_step(self):
        """Run only on the arm thread, so Cartesian and position writes cannot overlap."""
        with self.lock:
            self.raise_if_fault()
            request, self.position_request = self.position_request, None
        if request is None:
            # Once complete, keep the controller's position hold, never re-engage
            # impedance. The normal snapshot still checks health/freshness.
            if self.position_done.is_set() and self.snapshot()["arm_control_state"] != 1:
                raise RuntimeError("Right arm left position mode while holding the target")
            return
        initial, target, timeout, ratios = request
        # This also invalidates any old Cartesian reference. SDK move_joints
        # switches at measured joints, confirms state=1, then sends the goal once.
        self.executor.request_hold("Switching right arm to point-to-point position control")
        self.raise_if_fault()
        if ratios != self.position_ratios:
            from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile

            parameters = deepcopy(self.arm_profile.parameters)
            parameters["arms"]["right"].update(
                velocity_ratio=ratios[0], acceleration_ratio=ratios[1]
            )
            # configure confirms ratio readback. It does not re-engage impedance.
            self.driver.configure(
                ControlProfile("rail-receive-right-position", self.arm_profile.mode, parameters)
            )
            self.raise_if_fault()
            self.position_ratios = ratios
        state = self.snapshot()
        if np.max(np.abs(state["joints_rad"] - initial)) > self.config["arm_tracking_error_rad"]:
            raise ValueError("Right arm moved during position-mode handover; plan again")
        self.driver.move_joints(
            "right", tuple(target), cancel=self.position_cancel, timeout_s=timeout
        )
        # move_joints returns only after fresh position-mode, target echo,
        # low-speed and measured joint arrival checks, not just command acceptance.
        with self.lock:
            self.raise_if_fault()
            self.position_active = False
            self.position_done.set()

    def position_move_finished(self):
        self.raise_if_fault()
        return self.position_done.is_set()

    def _hand_loop(self):
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_NAMES
        from diffusion_policy.third_party.bimanual_teleop.types import DeviceCommand, JointTarget

        while not self.stop.is_set():
            started = time.monotonic()
            try:
                with self.lock:
                    blend = self.hand_blend
                    if blend is not None and not self.paused:
                        before, after, when, duration = blend
                        u = np.clip((started - when) / duration, 0.0, 1.0)
                        self.hand_target = before + (10 * u**3 - 15 * u**4 + 6 * u**5) * (
                            after - before
                        )
                    target = self.hand_target.copy()
                health = self.hand.health(check_latch=True)
                if not health.ready:
                    raise RuntimeError(health.detail)
                now = time.monotonic_ns()
                command = DeviceCommand(
                    self.hand.device_id,
                    f"receive-hand-{now}",
                    JointTarget(JOINT_NAMES, tuple(target)),
                    (),
                    now,
                    now + 50_000_000,
                    self.hand_profile.profile_id,
                )
                result = self.hand.submit(command)
                if not result.accepted:
                    raise RuntimeError(result.reason)
                with self.lock:
                    if (
                        blend is not None
                        and blend is self.hand_blend
                        and not self.paused
                        and started >= blend[2] + blend[3]
                        and self.hand_command_completed_at is None
                    ):
                        self.hand_command_completed_at = time.monotonic()
            except Exception as error:
                self.pause(str(error))
            self.stop.wait(max(0.0, 1 / 120 - (time.monotonic() - started)))

    def begin_hand(self, target):
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import (
            JOINT_LIMITS_RAD,
        )

        self.raise_if_fault()
        if not self.hand_enabled:
            raise RuntimeError("Right hand is not engaged")
        target = vector(target, 20)
        if any(not low <= q <= high for q, (low, high) in zip(target, JOINT_LIMITS_RAD)):
            raise ValueError("Right-hand target exceeds limits")
        with self.lock:
            self.hand_blend = (
                self.hand_target.copy(),
                target.copy(),
                time.monotonic(),
                self.config["hand_transition_s"],
            )
            self.hand_command_completed_at = None

    def hand_command_finished(self, now):
        """Final interpolated target was submitted, then held; not measured arrival."""
        self.raise_if_fault()
        with self.lock:
            if not self.hand_enabled or self.hand_blend is None:
                raise RuntimeError("No active right-hand transition")
            end = self.hand_blend[2] + self.hand_blend[3]
            completed_at = self.hand_command_completed_at
        if completed_at is None and now > end + self.config["hand_settle_timeout_s"]:
            raise TimeoutError("Right-hand final target was not submitted")
        return completed_at is not None and now - completed_at >= self.config["hand_settle_s"]

    def hand_transition_finished(self, samples, now, *, require_target=False):
        """Opening must dwell near its target; a loaded grasp only needs to stop moving."""
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_NAMES

        with self.lock:
            end = self.hand_blend[2] + self.hand_blend[3]
            target = self.hand_blend[1].copy()
        window = [
            (stamp, vector(q, 20))
            for stamp, q in samples
            if max(end, now - self.config["hand_settle_s"] - 0.1) <= stamp <= now
        ]
        recent = np.asarray([q for _, q in window])
        covered = (
            len(window) >= 3
            and window[-1][0] - window[0][0] >= self.config["hand_settle_s"]
            and now - window[-1][0] <= self.config["state_max_age_s"]
        )
        if now > end + self.config["hand_settle_timeout_s"]:
            detail = "no recent hand feedback"
            if len(recent):
                errors = np.abs(recent[-1] - target)
                ranges = np.ptp(np.asarray(recent), axis=0)
                worst, moving = int(np.argmax(errors)), int(np.argmax(ranges))
                detail = (
                    f"target error {JOINT_NAMES[worst]}={errors[worst]:.4f} rad "
                    f"(measured={recent[-1][worst]:.4f}, target={target[worst]:.4f}, limit=0.1000); "
                    f"motion range {JOINT_NAMES[moving]}={ranges[moving]:.4f} rad; "
                    f"required={'target dwell' if require_target else 'motion range < 0.0200 rad'}, "
                    f"feedback window covered={bool(covered)}"
                )
            raise TimeoutError(f"Right hand did not settle after its transition: {detail}")
        if not covered:
            return False
        # Small servo fluctuations inside the open pose tolerance do not mean
        # the hand is still closed. Require the entire dwell, not a single sample.
        return bool(
            np.max(np.abs(recent - target)) < 0.10
            if require_target
            else np.ptp(recent, axis=0).max() < 0.02
        )

    def jog(self, key):
        self.raise_if_fault()
        state = self.snapshot()
        with self.lock:
            if self.arm_mode == "position" and self.position_active:
                return None
            if key in ("w", "s", "r", "f", "d", "a", "i", "k", "j", "l", "u", "o"):
                # Each key is a step from the measured flange, so repeats cannot
                # accumulate an unexecuted distant goal in impedance control.
                self.jogger.target = state["T_B_F"].copy()
            target = self.jogger.propose(key)
            if target is not None and self.arm_mode == "position":
                from rail_common.planner import Planner

                path = Planner(self.config["move_above"], self.kinematics, side="right").plan(
                    state["joints_rad"], target
                )
                self.move_to_position(
                    path.points[0],
                    path.points[-1],
                    timeout_s=self.config["move_above"]["position_move_timeout_s"],
                )
        if target is not None:
            self.emit("keyboard_jog", key=key, target=target.tolist(), scale=self.jogger.scale)
        return target

    def move_left_to_initial_pose(self):
        """Return directly to the shifted fixed left target before right-arm engagement."""
        from rail_common.planner import Planner

        self.raise_if_fault()
        if not self.motion or self.arm_enabled:
            raise RuntimeError(
                "Left startup return requires motion mode before right-arm engagement"
            )
        self.left_initial_move_active = True
        try:
            target = initial_receiving_pose(self.config, self.kinematics, side="left")
            state = self._left_initial_state()
            settings = self.config["move_above"]
            path = Planner(settings, self.kinematics, side="left").plan(state["joints_rad"], target)
            self.raise_if_fault()
            self.driver.configure(self.left_initial_profile, cancel=self.position_cancel)
            self.raise_if_fault()
            self.emit(
                "left_initial_pose", **self.config["left_initial_pose"], target=target.tolist()
            )
            print(
                f"左臂直接回到新初始位置：目标 XYZ [m] = {np.round(target[:3, 3], 4).tolist()}；"
                "保持原初始化朝向，不累积偏移。",
                flush=True,
            )
            # Submit only the new endpoint, never the old ready pose or a relative move.
            self.driver.move_joints(
                "left",
                tuple(path.points[-1]),
                cancel=self.position_cancel,
                timeout_s=settings["position_move_timeout_s"],
            )
            began, settled = time.monotonic(), None
            while True:
                self.raise_if_fault()
                state = self._left_initial_state()
                now = time.monotonic()
                distance, angle = difference(state["T_B_F"], target)
                if (
                    distance <= self.config["initial_position_tolerance_m"]
                    and angle <= self.config["initial_rotation_tolerance_rad"]
                ):
                    settled = now if settled is None else settled
                    if now - settled >= settings["settle_s"]:
                        self.emit("left_initial_pose_complete", target=target.tolist())
                        print("左臂已到新初始位置。", flush=True)
                        return
                else:
                    settled = None
                if now - began > settings["reach_timeout_s"]:
                    raise TimeoutError(
                        "Left arm did not reach the fixed initial receiving pose: "
                        f"position error {distance*1000:.2f} mm, rotation error {np.degrees(angle):.2f} deg"
                    )
                self.stop.wait(0.01)
        except BaseException as error:
            self.driver.request_hold(str(error) or "Left startup return interrupted")
            raise
        finally:
            self.left_initial_move_active = False

    def _left_initial_state(self):
        health = self.driver.health(sides=("left",))
        if not health.ready:
            raise RuntimeError(health.detail)
        sample = self.driver.get_latest()
        if (
            sample is None
            or not sample.header.valid
            or not 0
            <= time.monotonic_ns() - sample.header.received_monotonic_ns
            <= self.config["state_max_age_s"] * 1e9
        ):
            raise RuntimeError("Left-arm startup feedback is invalid or stale")
        q = vector(sample.payload.arms["left"].joints.position_rad, 7)
        return {"joints_rad": q, "T_B_F": matrix(self.kinematics.fk("left", tuple(q)))}

    def move_to_initial_pose(self):
        """Return in SDK position mode before starting the impedance/hand loops."""
        from rail_common.planner import Planner

        self.raise_if_fault()
        if not self.motion or self.arm_enabled:
            raise RuntimeError(
                "Right startup return requires motion mode before right-arm engagement"
            )
        self.initial_move_active = True
        try:
            state = self.snapshot()
            target = initial_receiving_pose(self.config, self.kinematics)
            s = self.config["move_above"]
            path = Planner(s, self.kinematics, side="right").plan(state["joints_rad"], target)
            self.raise_if_fault()
            self.driver.configure(self.arm_profile, cancel=self.position_cancel)
            self.raise_if_fault()
            self.emit(
                "right_initial_pose",
                reference_joints_deg=self.config["initial_reference_joints_deg"],
                offset_base_m=self.config["initial_offset_base_m"],
                target=target.tolist(),
                control_mode="position",
            )
            print(
                f"启动回位：目标 XYZ [m] = {np.round(target[:3, 3], 4).tolist()}，"
                "使用 SDK 位置控制；到位停稳后再切换为阻抗控制。",
                flush=True,
            )
            # move_joints confirms position mode, target echo, measured joint
            # arrival and low speed before returning. No Cartesian writer runs yet.
            self.driver.move_joints(
                "right",
                tuple(path.points[-1]),
                cancel=self.position_cancel,
                timeout_s=s["position_move_timeout_s"],
            )
            began, settled = time.monotonic(), None
            position_limit = self.config["initial_position_tolerance_m"]
            rotation_limit = self.config["initial_rotation_tolerance_rad"]
            while True:
                self.raise_if_fault()
                now = time.monotonic()
                state = self.snapshot()
                if state["arm_control_state"] != 1:
                    raise RuntimeError("Right arm left position mode during startup arrival check")
                distance, angle = difference(state["T_B_F"], target)
                if (
                    state["arm_low_speed"] == 1
                    and distance <= position_limit
                    and angle <= rotation_limit
                ):
                    settled = now if settled is None else settled
                    if now - settled >= s["settle_s"]:
                        self.emit(
                            "right_initial_pose_complete",
                            target=target.tolist(),
                            control_mode="position",
                            measured_T_B_F=state["T_B_F"].tolist(),
                            position_error_m=distance,
                            rotation_error_rad=angle,
                        )
                        print("右臂位置回位已到位并停稳，准备切换为阻抗控制。", flush=True)
                        return
                else:
                    settled = None
                if now - began > s["reach_timeout_s"]:
                    raise TimeoutError(
                        "Right arm did not reach the fixed initial receiving pose: "
                        f"position error {distance*1000:.2f} mm (limit {position_limit*1000:.1f} mm), "
                        f"rotation error {np.degrees(angle):.2f} deg (limit {np.degrees(rotation_limit):.1f} deg)"
                    )
                self.stop.wait(0.01)
        except BaseException as error:
            self.driver.request_hold(str(error) or "Right startup return interrupted")
            raise
        finally:
            self.initial_move_active = False

    def raise_if_fault(self):
        if self.paused or self.stop.is_set():
            raise RuntimeError(self.error or "Receiver paused")

    def pause(self, reason):
        # Set before taking locks: mode confirmation must observe cancellation
        # even when request_hold has to wait for the driver's command lock.
        self.position_cancel.set()
        with self.lock:
            first = not self.paused
            self.paused = True
            if first:
                self.error = str(reason)
            self.hand_blend = None
        if first:
            try:
                self.emit("paused", reason=str(reason), hand_action="retain_last_target")
            finally:
                if self.arm_enabled:
                    self.executor.request_hold(str(reason))
                elif self.left_initial_move_active or self.initial_move_active:
                    self.driver.request_hold(str(reason))

    def close(self):
        self.stop.set()
        self.position_cancel.set()
        errors = []
        if self.arm_enabled:
            try:
                self.executor.request_hold("receiver exit")
            except Exception as error:
                errors.append(str(error))
        for thread in self.threads:
            thread.join(timeout=2)
        for close in (self.hand.close, self.driver.close, self.session.close):
            try:
                close()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("Receiver cleanup: " + "; ".join(errors))
