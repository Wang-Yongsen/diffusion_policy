"""Single right-hand owner with independent arm and hand holding loops."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from rail_preinsert.core import difference, matrix, transform, vector
from .logic import KeyboardJog


def _telemetry_value(value):
    """Keep even invalid fault samples JSON-safe without hiding missing values."""
    if isinstance(value, np.ndarray):
        return _telemetry_value(value.tolist())
    if isinstance(value, dict):
        return {key: _telemetry_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_telemetry_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def initial_receiving_pose(config, kinematics):
    """Fixed target anchored to the original ready pose, independent of feedback."""
    reference = np.radians(vector(config["initial_reference_joints_deg"], 7))
    target = matrix(kinematics.fk("right", tuple(reference)))
    target[:3, 3] += vector(config["initial_offset_base_m"], 3)
    return target


class JointTrackingGuard:
    """Allow brief tracking lag; stop sustained or large departures."""
    def __init__(self, config):
        self.limit = config["arm_tracking_error_rad"]
        self.timeout = config["arm_tracking_error_timeout_s"]
        self.hard_limit = config["arm_tracking_hard_error_rad"]
        self.reset()

    def reset(self):
        self.since = None
        self.peak = 0.

    def update(self, error, now):
        if error is None or not np.isfinite(error) or error < 0:
            raise RuntimeError("Invalid right-arm joint tracking feedback")
        if error > self.limit:
            if self.since is None:
                self.since = now
            self.peak = max(self.peak, error)
        else:
            self.reset()
        elapsed = 0. if self.since is None else max(0., now-self.since)
        reason = ("hard_limit" if error > self.hard_limit else
                  "persistent" if self.since is not None and elapsed >= self.timeout else None)
        return {"error_rad": error, "limit_rad": self.limit, "hard_limit_rad": self.hard_limit,
                "exceeded_for_s": elapsed, "timeout_s": self.timeout,
                "peak_error_rad": self.peak, "reason": reason,
                "exceeded": self.since is not None}


class RightHardware:
    def __init__(self, config, kinematics, emit, *, motion=False, devices=None, executor=None):
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import load_config as arm_config
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import load_config as hand_config, sdk_user_name
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import TianjiDriver
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import WujiSdkSession, WujiHandDriver
        from diffusion_policy.third_party.bimanual_teleop.control.arm.cartesian import TianjiCartesianExecutor
        from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile
        self.config, self.kinematics, self.emit, self.motion = config, kinematics, emit, motion
        # Position-only insertion must also start without an available force channel.
        self.record_force = config.get("lowering", {}).get("mode", "force_spiral") == "force_spiral"
        arms, hands = arm_config(config["arm_config"]), hand_config(config["hand_config"])
        parameters = deepcopy(arms["profile"]["parameters"])
        parameters["active_arms"] = ["right"]
        parameters["arms"] = {"right": parameters["arms"]["right"]}
        for key in ("velocity_ratio", "acceleration_ratio"):
            parameters["arms"]["right"][key] = min(parameters["arms"]["right"][key], config[key])
        self.arm_profile = ControlProfile("rail-receive-right", "cartesian_impedance", parameters)
        self.position_ratios = tuple(parameters["arms"]["right"][k]
                                     for k in ("velocity_ratio", "acceleration_ratio"))
        self.hand_profile = ControlProfile("rail-receive-right-hand", "mit", config["hand_parameters"])
        if devices is None:
            self.session = WujiSdkSession(user_name=sdk_user_name(hands))
            self.driver = TianjiDriver(arms["controller_ip"], record_force=self.record_force)
            self.hand = WujiHandDriver("right", hands["devices"]["right"]["hand"],
                                      feedback_hz=hands["feedback_hz"], timeout_s=hands["hand_timeout_s"])
        else:
            self.session, self.driver, self.hand = devices
        self.executor = executor or TianjiCartesianExecutor(self.driver, kinematics)
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.paused = False
        self.error = None
        self.arm_enabled = self.hand_enabled = False
        self.hand_target = self.hand_blend = self.jogger = None
        self.arm_path = None
        self.arm_mode = "cartesian_impedance"
        self.position_request = None
        self.position_active = False
        self.position_cancel = threading.Event()
        self.position_done = threading.Event()
        self.insertion_cancel = threading.Event()
        self.insertion_request = None
        self.insertion_controller = None
        self.insertion_profile = None
        self._insertion_report = None
        self._insertion_trace = deque(maxlen=2048)
        self._insertion_seq = 0
        self._insertion_last_log = 0.
        self._insertion_motion_sent = False
        self._insertion_motion_attempted = False
        self._insertion_acceptance_unknown = False
        self._insertion_last_state = None
        self._insertion_guard = JointTrackingGuard(config)
        self.initial_move_active = False
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
                    time.sleep(.02)
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
        return {"timestamp_ns": stamp, "joints_rad": q, "hand_position_rad": hand_q,
                "hand_current_a": hand.payload.motor_current_a,
                "T_B_F": matrix(self.kinematics.fk("right", tuple(q))),
                "arm_control_state": sample.payload.arms["right"].state,
                "arm_low_speed": sample.payload.arms["right"].low_speed,
                "arm_impedance_type": sample.payload.arms["right"].impedance_type,
                "right_sequence": sample.payload.arms["right"].source_sequence,
                "wrench": sample.payload.arms["right"].wrench if self.record_force else None}

    def engage(self):
        if not self.motion or self.paused:
            raise RuntimeError("Read-only or paused receiver cannot engage")
        self.snapshot()
        # Construct and validate the constrained servo before starting the watchdog.
        self.executor.configure(self.arm_profile)
        self.executor.engage()
        self.arm_enabled = True
        if self.paused or self.stop.is_set():
            self.executor.request_hold("Receiving engagement cancelled")
            raise RuntimeError("Receiving engagement cancelled")
        self.jogger = KeyboardJog(self.config, matrix(self.executor.engagement_poses["right"]))
        arm_thread = threading.Thread(target=self._arm_loop, daemon=True, name="receive-right-arm")
        self.threads.append(arm_thread)
        arm_thread.start()
        self.raise_if_fault()
        self.hand.configure(self.hand_profile)
        self.hand.engage(cancelled=lambda: self.paused or self.stop.is_set())
        with self.lock:
            self.hand_target = np.asarray(self.hand.last_target.position_rad).copy()
            self.hand_enabled = True
        hand_thread = threading.Thread(target=self._hand_loop, daemon=True, name="receive-right-hand")
        self.threads.append(hand_thread)
        hand_thread.start()

    def _arm_loop(self):
        from diffusion_policy.third_party.bimanual_teleop.types import Pose, RobotTarget
        last_log = 0.0
        guard = JointTrackingGuard(self.config)
        checked_path, previously_exceeded = None, False
        while not self.stop.is_set():
            started = time.monotonic()
            if self.paused:
                self.stop.wait(.005)
                continue
            try:
                state = self.snapshot()
                with self.lock:
                    inserting = self.arm_mode in ("switching_to_insertion", "insertion")
                    position_mode = self.arm_mode == "position"
                    T = self.jogger.target.copy()
                    path = self.arm_path
                    initial_move = self.initial_move_active
                if inserting:
                    # Feedback may arrive after the cycle's start timestamp.
                    self._insertion_step(state, time.monotonic())
                    self.stop.wait(max(0., .005 - (time.monotonic() - started)))
                    continue
                if position_mode:
                    self._position_step()
                    self.stop.wait(.01)
                    continue
                if path is not None:
                    trajectory, began = path
                    T = matrix(self.kinematics.fk("right", tuple(trajectory.at(started-began))))
                pose = Pose("tianji_right_base", "tianji_right_flange", tuple(T[:3, 3]),
                            tuple(Rotation.from_matrix(T[:3, :3]).as_quat()))
                now = time.monotonic_ns()
                result = self.executor.submit(RobotTarget(f"receive-{now}", {"right": pose}, (), now,
                                                          now + 50_000_000, self.arm_profile.profile_id),
                                              before_submit=self.raise_if_fault)
                if not result.accepted:
                    raise RuntimeError(result.reason)
                tracking = dict(self.executor.tracking_status["right"])
                if path is not checked_path:
                    guard.reset()
                    checked_path, previously_exceeded = path, False
                check = None
                # Startup/jog retain their existing endpoint/controller checks.
                # During loaded transport distinguish brief lag from persistent
                # tracking failure, with a separate immediate upper bound.
                if path is not None and not initial_move:
                    check = guard.update(tracking["max_joint_tracking_error_rad"], time.monotonic())
                    tracking["joint_guard"] = check
                else:
                    guard.reset()
                changed = check is not None and check["exceeded"] != previously_exceeded
                if started - last_log >= .05 or changed or (check and check["reason"]):
                    # These are measured FK errors, distinct from the servo's
                    # position_error_m (requested pose versus command reference).
                    tracking["requested_position_error_m"], tracking["requested_rotation_error_rad"] = difference(T, state["T_B_F"])
                    applied = self.executor.applied_poses.get("right")
                    if applied is not None:
                        applied_T = matrix(applied)
                        tracking["applied_T_B_F"] = applied_T.tolist()
                        tracking["applied_position_error_m"], tracking["applied_rotation_error_rad"] = difference(applied_T, state["T_B_F"])
                    serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in state.items()}
                    self.emit("right_state", **serializable, requested_T_B_F=T.tolist(), tracking=tracking)
                    last_log = started
                if changed:
                    self.emit("right_tracking_warning" if check["exceeded"] else "right_tracking_recovered",
                              **check)
                previously_exceeded = bool(check and check["exceeded"])
                if check and check["reason"]:
                    self.emit("right_tracking_fault", **check, tracking=tracking)
                    condition = (f"immediate limit {guard.hard_limit:.6f} rad" if check["reason"] == "hard_limit" else
                        f"limit {guard.limit:.6f} rad continuously for {check['exceeded_for_s']:.3f} s "
                        f"(allowed {guard.timeout:.3f} s)")
                    threshold = guard.hard_limit if check["reason"] == "hard_limit" else guard.limit
                    flange_detail = (f"; flange-to-command error {tracking['applied_position_error_m']*1000:.2f} mm / "
                        f"{np.degrees(tracking['applied_rotation_error_rad']):.2f} deg"
                        if "applied_position_error_m" in tracking else "")
                    raise RuntimeError("Right arm tracking fault: "
                        f"joint error {check['error_rad']:.9g} rad exceeds {condition} "
                        f"(excess {check['error_rad']-threshold:.3g} rad){flange_detail}")
            except Exception as error:
                if self._insertion_report is not None:
                    self._record_insertion_failure(error)
                self.pause(str(error))
            self.stop.wait(max(0.0, .005 - (time.monotonic() - started)))

    def move_to_position(self, initial_joints, target_joints, *, timeout_s,
                         velocity_ratio=None, acceleration_ratio=None):
        """Queue one position target for the sole arm owner; never stream IK points."""
        from rail_preinsert.planner import Planner
        self.raise_if_fault()
        if not self.motion or not self.arm_enabled:
            raise RuntimeError("Right arm must be engaged for a position move")
        with self.lock:
            if self._insertion_report is not None:
                raise RuntimeError("Insertion owns the arm; position moves cannot resume in this session")
        initial, target = vector(initial_joints, 7), vector(target_joints, 7)
        Planner(self.config["move_above"], self.kinematics, side="right").validate_path([initial, target])
        if isinstance(timeout_s, bool) or not np.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Position move timeout must be finite and positive")
        ratios = []
        for name, value in zip(("velocity_ratio", "acceleration_ratio"),
                               (velocity_ratio, acceleration_ratio)):
            ceiling = self.arm_profile.parameters["arms"]["right"][name]
            value = ceiling if value is None else value
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"Position {name} must be an integer from 1 to {ceiling}")
            ratios.append(value)
        state = self.snapshot()
        if np.max(np.abs(state["joints_rad"]-initial)) > self.config["arm_tracking_error_rad"]:
            raise ValueError("Right arm moved since planning; estimate and plan again")
        endpoint = matrix(self.kinematics.fk("right", tuple(target)))
        with self.lock:
            self.raise_if_fault()
            if self._insertion_report is not None:
                raise RuntimeError("Insertion owns the arm; position moves cannot resume in this session")
            if self.position_active:
                raise RuntimeError("A right-arm position move is already active")
            self.position_cancel.clear()
            self.position_done.clear()
            self.position_active = True
            self.position_request = (initial.copy(), target.copy(), timeout_s, tuple(ratios))
            self.arm_path = None
            self.jogger.target = endpoint
            self.arm_mode = "position"
        self.emit("right_position_queued", target_rad=target.tolist(), control_mode="position",
                  timeout_s=timeout_s)

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
            parameters["arms"]["right"].update(velocity_ratio=ratios[0], acceleration_ratio=ratios[1])
            # configure confirms ratio readback. It does not re-engage impedance.
            self.driver.configure(ControlProfile("rail-receive-right-position", self.arm_profile.mode, parameters))
            self.raise_if_fault()
            self.position_ratios = ratios
        state = self.snapshot()
        if np.max(np.abs(state["joints_rad"]-initial)) > self.config["arm_tracking_error_rad"]:
            raise ValueError("Right arm moved during position-mode handover; plan again")
        self.emit("right_position_started", target_rad=target.tolist(), control_mode="position",
                  velocity_ratio=ratios[0], acceleration_ratio=ratios[1])
        self.driver.move_joints("right", tuple(target), cancel=self.position_cancel, timeout_s=timeout)
        # move_joints returns only after fresh position-mode, target echo,
        # low-speed and measured joint arrival checks, not just command acceptance.
        with self.lock:
            self.raise_if_fault()
            self.position_active = False
            self.position_done.set()
        self.emit("right_position_complete", target_rad=target.tolist(), control_mode="position")

    def position_move_finished(self):
        self.raise_if_fault()
        return self.position_done.is_set()

    def _validate_insertion_start(self, request, state):
        from .insertion import InsertionController, validate_settings
        self.raise_if_fault()
        validate_settings(request["settings"], require_ready=True)
        if any(request["settings"]["impedance"][key] > self.config[key]
               for key in ("velocity_ratio", "acceleration_ratio")):
            raise ValueError("Insertion speed/acceleration ratios cannot exceed receiver limits")
        if not self.motion or not self.arm_enabled or not self.hand_enabled:
            raise RuntimeError("Insertion requires an engaged right arm and holding right hand")
        if (self.position_active or self.position_request is not None
                or not self.position_done.is_set()
                or state.get("arm_control_state") != 1 or state.get("arm_low_speed") != 1):
            raise RuntimeError("Insertion requires a completed, stationary position move")
        error = difference(transform(state["T_B_F"]) @ request["grasp"], request["above"])
        settings = self.config["move_above"]
        if error[0] > settings["position_tolerance_m"] or error[1] > settings["rotation_tolerance_rad"]:
            raise RuntimeError("Right arm left the verified above-goal insertion start")
        vector(state.get("wrench"), 6, "insertion wrench")
        if not isinstance(state.get("right_sequence"), (int, np.integer)):
            raise RuntimeError("Insertion requires a right-arm feedback sequence")
        if request["above"][1, 3] <= request["goal"][1, 3]:
            raise ValueError("Insertion start must be above the goal along right-base +Y")
        # Validate all pose/depth constraints before configuration can write to
        # the controller; the actual instance is recreated from engagement FK.
        preflight = InsertionController(request["settings"], request["grasp"],
                                        request["goal"], state["T_B_F"])
        preflight.hard_guard(state)

    def queue_insertion(self, request):
        """Queue an explicit handover; the caller never writes to the arm SDK."""
        request = deepcopy(request)
        for name in ("grasp", "goal", "above"):
            request[name] = transform(request[name]).copy()
        state = self.snapshot()
        self._validate_insertion_start(request, state)
        with self.lock:
            self.raise_if_fault()
            if self.arm_mode != "position" or self._insertion_report is not None:
                raise RuntimeError("An insertion attempt is already latched in this session")
            self.insertion_cancel.clear()
            self.insertion_request = request
            self.arm_path = None
            self.arm_mode = "switching_to_insertion"
            self._insertion_report = {"status": "switching_to_insertion", "reason": None,
                "insertion_executed": False, "assembly_verified": False,
                "automatic_release": False, "hand_action": "keep_grasp", "revision": 0}

    def insertion_snapshot(self, *, after_seq=0):
        """Return a bounded copy; GUI/file I/O never holds the control state lock."""
        with self.lock:
            if self._insertion_report is None:
                return None
            report = deepcopy(self._insertion_report)
            report["trace"] = [deepcopy(row) for row in self._insertion_trace
                               if row["trace_seq"] > after_seq]
            report["trace_first_seq"] = (self._insertion_trace[0]["trace_seq"]
                                         if self._insertion_trace else 0)
            return report

    def _check_insertion_cancel(self):
        self.raise_if_fault()
        if self.insertion_cancel.is_set():
            raise RuntimeError("Insertion cancelled")

    def _begin_insertion(self):
        from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile
        from .insertion import InsertionController
        with self.lock:
            request, self.insertion_request = self.insertion_request, None
        if request is None:
            raise RuntimeError("Missing insertion handover request")
        self._check_insertion_cancel()
        self._validate_insertion_start(request, self.snapshot())
        parameters = deepcopy(self.arm_profile.parameters)
        parameters["arms"]["right"].update({key: request["settings"]["impedance"][key]
            for key in ("stiffness", "damping", "nullspace_stiffness", "nullspace_damping",
                        "velocity_ratio", "acceleration_ratio")})
        profile = ControlProfile("rail-right-force-insertion", "cartesian_impedance", parameters)
        self.executor.request_hold("Switching right arm to contact insertion")
        self._check_insertion_cancel()
        self.executor.configure(profile, cancel=self.insertion_cancel)
        self._check_insertion_cancel()
        self.executor.engage(cancel=self.insertion_cancel)
        self._check_insertion_cancel()
        seed = matrix(self.executor.engagement_poses["right"])
        # Engagement's measured sample is the only valid new servo origin.
        controller = InsertionController(request["settings"], request["grasp"],
                                         request["goal"], seed, now=time.monotonic())
        with self.lock:
            self._check_insertion_cancel()
            self.insertion_profile = profile
            self.insertion_controller = controller
            self.jogger.target = seed.copy()
            self.arm_path = None
        self._submit_insertion_target(seed)
        with self.lock:
            self._check_insertion_cancel()
            self.arm_mode = "insertion"
        self._publish_insertion(self.snapshot(), seed, time.monotonic())

    def _insertion_final_guard(self):
        try:
            self._check_insertion_cancel()
            state = self.snapshot()
            self._insertion_last_state = deepcopy(state)
            if state.get("arm_control_state") != 3 or state.get("arm_impedance_type") != 2:
                raise RuntimeError("Right arm left Cartesian impedance insertion mode")
            self.insertion_controller.hard_guard(state, now=time.monotonic())
        except Exception as error:
            self._record_insertion_failure(error)
            raise

    def _record_insertion_failure(self, error):
        """Preserve the actual rejected sample; never issue an extra command."""
        controller = self.insertion_controller
        details = controller.summary() if controller is not None else {}
        state = self._insertion_last_state or {}
        details.update(hard_fault=str(error), fault_sample={
            key: state.get(key) for key in ("timestamp_ns", "right_sequence", "T_B_F", "wrench")})
        if "wrench" in state:
            details["raw_wrench"] = state["wrench"]
        if "T_B_F" in state:
            details["measured_T_B_F"] = state["T_B_F"]
        with self.lock:
            if self._insertion_report is not None:
                # Operator pause wins over a racing owner error.
                status, reason = self._insertion_report["status"], self._insertion_report.get("reason")
                self._insertion_report.update(_telemetry_value(details))
                self._insertion_report.update(status=status, reason=reason,
                    **self._insertion_evidence())

    def _insertion_evidence(self):
        return {"insertion_executed": (True if self._insertion_motion_sent else
                    None if self._insertion_acceptance_unknown else False),
                "motion_submission_attempted": self._insertion_motion_attempted,
                "motion_acceptance_unknown": self._insertion_acceptance_unknown}

    def _update_insertion_evidence(self, *, attempted=False, sent=False, unknown=False):
        with self.lock:
            self._insertion_motion_attempted |= attempted
            self._insertion_motion_sent |= sent
            self._insertion_acceptance_unknown |= unknown
            if self._insertion_report is not None:
                self._insertion_report.update(self._insertion_evidence())
                # A cancellation may be observed after the SDK accepted motion
                # but before the executor published its new reference. Preserve
                # this uncertainty even if pause already wrote its first trace.
                if self.paused:
                    self._insertion_report["revision"] += 1
                    self._insertion_seq += 1
                    self._insertion_trace.append(dict(deepcopy(self._insertion_report),
                        trace_seq=self._insertion_seq, host_monotonic_ns=time.monotonic_ns()))

    def _submit_insertion_target(self, target):
        from diffusion_policy.third_party.bimanual_teleop.types import Pose, RobotTarget
        self._insertion_final_guard()
        moving_reference = difference(target, self.snapshot()["T_B_F"])[0] > 1e-9
        pose = Pose("tianji_right_base", "tianji_right_flange", tuple(target[:3, 3]),
                    tuple(Rotation.from_matrix(target[:3, :3]).as_quat()))
        now = time.monotonic_ns()
        if moving_reference:
            self._update_insertion_evidence(attempted=True)
        try:
            result = self.executor.submit(RobotTarget(f"insert-{now}", {"right": pose}, (),
                now, now + 50_000_000, self.insertion_profile.profile_id),
                before_submit=self._insertion_final_guard)
        except Exception:
            if moving_reference:
                self._update_insertion_evidence(unknown=True)
            raise
        if not result.accepted:
            if moving_reference:
                self._update_insertion_evidence(unknown=True)
            raise RuntimeError(result.reason)
        if moving_reference:
            self._update_insertion_evidence(sent=True)
        self._check_insertion_cancel()
        applied = matrix(self.executor.applied_poses["right"])
        self.insertion_controller.accepted(applied)
        tracking = self.executor.tracking_status.get("right", {})
        check = self._insertion_guard.update(tracking.get("max_joint_tracking_error_rad"), time.monotonic())
        if check["reason"]:
            raise RuntimeError(f"Insertion joint tracking exceeded {check['reason']}: {check['error_rad']:.6g} rad")

    def _insertion_step(self, state, now):
        self._check_insertion_cancel()
        self._insertion_last_state = deepcopy(state)
        if self.arm_mode == "switching_to_insertion":
            self._begin_insertion()
            return
        state = dict(state)
        state["applied_target"] = matrix(self.executor.applied_poses["right"])
        state["servo_limited"] = self.executor.tracking_status.get("right", {}).get("limited", False)
        target = self.insertion_controller.step(state, now=now)
        self._submit_insertion_target(target)
        self._publish_insertion(state, target, now)

    def _publish_insertion(self, state, target, now):
        # The final guard may have received a newer sample during IK. Log that
        # exact raw sample beside its unfiltered compensated wrench; the filter
        # retains the distinct timestamp of the last controller step.
        state = self._insertion_last_state or state
        report = self.insertion_controller.summary()
        report.update(**self._insertion_evidence(),
            assembly_verified=False, automatic_release=False, hand_action="keep_grasp",
            side="right", control_mode="cartesian_impedance", collision_checked=False,
            host_monotonic_ns=time.monotonic_ns(), raw_wrench=list(state["wrench"]),
            filtered_wrench_timestamp_ns=self.insertion_controller.last_stamp,
            timestamp_ns=int(state["timestamp_ns"]), right_sequence=int(state["right_sequence"]),
            measured_T_B_F=transform(state["T_B_F"]).tolist(), requested_T_B_F=target.tolist())
        with self.lock:
            if self.paused or self.stop.is_set():
                return
            previous = self._insertion_report or {}
            changed = report["status"] != previous.get("status")
            report["revision"] = previous.get("revision", 0) + 1
            self._insertion_report = report
            if changed or now-self._insertion_last_log >= .05:
                self._insertion_seq += 1
                self._insertion_trace.append(dict(deepcopy(report), trace_seq=self._insertion_seq))
                self._insertion_last_log = now

    def _hand_loop(self):
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_NAMES
        from diffusion_policy.third_party.bimanual_teleop.types import DeviceCommand, JointTarget
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                with self.lock:
                    if self.hand_blend is not None and not self.paused:
                        before, after, when, duration = self.hand_blend
                        u = np.clip((started - when) / duration, 0.0, 1.0)
                        self.hand_target = before + (10*u**3 - 15*u**4 + 6*u**5) * (after - before)
                    target = self.hand_target.copy()
                health = self.hand.health(check_latch=True)
                if not health.ready:
                    raise RuntimeError(health.detail)
                now = time.monotonic_ns()
                command = DeviceCommand(self.hand.device_id, f"receive-hand-{now}",
                                        JointTarget(JOINT_NAMES, tuple(target)), (), now, now + 50_000_000,
                                        self.hand_profile.profile_id)
                result = self.hand.submit(command)
                if not result.accepted:
                    raise RuntimeError(result.reason)
            except Exception as error:
                self.pause(str(error))
            self.stop.wait(max(0.0, 1/120 - (time.monotonic() - started)))

    def begin_hand(self, target):
        from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import JOINT_LIMITS_RAD
        self.raise_if_fault()
        if not self.hand_enabled:
            raise RuntimeError("Right hand is not engaged")
        target = vector(target, 20)
        if any(not low <= q <= high for q, (low, high) in zip(target, JOINT_LIMITS_RAD)):
            raise ValueError("Right-hand target exceeds limits")
        with self.lock:
            if self._insertion_report is not None:
                raise RuntimeError("Hand posture is locked during and after an insertion attempt")
            self.hand_blend = (self.hand_target.copy(), target.copy(), time.monotonic(), self.config["hand_transition_s"])

    def hand_transition_finished(self, samples, now, *, require_target=False):
        """Require measured stability, without claiming physical grasp success."""
        with self.lock:
            end = self.hand_blend[2] + self.hand_blend[3]
        if now > end + self.config["hand_settle_timeout_s"]:
            raise TimeoutError("Right hand did not settle after its transition")
        recent = [q for stamp, q in samples if stamp >= now - self.config["hand_settle_s"] - .1]
        covered = samples and now - samples[0][0] >= self.config["hand_settle_s"]
        reached = bool(recent) and (not require_target or np.max(np.abs(recent[-1] - self.hand_blend[1])) < .10)
        return bool(now >= end + self.config["hand_settle_s"] and covered and len(recent) >= 3
                    and reached and np.ptp(np.asarray(recent), axis=0).max() < .02)

    def jog(self, key):
        self.raise_if_fault()
        state = self.snapshot()
        with self.lock:
            if self._insertion_report is not None:
                return None
            if self.arm_mode == "position" and self.position_active:
                return None
            self.arm_path = None
            if key in ("w", "s", "r", "f", "d", "a", "i", "k", "j", "l", "u", "o"):
                # Each key is a step from the measured flange, so repeats cannot
                # accumulate an unexecuted distant goal in impedance control.
                self.jogger.target = state["T_B_F"].copy()
            target = self.jogger.propose(key)
            if target is not None and self.arm_mode == "position":
                from rail_preinsert.planner import Planner
                path = Planner(self.config["move_above"], self.kinematics, side="right").plan(
                    state["joints_rad"], target)
                self.move_to_position(path.points[0], path.points[-1],
                                      timeout_s=self.config["move_above"]["position_move_timeout_s"])
        if target is not None:
            self.emit("keyboard_jog", key=key, target=target.tolist(), scale=self.jogger.scale)
        return target

    def follow_path(self, trajectory):
        """The existing arm owner follows planned FK references; hand holding continues."""
        self.raise_if_fault()
        if not self.motion or not self.arm_enabled:
            raise RuntimeError("Right arm must be engaged to follow a path")
        state = self.snapshot()
        if np.max(np.abs(state["joints_rad"]-trajectory.points[0])) > self.config["arm_tracking_error_rad"]:
            raise ValueError("Right arm moved since planning; estimate and plan again")
        endpoint = matrix(self.kinematics.fk("right", tuple(trajectory.points[-1])))
        with self.lock:
            self.raise_if_fault()
            if self._insertion_report is not None:
                raise RuntimeError("Insertion owns the arm; Cartesian transport cannot resume")
            if self.arm_mode == "position":
                raise RuntimeError("Cannot stream a Cartesian path after switching to position control")
            self.jogger.target = endpoint
            self.arm_path = (trajectory, time.monotonic())
        self.emit("right_path_started", duration_s=trajectory.duration, waypoints=len(trajectory.points))

    def move_to_initial_pose(self):
        """Plan from current joints to the same absolute receiving pose each time."""
        from rail_preinsert.planner import Planner
        self.raise_if_fault()
        if not self.motion or not self.arm_enabled:
            raise RuntimeError("Right arm must be engaged for startup movement")
        self.initial_move_active = True
        try:
            state = self.snapshot()
            target = initial_receiving_pose(self.config, self.kinematics)
            arm = self.kinematics.model.arm("right")
            profile = self.arm_profile.parameters["arms"]["right"]
            s = self.config["move_above"]
            settings = dict(s,
                joint_speed_rad_s=min(s["joint_speed_rad_s"], min(arm.max_joint_velocity_rad_s)
                                      * profile["velocity_ratio"]/100),
                joint_acceleration_rad_s2=min(s["joint_acceleration_rad_s2"],
                    min(np.radians(row[3]) for row in arm.limits_native)*profile["acceleration_ratio"]/100))
            path = Planner(settings, self.kinematics, side="right").plan(state["joints_rad"], target)
            self.emit("right_initial_pose", reference_joints_deg=self.config["initial_reference_joints_deg"],
                      offset_base_m=self.config["initial_offset_base_m"], target=target.tolist())
            print(f"启动回位：目标 XYZ [m] = {np.round(target[:3, 3], 4).tolist()}，"
                  f"参考轨迹约 {path.duration:.1f} 秒；按实测末端到位判断。", flush=True)
            self.follow_path(path)
            began, settled = time.monotonic(), None
            last_report = began-2.
            position_limit = self.config["initial_position_tolerance_m"]
            rotation_limit = self.config["initial_rotation_tolerance_rad"]
            while True:
                self.raise_if_fault()
                now = time.monotonic()
                distance, angle = difference(self.snapshot()["T_B_F"], target)
                if now-last_report >= 2.:
                    print(f"启动回位：已等待 {now-began:.1f} 秒，位置差 {distance*1000:.2f} mm"
                          f" / {position_limit*1000:.1f} mm，姿态差 {np.degrees(angle):.2f}°"
                          f" / {np.degrees(rotation_limit):.1f}°。", flush=True)
                    last_report = now
                # The servo follows Cartesian references, not the planner's
                # redundant joint branch. Finish on measured pose + stability;
                # do not wait for a nominal joint trajectory already achieved.
                if distance <= position_limit and angle <= rotation_limit:
                    settled = now if settled is None else settled
                    if now-settled >= s["settle_s"]:
                        with self.lock:
                            self.jogger.target = target.copy()
                            self.arm_path = None
                        self.emit("right_initial_pose_complete", target=target.tolist())
                        print("右臂已到接料初始位置，开始张手。", flush=True)
                        return
                else:
                    settled = None
                if now-began > path.duration+s["reach_timeout_s"]:
                    raise TimeoutError("Right arm did not reach the fixed initial receiving pose: "
                        f"position error {distance*1000:.2f} mm (limit {position_limit*1000:.1f} mm), "
                        f"rotation error {np.degrees(angle):.2f} deg (limit {np.degrees(rotation_limit):.1f} deg)")
                self.stop.wait(.01)
        finally:
            self.initial_move_active = False

    def raise_if_fault(self):
        if self.paused or self.stop.is_set():
            raise RuntimeError(self.error or "Receiver paused")

    def pause(self, reason):
        # Set before taking locks: mode confirmation must observe cancellation
        # even when request_hold has to wait for the driver's command lock.
        self.insertion_cancel.set()
        with self.lock:
            first = not self.paused
            self.paused = True
            if first:
                self.error = str(reason)
            self.position_cancel.set()
            self.hand_blend = None
            self.insertion_request = None
            if first and self._insertion_report is not None:
                previous = self._insertion_report
                self._insertion_report = dict(previous, status="paused", reason=str(reason),
                    previous_status=previous["status"], **self._insertion_evidence(),
                    revision=previous.get("revision", 0)+1)
                self._insertion_seq += 1
                self._insertion_trace.append(dict(deepcopy(self._insertion_report),
                    trace_seq=self._insertion_seq, host_monotonic_ns=time.monotonic_ns()))
        if first:
            try:
                self.emit("paused", reason=str(reason), hand_action="retain_last_target")
            finally:
                if self.arm_enabled:
                    self.executor.request_hold(str(reason))

    def close(self):
        self.stop.set()
        self.insertion_cancel.set()
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
