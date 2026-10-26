"""Fixed-attitude, feedback-gated insertion controller; no device access.

All poses use the right SDK base (+Y up). T_F_S maps the sensor's documented
measurement frame/origin into the flange. Positive compensated wrench denotes
external force on the tool. Baseline subtraction is valid only at fixed attitude
and with an unchanged, rigid grasp; it is not general gravity compensation.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np

from rail_preinsert.core import difference, transform, vector


DEFAULT_INSERTION = {
    "T_F_S": None, "wrench_sign": None, "hole_midpoint_cad_m": None,
    "impedance": None,
    "baseline_duration_s": 1.0, "baseline_min_samples": 30,
    "baseline_timeout_s": 5.0, "state_max_age_s": .1, "max_control_dt_s": .05,
    "force_filter_tau_s": .05, "contact_confirm_s": .1,
    "candidate_confirm_s": .2, "settle_s": .3,
    "baseline_force_std_n": None, "baseline_torque_std_nm": None,
    "baseline_position_tolerance_m": None, "orientation_tolerance_rad": None,
    "hand_shape_tolerance_rad": None, "max_reference_lead_m": None,
    "max_tracking_error_m": None,
    "approach_speed_m_s": None, "insert_speed_m_s": None, "retract_speed_m_s": None,
    "admittance_m_s_n": None, "contact_force_n": None, "target_force_n": None,
    "max_axial_force_n": None, "max_lateral_force_n": None, "max_tilt_torque_nm": None,
    "hard_force_n": None, "hard_torque_nm": None,
    # Optional absolute sensor limits include gravity and zero bias. They are
    # never treated as compensated contact force and never inferred from CAD.
    "raw_force_limit_n": None, "raw_torque_limit_nm": None,
    "pitch_m": None, "max_radius_m": None, "lateral_speed_m_s": None,
    "candidate_depth_m": None, "candidate_force_drop_n": None, "candidate_confirm_depth_m": None,
    "candidate_timeout_s": None, "max_descent_m": None,
    "endpoint_tolerance_m": None, "progress_epsilon_m": None,
    "stall_timeout_s": None, "phase_timeout_s": None,
    "total_timeout_s": None, "retract_timeout_s": None,
    "retract_unload_force_n": None, "retract_unload_torque_nm": None,
}
_OPTIONAL = {"raw_force_limit_n", "raw_torque_limit_nm"}
_NON_NUMERIC = {"T_F_S", "wrench_sign", "hole_midpoint_cad_m", "impedance"}


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"insertion.{name} must be finite and positive")


def validate_settings(settings, require_ready=False):
    """Return missing field names; malformed supplied values always raise.

    Missing calibration/load limits are allowed when loading a preview config,
    but require_ready=True refuses them before any control-mode change.
    """
    missing = []
    for name in DEFAULT_INSERTION:
        value = settings.get(name)
        if value is None:
            if name not in _OPTIONAL:
                missing.append(name)
        elif name not in _NON_NUMERIC:
            _positive(value, name)
    if settings.get("T_F_S") is not None:
        transform(settings["T_F_S"], "insertion.T_F_S")
    if settings.get("hole_midpoint_cad_m") is not None:
        vector(settings["hole_midpoint_cad_m"], 3, "insertion.hole_midpoint_cad_m")
    if settings.get("wrench_sign") is not None:
        if isinstance(settings["wrench_sign"], bool) or settings["wrench_sign"] not in (-1, 1):
            raise ValueError("insertion.wrench_sign must be +1 or -1")
    if settings.get("baseline_min_samples") is not None:
        if type(settings["baseline_min_samples"]) is not int or settings["baseline_min_samples"] < 2:
            raise ValueError("insertion.baseline_min_samples must be an integer >= 2")
    imp = settings.get("impedance")
    if imp is not None:
        if not isinstance(imp, dict):
            raise ValueError("insertion.impedance must be a mapping")
        for key, size in (("stiffness", 6), ("damping", 6)):
            if imp.get(key) is None:
                missing.append(f"impedance.{key}")
            else:
                values = vector(imp[key], size, f"insertion.impedance.{key}")
                if np.any(values < 0) or key == "damping" and np.any(values > 1):
                    raise ValueError(f"Invalid insertion.impedance.{key}")
        for key in ("nullspace_stiffness", "nullspace_damping", "velocity_ratio", "acceleration_ratio"):
            value = imp.get(key)
            if value is None:
                missing.append(f"impedance.{key}")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid insertion.impedance.{key}")
            elif key == "nullspace_damping" and value > 1:
                raise ValueError("insertion.impedance.nullspace_damping must be within [0, 1]")
            elif key.endswith("ratio") and (type(value) is not int or not 1 <= value <= 100):
                raise ValueError(f"insertion.impedance.{key} must be an integer within [1, 100]")
    for lower, upper in (("contact_force_n", "target_force_n"), ("target_force_n", "max_axial_force_n"),
                         ("max_axial_force_n", "hard_force_n"), ("max_lateral_force_n", "hard_force_n"),
                         ("max_tilt_torque_nm", "hard_torque_nm"),
                         ("max_reference_lead_m", "max_tracking_error_m"),
                         ("baseline_duration_s", "baseline_timeout_s"),
                         ("candidate_confirm_s", "candidate_timeout_s"),
                         ("settle_s", "stall_timeout_s"),
                         ("retract_unload_force_n", "hard_force_n"),
                         ("retract_unload_torque_nm", "hard_torque_nm"),
                         ("endpoint_tolerance_m", "candidate_depth_m"),
                         ("progress_epsilon_m", "candidate_depth_m")):
        if settings.get(lower) is not None and settings.get(upper) is not None and settings[lower] >= settings[upper]:
            raise ValueError(f"insertion.{lower} must be less than {upper}")
    if require_ready and missing:
        raise ValueError("Insertion configuration incomplete: " + ", ".join(missing))
    return missing


class InsertionFault(RuntimeError):
    """Hard stop: the owner must hold/pause, never automatically retreat."""


class InsertionController:
    """Pure controller. step proposes poses; accepted records applied poses.

    state: timestamp_ns (host monotonic), right_sequence, T_B_F, wrench,
    hand_position_rad; optional applied_target and servo_limited. Call
    hard_guard again immediately before sending a target to the device.
    """
    def __init__(self, settings, grasp, goal, start_flange, now=None):
        self.settings = deepcopy(settings)
        validate_settings(self.settings, require_ready=True)
        self.grasp, self.goal, self.start = [transform(v).copy() for v in (grasp, goal, start_flange)]
        self.goal_flange = self.goal @ np.linalg.inv(self.grasp)
        s = self.settings
        if difference(self.start, self.goal_flange)[1] > s["orientation_tolerance_rad"]:
            raise ValueError("Insertion start and taught goal orientations disagree")
        self.depth_goal = float(self.start[1, 3] - self.goal_flange[1, 3])
        if not s["endpoint_tolerance_m"] < self.depth_goal <= s["max_descent_m"]:
            raise ValueError("Insertion goal must be below start and within max_descent_m")
        if np.linalg.norm((self.start-self.goal_flange)[[0, 2], 3]) > s["max_radius_m"]:
            raise ValueError("Insertion start is outside the taught lateral corridor")
        self.T_F_S = transform(s["T_F_S"])
        self.point_F = (self.grasp @ np.r_[s["hole_midpoint_cad_m"], 1.])[:3]
        self.started = time.monotonic() if now is None else float(now)
        self.phase_started = self.started
        self.last_now = self.started
        self.status = "baseline"
        self.target = self.start.copy()
        self.applied_target = self.start.copy()
        self.baseline = None
        self.baseline_samples = []
        self.baseline_first_time = None
        self.baseline_hand = None
        self.last_sequence = None
        self.last_stamp = None
        self.guard_sequence = None
        self.guard_advanced_at = self.started
        self.sequence = 0
        self.filtered = None
        self.last_wrench = None
        self.last_state = None
        self.contact_since = self.endpoint_since = None
        self.terminal_contact_since = None
        self.contact_y = self.capture_y = None
        self.search_center = None
        self.theta = 0.
        self.search_peak_axial = 0.
        self.reference_blocked = False
        self.progress_y = float(self.start[1, 3])
        self.progress_time = self.started
        self.reason = None
        self.events = []
        self.contact_verified = False
        self.hard_fault = None

    @property
    def stage(self):
        return self.status

    @property
    def done(self):
        return self.status in ("awaiting_confirmation", "retracted")

    @property
    def failed(self):
        return self.reason is not None or self.hard_fault is not None

    def _fault(self, reason):
        self.hard_fault = reason
        raise InsertionFault(reason)

    def _phase(self, status, now):
        self.status, self.phase_started = status, now
        self.events.append({"sequence": self.sequence, "time_s": now-self.started, "status": status})
        self.endpoint_since = None
        self.progress_time = now

    def _measured(self, state, now):
        try:
            T = transform(state["T_B_F"])
            w = vector(state["wrench"], 6, "wrench")
            hand = vector(state["hand_position_rad"], 20, "hand_position_rad")
            stamp, seq = state["timestamp_ns"], state["right_sequence"]
            if isinstance(stamp, bool) or not isinstance(stamp, (int, np.integer)):
                raise ValueError("timestamp_ns must be an integer")
            if isinstance(seq, bool) or not isinstance(seq, (int, np.integer)):
                raise ValueError("right_sequence must be an integer")
        except (KeyError, TypeError, ValueError) as e:
            self._fault(f"Invalid insertion feedback: {e}")
        age = now-int(stamp)*1e-9
        if not -1e-6 <= age <= self.settings["state_max_age_s"]:
            self._fault("Insertion feedback is stale or future-dated")
        return T, w, hand, int(stamp), int(seq)

    def compensated_wrench(self, T, raw):
        """External wrench in right-base axes, torque about the CAD hole midpoint."""
        delta = self.settings["wrench_sign"] * (raw-self.baseline)
        R_B_S = T[:3, :3] @ self.T_F_S[:3, :3]
        f = R_B_S @ delta[:3]
        sensor_from_point_B = T[:3, :3] @ (self.T_F_S[:3, 3]-self.point_F)
        tau = R_B_S @ delta[3:] + np.cross(sensor_from_point_B, f)
        return np.r_[f, tau]

    def hard_guard(self, state, now=None):
        now = time.monotonic() if now is None else float(now)
        if self.hard_fault is not None:
            raise InsertionFault(self.hard_fault)
        s = self.settings
        T, raw, hand, stamp, seq = self._measured(state, now)
        # Match the driver's vendor counter modulus, independent of the global
        # packet timestamp (which may advance for the other arm only).
        if not 0 <= seq < 1_000_000:
            self._fault("Insertion right-arm source counter out of range")
        if self.guard_sequence is None or seq != self.guard_sequence:
            if self.guard_sequence is not None and (seq-self.guard_sequence) % 1_000_000 >= 500_000:
                self._fault("Insertion right-arm source counter reset or reordered")
            self.guard_sequence, self.guard_advanced_at = seq, now
        elif now-self.guard_advanced_at > s["state_max_age_s"]:
            self._fault("Insertion right-arm source counter is stale")
        self.last_state = T.copy()
        if difference(T, self.start)[1] > s["orientation_tolerance_rad"]:
            self._fault("Insertion attitude changed; fixed-attitude baseline invalid")
        if self.baseline_hand is not None and np.max(np.abs(hand-self.baseline_hand)) > s["hand_shape_tolerance_rad"]:
            self._fault("Insertion grasp shape changed; baseline invalid")
        for offset, limit in ((0, s["raw_force_limit_n"]), (3, s["raw_torque_limit_nm"])):
            if limit is not None and np.linalg.norm(raw[offset:offset+3]) >= limit:
                self._fault("Insertion absolute raw sensor hard limit exceeded")
        y = T[1, 3]
        if (y < self.goal_flange[1, 3]-s["endpoint_tolerance_m"]
                or y < self.start[1, 3]-s["max_descent_m"]-s["endpoint_tolerance_m"]
                or y > self.start[1, 3]+s["endpoint_tolerance_m"]):
            self._fault("Insertion exceeded vertical workspace")
        if np.linalg.norm((T-self.start)[[0, 2], 3]) > s["max_radius_m"]+s["max_tracking_error_m"]:
            self._fault("Insertion exceeded lateral workspace")
        if np.linalg.norm(T[:3, 3]-self.applied_target[:3, 3]) > s["max_tracking_error_m"]:
            self._fault("Insertion actual pose departed from accepted reference")
        if self.baseline is not None:
            delta = raw-self.baseline
            # Sensor-origin raw delta limits precede low-pass filtering; point
            # torque additionally detects the relevant contact lever arm.
            w = self.compensated_wrench(T, raw)
            self.last_wrench = w.copy()
            if np.linalg.norm(delta[:3]) >= s["hard_force_n"] or max(np.linalg.norm(delta[3:]), np.linalg.norm(w[3:])) >= s["hard_torque_nm"]:
                self._fault("Insertion compensated wrench hard limit exceeded")
        return T, raw, hand, stamp, seq

    def accepted(self, target):
        self.applied_target = transform(target, "accepted insertion reference").copy()

    def _next_theta(self, distance):
        """Advance by bounded arc length, including the spiral's centre."""
        a = self.settings["pitch_m"]/(2*math.pi)
        def length(theta):
            return .5*a*(theta*math.sqrt(1+theta*theta)+math.asinh(theta))
        wanted = length(self.theta)+distance
        lo, hi = self.theta, self.theta+distance/a
        for _ in range(30):
            middle = .5*(lo+hi)
            if length(middle) < wanted:
                lo = middle
            else:
                hi = middle
        return .5*(lo+hi)

    def _retract(self, T, now, reason):
        self.reason = reason
        self.target = T.copy()
        self.target[:3, :3] = self.start[:3, :3]
        self._phase("retracting", now)

    def abort(self, state, reason, now=None):
        """Request a bounded soft retreat, provided current feedback is safe."""
        now = time.monotonic() if now is None else float(now)
        T, *_ = self.hard_guard(state, now)
        self._retract(T, now, str(reason))

    def _bounded(self, desired, actual):
        delta = desired[:3, 3]-actual[:3, 3]
        length = np.linalg.norm(delta)
        if length > self.settings["max_reference_lead_m"]:
            desired[:3, 3] = actual[:3, 3]+delta*self.settings["max_reference_lead_m"]/length
        desired[:3, :3] = self.start[:3, :3]
        desired[1, 3] = np.clip(desired[1, 3], self.goal_flange[1, 3], self.start[1, 3])
        self.target = desired
        return desired.copy()

    def step(self, state, now=None):
        now = time.monotonic() if now is None else float(now)
        if state.get("applied_target") is not None:
            self.accepted(state["applied_target"])
        T, raw, hand, stamp, seq = self.hard_guard(state, now)
        s = self.settings
        if now < self.last_now:
            self._fault("Insertion controller clock moved backwards")
        if now-self.last_now > s["max_control_dt_s"] and self.last_sequence is not None and not self.done:
            self._fault("Insertion control loop deadline missed")
        if self.last_stamp is not None and stamp < self.last_stamp:
            self._fault("Insertion feedback timestamp moved backwards")
        if self.last_sequence == seq or self.last_stamp == stamp:
            # Duplicate feedback must neither move nor count toward durations.
            return self.target.copy()
        dt = min(now-self.last_now, s["max_control_dt_s"])
        self.last_now, self.last_stamp, self.last_sequence = now, stamp, seq
        self.sequence += 1
        if self.status == "baseline":
            if now-self.started > s["baseline_timeout_s"]:
                self._fault("Insertion baseline acquisition timed out")
            if difference(T, self.start)[0] > s["baseline_position_tolerance_m"]:
                self._fault("Insertion moved during unloaded baseline acquisition")
            if self.baseline_hand is None:
                self.baseline_hand = hand.copy()
                self.baseline_first_time = now
            self.baseline_samples.append(raw.copy())
            if len(self.baseline_samples) >= s["baseline_min_samples"] and now-self.baseline_first_time >= s["baseline_duration_s"]:
                samples = np.array(self.baseline_samples)
                spread = samples.std(axis=0, ddof=1)
                if max(spread[:3]) > s["baseline_force_std_n"] or max(spread[3:]) > s["baseline_torque_std_nm"]:
                    self._fault("Insertion suspended baseline is not stable")
                self.baseline = np.mean(samples, axis=0)
                self.filtered = np.zeros(6)
                self._phase("approaching", now)
            return self.target.copy()
        w = self.compensated_wrench(T, raw)
        self.last_wrench = w.copy()
        alpha = dt/(s["force_filter_tau_s"]+dt)
        self.filtered += alpha*(w-self.filtered)
        axial = float(self.filtered[1])  # Reaction against descent along -Y.
        lateral = float(np.linalg.norm(self.filtered[[0, 2]]))
        tilt = float(np.linalg.norm(self.filtered[[3, 5]]))
        contact_limit = (axial > s["max_axial_force_n"] or lateral > s["max_lateral_force_n"]
                         or tilt > s["max_tilt_torque_nm"])
        if self.done:
            # Completion freezes the reference, not sensing. A newly loaded
            # terminal hold requires operator attention instead of restarting
            # an automatic retreat after the completed motion sequence.
            if contact_limit:
                self.terminal_contact_since = now if self.terminal_contact_since is None else self.terminal_contact_since
                if now-self.terminal_contact_since >= s["contact_confirm_s"]:
                    self._fault("Insertion terminal contact limit persisted; hold for operator review")
            else:
                self.terminal_contact_since = None
            return self.target.copy()
        desired = self.applied_target.copy()
        tolerance = s["progress_epsilon_m"]
        self.reference_blocked = bool(
            np.linalg.norm(self.target[:3, 3]-self.applied_target[:3, 3]) > tolerance
            or np.linalg.norm(self.applied_target[:3, 3]-T[:3, 3]) > s["max_reference_lead_m"]*.5)
        if self.reference_blocked:
            desired = self.target.copy()  # Retain the pending reference until it is followed.
        # Hold the latest measured lateral coordinates throughout a retreat.
        if self.status != "retracting":
            if contact_limit:
                self._retract(T, now, "contact_limit")
            elif now-self.started > s["total_timeout_s"]:
                self._retract(T, now, "total_timeout")
            elif now-self.phase_started > s["phase_timeout_s"]:
                self._retract(T, now, "phase_timeout")
        if self.status == "retracting":
            if now-self.phase_started > s["retract_timeout_s"]:
                self._fault("Insertion retreat timed out")
            desired = self.target.copy()
            # An old, downward reference may never delay unloading. Bound the
            # upward lead from measured feedback even while the servo is limited.
            desired[1, 3] = min(self.start[1, 3], max(T[1, 3], self.applied_target[1, 3])+s["retract_speed_m_s"]*dt)
            unloaded = (np.linalg.norm(w[:3]) <= s["retract_unload_force_n"]
                        and np.linalg.norm(w[3:]) <= s["retract_unload_torque_nm"])
            if abs(T[1, 3]-self.start[1, 3]) <= s["endpoint_tolerance_m"] and unloaded:
                self.endpoint_since = now if self.endpoint_since is None else self.endpoint_since
                if now-self.endpoint_since >= s["settle_s"]:
                    self._phase("retracted", now)
                    self.target = T.copy()
                    self.target[:3, :3] = self.start[:3, :3]
                    return self.target.copy()
            else:
                self.endpoint_since = None
            return self._bounded(desired, T)
        if self.status == "approaching":
            if axial >= s["contact_force_n"]:
                self.contact_since = now if self.contact_since is None else self.contact_since
                desired = T.copy()  # No further penetration while confirming contact.
                if now-self.contact_since >= s["contact_confirm_s"]:
                    self.contact_verified = True
                    self.contact_y = float(T[1, 3])
                    self.search_center = T[[0, 2], 3].copy()
                    self.search_peak_axial = axial
                    self.progress_y = self.contact_y
                    self._phase("searching", now)
            else:
                self.contact_since = None
                if not self.reference_blocked:
                    desired[1, 3] -= s["approach_speed_m_s"]*dt
            if T[1, 3] <= self.goal_flange[1, 3]+s["endpoint_tolerance_m"]:
                self._retract(T, now, "goal_reached_without_contact")
                desired = self.target.copy()
        elif self.status in ("searching", "capture", "inserting"):
            velocity = np.clip(s["admittance_m_s_n"]*(s["target_force_n"]-axial), -s["retract_speed_m_s"], s["insert_speed_m_s"])
            if velocity < 0:
                desired[1, 3] = max(T[1, 3], desired[1, 3])-float(velocity)*dt
            elif not self.reference_blocked:
                desired[1, 3] -= float(velocity)*dt
            if self.status == "searching":
                self.search_peak_axial = max(self.search_peak_axial, axial)
                if (self.contact_y-T[1, 3] >= s["candidate_depth_m"]
                        or self.search_peak_axial-w[1] >= s["candidate_force_drop_n"]):
                    self.capture_y = float(T[1, 3])
                    self.capture_xz = T[[0, 2], 3].copy()
                    desired = T.copy()  # Latch actual entry; never pull back to a stale search reference.
                    desired[[0, 2], 3] = self.capture_xz
                    self._phase("capture", now)
                elif not self.reference_blocked:
                    a = s["pitch_m"]/(2*math.pi)
                    next_theta = self._next_theta(s["lateral_speed_m_s"]*dt)
                    radius = a*next_theta
                    if radius > s["max_radius_m"]:
                        self._retract(T, now, "search_radius_exhausted")
                        desired = self.target.copy()
                    else:
                        self.theta = next_theta
                        desired[[0, 2], 3] = self.search_center+radius*np.array([math.cos(self.theta), math.sin(self.theta)])
            elif self.status == "capture":
                desired[[0, 2], 3] = self.capture_xz
                if now-self.phase_started > s["candidate_timeout_s"]:
                    self._retract(T, now, "unconfirmed_capture")
                    desired = self.target.copy()
                elif self.capture_y-T[1, 3] >= s["candidate_confirm_depth_m"] and now-self.phase_started >= s["candidate_confirm_s"]:
                    self.progress_y = float(T[1, 3])
                    self._phase("inserting", now)
            else:
                desired[[0, 2], 3] = self.capture_xz
            if self.status in ("capture", "inserting"):
                if self.progress_y-T[1, 3] >= s["progress_epsilon_m"]:
                    self.progress_y, self.progress_time = float(T[1, 3]), now
                elif now-self.progress_time > s["stall_timeout_s"]:
                    self._retract(T, now, "insertion_stalled")
                    desired = self.target.copy()
            if self.status == "inserting":
                at_depth = abs(T[1, 3]-self.goal_flange[1, 3]) <= s["endpoint_tolerance_m"]
                if at_depth:
                    desired = T.copy()
                    self.endpoint_since = now if self.endpoint_since is None else self.endpoint_since
                    if now-self.endpoint_since >= s["settle_s"]:
                        self._phase("awaiting_confirmation", now)
                        self.target = T.copy()
                        self.target[:3, :3] = self.start[:3, :3]
                        return self.target.copy()
                else:
                    self.endpoint_since = None
        return self._bounded(desired, T)

    def summary(self):
        return {"schema_version": 1, "sequence": self.sequence, "status": self.status,
                "done": self.done, "failed": self.failed, "reason": self.reason,
                "hard_fault": self.hard_fault, "assembly_verified": False,
                "contact_verified": self.contact_verified, "automatic_release": False,
                "hand_action": "keep_grasp", "control_mode": "cartesian_impedance",
                "force_controlled": True, "baseline_mode": "fixed_attitude_suspended",
                "baseline_samples": len(self.baseline_samples),
                "baseline_wrench_sensor": None if self.baseline is None else self.baseline.tolist(),
                "wrench_at_hole_midpoint_base": None if self.last_wrench is None else self.last_wrench.tolist(),
                "filtered_wrench_at_hole_midpoint_base": None if self.filtered is None else self.filtered.tolist(),
                "T_B_F_target": self.target.tolist(), "T_B_F_applied": self.applied_target.tolist(),
                "T_B_F_measured": None if self.last_state is None else self.last_state.tolist(),
                "search_radius_m": self.settings["pitch_m"]*self.theta/(2*math.pi),
                "reference_blocked": self.reference_blocked,
                "depth_m": None if self.last_state is None else float(self.start[1, 3]-self.last_state[1, 3]),
                "events": list(self.events)}
