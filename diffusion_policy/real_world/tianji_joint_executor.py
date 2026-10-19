"""Direct joint-target following for joint-space policies.

Uses the same engaged driver mode and joint command as Cartesian following,
without IK. Each 200 Hz step limits joint velocity and acceleration with the
active profile ratios and stays inside the M6 joint and coupled-wrist limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Mapping

import numpy as np

from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import TianjiDriver, TianjiJointCommand
from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile, DeviceCommand, SampleRef, Side, Submission

CONTROL_HZ = 200
PERIOD_NS = 1_000_000_000 // CONTROL_HZ
# Proportional approach gain (1/s), equal to the Cartesian servo task gain.
TRACKING_GAIN = 16.
WRIST_LIMIT_RAD = math.radians(110.5)
# The controller receives float32 degrees; plan this far inside every bound,
# then check the encoded command against the physical bounds.
PLANNING_MARGIN_RAD = math.radians(1e-5)
PINNED_WRIST = ((0., -1.025, 110.5), (0., 1.025, 110.5), (0., -1.025, -110.5), (0., 1.025, -110.5))


def model_joint_limits(model, side):
    arm = model.arm(side)
    if not np.allclose(arm.bd67_native, PINNED_WRIST, rtol=0, atol=1e-10):
        raise ValueError("Joint following requires the pinned linear M6 wrist envelope")
    return np.asarray(arm.lower_rad), np.asarray(arm.upper_rad)


def joints_within_limits(q, lower, upper, margin=PLANNING_MARGIN_RAD):
    """Rows of seven joints inside the box and 1.025*|J6| + |J7| <= 110.5 deg, less margin."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    return bool(np.all(q >= lower + margin) and np.all(q <= upper - margin)
                and np.all(1.025 * np.abs(q[:, 5]) + np.abs(q[:, 6]) <= WRIST_LIMIT_RAD - margin))


@dataclass(frozen=True)
class JointArmTarget:
    command_id: str
    joints: Mapping[Side, tuple[float, ...]]
    source_refs: tuple[SampleRef, ...]
    created_monotonic_ns: int
    expires_monotonic_ns: int
    control_profile_id: str


class JointFollower:
    """Per-joint braking-distance approach; a proposal advances only after accept()."""

    def __init__(self, model, side, *, velocity_ratio=100, acceleration_ratio=100):
        if not (0 < velocity_ratio <= 100 and 0 < acceleration_ratio <= 100):
            raise ValueError("velocity/acceleration ratios must be in (0, 100]")
        self.side = side
        self.lower, self.upper = model_joint_limits(model, side)
        arm = model.arm(side)
        self.vmax = np.asarray(arm.max_joint_velocity_rad_s) * velocity_ratio / 100
        self.amax = np.radians([row[3] for row in arm.limits_native]) * acceleration_ratio / 100
        self.q = self.v = None

    def reset(self, q_rad):
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError(f"{self.side}: joint following requires seven finite actual joints")
        if not joints_within_limits(q, self.lower, self.upper):
            raise ValueError(f"{self.side}: actual joints are outside the M6 joint/wrist limits")
        self.q, self.v = q.copy(), np.zeros(7)

    def propose(self, goal_rad, dt_s):
        if self.q is None:
            raise RuntimeError("Reset joint following from actual joints before use")
        goal = np.asarray(goal_rad, dtype=float)
        if goal.shape != (7,) or not np.isfinite(goal).all():
            raise ValueError(f"{self.side}: joint target must contain seven finite values")
        if not joints_within_limits(goal, self.lower, self.upper):
            raise ValueError(f"{self.side}: joint target is outside the M6 joint/wrist limits")
        if not math.isfinite(dt_s) or dt_s < 0:
            raise ValueError("Joint following dt must be finite and nonnegative")
        error = goal - self.q
        distance = np.abs(error)
        # Largest speed that still stops within distance when it drops by amax*dt per step.
        braking = self.amax * (np.sqrt(dt_s ** 2 / 4 + 2 * distance / self.amax) - dt_s / 2)
        speed = np.minimum.reduce([self.vmax, TRACKING_GAIN * distance, braking])
        v = np.clip(np.sign(error) * speed, self.v - self.amax * dt_s, self.v + self.amax * dt_s)
        v = np.clip(v, -self.vmax, self.vmax)
        q = self.q + v * dt_s
        if not joints_within_limits(q, self.lower, self.upper):
            # The feasible set is convex and contains self.q; shorten the step.
            low, high = 0., 1.
            for _ in range(30):
                middle = (low + high) / 2
                if joints_within_limits(self.q + middle * v * dt_s, self.lower, self.upper):
                    low = middle
                else:
                    high = middle
            v = v * low
            q = self.q + v * dt_s
        encoded = np.radians(np.degrees(q).astype(np.float32).astype(float))
        if not joints_within_limits(encoded, self.lower, self.upper, margin=0.):
            raise ValueError(f"{self.side}: float32 joint command exceeds M6 joint/wrist limits")
        return q, v

    def accept(self, q, v):
        self.q, self.v = np.asarray(q, dtype=float), np.asarray(v, dtype=float)


class TianjiJointExecutor:
    """Same lifecycle as TianjiCartesianExecutor, with joint goals instead of flange poses."""

    def __init__(self, driver: TianjiDriver, *, clock_ns=None) -> None:
        self.clock_ns = clock_ns or (lambda: time.monotonic_ns())
        self.driver = driver
        self._followers: dict[Side, JointFollower] = {}
        self._engagement_joints: dict[Side, tuple[float, ...]] = {}
        self._engagement_ref: SampleRef | None = None
        self._last_step_ns = None
        self.tracking_status = {}
        self.timing_status = {}
        self._state_lock = threading.RLock()
        self._generation = 0

    @property
    def engagement_joints(self) -> dict[Side, tuple[float, ...]]:
        """Measured joints from the exact sample used by the driver's engagement."""
        return dict(self._engagement_joints)

    @property
    def engagement_ref(self) -> SampleRef | None:
        return self._engagement_ref

    def configure(self, profile: ControlProfile) -> None:
        self.driver.configure(profile)
        with self._state_lock:
            self._generation += 1
            self._followers.clear()
            self._engagement_joints.clear()
            self._engagement_ref = None
            self._last_step_ns = None
            self.tracking_status.clear()

    def _resolve_reference(self, sample):
        profile = self.driver.profile
        joints, followers = {}, {}
        for side in profile.active_arms:
            q = sample.payload.arms[side].joints.position_rad
            if len(q) != 7 or any(value is None or not math.isfinite(value) for value in q):
                raise ValueError(f"{side}: actual joints are invalid")
            arm = profile.arms[side]
            follower = JointFollower(profile.model, side, velocity_ratio=arm.velocity_ratio,
                                     acceleration_ratio=arm.acceleration_ratio)
            follower.reset(q)
            joints[side], followers[side] = tuple(float(value) for value in q), follower
        return joints, followers

    def engage(self) -> None:
        generation = self._generation
        sample = self.driver.get_latest()
        if self.driver.profile is None or sample is None:
            raise RuntimeError("configure and receive actual joints before engagement")
        if not self.driver.health().ready:
            raise RuntimeError("feedback is not ready for engagement")
        self._resolve_reference(sample)
        self.driver.engage()
        try:
            sample = self.driver.engagement_sample
            if sample is None:
                raise RuntimeError("driver did not retain its engagement sample")
            joints, followers = self._resolve_reference(sample)
            with self._state_lock:
                if self._generation != generation or not self.driver.engaged:
                    raise RuntimeError("Joint engagement was cancelled")
                self._generation += 1
                self._engagement_joints = joints
                self._engagement_ref = sample.header.ref
                self._followers = followers
                self._last_step_ns = None
                self.tracking_status = {}
        except (RuntimeError, ValueError):
            self.request_hold("driver engagement reference could not initialize joint following")
            raise

    def _reject(self, target: JointArmTarget, reason: str) -> Submission:
        self.timing_status["elapsed_ns"] = self.clock_ns() - self.timing_status["started_ns"]
        self.request_hold(f"Joint target {target.command_id}: {reason}")
        return Submission(target.command_id, False, reason)

    def _ended_reason(self, fallback: str) -> str:
        stop = getattr(self.driver, "motion_stop", None)
        return stop["reason"] if stop else fallback

    def submit(self, target: JointArmTarget) -> Submission:
        profile = self.driver.profile
        now = self.clock_ns()
        self.timing_status = {"started_ns": now, "stage": "validate", "command_id": target.command_id,
                              "created_ns": target.created_monotonic_ns, "expires_ns": target.expires_monotonic_ns}
        with self._state_lock:
            if not self.driver.engaged or not self._followers or profile is None:
                return Submission(target.command_id, False, self._ended_reason("executor is not engaged"))
            generation, followers, last_step_ns = self._generation, dict(self._followers), self._last_step_ns
        if target.control_profile_id != profile.profile_id:
            return self._reject(target, "control profile does not match")
        if not target.created_monotonic_ns <= now < target.expires_monotonic_ns:
            return self._reject(target, "target is expired or created in the future")
        if set(target.joints) != set(profile.active_arms):
            return self._reject(target, "target must contain exactly the configured arms")
        dt = PERIOD_NS / 1e9 if last_step_ns is None else (now - last_step_ns) / 1e9
        try:
            steps = {side: followers[side].propose(target.joints[side], dt) for side in profile.active_arms}
        except (ValueError, RuntimeError) as error:
            return self._reject(target, str(error))
        if self.clock_ns() >= target.expires_monotonic_ns:
            return self._reject(target, "target expired during planning")
        targets = {side: tuple(float(value) for value in q) for side, (q, _) in steps.items()}
        command = DeviceCommand(
            device_id="tianji", command_id=target.command_id,
            payload=TianjiJointCommand(targets=targets, cartesian_targets={}),
            source_refs=tuple(target.source_refs), created_monotonic_ns=target.created_monotonic_ns,
            expires_monotonic_ns=target.expires_monotonic_ns, control_profile_id=target.control_profile_id)
        self.timing_status["stage"] = "driver_submit"
        started = self.clock_ns()
        feedback = self.driver.get_latest()
        result = self.driver.submit(command)
        self.timing_status["driver_submit_ns"] = self.clock_ns() - started
        if result.accepted:
            with self._state_lock:
                if self._generation != generation or not self.driver.engaged:
                    return Submission(target.command_id, False,
                                      self._ended_reason("Joint engagement ended during submission"))
                for side, (q, v) in steps.items():
                    followers[side].accept(q, v)
                self._last_step_ns = now
                self.tracking_status = {side: {
                    "goal_error_rad": float(np.max(np.abs(np.asarray(target.joints[side]) - q))),
                    "max_joint_tracking_error_rad": float(np.max(np.abs(
                        q - np.asarray(feedback.payload.arms[side].joints.position_rad)))) if feedback else None}
                    for side, (q, _) in steps.items()}
        else:
            self.request_hold(f"Joint target {target.command_id}: {result.reason}")
        self.timing_status["stage"] = "complete" if result.accepted else "rejected"
        self.timing_status["elapsed_ns"] = self.clock_ns() - now
        return result

    def request_hold(self, reason: str) -> None:
        with self._state_lock:
            self._generation += 1
            self._followers.clear()
            self._last_step_ns = None
        self.driver.request_hold(reason)
