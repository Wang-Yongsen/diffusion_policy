"""Constrained flange following, submitted atomically as one arm group."""

from __future__ import annotations

from dataclasses import asdict
import math
import threading
import time

from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import TianjiDriver, TianjiFrame, TianjiJointCommand
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import KinematicsError, TianjiKinematics
from diffusion_policy.third_party.bimanual_teleop.control.arm.servo import CartesianServo
from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile, DeviceCommand, Pose, RobotTarget, Sample, SampleRef, Side, Submission


CONTROL_HZ = 200
PERIOD_NS = 1_000_000_000 // CONTROL_HZ


class TianjiCartesianExecutor:
    """Advance each constrained servo only after a successful group submission."""

    def __init__(self, driver: TianjiDriver, kinematics: TianjiKinematics, *, clock_ns=None) -> None:
        self.clock_ns = clock_ns or (lambda: time.monotonic_ns())
        self.driver = driver
        self.kinematics = kinematics
        self._reference: dict[Side, tuple[float, ...]] = {}
        self._engagement_poses: dict[Side, Pose] = {}
        self._engagement_ref: SampleRef | None = None
        self._servos = {}
        self._last_step_ns = None
        self._applied_poses = {}
        self.tracking_status = {}
        self.timing_status = {}
        self._state_lock = threading.RLock()
        self._generation = 0

    @property
    def applied_poses(self):
        return dict(self._applied_poses)

    @property
    def engagement_poses(self) -> dict[Side, Pose]:
        """Origins from the exact sample used by the driver's engagement."""
        return dict(self._engagement_poses)

    @property
    def engagement_ref(self) -> SampleRef | None:
        return self._engagement_ref

    @staticmethod
    def _check_cancelled(cancel, operation):
        if cancel is not None and cancel.is_set():
            raise RuntimeError(f"Cartesian {operation} was cancelled")

    def configure(self, profile: ControlProfile, *, cancel=None) -> None:
        self._check_cancelled(cancel, "configuration")
        try:
            self.driver.configure(profile, **({"cancel": cancel} if cancel is not None else {}))
            with self._state_lock:
                self._generation += 1
                self._reference.clear()
                self._engagement_poses.clear()
                self._engagement_ref = None
                self._servos.clear()
                self._last_step_ns = None
                self._applied_poses.clear()
                self.tracking_status.clear()
                self._check_cancelled(cancel, "configuration")
        except (RuntimeError, ValueError):
            if cancel is not None and cancel.is_set():
                self.request_hold("Cartesian configuration was cancelled")
            raise

    def _resolve_reference(self, sample: Sample[TianjiFrame]) -> tuple[dict, dict, dict]:
        reference, poses, servos = {}, {}, {}
        for side in self.driver.profile.active_arms:
            joints = sample.payload.arms[side].joints.position_rad
            if len(joints) != 7 or any(q is None or not math.isfinite(q) for q in joints):
                raise ValueError(f"{side}: actual joints are invalid")
            reference[side] = tuple(float(q) for q in joints)
            poses[side] = self.kinematics.fk(side, reference[side])
            self.kinematics.jacobian(side, reference[side])
            arm = self.driver.profile.arms[side]
            servo = CartesianServo(self.kinematics, side, self.driver.profile.model,
                velocity_ratio=arm.velocity_ratio, acceleration_ratio=arm.acceleration_ratio)
            servo.reset(reference[side])
            servos[side] = servo
        return reference, poses, servos

    def engage(self, *, cancel=None) -> None:
        self._check_cancelled(cancel, "engagement")
        generation = self._generation
        profile = self.driver.profile
        sample = self.driver.get_latest()
        if profile is None or sample is None:
            raise RuntimeError("configure and receive actual joints before engagement")
        if not self.driver.health().ready:
            raise RuntimeError("feedback is not ready for engagement")
        self._resolve_reference(sample)
        self._check_cancelled(cancel, "engagement")
        try:
            self.driver.engage(**({"cancel": cancel} if cancel is not None else {}))
            self._check_cancelled(cancel, "engagement")
            sample = self.driver.engagement_sample
            if sample is None:
                raise RuntimeError("driver did not retain its engagement sample")
            reference, poses, servos = self._resolve_reference(sample)
            with self._state_lock:
                self._check_cancelled(cancel, "engagement")
                if self._generation != generation or not self.driver.engaged:
                    raise RuntimeError("Cartesian engagement was cancelled")
                self._generation += 1
                self._reference = reference
                self._engagement_poses = dict(poses)
                self._engagement_ref = sample.header.ref
                self._servos = servos
                self._last_step_ns = None
                self._applied_poses = dict(poses)
                self.tracking_status = {}
        except (RuntimeError, ValueError):
            self.request_hold("driver engagement reference could not initialize Cartesian servos")
            raise

    def _reject(self, target: RobotTarget, reason: str) -> Submission:
        self.timing_status["elapsed_ns"] = self.clock_ns() - self.timing_status["started_ns"]
        self.request_hold(f"Cartesian target {target.command_id}: {reason}")
        return Submission(target.command_id, False, reason)

    def _ended_reason(self, fallback: str) -> str:
        stop = getattr(self.driver, "motion_stop", None)
        return stop["reason"] if stop else fallback

    def submit(self, target: RobotTarget, *, before_submit=None) -> Submission:
        profile = self.driver.profile
        now = self.clock_ns()
        self.timing_status = {"started_ns": now, "stage": "validate", "command_id": target.command_id,
                              "created_ns": target.created_monotonic_ns,
                              "expires_ns": target.expires_monotonic_ns}
        with self._state_lock:
            if not self.driver.engaged or not self._reference or profile is None:
                return Submission(target.command_id, False, self._ended_reason("executor is not engaged"))
            generation, servos, last_step_ns = self._generation, dict(self._servos), self._last_step_ns
        if target.control_profile_id != profile.profile_id:
            return self._reject(target, "control profile does not match")
        if not target.created_monotonic_ns <= now < target.expires_monotonic_ns:
            return self._reject(target, "target is expired or created in the future")
        if set(target.tool_poses) != set(profile.active_arms):
            return self._reject(target, "target must contain exactly the configured arms")
        poses = dict(target.tool_poses)
        try:
            dt = PERIOD_NS / 1e9 if last_step_ns is None else (now - last_step_ns) / 1e9
            feedback = self.driver.get_latest()
            steps = {}
            for side in profile.active_arms:
                started = self.clock_ns()
                self.timing_status["stage"] = f"{side}_servo"
                try:
                    steps[side] = servos[side].propose(poses[side], dt,
                        actual_rad=feedback.payload.arms[side].joints.position_rad if feedback else None)
                finally:
                    self.timing_status[f"{side}_servo_ns"] = self.clock_ns() - started
            targets = {side: step.joints for side, step in steps.items()}
            self.timing_status["stage"] = "input_recheck"
            if before_submit is not None:
                before_submit()
            if self._generation != generation or not self.driver.engaged:
                raise RuntimeError(self._ended_reason("Cartesian engagement ended during planning"))
        except (ValueError, RuntimeError) as error:
            if isinstance(error, KinematicsError) and error.diagnostic is not None:
                error.diagnostic.update(
                    command_id=target.command_id, control_profile_id=target.control_profile_id,
                    source_refs=[asdict(ref) for ref in target.source_refs],
                    created_monotonic_ns=target.created_monotonic_ns,
                    expires_monotonic_ns=target.expires_monotonic_ns,
                    ik_started_monotonic_ns=now,
                )
                feedback = self.driver.get_latest()
                if feedback is not None:
                    side = error.diagnostic["side"]
                    joints = feedback.payload.arms[side].joints.position_rad
                    error.diagnostic["feedback"] = {
                        "ref": asdict(feedback.header.ref),
                        "received_monotonic_ns": feedback.header.received_monotonic_ns,
                        "joints_deg": [math.degrees(q) if q is not None and math.isfinite(q)
                                       else None for q in joints],
                    }
            return self._reject(target, str(error))
        if self.clock_ns() >= target.expires_monotonic_ns:
            return self._reject(target, "target expired during IK")
        command = DeviceCommand(
            device_id="tianji", command_id=target.command_id,
            payload=TianjiJointCommand(targets=targets, cartesian_targets={s: step.pose for s, step in steps.items()},
                                      requested_cartesian_targets=poses),
            source_refs=tuple(target.source_refs), created_monotonic_ns=target.created_monotonic_ns,
            expires_monotonic_ns=target.expires_monotonic_ns,
            control_profile_id=target.control_profile_id,
        )
        self.timing_status["stage"] = "driver_submit"
        started = self.clock_ns()
        result = self.driver.submit(command)
        self.timing_status["driver_submit_ns"] = self.clock_ns() - started
        if result.accepted:
            with self._state_lock:
                if self._generation != generation or not self.driver.engaged:
                    return Submission(target.command_id, False,
                                      self._ended_reason("Cartesian engagement ended during submission"))
                for side, step in steps.items():
                    servos[side].accept(step)
                self._reference = targets
                self._last_step_ns = now
                self._applied_poses = {s: step.pose for s, step in steps.items()}
                self.tracking_status = {s: {"limited": step.limited, "progress": step.progress,
                    "position_error_m": step.position_error_m,
                    "orientation_error_rad": step.orientation_error_rad,
                    "max_joint_tracking_error_rad": max(abs(q - actual) for q, actual in zip(
                        step.joints, feedback.payload.arms[s].joints.position_rad)) if feedback else None}
                    for s, step in steps.items()}
        else:
            self.request_hold(f"Cartesian target {target.command_id}: {result.reason}")
        self.timing_status["stage"] = "complete" if result.accepted else "rejected"
        self.timing_status["elapsed_ns"] = self.clock_ns() - now
        return result

    def request_hold(self, reason: str) -> None:
        with self._state_lock:
            self._generation += 1
            self._reference.clear()
            self._servos.clear()
            self._last_step_ns = None
        self.driver.request_hold(reason)
