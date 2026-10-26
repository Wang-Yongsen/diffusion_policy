"""Three RGB views plus measured flange/hand histories, in one host clock."""

import threading
import time
from collections import deque
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.third_party.bimanual_teleop.control.arm.cartesian import (
    TianjiCartesianExecutor,
)
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import (
    load_config as load_arm_config,
)
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import (
    TianjiDriver,
)
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import (
    TianjiKinematics,
)
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import (
    JOINT_NAMES,
    WujiHandDriver,
    WujiSdkSession,
)
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import (
    load_config as load_hand_config,
)
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import (
    sdk_user_name,
)
from diffusion_policy.third_party.bimanual_teleop.recording.camera import CameraRig
from diffusion_policy.third_party.bimanual_teleop.recording.config import (
    load_config as load_camera_config,
)
from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile

from .joint_executor import TianjiJointExecutor, model_joint_limits
from .observation import (
    ObservationNotReady,
    align_observation,
    align_latest_observation,
    _fixed_queries,
    _check_fixed_coverage,
    CAMERA_KEYS,
    STATE_KEYS,
)
from .runtime import (
    SIDES,
    TianjiPolicyRuntime,
    future_actions,
    retimed_actions,
)


class TianjiRailEnv:
    def __init__(self, config, n_obs_steps, *, motion=False):
        if config["inference_mode"] == "sync" and n_obs_steps not in (1, 2):
            raise ValueError("Synchronous inference requires checkpoint n_obs_steps in (1, 2)")
        (self.config, self.n_obs_steps, self.motion) = (config, n_obs_steps, motion)
        self.recorder = None
        self.trace = None
        self.record_observations = False
        self._measured_log_at = 0.0
        self.error = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._collector = None
        self._arm_sequences = {}
        self._shadow_chunk_end_ns = 0
        self._shadow_source_chunk_id = None
        self._shadow_chunk_id = 0
        self._shadow_prediction_id = None
        self.histories = {
            key: deque()
            for key in (
                "camera_0",
                "camera_1",
                "camera_2",
                "arms/left",
                "arms/right",
                "arm_joints/left",
                "arm_joints/right",
                "hands/left",
                "hands/right",
            )
        }
        arm_config = load_arm_config(config["arm_config"])
        hand_config = load_hand_config(config["hand_config"])
        self.arm_profile = ControlProfile(**arm_config["profile"])
        self.hand_profile = ControlProfile(
            hand_config["profile_id"], "mit", hand_config["parameters"]
        )
        self.kinematics = TianjiKinematics()
        if self.kinematics.model.digest != config["expected_model_sha256"]:
            raise ValueError("Loaded FK model differs from deployment expected_model_sha256")
        self.driver = TianjiDriver(arm_config["controller_ip"])
        self.action_space = config.get("action_space", "eef")
        joint_limits = None
        if self.action_space == "joint":
            self.arm = TianjiJointExecutor(self.driver)
            joint_limits = {
                side: model_joint_limits(self.kinematics.model, side) for side in SIDES
            }
        else:
            self.arm = TianjiCartesianExecutor(self.driver, self.kinematics)
        self.session = WujiSdkSession(user_name=sdk_user_name(hand_config))
        self.hands = {
            side: WujiHandDriver(
                side,
                hand_config["devices"][side]["hand"],
                feedback_hz=hand_config.get("feedback_hz", 200),
                timeout_s=config["max_state_age_s"],
            )
            for side in SIDES
        }
        camera_config = replace(load_camera_config(config["camera_config"]), main_depth=False)
        self.cameras = CameraRig(camera_config)
        self.runtime = TianjiPolicyRuntime(
            self.arm,
            self.hands,
            config,
            self.health_problem,
            self.emit,
            joint_limits=joint_limits,
        )

    def enable_trace(self, trace):
        if self.trace is not None:
            raise RuntimeError("A trace is already attached")
        self.trace = trace
        self.runtime.trace = trace
        trace.install_on(self.arm)

    def emit(self, event, **values):
        recorder = self.recorder
        if recorder is not None:
            recorder.event(event, monotonic_ns=time.monotonic_ns(), **values)

    def start(self):
        try:
            self.session.open()
            sink = None if self.trace is None else self.trace.sink
            if sink is None:
                self.driver.start()
                for hand in self.hands.values():
                    hand.start()
            else:
                self.driver.start(sink)
                for hand in self.hands.values():
                    hand.start(sink)
            self.cameras.start()
            self.runtime.start()
            self._shadow_chunk_end_ns = time.monotonic_ns()
            self._collector = threading.Thread(
                target=self._collect, name="policy-observation", daemon=True
            )
            self._collector.start()
        except Exception:
            self.close()
            raise

    def _append(self, key, stamp, values):
        with self._lock:
            history = self.histories[key]
            if history and stamp <= history[-1][0]:
                if stamp < history[-1][0]:
                    raise RuntimeError(f"{key}: host timestamp moved backwards")
                return
            history.append((int(stamp), values))
            while history and stamp - history[0][0] > self.config["history_s"] * 1e9:
                history.popleft()

    def _collect(self):
        next_state = 0.0
        try:
            while not self._stop.is_set():
                for name, kind, image, record in self.cameras.poll():
                    if kind == "rgb":
                        self._append(name, record.time_ns, image)
                if time.monotonic() >= next_state:
                    next_state = time.monotonic() + 1 / self.config["state_hz"]
                    sample = self.driver.get_latest()
                    if sample is not None and sample.header.valid:
                        for side in SIDES:
                            sequence = sample.payload.arms[side].source_sequence
                            if self._arm_sequences.get(side) == sequence:
                                continue
                            self._arm_sequences[side] = sequence
                            q = np.asarray(
                                sample.payload.arms[side].joints.position_rad, dtype=float
                            )
                            if q.shape != (7,) or not np.isfinite(q).all():
                                raise RuntimeError(f"{side}: invalid measured arm joints")
                            pose = self.kinematics.fk(side, tuple(q))
                            value = np.r_[
                                pose.position_m,
                                Rotation.from_quat(pose.orientation_xyzw).as_rotvec(),
                            ]
                            self._append(
                                f"arms/{side}", sample.header.received_monotonic_ns, value
                            )
                            self._append(
                                f"arm_joints/{side}", sample.header.received_monotonic_ns, q
                            )
                    for side, hand in self.hands.items():
                        sample = hand.get_latest()
                        if sample is not None and sample.header.valid:
                            q = np.asarray(sample.payload.position_rad, dtype=float)
                            if (
                                sample.payload.joint_names != JOINT_NAMES
                                or q.shape != (20,)
                                or not np.isfinite(q).all()
                            ):
                                raise RuntimeError(f"{side}: invalid measured hand joints")
                            self._append(
                                f"hands/{side}", sample.header.received_monotonic_ns, q
                            )
                    self._log_measured()
                if self.runtime.active or self.runtime.sync_waiting:
                    problem = self.health_problem()
                    if problem:
                        raise RuntimeError(problem)
                self._stop.wait(0.001)
        except Exception as error:
            self.error = str(error)
            self.runtime.pause(error, fault=True)

    def health_problem(self):
        if self.error:
            return self.error
        for device in (self.session, self.driver, *self.hands.values()):
            health = device.health()
            if not health.ready:
                return str(health.detail)
        with self._lock:
            now = time.monotonic_ns()
            for key, history in self.histories.items():
                budget = self.config[
                    "max_camera_age_s" if key.startswith("camera") else "max_state_age_s"
                ]
                if not history or now - history[-1][0] > budget * 1e9:
                    return f"{key}: missing/stale history"
        return None

    def latest_measured(self):
        """Latest FK poses and hand joints already stored by the collector. No new FK."""
        with self._lock:
            sample = {}
            for key in ("arms/left", "arms/right", "hands/left", "hands/right"):
                history = self.histories[key]
                if not history:
                    return None
                stamp, value = history[-1]
                sample[key.replace("/", "_")] = {
                    "stamp_ns": int(stamp),
                    "value": np.asarray(value, dtype=np.float64).copy(),
                }
            return sample

    def _log_measured(self):
        if (
            not self.record_observations
            or self.recorder is None
            or time.monotonic() < self._measured_log_at
        ):
            return
        self._measured_log_at = time.monotonic() + 1.0 / self.config["policy_hz"]
        measured = self.latest_measured()
        if measured is not None:
            self.emit("measured", **measured)

    def latest_anchor_ns(self):
        """Latest main-camera anchor covered by states, without image copies."""
        with self._lock:
            if any(not history for history in self.histories.values()):
                return None
            state_end = min(
                self.histories[f"{kind}/{side}"][-1][0]
                for kind in ("arms", "hands")
                for side in SIDES
            )
            return next(
                (
                    stamp
                    for stamp, _ in reversed(self.histories["camera_0"])
                    if stamp <= state_end
                ),
                None,
            )

    def get_obs(self, *, query_ns=None):
        if self.error:
            raise RuntimeError(self.error)
        problem = self.health_problem()
        if problem:
            raise ObservationNotReady(problem)
        with self._lock:
            histories = {key: list(history) for key, history in self.histories.items()}
        if query_ns is None and self.config["inference_mode"] == "sync":
            return align_latest_observation(
                histories, self.n_obs_steps, self.config, now_ns=time.monotonic_ns()
            )
        return align_observation(
            histories,
            self.n_obs_steps,
            self.config,
            now_ns=time.monotonic_ns(),
            query_ns=query_ns,
        )

    def engage(self):
        if not self.motion:
            raise RuntimeError("Shadow mode cannot engage devices")
        if any(self.config["workspace"][side] is None for side in SIDES):
            raise ValueError("Fill both measured workspaces before motion")
        obs, _, _ = self.get_obs()
        arm_key = "robot_joint" if self.action_space == "joint" else "robot_eef_pose"
        self.runtime.seed(np.r_[obs[arm_key][-1], obs["hand_joint"][-1]])
        try:
            for hand in self.hands.values():
                if hand.profile is None:
                    hand.configure(self.hand_profile)
                hand.engage()
            if self.driver.profile is None:
                self.arm.configure(self.arm_profile)
            self.arm.engage()
            # Use the exact arm engagement reference, not a previous aligned camera pose.
            if self.action_space == "joint":
                arms = [np.asarray(self.arm.engagement_joints[side]) for side in SIDES]
            else:
                arms = [
                    np.r_[
                        self.arm.engagement_poses[side].position_m,
                        Rotation.from_quat(
                            self.arm.engagement_poses[side].orientation_xyzw
                        ).as_rotvec(),
                    ]
                    for side in SIDES
                ]
            seed = np.concatenate(
                arms + [np.asarray(self.hands[side].last_target.position_rad) for side in SIDES]
            )
            self.runtime.seed(seed, arm_ready=True)
        except Exception as error:
            self.runtime.pause(error, fault=True)
            raise

    def exec_actions(self, actions, anchor_ns, prediction_id=None):
        if self.motion:
            return self.runtime.submit(actions, anchor_ns, prediction_id=prediction_id)
        if self.config["inference_mode"] == "sync":
            selected, times, dropped = retimed_actions(
                actions,
                time.monotonic_ns(),
                self.config["policy_hz"],
                self.config["submit_margin_s"],
            )
            with self._lock:
                self._shadow_chunk_end_ns = int(times[-1]) + round(
                    1e9 / self.config["policy_hz"]
                )
                self._shadow_chunk_id += 1
                self._shadow_source_chunk_id = self._shadow_chunk_id
                self._shadow_prediction_id = prediction_id
        else:
            selected, times, dropped = future_actions(
                actions,
                anchor_ns,
                time.monotonic_ns(),
                self.config["policy_hz"],
                self.config["submit_margin_s"],
            )
        self.emit(
            "shadow_schedule",
            times_ns=times,
            actions=selected,
            dropped_steps=dropped,
            **(
                {"generation": self._shadow_chunk_id, "prediction_id": prediction_id}
                if self.config["inference_mode"] == "sync"
                else {}
            ),
        )
        return dropped

    def chunk_completed_ns(self):
        """Sync mode: when the robot entered hold after the last chunk, or None while a chunk executes."""
        if self.motion:
            return self.runtime.chunk_completed_ns
        return (
            self._shadow_chunk_end_ns
            if time.monotonic_ns() >= self._shadow_chunk_end_ns
            else None
        )

    def pause(self, reason="Operator pause"):
        with self._lock:
            self._shadow_chunk_end_ns = time.monotonic_ns()
            self._shadow_source_chunk_id = None
            self._shadow_prediction_id = None
        self.runtime.pause(reason)

    def close(self):
        self._stop.set()
        if self._collector is not None:
            self._collector.join(timeout=2)
        try:
            self.runtime.close()
        finally:
            for device in (*self.hands.values(), self.driver, self.cameras, self.session):
                try:
                    device.close()
                except Exception as error:
                    self.error = self.error or str(error)

    def observation_ready(self, query_ns):
        """Check fixed-window coverage without image copies; lost history is fatal."""
        if self.error:
            raise RuntimeError(self.error)
        problem = self.health_problem()
        if problem:
            raise RuntimeError(problem)
        queries = _fixed_queries(query_ns, self.n_obs_steps)
        with self._lock:
            try:
                _check_fixed_coverage(self.histories, queries, self.config)
            except ObservationNotReady:
                return False
        return True

    def sync_observation_window(self):
        """Completion context for choosing a fresh observation, including bootstrap."""
        if self.motion:
            return self.runtime.sync_observation_window()
        with self._lock:
            if time.monotonic_ns() < self._shadow_chunk_end_ns:
                return None
            return {
                "completed_ns": self._shadow_chunk_end_ns,
                "source_chunk_id": self._shadow_source_chunk_id,
                "source_prediction_id": (
                    self._shadow_prediction_id
                    if self._shadow_source_chunk_id is not None
                    else None
                ),
            }

    def feedback_since(self, since_ns):
        """Copy actual flange poses and hand joints at or after ``since_ns``. No new FK."""
        since_ns = int(since_ns)
        with self._lock:

            def take(key):
                return [
                    (int(stamp), np.asarray(value, dtype=np.float64).copy())
                    for stamp, value in self.histories[key]
                    if stamp >= since_ns
                ]

            return {
                "arms": {side: take(f"arms/{side}") for side in SIDES},
                "hands": {side: take(f"hands/{side}") for side in SIDES},
            }
