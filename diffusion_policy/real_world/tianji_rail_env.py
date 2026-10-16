"""Three RGB views plus measured flange/hand histories, in one host clock."""

from collections import deque
from dataclasses import replace
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from diffusion_policy.third_party.bimanual_teleop.control.arm.cartesian import TianjiCartesianExecutor
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import load_config as load_arm_config
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.driver import TianjiDriver
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.adapter import WujiHandDriver, WujiSdkSession, JOINT_NAMES
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import load_config as load_hand_config, sdk_user_name
from diffusion_policy.third_party.bimanual_teleop.recording.camera import CameraRig
from diffusion_policy.third_party.bimanual_teleop.recording.config import load_config as load_camera_config
from diffusion_policy.third_party.bimanual_teleop.types import ControlProfile

from .tianji_policy_runtime import SIDES, TianjiPolicyRuntime, future_actions

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ObservationNotReady(RuntimeError):
    pass


def load_deployment_config(path):
    with Path(path).open() as file:
        config = yaml.safe_load(file)
    for key, value in config.items():
        if key.endswith(("_s", "_hz", "_rad", "_m", "_m_s", "_rad_s")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
    if config["policy_hz"] != 30 or config["arm_hz"] != 200:
        raise ValueError("First deployment requires policy_hz=30 and arm_hz=200")
    digest = config["expected_model_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("expected_model_sha256 must be a lowercase SHA256 digest")
    for key in ("replanning_steps", "record_queue_capacity"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for side in SIDES:
        box = config["workspace"][side]
        if box is not None:
            low, high = np.asarray(box["min"]), np.asarray(box["max"])
            if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low >= high):
                raise ValueError(f"workspace.{side} needs finite 3-D min < max")
    for key in ("arm_config", "hand_config", "camera_config"):
        config[key] = str((PROJECT_ROOT / config[key]).resolve())
    config["output_dir"] = str((PROJECT_ROOT / config["output_dir"]).resolve())
    return config


def interpolate_state(history, queries, *, pose=False, max_gap_ns=50_000_000):
    stamps = np.asarray([item[0] for item in history], dtype=np.int64)
    if len(stamps) < 2 or np.any(np.diff(stamps) <= 0) or queries[0] < stamps[0] or queries[-1] > stamps[-1]:
        raise ObservationNotReady("State history does not cover requested observation times")
    values = np.stack([item[1] for item in history])
    result, used = [], []
    for query in queries:
        upper = int(np.searchsorted(stamps, query, side="left"))
        if stamps[upper] == query:
            result.append(values[upper])
            used.append((int(stamps[upper]), int(stamps[upper])))
            continue
        lower = upper - 1
        gap = int(stamps[upper] - stamps[lower])
        if gap > max_gap_ns:
            raise ObservationNotReady("State interpolation interval is too large")
        fraction = (int(query) - int(stamps[lower])) / gap
        value = values[lower] + fraction * (values[upper] - values[lower])
        if pose:
            rotations = Rotation.from_rotvec(values[[lower, upper], 3:6])
            value[3:6] = Slerp([0., 1.], rotations)([fraction]).as_rotvec()[0]
        result.append(value)
        used.append((int(stamps[lower]), int(stamps[upper])))
    return np.asarray(result, dtype=np.float32), used


def align_observation(histories, n_obs_steps, config, *, now_ns):
    if any(not history for history in histories.values()):
        raise ObservationNotReady("Waiting for all seven observation streams")
    state_keys = [f"{kind}/{side}" for kind in ("arms", "hands") for side in SIDES]
    state_end = min(histories[key][-1][0] for key in state_keys)
    anchors = [item[0] for item in histories["camera_0"] if item[0] <= state_end]
    if not anchors:
        raise ObservationNotReady("No camera anchor covered by measured states")
    anchor = anchors[-1]
    if anchor > now_ns:
        raise ObservationNotReady("Observation anchor is in the future")
    queries = anchor - np.rint(np.arange(n_obs_steps - 1, -1, -1) * 1e9 / config["policy_hz"]).astype(np.int64)
    if now_ns - anchor > config["max_camera_age_s"] * 1e9:
        raise ObservationNotReady("Aligned observation is too old")
    obs, metadata = {}, {"query_ns": queries.tolist(), "camera_ns": {}, "state_brackets_ns": {}}
    camera_stamps = []
    for key in ("camera_0", "camera_1", "camera_2"):
        history = histories[key]
        if now_ns - history[-1][0] > config["max_camera_age_s"] * 1e9:
            raise ObservationNotReady(f"{key} is stale")
        stamps = np.asarray([item[0] for item in history], dtype=np.int64)
        indices = [int(np.abs(stamps - query).argmin()) for query in queries]
        selected = stamps[indices]
        if np.any(np.abs(selected - queries) > config["max_image_skew_s"] * 1e9):
            raise ObservationNotReady(f"{key} has no image near the query times")
        obs[key] = np.stack([history[i][1] for i in indices])
        metadata["camera_ns"][key] = selected.tolist()
        camera_stamps.append(selected)
    if np.any(np.ptp(np.asarray(camera_stamps), axis=0) > config["max_camera_span_s"] * 1e9):
        raise ObservationNotReady("Three-camera timestamp span exceeds budget")
    for kind, output in (("arms", "robot_eef_pose"), ("hands", "hand_joint")):
        values = []
        for side in SIDES:
            key = f"{kind}/{side}"
            if now_ns - histories[key][-1][0] > config["max_state_age_s"] * 1e9:
                raise ObservationNotReady(f"{key} is stale")
            value, brackets = interpolate_state(histories[key], queries, pose=kind == "arms",
                                                max_gap_ns=round(config["max_state_gap_s"] * 1e9))
            values.append(value)
            metadata["state_brackets_ns"][key] = brackets
        obs[output] = np.concatenate(values, axis=-1)
    return obs, anchor, metadata


def preprocess_observation(obs, shape_meta, dataset_config):
    import cv2
    interpolation = {"area": cv2.INTER_AREA, "bilinear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}
    result = {}
    if set(obs) != set(shape_meta["obs"]):
        raise ValueError("Checkpoint observation keys differ from the real rig")
    for key, spec in shape_meta["obs"].items():
        values, shape = np.asarray(obs[key]), tuple(spec["shape"])
        if spec.get("type", "low_dim") == "rgb":
            if values.dtype != np.uint8 or values.ndim != 4 or values.shape[-1] != 3:
                raise ValueError(f"{key}: require RGB uint8 NHWC")
            if dataset_config.get("resize_enabled", True):
                values = np.stack([cv2.resize(image, (shape[2], shape[1]),
                    interpolation=interpolation[dataset_config.get("resize_interpolation", "area")]) for image in values])
            result[key] = np.ascontiguousarray(np.moveaxis(values, -1, 1), dtype=np.float32) / 255.
        else:
            result[key] = values.astype(np.float32)
        if result[key].shape[1:] != shape or not np.isfinite(result[key]).all():
            raise ValueError(f"{key}: shape/finite check failed against checkpoint")
    return result


class TianjiRailEnv:
    def __init__(self, config, n_obs_steps, *, motion=False):
        self.config, self.n_obs_steps, self.motion = config, n_obs_steps, motion
        self.recorder = None
        self.error = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._collector = None
        self._arm_sequences = {}
        self.histories = {key: deque() for key in ("camera_0", "camera_1", "camera_2",
                         "arms/left", "arms/right", "hands/left", "hands/right")}
        arm_config = load_arm_config(config["arm_config"])
        hand_config = load_hand_config(config["hand_config"])
        self.arm_profile = ControlProfile(**arm_config["profile"])
        self.hand_profile = ControlProfile(hand_config["profile_id"], "mit", hand_config["parameters"])
        self.kinematics = TianjiKinematics()
        if self.kinematics.model.digest != config["expected_model_sha256"]:
            raise ValueError("Loaded FK model differs from deployment expected_model_sha256")
        self.driver = TianjiDriver(arm_config["controller_ip"])
        self.arm = TianjiCartesianExecutor(self.driver, self.kinematics)
        self.session = WujiSdkSession(user_name=sdk_user_name(hand_config))
        self.hands = {side: WujiHandDriver(side, hand_config["devices"][side]["hand"],
                     feedback_hz=hand_config.get("feedback_hz", 200), timeout_s=config["max_state_age_s"])
                     for side in SIDES}
        camera_config = replace(load_camera_config(config["camera_config"]), main_depth=False)
        self.cameras = CameraRig(camera_config)
        self.runtime = TianjiPolicyRuntime(self.arm, self.hands, config, self.health_problem, self.emit)

    def emit(self, event, **values):
        recorder = self.recorder
        if recorder is not None:
            recorder.event(event, monotonic_ns=time.monotonic_ns(), **values)

    def start(self):
        try:
            self.session.open()
            self.driver.start()
            for hand in self.hands.values():
                hand.start()
            self.cameras.start()
            self.runtime.start()
            self._collector = threading.Thread(target=self._collect, name="policy-observation", daemon=True)
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
        next_state = 0.
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
                            q = np.asarray(sample.payload.arms[side].joints.position_rad, dtype=float)
                            if q.shape != (7,) or not np.isfinite(q).all():
                                raise RuntimeError(f"{side}: invalid measured arm joints")
                            pose = self.kinematics.fk(side, tuple(q))
                            value = np.r_[pose.position_m, Rotation.from_quat(pose.orientation_xyzw).as_rotvec()]
                            self._append(f"arms/{side}", sample.header.received_monotonic_ns, value)
                    for side, hand in self.hands.items():
                        sample = hand.get_latest()
                        if sample is not None and sample.header.valid:
                            q = np.asarray(sample.payload.position_rad, dtype=float)
                            if sample.payload.joint_names != JOINT_NAMES or q.shape != (20,) or not np.isfinite(q).all():
                                raise RuntimeError(f"{side}: invalid measured hand joints")
                            self._append(f"hands/{side}", sample.header.received_monotonic_ns, q)
                if self.runtime.active:
                    problem = self.health_problem()
                    if problem:
                        raise RuntimeError(problem)
                self._stop.wait(.001)
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
                budget = self.config["max_camera_age_s" if key.startswith("camera") else "max_state_age_s"]
                if not history or now - history[-1][0] > budget * 1e9:
                    return f"{key}: missing/stale history"
        return None

    def get_obs(self):
        if self.error:
            raise RuntimeError(self.error)
        problem = self.health_problem()
        if problem:
            raise ObservationNotReady(problem)
        with self._lock:
            histories = {key: list(history) for key, history in self.histories.items()}
        return align_observation(histories, self.n_obs_steps, self.config, now_ns=time.monotonic_ns())

    def engage(self):
        if not self.motion:
            raise RuntimeError("Shadow mode cannot engage devices")
        if any(self.config["workspace"][side] is None for side in SIDES):
            raise ValueError("Fill both measured workspaces before motion")
        obs, _, _ = self.get_obs()
        self.runtime.seed(np.r_[obs["robot_eef_pose"][-1], obs["hand_joint"][-1]])
        try:
            for hand in self.hands.values():
                if hand.profile is None:
                    hand.configure(self.hand_profile)
                hand.engage()
            if self.driver.profile is None:
                self.arm.configure(self.arm_profile)
            self.arm.engage()
            # Use the exact arm engagement reference, not a previous aligned camera pose.
            seed = np.concatenate([np.r_[self.arm.engagement_poses[side].position_m,
                       Rotation.from_quat(self.arm.engagement_poses[side].orientation_xyzw).as_rotvec()]
                       for side in SIDES] + [np.asarray(self.hands[side].last_target.position_rad) for side in SIDES])
            self.runtime.seed(seed, arm_ready=True)
        except Exception as error:
            self.runtime.pause(error, fault=True)
            raise

    def exec_actions(self, actions, anchor_ns):
        if self.motion:
            return self.runtime.submit(actions, anchor_ns)
        selected, times, dropped = future_actions(actions, anchor_ns, time.monotonic_ns(),
            self.config["policy_hz"], self.config["submit_margin_s"])
        self.emit("shadow_schedule", times_ns=times, actions=selected, dropped_steps=dropped)
        return dropped

    def pause(self, reason="Operator pause"):
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
