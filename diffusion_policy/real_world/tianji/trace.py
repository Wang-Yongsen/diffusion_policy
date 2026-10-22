"""Bounded in-memory trace of one short Tianji rail trial.

Images stay in preallocated arrays and are written only after motion stops.
The control thread copies numbers already produced by the servo; it does not
compress, write NPZ, or run extra forward kinematics. A full buffer marks the
trace invalid and drops the new item. Missing joints stay missing.
"""

import json
import subprocess
import threading
import time
from functools import wraps
from pathlib import Path

import numpy as np

from diffusion_policy.common.file_util import file_sha256

CAMERAS = ("camera_0", "camera_1", "camera_2")
SIDES = ("left", "right")
SCHEMA = {
    "robot_eef_pose": "录制／控制坐标：双臂各 xyz [m]＋rotvec [rad]，12维；模型预处理转为14维四元数观测。",
    "actions": "解码后的52维绝对控制目标（双臂 xyz＋rotvec、双手关节）；模型原始动作是58维 xyz＋6D旋转及手关节。",
    "command_fk": "已接受关节命令的正解，来自 cartesian_targets / applied_poses，不是实测法兰。",
    "position_error_m": "请求位姿到命令正解的剩余距离，不是实测跟踪误差。",
    "controller_target_rad": "控制器回报的关节命令。缺关节保持缺失，不补 0。",
    "measured_q": "该采样时刻的实测关节。实测正解在离线阶段计算。",
    "real_images": "real_present 为 false 的图像行不是观测。",
    "sdk_receipt": "SDK 接受命令，不是 UDP 送达，也不是物理到位。",
}


def finite_joints(values, width):
    """Return a float vector, or None when any joint is missing or non-finite."""
    if values is None or len(values) != width:
        return None
    result = np.empty(width, np.float64)
    for index, value in enumerate(values):
        if value is None or not np.isfinite(value):
            return None
        result[index] = float(value)
    return result


def runtime_files():
    """Files actually imported for this deployment, not a remembered list of names."""
    import diffusion_policy.env_runner.tianji_rail_real_runner as runner
    import diffusion_policy.real_world.tianji.env as env
    import diffusion_policy.real_world.tianji.inference as inference
    import diffusion_policy.real_world.tianji.runtime as runtime
    import diffusion_policy.real_world.tianji.trace as trace
    from diffusion_policy.common import image_preprocessing, tianji_rail_pose
    from diffusion_policy.third_party.bimanual_teleop.control.arm import (
        cartesian,
        servo,
    )
    from diffusion_policy.third_party.bimanual_teleop.devices.tianji import driver

    from . import checkpoint, config, observation, recording

    modules = {
        "tianji_rail_env": env,
        "tianji_policy_runtime": runtime,
        "policy_inference_process": inference,
        "tianji_rail_real_runner": runner,
        "decision_trace": trace,
        "cartesian": cartesian,
        "servo": servo,
        "driver": driver,
        "checkpoint": checkpoint,
        "deployment_config": config,
        "observation": observation,
        "recording": recording,
        "image_preprocessing": image_preprocessing,
        "tianji_rail_pose": tianji_rail_pose,
    }
    return {name: Path(module.__file__).resolve() for name, module in modules.items()}


def write_provenance(directory, files, root):
    """Commit, dirty diff, and hashes of the modules this process imported."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    commit = _git(root, "rev-parse", "HEAD")
    diff = _git(root, "diff", "HEAD")
    status = _git(root, "status", "--porcelain")
    hashed = {
        name: {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}
        for name, path in files.items()
    }
    payload = {
        "commit": commit.strip(),
        "dirty": bool(status.strip()),
        "status": status,
        "files": hashed,
        "schema": SCHEMA,
    }
    (directory / "source_provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    (directory / "source.diff").write_text(diff)
    return payload


def snapshot_arm_feedback(sample):
    """Plain numbers from the sample a submit just read. None stays missing."""
    if sample is None:
        return {"present": False}
    arms = {}
    for side in SIDES:
        arm = sample.payload.arms.get(side) if hasattr(sample.payload, "arms") else None
        if arm is None:
            arms[side] = {"present": False}
            continue
        arms[side] = {
            "present": True,
            "source_sequence": int(arm.source_sequence),
            "input_sequence": int(arm.input_sequence),
            "q": finite_joints(arm.joints.position_rad, 7),
            "controller_target": finite_joints(arm.controller_target_rad, 7),
        }
    return {
        "present": True,
        "packet_index": int(sample.payload.packet_index),
        "received_monotonic_ns": int(sample.header.received_monotonic_ns),
        "ref_sequence": int(sample.header.ref.sequence),
        "arms": arms,
    }


class TraceSink:
    """Driver observer. It always accepts; a copy failure invalidates the trace."""

    def __init__(self, trace):
        self.trace = trace

    def try_publish(self, sample):
        try:
            self.trace.append_sample(sample)
        except Exception as error:
            self.trace.mark_invalid(f"sample copy failed: {error}")
        return True

    def try_event(self, event):
        try:
            kind = getattr(event, "kind", None)
            if kind == "tianji_command_submitted":
                self.trace.record_driver_command(event)
            elif hasattr(event, "command_id") and hasattr(event, "status"):
                # A hand receipt is not an arm submission with accepted=False.
                if getattr(event, "device_id", None) == "tianji":
                    self.trace.record_receipt(event)
            else:
                self.trace.record_status(event)
        except Exception as error:
            self.trace.mark_invalid(f"event copy failed: {error}")
        return True


def _serialized_trace_write(method):
    """Commit each row before begin/reset/flush can replace or save its arrays."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


class SessionTrace:
    def __init__(
        self,
        *,
        decision_capacity=48,
        feedback_capacity=60000,
        command_capacity=20000,
        hand_capacity=20000,
        status_capacity=1024,
    ):
        for name, value in (
            ("decision_capacity", decision_capacity),
            ("feedback_capacity", feedback_capacity),
            ("command_capacity", command_capacity),
            ("hand_capacity", hand_capacity),
            ("status_capacity", status_capacity),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.decision_capacity = decision_capacity
        self.sink = TraceSink(self)
        self._lock = threading.RLock()
        self._tls = threading.local()
        self._frozen = False
        self.valid = True
        self.invalid_reason = None
        self.dropped = {"decisions": 0, "feedback": 0, "commands": 0, "hands": 0, "status": 0}
        self.decisions_ready = False
        self.capture_real = False
        self.fb_count = 0
        self.cmd_count = 0
        self.hand_count = 0
        self.hfb_count = 0
        self.decision_count = 0
        self.meta = []
        self._cmd_index = {}
        self._status = []
        self.status_summary = {}
        self.recording_start_ns = None
        self.recording_end_ns = None
        self._status_capacity = status_capacity
        self._allocate_feedback(feedback_capacity)
        self._allocate_commands(command_capacity)
        self._allocate_hands(hand_capacity)

    def mark_invalid(self, reason):
        with self._lock:
            self.valid = False
            if self.invalid_reason is None:
                self.invalid_reason = str(reason)

    def begin_trial(self):
        """Call while disengaged, after allocation, before starting the trial."""
        with self._lock:
            self._reset_counts()
            self.recording_start_ns = time.monotonic_ns()
            self.recording_end_ns = None

    def install_on(self, executor):
        """Remember the feedback sample cartesian submit reads, without changing the servo."""
        if getattr(executor, "_trace_wrapped", False):
            return
        real_submit = executor.submit
        real_latest = executor.driver.get_latest
        trace = self

        def wrapped_latest():
            sample = real_latest()
            box = getattr(trace._tls, "box", None)
            if box is not None and "feedback" not in box:
                try:
                    box["feedback"] = snapshot_arm_feedback(sample)
                except Exception as error:
                    box["feedback_error"] = str(error)
            return sample

        def submit(target, *args, **kwargs):
            box = {}
            trace._tls.box = box
            executor.driver.get_latest = wrapped_latest
            try:
                result = real_submit(target, *args, **kwargs)
            except Exception:
                try:
                    trace.record_submit_result(target, None, box, executor)
                except Exception as error:
                    trace.mark_invalid(error)
                raise
            finally:
                executor.driver.get_latest = real_latest
                trace._tls.box = None
            try:
                trace.record_submit_result(target, result, box, executor)
            except Exception as error:
                trace.mark_invalid(error)
            return result

        executor.submit = submit
        executor._trace_wrapped = True

    def allocate_decisions(self, example, n_action, *, capture_real=False):
        if self.decisions_ready:
            raise RuntimeError("Decision buffers are already allocated")
        image = np.asarray(example["camera_0"])
        if image.dtype != np.uint8 or image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("camera_0 must be uint8 NHWC")
        n_obs, height, width, _ = image.shape
        if type(n_action) is not int or n_action < 1:
            raise ValueError("n_action must be a positive integer")
        for camera in CAMERAS:
            other = np.asarray(example[camera])
            if other.shape != image.shape or other.dtype != np.uint8:
                raise ValueError(f"{camera} does not match camera_0")
        capacity = self.decision_capacity
        self.images = {
            camera: np.empty((capacity, n_obs, height, width, 3), np.uint8) for camera in CAMERAS
        }
        self.eef = np.empty((capacity, n_obs, 12), np.float32)
        self.hands = np.empty((capacity, n_obs, 40), np.float32)
        self.actions = np.full((capacity, n_action, 52), np.nan, np.float64)
        self.action_rows = np.zeros(capacity, np.int16)
        self.anchor_ns = np.full(capacity, -1, np.int64)
        self.started_ns = np.full(capacity, -1, np.int64)
        self.observed_ns = np.full(capacity, -1, np.int64)
        self.preprocessed_ns = np.full(capacity, -1, np.int64)
        self.inputs_ready_ns = np.full(capacity, -1, np.int64)
        self.returned_ns = np.full(capacity, -1, np.int64)
        self.window_start = np.full(capacity, -1, np.int64)
        self.present = np.zeros(capacity, bool)
        self.meta = [None] * capacity
        self.capture_real = bool(capture_real)
        if self.capture_real:
            self.real_images = {camera: np.empty_like(self.images[camera]) for camera in CAMERAS}
            self.real_eef = np.empty_like(self.eef)
            self.real_hands = np.empty_like(self.hands)
            self.real_anchor_ns = np.full(capacity, -1, np.int64)
            self.real_present = np.zeros(capacity, bool)
        else:
            self.real_images = self.real_eef = self.real_hands = self.real_anchor_ns = None
            self.real_present = np.zeros(capacity, bool)
        # Allocation is done while disengaged. Touch pages now, not in a formal decision.
        for buffer in self.images.values():
            buffer.fill(0)
        if self.real_images is not None:
            for buffer in self.real_images.values():
                buffer.fill(0)
        self.n_obs = n_obs
        self.n_action = n_action
        self.decisions_ready = True

    @property
    def image_mebibytes(self):
        if not self.decisions_ready:
            return 0.0
        arrays = list(self.images.values())
        if self.real_images is not None:
            arrays.extend(self.real_images.values())
        return sum(array.nbytes for array in arrays) / (1024**2)

    @_serialized_trace_write
    def store_decision(
        self,
        *,
        obs,
        alignment,
        anchor_ns,
        actions,
        started_ns,
        observed_ns,
        preprocessed_ns,
        inputs_ready_ns,
        returned_ns,
        rng=None,
        real_obs=None,
        real_alignment=None,
        real_anchor_ns=None,
        window_start=None,
        real_missing=None,
        control_input="real",
        live_capture_started_ns=None,
        live_capture_returned_ns=None,
    ):
        """Copy one decision into the next free slot. Overflow does not raise."""
        if not self.decisions_ready:
            raise RuntimeError("Allocate decision buffers before storing")
        with self._lock:
            if self._frozen:
                return None
            if self.decision_count >= self.decision_capacity:
                self.dropped["decisions"] += 1
                self.mark_invalid("decision buffer full")
                return None
            index = self.decision_count
            self.decision_count += 1
        self._copy_observation(self.images, self.eef, self.hands, index, obs)
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 52 or actions.shape[0] > self.n_action:
            self.mark_invalid("decision actions do not fit the buffer")
            return None
        self.actions[index].fill(np.nan)
        self.actions[index, : len(actions)] = actions
        self.action_rows[index] = len(actions)
        self.anchor_ns[index] = int(anchor_ns)
        self.started_ns[index] = int(started_ns)
        self.observed_ns[index] = int(observed_ns)
        self.preprocessed_ns[index] = int(preprocessed_ns)
        self.inputs_ready_ns[index] = int(inputs_ready_ns)
        self.returned_ns[index] = int(returned_ns)
        self.window_start[index] = -1 if window_start is None else int(window_start)
        if self.real_anchor_ns is not None:
            self.real_anchor_ns[index] = -1
        self.real_present[index] = False
        real_present = False
        if real_obs is not None:
            if self.real_images is None:
                raise RuntimeError("This trace was allocated without a real-observation buffer")
            self._copy_observation(
                self.real_images, self.real_eef, self.real_hands, index, real_obs
            )
            if real_anchor_ns is not None:
                self.real_anchor_ns[index] = int(real_anchor_ns)
            real_present = True
            self.real_present[index] = True
        rng_cpu, rng_cuda = _split_rng(rng)
        self.meta[index] = {
            "prediction_id": index,
            "control_input": control_input,
            "alignment": _plain(alignment),
            "real_alignment": _plain(real_alignment),
            "real_missing": real_missing,
            "real_present": real_present,
            "rng_device": None if not isinstance(rng, dict) else rng.get("device"),
            "rng_cuda_present": rng_cuda is not None,
            "live_capture_started_ns": live_capture_started_ns,
            "live_capture_returned_ns": live_capture_returned_ns,
        }
        self._rng_cpu[index] = rng_cpu
        self._rng_cuda[index] = rng_cuda
        self.present[index] = True
        return index

    @_serialized_trace_write
    def append_sample(self, sample):
        payload = getattr(sample, "payload", None)
        if payload is None:
            return
        if hasattr(payload, "arms"):
            self._append_arm(snapshot_arm_feedback(sample))
            return
        side = _hand_index(sample)
        if side is None:
            return
        joints = finite_joints(getattr(payload, "position_rad", None), 20)
        header = sample.header
        with self._lock:
            if self._frozen:
                return
            if self.hfb_count >= len(self.hfb_ns):
                self.dropped["hands"] += 1
                self.mark_invalid("hand feedback buffer full")
                return
            row = self.hfb_count
            self.hfb_count += 1
        self.hfb_ns[row] = int(header.received_monotonic_ns)
        self.hfb_seq[row] = -1 if header.source_sequence is None else int(header.source_sequence)
        self.hfb_side[row] = side
        if joints is not None:
            self.hfb_q[row] = joints
            self.hfb_present[row] = True

    def record_driver_command(self, event):
        command = event.details["command"]
        payload = command.payload
        sink_ns = time.monotonic_ns()
        with self._lock:
            row = self._row_for(command.command_id)
            if row is None:
                return
            self.cmd_driver_stamp[row] = int(event.observed_monotonic_ns)
            self.cmd_sink_ns[row] = int(sink_ns)
            for index, side in enumerate(SIDES):
                joints = finite_joints(payload.targets.get(side), 7)
                if joints is not None:
                    self.cmd_q[row, index] = joints
                    self.cmd_q_present[row, index] = True
                _write_pose(
                    self.cmd_fk_xyz,
                    self.cmd_fk_quat,
                    self.cmd_fk_present,
                    row,
                    index,
                    payload.cartesian_targets.get(side),
                )
                requested = getattr(payload, "requested_cartesian_targets", {}) or {}
                _write_pose(
                    self.req_xyz, self.req_quat, self.req_present, row, index, requested.get(side)
                )

    def record_receipt(self, event):
        status = event.status.value if hasattr(event.status, "value") else str(event.status)
        with self._lock:
            row = self._row_for(event.command_id)
            if row is None:
                return
            self.cmd_receipt[row] = status
            self.cmd_receipt_ns[row] = int(event.observed_monotonic_ns)
            if status == "accepted":
                self.cmd_sdk_accepted[row] = True

    def record_status(self, event):
        kind = getattr(event, "kind", type(event).__name__)
        details = getattr(event, "details", {}) or {}
        safe = {
            key: value
            for key, value in details.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        stamp = getattr(event, "observed_monotonic_ns", None)
        with self._lock:
            if self._frozen:
                return
            key = str(kind)
            summary = self.status_summary.setdefault(
                key, {"count": 0, "first_ns": stamp, "last_ns": stamp, "last_details": {}}
            )
            summary["count"] += 1
            summary["last_ns"] = stamp
            summary["last_details"] = safe
            # Source-cache sampling gaps are counted, not allowed to evict faults.
            if key == "tianji.source_gap":
                return
            if len(self._status) >= self._status_capacity:
                self.dropped["status"] += 1
                self.mark_invalid("status buffer full")
                return
            self._status.append(
                {
                    "kind": str(kind),
                    "observed_monotonic_ns": None if stamp is None else int(stamp),
                    "details": safe,
                }
            )

    def record_submit_result(self, target, result, box, executor):
        """Attach the feedback read inside this submit. Rejected calls do not reuse an older FK."""
        command_id = getattr(target, "command_id", None)
        accepted = bool(result is not None and result.accepted)
        feedback = box.get("feedback") if isinstance(box, dict) else None
        with self._lock:
            row = self._row_for(command_id)
            if row is None:
                return
            self.cmd_accepted[row] = accepted
            self.cmd_accepted_present[row] = result is not None
            self._write_feedback(row, feedback)
            if not accepted:
                return
            tracking = getattr(executor, "tracking_status", {}) or {}
            for index, side in enumerate(SIDES):
                _write_pose(
                    self.cmd_fk_xyz,
                    self.cmd_fk_quat,
                    self.cmd_fk_present,
                    row,
                    index,
                    (getattr(executor, "applied_poses", {}) or {}).get(side),
                )
                info = tracking.get(side) if isinstance(tracking, dict) else None
                if not isinstance(info, dict):
                    continue
                if info.get("limited") is not None:
                    self.cmd_limited[row, index] = bool(info["limited"])
                    self.cmd_limited_present[row, index] = True
                progress = info.get("progress")
                if progress is not None and np.isfinite(progress):
                    self.cmd_progress[row, index] = float(progress)
                error = info.get("position_error_m")
                if error is not None and np.isfinite(error):
                    self.cmd_position_error_m[row, index] = float(error)
            elapsed = (getattr(executor, "timing_status", {}) or {}).get("elapsed_ns")
            if elapsed is not None and np.isfinite(elapsed):
                self.cmd_elapsed_ns[row] = int(elapsed)

    def annotate_command(self, command_id, *, prediction_id, chunk_id, step_id, started_ns):
        with self._lock:
            row = self._row_for(command_id)
            if row is None:
                return
            if prediction_id is not None:
                self.cmd_prediction_id[row] = int(prediction_id)
            if chunk_id is not None:
                self.cmd_chunk_id[row] = int(chunk_id)
            if step_id is not None and step_id >= 0:
                self.cmd_step_id[row] = int(step_id)
                self.cmd_step_present[row] = True
            if started_ns is not None:
                self.cmd_started_ns[row] = int(started_ns)

    @_serialized_trace_write
    def note_hand_command(
        self, *, command_id, side, q, prediction_id, chunk_id, started_ns, accepted
    ):
        joints = finite_joints(q, 20)
        with self._lock:
            if self._frozen:
                return
            if self.hand_count >= len(self.hand_ns):
                self.dropped["hands"] += 1
                self.mark_invalid("hand command buffer full")
                return
            row = self.hand_count
            self.hand_count += 1
        self.hand_id[row] = "" if command_id is None else str(command_id)
        self.hand_side[row] = SIDES.index(side)
        self.hand_ns[row] = int(started_ns)
        self.hand_accepted[row] = bool(accepted)
        self.hand_present[row] = joints is not None
        if joints is not None:
            self.hand_q[row] = joints
        if prediction_id is not None:
            self.hand_prediction[row] = int(prediction_id)
        if chunk_id is not None:
            self.hand_chunk[row] = int(chunk_id)

    def flush(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._frozen = True
            self.recording_end_ns = time.monotonic_ns()
            counts = {
                "decisions": self.decision_count,
                "feedback": self.fb_count,
                "commands": self.cmd_count,
                "hand_commands": self.hand_count,
                "hand_feedback": self.hfb_count,
            }
            valid, reason, dropped = self.valid, self.invalid_reason, dict(self.dropped)
            status = list(self._status)
            status_summary = _plain(self.status_summary)
            recording_start_ns = self.recording_start_ns
            recording_end_ns = self.recording_end_ns
        try:
            self._write_arrays(directory, counts)
            meta = {
                "schema": SCHEMA,
                "valid": valid and reason is None,
                "invalid_reason": reason,
                "dropped": dropped,
                "counts": counts,
                "capture_real": self.capture_real,
                "status": status,
                "decisions": [item for item in self.meta[: counts["decisions"]]],
            }
            meta.update(
                recording_start_ns=recording_start_ns,
                recording_end_ns=recording_end_ns,
                status_summary=status_summary,
                critical_status_dropped=dropped["status"],
            )
            # A frozen drop during the copy is part of this trial.
            with self._lock:
                meta["valid"] = self.valid and self.invalid_reason is None
                meta["invalid_reason"] = self.invalid_reason
                meta["dropped"] = dict(self.dropped)
            (directory / "trace_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n"
            )
            return meta
        finally:
            with self._lock:
                self._reset_counts()
                self._frozen = False

    def _copy_observation(self, images, eef, hands, index, obs):
        for camera in CAMERAS:
            values = np.asarray(obs[camera])
            if values.shape != self.images[camera][index].shape or values.dtype != np.uint8:
                raise ValueError(f"{camera} does not fit the decision buffer")
            np.copyto(images[camera][index], values)
        pose = np.asarray(obs["robot_eef_pose"], dtype=np.float32)
        joints = np.asarray(obs["hand_joint"], dtype=np.float32)
        if pose.shape != eef[index].shape or joints.shape != hands[index].shape:
            raise ValueError("Low-dimensional observation does not fit the decision buffer")
        np.copyto(eef[index], pose)
        np.copyto(hands[index], joints)

    @_serialized_trace_write
    def _append_arm(self, snap):
        with self._lock:
            if self._frozen:
                return
            if self.fb_count >= len(self.fb_ns):
                self.dropped["feedback"] += 1
                self.mark_invalid("arm feedback buffer full")
                return
            row = self.fb_count
            self.fb_count += 1
        if not snap.get("present"):
            return
        self.fb_ns[row] = int(snap["received_monotonic_ns"])
        self.fb_packet[row] = int(snap["packet_index"])
        self.fb_ref[row] = int(snap["ref_sequence"])
        for index, side in enumerate(SIDES):
            arm = snap["arms"].get(side) or {}
            if not arm.get("present"):
                continue
            self.fb_source_seq[row, index] = int(arm["source_sequence"])
            if arm["q"] is not None:
                self.fb_q[row, index] = arm["q"]
                self.fb_q_present[row, index] = True
            if arm["controller_target"] is not None:
                self.fb_target[row, index] = arm["controller_target"]
                self.fb_target_present[row, index] = True

    def _write_feedback(self, row, feedback):
        if not isinstance(feedback, dict) or not feedback.get("present"):
            return
        self.cmd_feedback_packet[row] = int(feedback["packet_index"])
        self.cmd_feedback_ns[row] = int(feedback["received_monotonic_ns"])
        for index, side in enumerate(SIDES):
            arm = feedback["arms"].get(side) or {}
            if not arm.get("present"):
                continue
            self.cmd_feedback_seq[row, index] = int(arm["source_sequence"])
            if arm["q"] is not None:
                self.cmd_feedback_q[row, index] = arm["q"]
                self.cmd_feedback_q_present[row, index] = True
            if arm["controller_target"] is not None:
                self.cmd_controller_target[row, index] = arm["controller_target"]
                self.cmd_controller_present[row, index] = True

    def _row_for(self, command_id):
        if self._frozen:
            return None
        if command_id is None:
            self.mark_invalid("command is missing command_id")
            return None
        command_id = str(command_id)
        existing = self._cmd_index.get(command_id)
        if existing is not None:
            return existing
        if self.cmd_count >= len(self.cmd_id):
            self.dropped["commands"] += 1
            self.mark_invalid("command buffer full")
            return None
        row = self.cmd_count
        self.cmd_count += 1
        self.cmd_id[row] = command_id
        self._cmd_index[command_id] = row
        self.cmd_row_present[row] = True
        return row

    def _allocate_feedback(self, capacity):
        self.fb_q = np.full((capacity, 2, 7), np.nan)
        self.fb_target = np.full((capacity, 2, 7), np.nan)
        self.fb_q_present = np.zeros((capacity, 2), bool)
        self.fb_target_present = np.zeros((capacity, 2), bool)
        self.fb_ns = np.full(capacity, -1, np.int64)
        self.fb_packet = np.full(capacity, -1, np.int64)
        self.fb_ref = np.full(capacity, -1, np.int64)
        self.fb_source_seq = np.full((capacity, 2), -1, np.int64)

    def _allocate_commands(self, capacity):
        self.cmd_id = np.empty(capacity, dtype="U80")
        self.cmd_q = np.full((capacity, 2, 7), np.nan)
        self.cmd_q_present = np.zeros((capacity, 2), bool)
        self.cmd_fk_xyz = np.full((capacity, 2, 3), np.nan)
        self.cmd_fk_quat = np.full((capacity, 2, 4), np.nan)
        self.cmd_fk_present = np.zeros((capacity, 2), bool)
        self.req_xyz = np.full((capacity, 2, 3), np.nan)
        self.req_quat = np.full((capacity, 2, 4), np.nan)
        self.req_present = np.zeros((capacity, 2), bool)
        self.cmd_limited = np.zeros((capacity, 2), bool)
        self.cmd_limited_present = np.zeros((capacity, 2), bool)
        self.cmd_progress = np.full((capacity, 2), np.nan)
        self.cmd_position_error_m = np.full((capacity, 2), np.nan)
        self.cmd_accepted = np.zeros(capacity, bool)
        self.cmd_accepted_present = np.zeros(capacity, bool)
        self.cmd_device_id = np.full(capacity, "tianji", dtype="U16")
        self.cmd_row_present = np.zeros(capacity, bool)
        self.cmd_feedback_packet = np.full(capacity, -1, np.int64)
        self.cmd_feedback_ns = np.full(capacity, -1, np.int64)
        self.cmd_feedback_seq = np.full((capacity, 2), -1, np.int64)
        self.cmd_feedback_q = np.full((capacity, 2, 7), np.nan)
        self.cmd_feedback_q_present = np.zeros((capacity, 2), bool)
        self.cmd_controller_target = np.full((capacity, 2, 7), np.nan)
        self.cmd_controller_present = np.zeros((capacity, 2), bool)
        self.cmd_started_ns = np.full(capacity, -1, np.int64)
        self.cmd_sink_ns = np.full(capacity, -1, np.int64)
        self.cmd_driver_stamp = np.full(capacity, -1, np.int64)
        self.cmd_receipt_ns = np.full(capacity, -1, np.int64)
        self.cmd_elapsed_ns = np.full(capacity, -1, np.int64)
        self.cmd_prediction_id = np.full(capacity, -1, np.int64)
        self.cmd_chunk_id = np.full(capacity, -1, np.int64)
        self.cmd_step_id = np.full(capacity, -1, np.int64)
        self.cmd_step_present = np.zeros(capacity, bool)
        self.cmd_sdk_accepted = np.zeros(capacity, bool)
        self.cmd_receipt = np.empty(capacity, dtype="U32")

    def _allocate_hands(self, capacity):
        self.hand_id = np.empty(capacity, dtype="U80")
        self.hand_q = np.full((capacity, 20), np.nan)
        self.hand_side = np.full(capacity, -1, np.int8)
        self.hand_ns = np.full(capacity, -1, np.int64)
        self.hand_prediction = np.full(capacity, -1, np.int64)
        self.hand_chunk = np.full(capacity, -1, np.int64)
        self.hand_present = np.zeros(capacity, bool)
        self.hand_accepted = np.zeros(capacity, bool)
        self.hfb_q = np.full((capacity, 20), np.nan)
        self.hfb_side = np.full(capacity, -1, np.int8)
        self.hfb_ns = np.full(capacity, -1, np.int64)
        self.hfb_seq = np.full(capacity, -1, np.int64)
        self.hfb_present = np.zeros(capacity, bool)
        self._rng_cpu = [None] * self.decision_capacity
        self._rng_cuda = [None] * self.decision_capacity

    def _reset_counts(self):
        feedback, commands, hands = len(self.fb_ns), len(self.cmd_id), len(self.hand_ns)
        self._allocate_feedback(feedback)
        self._allocate_commands(commands)
        self._allocate_hands(hands)
        self.fb_count = self.cmd_count = self.hand_count = self.hfb_count = self.decision_count = 0
        self._cmd_index.clear()
        self._status.clear()
        self.status_summary.clear()
        self.valid = True
        self.invalid_reason = None
        self.dropped = {key: 0 for key in self.dropped}
        if self.decisions_ready:
            self.present[:] = False
            self.real_present[:] = False
            self.meta = [None] * self.decision_capacity

    def _write_arrays(self, directory, counts):
        decisions = counts["decisions"]
        payload = (
            dict(
                present=self.present[:decisions],
                camera_0=self.images["camera_0"][:decisions],
                camera_1=self.images["camera_1"][:decisions],
                camera_2=self.images["camera_2"][:decisions],
                robot_eef_pose=self.eef[:decisions],
                hand_joint=self.hands[:decisions],
                actions=self.actions[:decisions],
                action_rows=self.action_rows[:decisions],
                anchor_ns=self.anchor_ns[:decisions],
                started_ns=self.started_ns[:decisions],
                observed_ns=self.observed_ns[:decisions],
                preprocessed_ns=self.preprocessed_ns[:decisions],
                inputs_ready_ns=self.inputs_ready_ns[:decisions],
                returned_ns=self.returned_ns[:decisions],
                window_start=self.window_start[:decisions],
                real_present=self.real_present[:decisions],
            )
            if self.decisions_ready
            else {"present": np.zeros(0, bool)}
        )
        if self.decisions_ready and self.capture_real:
            payload.update(
                real_camera_0=self.real_images["camera_0"][:decisions],
                real_camera_1=self.real_images["camera_1"][:decisions],
                real_camera_2=self.real_images["camera_2"][:decisions],
                real_robot_eef_pose=self.real_eef[:decisions],
                real_hand_joint=self.real_hands[:decisions],
                real_anchor_ns=self.real_anchor_ns[:decisions],
            )
        cpus = self._rng_cpu[:decisions]
        if cpus and all(item is not None and item.shape == cpus[0].shape for item in cpus):
            payload["rng_cpu"] = np.stack(cpus)
        cudas = self._rng_cuda[:decisions]
        if cudas and all(item is not None and item.shape == cudas[0].shape for item in cudas):
            payload["rng_cuda"] = np.stack(cudas)
        np.savez(directory / "trace_decisions.npz", **payload)
        feedback = counts["feedback"]
        np.savez(
            directory / "trace_feedback.npz",
            received_ns=self.fb_ns[:feedback],
            packet=self.fb_packet[:feedback],
            ref_sequence=self.fb_ref[:feedback],
            source_sequence=self.fb_source_seq[:feedback],
            q=self.fb_q[:feedback],
            q_present=self.fb_q_present[:feedback],
            controller_target=self.fb_target[:feedback],
            controller_present=self.fb_target_present[:feedback],
        )
        commands = counts["commands"]
        np.savez(
            directory / "trace_commands.npz",
            command_id=self.cmd_id[:commands],
            q_cmd=self.cmd_q[:commands],
            q_cmd_present=self.cmd_q_present[:commands],
            command_fk_xyz=self.cmd_fk_xyz[:commands],
            command_fk_quat=self.cmd_fk_quat[:commands],
            command_fk_present=self.cmd_fk_present[:commands],
            requested_xyz=self.req_xyz[:commands],
            requested_quat=self.req_quat[:commands],
            requested_present=self.req_present[:commands],
            limited=self.cmd_limited[:commands],
            limited_present=self.cmd_limited_present[:commands],
            progress=self.cmd_progress[:commands],
            position_error_m=self.cmd_position_error_m[:commands],
            accepted=self.cmd_accepted[:commands],
            accepted_present=self.cmd_accepted_present[:commands],
            device_id=self.cmd_device_id[:commands],
            started_ns=self.cmd_started_ns[:commands],
            sink_received_ns=self.cmd_sink_ns[:commands],
            driver_event_stamp=self.cmd_driver_stamp[:commands],
            receipt=self.cmd_receipt[:commands],
            receipt_ns=self.cmd_receipt_ns[:commands],
            sdk_accepted=self.cmd_sdk_accepted[:commands],
            elapsed_ns=self.cmd_elapsed_ns[:commands],
            prediction_id=self.cmd_prediction_id[:commands],
            chunk_id=self.cmd_chunk_id[:commands],
            step_id=self.cmd_step_id[:commands],
            step_present=self.cmd_step_present[:commands],
            feedback_packet=self.cmd_feedback_packet[:commands],
            feedback_received_ns=self.cmd_feedback_ns[:commands],
            feedback_source_sequence=self.cmd_feedback_seq[:commands],
            feedback_q=self.cmd_feedback_q[:commands],
            feedback_q_present=self.cmd_feedback_q_present[:commands],
            controller_target=self.cmd_controller_target[:commands],
            controller_present=self.cmd_controller_present[:commands],
        )
        hands = counts["hand_commands"]
        seen = counts["hand_feedback"]
        np.savez(
            directory / "trace_hands.npz",
            command_id=self.hand_id[:hands],
            q=self.hand_q[:hands],
            side=self.hand_side[:hands],
            started_ns=self.hand_ns[:hands],
            prediction_id=self.hand_prediction[:hands],
            chunk_id=self.hand_chunk[:hands],
            present=self.hand_present[:hands],
            accepted=self.hand_accepted[:hands],
            feedback_q=self.hfb_q[:seen],
            feedback_side=self.hfb_side[:seen],
            feedback_received_ns=self.hfb_ns[:seen],
            feedback_source_sequence=self.hfb_seq[:seen],
            feedback_present=self.hfb_present[:seen],
        )


def _write_pose(xyz, quat, present, row, index, pose):
    if pose is None:
        return
    position = np.asarray(pose.position_m, dtype=np.float64)
    orientation = np.asarray(pose.orientation_xyzw, dtype=np.float64)
    if (
        position.shape != (3,)
        or orientation.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(orientation).all()
    ):
        return
    xyz[row, index] = position
    quat[row, index] = orientation
    present[row, index] = True


def _hand_index(sample):
    stream = str(
        getattr(getattr(sample, "header", None), "ref", None).stream
        if getattr(sample, "header", None)
        else ""
    )
    payload = sample.payload
    if not hasattr(payload, "position_rad") or not hasattr(payload, "joint_names"):
        return None
    if not stream.endswith("/joints"):
        return None
    if stream.startswith("wuji_left"):
        return 0
    if stream.startswith("wuji_right"):
        return 1
    return None


def _split_rng(rng):
    if not isinstance(rng, dict):
        return None, None
    cpu = rng.get("cpu")
    cuda = rng.get("cuda")
    cpu = None if cpu is None else np.asarray(cpu).copy()
    cuda = None if cuda is None else np.asarray(cuda).copy()
    return cpu, cuda


def _plain(value):
    if value is None:
        return None
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return result.stdout
