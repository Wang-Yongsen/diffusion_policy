"""Trial event logging and inference. The operator loop remains responsive to stop."""

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from queue import Full, Queue
import select
import sys
import threading
import time

import numpy as np

from diffusion_policy.real_world.tianji_rail_env import ObservationNotReady, preprocess_observation


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=json_value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def timing_report(path):
    samples, arrivals = {}, {}
    def add(key, value):
        samples.setdefault(key, []).append(value)
    with (Path(path) / "events.jsonl").open() as file:
        for line in file:
            item = json.loads(line)
            event = item["event"]
            if event == "prediction":
                add("preprocess_inference_transfer_ms", (item["returned_ns"] - item["started_ns"]) / 1e6)
                add("observation_age_at_prediction_return_ms", (item["returned_ns"] - item["anchor_ns"]) / 1e6)
                # Older sessions only contain the combined duration above.
                for name, start, end in (
                        ("observation_read_ms", "started_ns", "observed_ns"),
                        ("preprocess_ms", "observed_ns", "preprocessed_ns"),
                        ("input_transfer_ms", "preprocessed_ns", "inputs_ready_ns"),
                        ("inference_output_transfer_ms", "inputs_ready_ns", "returned_ns")):
                    if start in item and end in item:
                        add(name, (item[end] - item[start]) / 1e6)
                add("observation_age_at_processing_start_ms", (item["started_ns"] - item["anchor_ns"]) / 1e6)
                if item.get("chunk_completed_ns") is not None:
                    add("sync_hold_ms", (item["returned_ns"] - item["chunk_completed_ns"]) / 1e6)
            elif event == "submission":
                control = item.get("side") or item["kind"]
                add(f"{control}_submission_call_ms", (item["returned_ns"] - item["started_ns"]) / 1e6)
                arrivals.setdefault(f"{control}_control", []).append(item["started_ns"])
                if item["mode"] == "policy" and item["planned_ns"] is not None:
                    add(f"{item['kind']}_target_age_at_submission_ms", (item["started_ns"] - item["planned_ns"]) / 1e6)
    for key, stamps in arrivals.items():
        if len(stamps) > 1:
            samples[f"{key}_interval_ms"] = (np.diff(stamps) / 1e6).tolist()
    report = {key: {"count": len(values), "median": float(np.median(values)),
                   "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
                   "max": float(np.max(values))} for key, values in samples.items() if values}
    report["scope"] = "host monotonic timestamps; SDK acceptance does not establish physical arrival"
    write_json(Path(path) / "timing.json", report)
    return report


class TrialRecorder:
    """Bounded asynchronous event log. Logging problems are counted, never raised into control."""

    def __init__(self, path, metadata, capacity=256):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.error = None
        self.dropped = 0
        self.predictions = 0
        self._closed = False
        self._drop_lock = threading.Lock()
        self._queue = Queue(capacity)
        write_json(self.path / "trial.json", {**metadata, "status": "running"})
        self.metadata = metadata
        self._thread = threading.Thread(target=self._write, name="trial-recorder", daemon=True)
        self._thread.start()

    def event(self, event, **values):
        if self._closed:
            return
        if event == "prediction":
            self.predictions += 1
        if not self.error:
            try:
                self._queue.put_nowait({"event": event, **values})
                return
            except Full:
                pass
        with self._drop_lock:
            self.dropped += 1

    def snapshot(self, name, *, anchor_ns, prediction_index, **arrays):
        """Queue one observation window. The writer thread compresses it off the control loops."""
        if self._closed:
            return
        payload = {"event": "observation_snapshot", "name": str(name), "anchor_ns": int(anchor_ns),
                   "prediction_index": int(prediction_index), "monotonic_ns": time.monotonic_ns(),
                   "_arrays": {key: np.array(value, copy=True) for key, value in arrays.items()}}
        if self.error:
            with self._drop_lock:
                self.dropped += 1
            return
        try:
            self._queue.put_nowait(payload)
        except Full:
            with self._drop_lock:
                self.dropped += 1

    def _write(self):
        try:
            with (self.path / "events.jsonl").open("w") as file:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    arrays = item.pop("_arrays", None)
                    if arrays is not None:
                        filename = f"{Path(str(item['name'])).name}.npz"
                        np.savez_compressed(self.path / filename, **arrays)
                        item["file"] = filename
                    file.write(json.dumps(item, default=json_value, ensure_ascii=False, allow_nan=False) + "\n")
        except Exception as error:
            self.error = f"Trial writer failed: {error}"

    def close(self, outcome, reason=None):
        self._closed = True
        if self._thread.is_alive():
            try:
                self._queue.put(None, timeout=5)
            except Full:
                self.error = self.error or "Recording flush timed out"
            self._thread.join(timeout=15)
        if self._thread.is_alive():
            self.error = self.error or "Recording writer did not stop"
        if outcome == "shadow" and not self.predictions:
            outcome, reason = "device_abort", "Shadow trial produced no valid model predictions"
        if not self.error:
            timing_report(self.path)
        result = {**self.metadata, "status": "complete", "outcome": outcome, "reason": reason,
                  "dropped_events": self.dropped, "record_error": self.error}
        write_json(self.path / "trial.json", result)
        return result


def summarize_trials(output):
    results = [json.loads(path.read_text()) for path in sorted(Path(output).glob("trial_*/trial.json"))]
    motion = [r for r in results if r["mode"] == "motion"]
    successes = sum(r.get("outcome") == "success" and r["status"] == "complete" for r in motion)
    report = {"trials": results, "motion_trials": len(motion), "successes": successes,
              "success_rate": successes / len(motion) if motion else None,
              "denominator": "all started motion trials, including timeout/takeover/device_abort"}
    write_json(Path(output) / "summary.json", report)
    return report


class TianjiRailRealRunner:
    def __init__(self, env, policy, checkpoint_config, metadata, output_dir, *, record_observations=False, trace=None):
        self.env, self.policy, self.cfg = env, policy, checkpoint_config
        self.record_observations = bool(record_observations) and trace is None
        self.trace = trace
        self.metadata = metadata
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.glob("trial_*")) or (self.output / "session.json").exists():
            raise FileExistsError(f"Evaluation output already contains a trial: {self.output}")
        write_json(self.output / "session.json", metadata)
        self._stop = threading.Event()
        self._enabled = threading.Event()
        self._inference_lock = threading.Lock()
        self._generation = 0
        self._snapshot_index = 0
        self._trial = None
        self._started = None
        self._count = 0
        self.error = None
        self.batch_hooks = None
        self._batch_stop_reason = None

    def _begin(self):
        if self._trial is not None:
            return
        self._count += 1
        self._snapshot_index = 0
        self._trial = TrialRecorder(self.output / f"trial_{self._count:04d}",
            {**self.metadata, "mode": "motion" if self.env.motion else "shadow",
             "record_observations": self.record_observations, "trace": self.trace is not None,
             "started_monotonic_ns": time.monotonic_ns()}, self.env.config["record_queue_capacity"])
        self.env.recorder = self._trial
        self._generation += 1
        self.error = None
        try:
            if self.env.motion and self.batch_hooks is not None and self.batch_hooks.check_init is not None:
                report = self.batch_hooks.check_init(self.env)
                self.batch_hooks.init_report = report
                if not report.get("comparable"):
                    self._batch_stop_reason = "initialization mismatch"
                    self.env.recorder = None
                    trial, self._trial = self._trial, None
                    if trial is not None:
                        trial.close("invalid_comparison", "initialization mismatch")
                    print("初始化超出冻结容差，本段记为无效比较，未接合。")
                    return
            if self.trace is not None and not self.trace.decisions_ready:
                obs, _, _ = self.env.get_obs()
                self.trace.allocate_decisions(obs, self._n_action_steps(), capture_real=False)
                print(f"完整记录缓冲：{self.trace.decision_capacity} 次决策，图像约 {self.trace.image_mebibytes:.0f} MiB。"
                      "缓冲满会把本段标为无效，不会停住控制。")
            if self.trace is not None:
                self.trace.begin_trial()
            self._started = time.monotonic()
            if self.env.motion:
                self.env.engage()
            # Inference/reset are serialized inside the worker, never in the control loops.
            with self._inference_lock:
                if self.metadata.get("batch_seed") is not None:
                    self.policy.reseed(int(self.metadata["batch_seed"]))
                self._enabled.set()
        except Exception as error:
            self._end("device_abort", str(error))
            print(f"试验未能开始：{error}")

    def _end(self, outcome, reason=None):
        with self._inference_lock:
            self._enabled.clear()
            self._generation += 1
            try:
                self.env.pause(reason or outcome)
            finally:
                self.env.recorder = None
                trial, self._trial = self._trial, None
        if trial is not None:
            trial.close(outcome, reason)
            if self.trace is not None:
                self._flush_trace(trial.path)
            summarize_trials(self.output)

    def _infer(self):
        import torch
        generation = -1
        anchor_previous = None
        sync = self.env.config["inference_mode"] == "sync"
        try:
            while not self._stop.is_set():
                if not self._enabled.wait(timeout=.05):
                    continue
                current = self._generation
                if generation != current:
                    self.policy.reset()
                    generation, anchor_previous = current, None
                completed = None
                if sync:
                    completed = self.env.chunk_completed_ns()
                    if completed is None:
                        self._stop.wait(.002)
                        continue
                    # Full alignment competes with the arm loop for the GIL; skip it until a fresh anchor exists.
                    latest = self.env.latest_anchor_ns()
                    if latest is None or latest <= completed:
                        self._stop.wait(.003)
                        continue
                hold = self._before_observation()
                if hold == "stop":
                    return
                started = time.monotonic_ns()
                try:
                    obs, anchor, alignment = self.env.get_obs()
                except ObservationNotReady as error:
                    # In sync mode the runtime is holding here, so not-ready observations are retried.
                    if self.env.motion and self.env.runtime.active:
                        raise RuntimeError(str(error)) from error
                    self._stop.wait(.01)
                    continue
                observed = time.monotonic_ns()
                if sync:
                    # The observation must show the robot after the previous chunk, not during it.
                    if anchor <= completed:
                        self._stop.wait(.003)
                        continue
                elif anchor_previous is not None and anchor - anchor_previous < self.env.config["replanning_steps"] * 1e9 / self.env.config["policy_hz"]:
                    self._stop.wait(.003)
                    continue
                process_predict = getattr(self.policy, "predict_observation", None)
                process_timing = None
                if process_predict is not None:
                    prediction = process_predict(obs)
                    process_timing = self.policy.last_timing
                    preprocessed = process_timing["preprocessed_ns"]
                    inputs_ready = process_timing["inputs_ready_ns"]
                else:
                    prepared = preprocess_observation(obs, self.cfg.task.shape_meta, self.cfg.task.dataset)
                    preprocessed = time.monotonic_ns()
                    tensor_obs = {key: torch.from_numpy(value).unsqueeze(0).to(self.policy.device) for key, value in prepared.items()}
                    inputs_ready = time.monotonic_ns()
                    with torch.inference_mode():
                        prediction = self.policy.predict_action(tensor_obs)
                actions = prediction["action"][0].detach().cpu().numpy()
                returned = time.monotonic_ns()
                stability, submit = self._prediction_gate(hold, returned)
                # A stopped/previous trial cannot publish into a new trial.
                with self._inference_lock:
                    if generation != self._generation or not self._enabled.is_set():
                        continue
                    prediction_id = None
                    if self.trace is not None:
                        prediction_id = self.trace.store_decision(
                            obs=obs, alignment=alignment, anchor_ns=anchor, actions=actions,
                            started_ns=started, observed_ns=observed, preprocessed_ns=preprocessed,
                            inputs_ready_ns=inputs_ready, returned_ns=returned,
                            rng=getattr(self.policy, "last_rng", None), control_input="real",
                            live_capture_started_ns=started, live_capture_returned_ns=observed,
                            stability=stability)
                    observation = {} if self.trace is not None else self._observation_record(obs)
                    if prediction_id is not None:
                        observation = {**observation, "prediction_id": prediction_id}
                    if stability is not None:
                        observation = {**observation, "stability": stability}
                    self.env.emit("prediction", anchor_ns=anchor, started_ns=started, returned_ns=returned,
                                  observed_ns=observed, preprocessed_ns=preprocessed,
                                  inputs_ready_ns=inputs_ready, process_timing=process_timing,
                                  actions=actions, alignment=alignment, chunk_completed_ns=completed,
                                  **observation)
                    if not submit:
                        self.env.emit("prediction_rejected", reason="post-inference excursion",
                                      stability=stability)
                        if self._batch_stop_reason:
                            return
                        continue
                    if prediction_id is None:
                        dropped = self.env.exec_actions(actions, anchor)
                    else:
                        dropped = self.env.exec_actions(actions, anchor, prediction_id=prediction_id)
                    if self.trace is None:
                        self._save_observation_window(obs, anchor)
                    self.env.emit("prediction_coverage", dropped_steps=dropped,
                                  remaining_steps=len(actions) - dropped)
                anchor_previous = anchor
        except Exception as error:
            with self._inference_lock:
                if generation == self._generation:
                    self.error = str(error)
                    self._enabled.clear()
                    self.env.runtime.pause(error, fault=True)

    def _batch_blocked(self):
        hooks = self.batch_hooks
        if self._batch_stop_reason:
            return True
        if hooks is None or self.trace is None:
            return False
        if self.trace.decision_count >= hooks.max_decisions:
            self._batch_stop_reason = "decision cap"
            self._enabled.clear()
            return True
        return False

    def _before_observation(self):
        if self._batch_blocked():
            return "stop"
        hooks = self.batch_hooks
        if hooks is None or not hooks.stable:
            return None
        verdict = hooks.gate.wait_stable(
            self.env.feedback_since, time.monotonic_ns, self._stop.wait,
            lambda: self._enabled.is_set() and not self._stop.is_set() and self._batch_stop_reason is None)
        public = {key: value for key, value in verdict.items() if key != "snapshot"}
        self.env.emit("stability_wait", **public)
        if not verdict["stable"]:
            self._batch_stop_reason = verdict.get("reason") or "stability wait failed"
            self._enabled.clear()
            return "stop"
        return verdict

    def _prediction_gate(self, verdict, returned_ns):
        from diffusion_policy.real_world.batch_factors import latest_snapshot
        if not isinstance(verdict, dict):
            return None, True
        after = latest_snapshot(self.env.feedback_since(int(returned_ns) - int(0.25e9)))
        post = self.batch_hooks.gate.postcheck(verdict.get("snapshot"), after)
        if post["ok"]:
            self.batch_hooks.gate.note_accept()
        elif not self.batch_hooks.gate.note_reject():
            self._batch_stop_reason = "post-inference excursion"
            self._enabled.clear()
        info = {"accepted": bool(post["ok"]), "resamples": self.batch_hooks.gate.resamples,
                "wait": {key: value for key, value in verdict.items() if key != "snapshot"}, "post": post}
        return info, bool(post["ok"])

    def _n_action_steps(self):
        preflight = getattr(self.policy, "preflight", None)
        if isinstance(preflight, dict) and "n_action_steps" in preflight:
            return int(preflight["n_action_steps"])
        return int(self.policy.n_action_steps)

    def _flush_trace(self, path):
        try:
            meta = self.trace.flush(path)
        except Exception as error:
            (Path(path) / "trace_flush_error.txt").write_text(str(error))
            return
        trial_path = Path(path) / "trial.json"
        if trial_path.is_file() and meta is not None:
            trial = json.loads(trial_path.read_text())
            trial["trace_valid"] = meta["valid"]
            trial["trace_invalid_reason"] = meta["invalid_reason"]
            trial["trace_dropped"] = meta["dropped"]
            write_json(trial_path, trial)
            print("完整记录有效。" if meta["valid"] else f"完整记录无效：{meta['invalid_reason']}。不能据此判定根因。")

    def _observation_record(self, obs):
        if not self.record_observations:
            return {}
        index = self._snapshot_index
        name = f"obs_{index:02d}" if index < 3 else f"obs_recent_{index % 3:02d}"
        return {"observation_file": f"{name}.npz", "prediction_index": index,
                "input_robot_eef_pose": np.asarray(obs["robot_eef_pose"], dtype=np.float32),
                "input_hand_joint": np.asarray(obs["hand_joint"], dtype=np.float32),
                "measured": self.env.latest_measured()}

    def _save_observation_window(self, obs, anchor):
        if not self.record_observations or self._trial is None:
            return
        index = self._snapshot_index
        self._snapshot_index += 1
        name = f"obs_{index:02d}" if index < 3 else f"obs_recent_{index % 3:02d}"
        self._trial.snapshot(name, anchor_ns=anchor, prediction_index=index,
                             camera_0=obs["camera_0"], camera_1=obs["camera_1"], camera_2=obs["camera_2"],
                             robot_eef_pose=obs["robot_eef_pose"], hand_joint=obs["hand_joint"])

    def run(self):
        thread = threading.Thread(target=self._infer, name="policy-inference", daemon=True)
        thread.start()
        if not self.env.motion:
            self._begin()
        print("s 开始试验；p 暂停并记接管；y 成功；n 失败；q 退出（终端输入后按 Enter）。")
        print("默认影子模式，不使能设备。" if not self.env.motion else
              "运动模式：每次仍需手动 s 开始。")
        if self.trace is not None and self.env.motion:
            print("出现明显偏离就可以停，不必凑满试验时限。有人递送导轨或扶机器人时，用 p 记下接管时间。")
        try:
            while not self._stop.wait(.02):
                if self._batch_stop_reason:
                    if self._trial is not None:
                        kind = "invalid_comparison" if self._batch_stop_reason == "initialization mismatch" else "timeout"
                        self._end(kind, self._batch_stop_reason)
                        print(f"试验结束：{self._batch_stop_reason}")
                    break
                if self._trial is not None:
                    problem = self.error or self.env.error or self.env.runtime.fault
                    if not thread.is_alive():
                        problem = problem or "Inference worker exited"
                    if problem:
                        self._end("device_abort", problem)
                        print(f"试验中止：{problem}")
                        if not self.env.motion:
                            break
                    elif self._started is not None and time.monotonic() - self._started >= self.env.config["trial_timeout_s"]:
                        self._end("timeout" if self.env.motion else "shadow", "Trial time limit")
                        if not self.env.motion:
                            break
                if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                    command = sys.stdin.readline().strip().lower()
                    if command == "s":
                        if not thread.is_alive():
                            print("推理线程已退出，请退出并重新启动入口。")
                        else:
                            self._begin()
                    elif command in ("p", "y", "n"):
                        self._end({"p": "takeover", "y": "success", "n": "failure"}[command]
                                  if self.env.motion else "shadow", f"Operator {command}")
                    elif command == "q":
                        break
        finally:
            self._stop.set()
            self._end("takeover" if self.env.motion else "shadow", "Runner exit")
            thread.join(timeout=2)
        return summarize_trials(self.output)
