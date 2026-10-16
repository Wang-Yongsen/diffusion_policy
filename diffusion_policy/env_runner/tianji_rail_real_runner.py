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

    def _write(self):
        try:
            with (self.path / "events.jsonl").open("w") as file:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
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
    def __init__(self, env, policy, checkpoint_config, metadata, output_dir):
        self.env, self.policy, self.cfg = env, policy, checkpoint_config
        self.metadata = metadata
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=False)
        write_json(self.output / "session.json", metadata)
        self._stop = threading.Event()
        self._enabled = threading.Event()
        self._inference_lock = threading.Lock()
        self._generation = 0
        self._trial = None
        self._started = None
        self._count = 0
        self.error = None

    def _begin(self):
        if self._trial is not None:
            return
        self._count += 1
        self._trial = TrialRecorder(self.output / f"trial_{self._count:04d}",
            {**self.metadata, "mode": "motion" if self.env.motion else "shadow",
             "started_monotonic_ns": time.monotonic_ns()}, self.env.config["record_queue_capacity"])
        self.env.recorder = self._trial
        self._started = time.monotonic()
        self._generation += 1
        self.error = None
        try:
            if self.env.motion:
                self.env.engage()
            # Inference/reset are serialized inside the worker, never in the control loops.
            with self._inference_lock:
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
            summarize_trials(self.output)

    def _infer(self):
        import torch
        generation = -1
        anchor_previous = None
        try:
            while not self._stop.is_set():
                if not self._enabled.wait(timeout=.05):
                    continue
                current = self._generation
                if generation != current:
                    self.policy.reset()
                    generation, anchor_previous = current, None
                started = time.monotonic_ns()
                try:
                    obs, anchor, alignment = self.env.get_obs()
                except ObservationNotReady as error:
                    if self.env.motion and self.env.runtime.active:
                        raise RuntimeError(str(error)) from error
                    self._stop.wait(.01)
                    continue
                observed = time.monotonic_ns()
                if anchor_previous is not None and anchor - anchor_previous < self.env.config["replanning_steps"] * 1e9 / 30:
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
                # A stopped/previous trial cannot publish into a new trial.
                with self._inference_lock:
                    if generation != self._generation or not self._enabled.is_set():
                        continue
                    self.env.emit("prediction", anchor_ns=anchor, started_ns=started, returned_ns=returned,
                                  observed_ns=observed, preprocessed_ns=preprocessed,
                                  inputs_ready_ns=inputs_ready, process_timing=process_timing,
                                  actions=actions, alignment=alignment)
                    dropped = self.env.exec_actions(actions, anchor)
                    self.env.emit("prediction_coverage", dropped_steps=dropped,
                                  remaining_steps=len(actions) - dropped)
                anchor_previous = anchor
        except Exception as error:
            with self._inference_lock:
                if generation == self._generation:
                    self.error = str(error)
                    self._enabled.clear()
                    self.env.runtime.pause(error, fault=True)

    def run(self):
        thread = threading.Thread(target=self._infer, name="policy-inference", daemon=True)
        thread.start()
        if not self.env.motion:
            self._begin()
        print("s 开始试验；p 暂停并记接管；y 成功；n 失败；q 退出（终端输入后按 Enter）。")
        print("默认影子模式，不使能设备。" if not self.env.motion else "运动模式：每次仍需手动 s 开始。")
        try:
            while not self._stop.wait(.02):
                if self._trial is not None:
                    problem = self.error or self.env.error or self.env.runtime.fault
                    if not thread.is_alive():
                        problem = problem or "Inference worker exited"
                    if problem:
                        self._end("device_abort", problem)
                        print(f"试验中止：{problem}")
                        if not self.env.motion:
                            break
                    elif time.monotonic() - self._started >= self.env.config["trial_timeout_s"]:
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
