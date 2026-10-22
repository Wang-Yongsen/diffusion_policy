"""Bounded event recording, timing reports and trial summaries."""

import json
import threading
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from queue import Full, Queue

import numpy as np


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
    Path(path).write_text(
        json.dumps(value, default=json_value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def timing_report(path):
    samples, arrivals = {}, {}

    def add(key, value):
        samples.setdefault(key, []).append(value)

    with (Path(path) / "events.jsonl").open() as file:
        for line in file:
            item = json.loads(line)
            event = item["event"]
            if event == "prediction":
                add(
                    "preprocess_inference_transfer_ms",
                    (item["returned_ns"] - item["started_ns"]) / 1e6,
                )
                add(
                    "observation_age_at_prediction_return_ms",
                    (item["returned_ns"] - item["anchor_ns"]) / 1e6,
                )
                # Older sessions only contain the combined duration above.
                for name, start, end in (
                    ("observation_read_ms", "started_ns", "observed_ns"),
                    ("preprocess_ms", "observed_ns", "preprocessed_ns"),
                    ("input_transfer_ms", "preprocessed_ns", "inputs_ready_ns"),
                    ("inference_output_transfer_ms", "inputs_ready_ns", "returned_ns"),
                ):
                    if start in item and end in item:
                        add(name, (item[end] - item[start]) / 1e6)
                add(
                    "observation_age_at_processing_start_ms",
                    (item["started_ns"] - item["anchor_ns"]) / 1e6,
                )
                if item.get("chunk_completed_ns") is not None:
                    add("sync_hold_ms", (item["returned_ns"] - item["chunk_completed_ns"]) / 1e6)
            elif event == "submission":
                control = item.get("side") or item["kind"]
                add(
                    f"{control}_submission_call_ms",
                    (item["returned_ns"] - item["started_ns"]) / 1e6,
                )
                arrivals.setdefault(f"{control}_control", []).append(item["started_ns"])
                if item["mode"] == "policy" and item["planned_ns"] is not None:
                    add(
                        f"{item['kind']}_target_age_at_submission_ms",
                        (item["started_ns"] - item["planned_ns"]) / 1e6,
                    )
    for key, stamps in arrivals.items():
        if len(stamps) > 1:
            samples[f"{key}_interval_ms"] = (np.diff(stamps) / 1e6).tolist()
    report = {
        key: {
            "count": len(values),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }
        for key, values in samples.items()
        if values
    }
    report["scope"] = (
        "host monotonic timestamps; SDK acceptance does not establish physical arrival"
    )
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
        payload = {
            "event": "observation_snapshot",
            "name": str(name),
            "anchor_ns": int(anchor_ns),
            "prediction_index": int(prediction_index),
            "monotonic_ns": time.monotonic_ns(),
            "_arrays": {key: np.array(value, copy=True) for key, value in arrays.items()},
        }
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
                    file.write(
                        json.dumps(item, default=json_value, ensure_ascii=False, allow_nan=False)
                        + "\n"
                    )
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
        result = {
            **self.metadata,
            "status": "complete",
            "outcome": outcome,
            "reason": reason,
            "dropped_events": self.dropped,
            "record_error": self.error,
        }
        write_json(self.path / "trial.json", result)
        return result


def summarize_trials(output):
    results = [
        json.loads(path.read_text()) for path in sorted(Path(output).glob("trial_*/trial.json"))
    ]
    motion = [r for r in results if r["mode"] == "motion"]
    successes = sum(r.get("outcome") == "success" and r["status"] == "complete" for r in motion)
    report = {
        "trials": results,
        "motion_trials": len(motion),
        "successes": successes,
        "success_rate": successes / len(motion) if motion else None,
        "denominator": "all started motion trials, including timeout/takeover/device_abort",
    }
    write_json(Path(output) / "summary.json", report)
    return report
