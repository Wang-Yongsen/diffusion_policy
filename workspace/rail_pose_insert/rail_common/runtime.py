"""Compact asynchronous event recording and access to the SDK kinematics."""

import json
from pathlib import Path
from queue import Queue
import threading
import time

from .core import read_json, write_json


class Log:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.queue = Queue()
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            with (self.directory / "events.jsonl").open("w", encoding="utf-8") as stream:
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
        except Exception as error:
            self.error = str(error)

    def emit(self, event, **values):
        if self.error:
            raise RuntimeError(f"Run log failed: {self.error}")
        self.queue.put(dict(event=event, host_monotonic_ns=time.monotonic_ns(), **values))

    def finish(self, stage, *, exit_requested, errors, feedback=None):
        path = self.directory / "result.json"
        result = (
            read_json(path)
            if path.exists()
            else dict(status=stage, side="right", assembly_verified=False)
        )
        result.update(
            last_stage=stage,
            exit_key_received=exit_requested,
            exit_hand_action="disabled_on_exit" if not errors else "inspect_cleanup_errors",
            cleanup_errors=errors,
        )
        if feedback is not None:
            result["last_feedback"] = {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in feedback.items()
            }
        write_json(path, result)
        self.emit("exit", stage=stage, cleanup_errors=errors)

    def close(self):
        self.queue.put(None)
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError("Run log did not stop")
        if self.error:
            raise RuntimeError(f"Run log failed: {self.error}")


def kinematics():
    from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics

    return TianjiKinematics()


def execution_summary(records):
    return dict(
        attempted_steps=len(records),
        verified_steps=sum(bool(r.get("endpoint_verified")) for r in records),
        last_step=records[-1].copy() if records else None,
    )
