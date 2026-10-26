"""Nonblocking latest-observation bridge; perception never owns an actuator."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4

from rail_preinsert.core import HERE, read_json, transform, write_json
from rail_preinsert.perception import python_command


class ReceivePerception:
    def __init__(self, config, directory, serial):
        self.config, self.directory = config, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.log = (self.directory / "worker.log").open("w")
        self.process = None
        self.generation = uuid4().hex
        self.last_context_write = 0.0
        try:
            write_json(self.directory / "config.json", config)
            write_json(self.directory / "request.json", {"id": self.generation, "command": "initialize"})
            command = python_command(config) + [str(HERE / "receive_perception_worker.py"),
                "--foundationpose-root", config["foundationpose_root"], "--mesh", config["mesh"],
                "--camera-serial", serial, "--serve", str(self.directory),
                "--config", str(self.directory / "config.json")]
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT, env=env)
            self.started = time.monotonic()
        except BaseException:
            self.close()
            raise

    def ready(self):
        if self.process.poll() is not None:
            raise RuntimeError(f"Receiving perception stopped; inspect {self.directory / 'worker.log'}")
        path = self.directory / "ready.json"
        if not path.is_file():
            if time.monotonic() - self.started > self.config["perception_timeout_s"]:
                raise TimeoutError("Receiving perception initialization timed out")
            return False
        result = read_json(path)
        if not result["ready"]:
            raise RuntimeError(result.get("error", "Receiving perception initialization failed"))
        return True

    def initialize(self, *, predicted_pose=None):
        request = {"command": "initialize"}
        if predicted_pose is not None:
            request["T_C_O_predicted"] = transform(predicted_pose, "T_C_O_predicted").tolist()
        if not self.ready():
            raise ValueError("Camera preview is still initializing")
        self.generation = uuid4().hex
        write_json(self.directory / "request.json", dict(request, id=self.generation))
        return self.generation

    def publish_context(self, state, T_C_G, *, enabled, epoch, mode="receive"):
        now = time.monotonic()
        if now-self.last_context_write < self.config["automatic_detection"]["context_publish_period_s"]:
            return
        write_json(self.directory / "context.json", {
            "timestamp_ns": int(state["timestamp_ns"]),
            "T_C_G": T_C_G.tolist(), "enabled": bool(enabled),
            "epoch": epoch, "mode": mode})
        self.last_context_write = now

    def latest(self):
        if not self.ready():
            return None
        path = self.directory / "latest.json"
        if path.is_file():
            record = read_json(path)
            if record.get("generation") == self.generation:
                return record
        return None

    def stop_tracking(self):
        write_json(self.directory / "request.json", {"id": uuid4().hex, "command": "stop"})
        self.generation = None

    def close(self):
        if self.process is not None and self.process.poll() is None:
            write_json(self.directory / "request.json", {"id": uuid4().hex, "command": "shutdown"})
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        self.log.close()
