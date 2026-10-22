"""File-based request/reply bridge to the isolated FoundationPose environment."""
from __future__ import annotations

import subprocess
import os
import time
from pathlib import Path
from uuid import uuid4

from .core import HERE, read_json, write_json


class Perception:
    def __init__(self, config, directory, camera_serial, cancelled=lambda: False):
        self.config, self.directory = config, Path(directory)
        self.cancelled, self.process = cancelled, None
        self.directory.mkdir(parents=True, exist_ok=False)
        self.log = (self.directory / "worker.log").open("w")
        command = [config["foundationpose_python"], str(HERE / "perception_worker.py"),
                   "--foundationpose-root", config["foundationpose_root"], "--mesh", config["mesh"],
                   "--camera-serial", camera_serial, "--serve", str(self.directory)]
        try:
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT, env=env)
            deadline = time.monotonic() + config["perception_timeout_s"]
            while not (self.directory / "ready.json").is_file():
                self._check(deadline)
                time.sleep(.05)
            ready = read_json(self.directory / "ready.json")
            if not ready.get("ready"):
                raise RuntimeError(ready.get("error", "Perception initialization failed"))
        except BaseException:
            self.close(); raise

    def _check(self, deadline):
        if self.cancelled():
            raise RuntimeError("Perception request cancelled; arm paused")
        if self.process.poll() is not None:
            raise RuntimeError(f"FoundationPose worker exited; inspect {self.directory / 'worker.log'}")
        if time.monotonic() > deadline:
            raise TimeoutError("FoundationPose response timed out")

    def measure(self, samples=3):
        request_id = uuid4().hex
        write_json(self.directory / "request.json", {"id": request_id, "command": "measure", "samples": samples})
        deadline = time.monotonic() + self.config["perception_timeout_s"]
        while True:
            self._check(deadline)
            response_path = self.directory / "response.json"
            if response_path.is_file():
                response = read_json(response_path)
                if response.get("id") == request_id:
                    if not response.get("ok"):
                        raise RuntimeError(response.get("error", "Pose estimation failed"))
                    return response["observation"]
            time.sleep(.03)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            write_json(self.directory / "request.json", {"id": uuid4().hex, "command": "shutdown"})
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait(timeout=3)
        self.log.close()
