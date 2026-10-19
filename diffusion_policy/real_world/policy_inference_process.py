"""Run model preprocessing and GPU inference outside the device-control GIL."""

import multiprocessing as mp
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

import torch
from threadpoolctl import threadpool_limits


def _rng_snapshot(device):
    """Generator state before sampling. A CPU run leaves CUDA explicitly absent."""
    snapshot = {"cpu": torch.get_rng_state().detach().cpu().numpy().copy(), "cuda": None, "device": str(device)}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device)
        if index.index is None:
            index = torch.device("cuda", torch.cuda.current_device())
        snapshot["cuda"] = torch.cuda.get_rng_state(index).detach().cpu().numpy().copy()
    return snapshot


def _inference_worker(connection, checkpoint, device, capture_rng=False):
    # The parent owns terminal interrupts, devices and their stop path.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        from eval_tianji_rail import check_inference, load_policy
        from diffusion_policy.real_world.tianji_rail_env import preprocess_observation
        with threadpool_limits(limits=1):
            torch.set_num_threads(1)
            policy, cfg, _ = load_policy(checkpoint, device)
            preflight = check_inference(policy, cfg)
            connection.send({"ok": True, "preflight": preflight})
            while True:
                command, payload = connection.recv()
                if command == "close":
                    return
                if command == "reset":
                    policy.reset()
                    connection.send({"ok": True})
                    continue
                if command == "reseed":
                    torch.manual_seed(int(payload))
                    if str(device).startswith("cuda"):
                        torch.cuda.manual_seed(int(payload))
                    policy.reset()
                    connection.send({"ok": True})
                    continue
                if command != "predict":
                    raise ValueError(f"Unknown inference command: {command}")
                received = time.monotonic_ns()
                prepared = preprocess_observation(payload, cfg.task.shape_meta, cfg.task.dataset)
                preprocessed = time.monotonic_ns()
                tensor_obs = {key: torch.from_numpy(value).unsqueeze(0).to(policy.device)
                              for key, value in prepared.items()}
                inputs_ready = time.monotonic_ns()
                rng = _rng_snapshot(device) if capture_rng else None
                with torch.inference_mode():
                    actions = policy.predict_action(tensor_obs)["action"].detach().cpu().numpy()
                returned = time.monotonic_ns()
                connection.send({"ok": True, "actions": actions, "rng": rng, "timing": {
                    "received_ns": received, "preprocessed_ns": preprocessed,
                    "inputs_ready_ns": inputs_ready, "returned_ns": returned}})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send({"ok": False, "error": traceback.format_exc()})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class PolicyInferenceProcess:
    """One in-flight observation; a fresh CUDA process never controls devices."""

    def __init__(self, checkpoint, device, *, startup_timeout_s=60., request_timeout_s=10., capture_rng=False):
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._request_timeout_s = request_timeout_s
        self.last_timing = None
        self.last_rng = None
        context = mp.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_inference_worker,
                                        args=(child, str(checkpoint), str(device), bool(capture_rng)),
                                        name="policy-inference", daemon=True)
        self._process.start()
        child.close()
        try:
            self.preflight = self._receive(startup_timeout_s)["preflight"]
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while not self._closed.is_set():
            if self._connection.poll(.02):
                try:
                    result = self._connection.recv()
                except EOFError as error:
                    raise RuntimeError("Inference process connection closed") from error
                if not result["ok"]:
                    raise RuntimeError(f"Inference process failed:\n{result['error']}")
                return result
            if not self._process.is_alive():
                raise RuntimeError(f"Inference process exited with code {self._process.exitcode}")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Inference process did not respond within {timeout_s:g} s")
        raise RuntimeError("Inference process closed")

    def reset(self):
        with self._lock:
            self._connection.send(("reset", None))
            self._receive(self._request_timeout_s)

    def predict_observation(self, obs):
        with self._lock:
            started = time.monotonic_ns()
            self._connection.send(("predict", obs))
            sent = time.monotonic_ns()
            result = self._receive(self._request_timeout_s)
            self.last_timing = {**result["timing"], "request_started_ns": started,
                                "request_sent_ns": sent, "response_received_ns": time.monotonic_ns()}
            self.last_rng = result.get("rng")
            return {"action": torch.from_numpy(result["actions"])}

    def reseed(self, seed):
        """Explicit batch-only reset after warmup/initialization, before formal decision 0."""
        with self._lock:
            self._connection.send(("reseed", int(seed)))
            self._receive(self._request_timeout_s)

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        if self._lock.acquire(timeout=.1):
            try:
                if self._process.is_alive():
                    self._connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                self._lock.release()
        self._process.join(timeout=1.)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.)
        self._connection.close()
