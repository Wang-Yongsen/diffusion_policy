"""Exercise spawned inference IPC, reset, failures and cancellation without devices."""

import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

import diffusion_policy.real_world.policy_inference_process as module


def echo_worker(connection, checkpoint, device, capture_rng=False):
    connection.send({"ok": True, "preflight": {"scope": "fake model, no devices"}})
    count = 0
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                return
            if command == "reset":
                count = 0
                connection.send({"ok": True})
                continue
            if payload.get("fail", False):
                connection.send({"ok": False, "error": "fake model failure"})
                return
            if payload.get("block", False):
                time.sleep(30)
                return
            count += 1
            now = time.monotonic_ns()
            connection.send({"ok": True, "actions": np.full((1, 8, 52), count, np.float32),
                             "timing": {"received_ns": now, "preprocessed_ns": now,
                                        "inputs_ready_ns": now, "returned_ns": now}})
    finally:
        connection.close()


class PolicyInferenceProcessTests(unittest.TestCase):
    def client(self):
        with patch.object(module, "_inference_worker", echo_worker):
            client = module.PolicyInferenceProcess("unused.ckpt", "cpu", startup_timeout_s=15.)
        self.addCleanup(client.close)
        return client

    def test_round_trip_and_reset_use_the_same_process(self):
        client = self.client()
        obs = {"camera_0": np.zeros((2, 8, 8, 3), np.uint8)}
        self.assertEqual(client.predict_observation(obs)["action"].shape, (1, 8, 52))
        self.assertEqual(client.predict_observation(obs)["action"][0, 0, 0].item(), 2)
        client.reset()
        self.assertEqual(client.predict_observation(obs)["action"][0, 0, 0].item(), 1)
        self.assertLessEqual(client.last_timing["request_started_ns"], client.last_timing["response_received_ns"])
        client.close()
        self.assertFalse(client._process.is_alive())

    def test_model_failure_reaches_the_parent(self):
        client = self.client()
        with self.assertRaisesRegex(RuntimeError, "fake model failure"):
            client.predict_observation({"fail": True})

    def test_close_interrupts_an_inflight_prediction_and_stops_worker(self):
        client = self.client()
        failures = []
        def predict():
            try:
                client.predict_observation({"block": True})
            except RuntimeError as error:
                failures.append(str(error))
        thread = threading.Thread(target=predict)
        thread.start()
        deadline = time.monotonic() + 2
        while not client._lock.locked() and time.monotonic() < deadline:
            time.sleep(.001)
        client.close()
        thread.join(timeout=2.)
        self.assertFalse(thread.is_alive())
        self.assertFalse(client._process.is_alive())
        self.assertTrue(failures)

    def test_cpu_rng_snapshot_leaves_cuda_absent(self):
        snapshot = module._rng_snapshot("cpu")
        self.assertIsNone(snapshot["cuda"])
        self.assertEqual(snapshot["device"], "cpu")
        self.assertGreater(snapshot["cpu"].nbytes, 0)


if __name__ == "__main__":
    unittest.main()
