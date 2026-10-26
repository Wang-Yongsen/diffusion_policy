"""Fake-clock and stop-path checks for the fixed batch. No devices or checkpoints."""

import unittest

import numpy as np

from diffusion_policy.real_world.tianji.batch_factors import (
    STABILITY_PARAMETERS,
    StabilityGate,
    run_fake_preflight,
    submission_decision,
)


class BatchFactorTests(unittest.TestCase):
    def target(self):
        return {
            "arm_q": {"left": [0.1] * 7, "right": [-0.1] * 7},
            "eef": {
                "left": [0.62, 0.14, 0.23, 0.08, 1.62, -0.1],
                "right": [0.63, -0.14, 0.24, -0.02, 1.62, 0.04],
            },
            "hand": {"left": [0.2] * 20, "right": [0.3] * 20},
        }

    def test_fake_clock_stop_path_and_record(self):
        report = run_fake_preflight(self.target())
        self.assertFalse(report["hardware_started"])
        self.assertTrue(report["fake_clock"])
        self.assertTrue(report["stability_stop"])
        self.assertTrue(report["record_integrity"])
        self.assertTrue(report["visual_candidate_refused"])

    def test_resample_budget_is_two(self):
        self.assertEqual(submission_decision(False, 0, 2), "resample")
        self.assertEqual(submission_decision(False, 1, 2), "resample")
        self.assertEqual(submission_decision(False, 2, 2), "stop")
        gate = StabilityGate(STABILITY_PARAMETERS)
        self.assertTrue(gate.note_reject())
        self.assertTrue(gate.note_reject())
        self.assertFalse(gate.note_reject())
        self.assertEqual(gate.resamples, 2)
        gate.note_accept()
        self.assertEqual(gate.resamples, 0)

    def test_postcheck_uses_rotation_not_rotvec_difference(self):
        now = 2_000_000_000
        from diffusion_policy.real_world.tianji.batch_factors import (
            _constant_histories,
            assess_feedback,
            snapshot_delta,
        )

        quiet = _constant_histories(now - int(0.2e9), now, 10_000_000)
        before = assess_feedback(quiet, now)["snapshot"]
        after = {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in before.items()
        }
        after["left_rotvec"] = np.array([np.pi, 0.0, 0.0])
        before["left_rotvec"] = np.array([-np.pi, 0.0, 0.0])
        delta = snapshot_delta(before, after)
        self.assertLess(delta["measurements"]["left_rotation_deg"], 1e-4)
        self.assertTrue(delta["ok"])


if __name__ == "__main__":
    unittest.main()
