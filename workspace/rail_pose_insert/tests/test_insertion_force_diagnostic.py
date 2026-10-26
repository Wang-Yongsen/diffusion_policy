"""Wrench quality, sensor-origin transforms, replay and feedback-only ownership."""
import builtins
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import read_insertion_force as diagnostic


def settings(**changes):
    return dict(state_max_age_s=.1, baseline_duration_s=.1, baseline_min_samples=3,
                T_F_S=None, wrench_sign=None, baseline_position_tolerance_m=None,
                orientation_tolerance_rad=None, **changes)


def row(index, *, trace=False, **changes):
    stamp = 1_000_000_000 + index*50_000_000
    result = {"event": "right_state", "timestamp_ns": stamp, "host_monotonic_ns": stamp+1_000_000,
              "right_sequence": index, "wrench": [10., 20., 30., 1., 2., 3.],
              "T_B_F": np.eye(4).tolist()}
    if trace:
        result.pop("event")
        result["raw_wrench"] = result.pop("wrench")
        result["measured_T_B_F"] = result.pop("T_B_F")
    result.update(changes)
    return result


class ForceDiagnosticTests(unittest.TestCase):
    def test_unique_fresh_samples_define_baseline_not_poll_count(self):
        duplicate = row(0, timestamp_ns=1_010_000_000, host_monotonic_ns=1_011_000_000)
        stale = row(1, host_monotonic_ns=1_300_000_000)
        invalid = row(1, wrench=[0., 0., float("nan"), 0., 0., 0.])
        records = [row(0), duplicate, stale, invalid, row(1), row(2), {"event": "paused"}]
        result = diagnostic.analyze_records(records, settings())
        self.assertEqual(result["status"], "sampled")
        self.assertEqual(result["counts"]["valid_samples"], 3)
        self.assertEqual(result["counts"]["duplicate_source_sequence"], 1)
        self.assertEqual(result["counts"]["stale_or_future"], 1)
        self.assertEqual(result["counts"]["invalid_wrench_or_pose"], 1)
        self.assertEqual(result["sample_interval_ms"]["mean"], 50.)
        np.testing.assert_allclose(result["raw_sensor_wrench"]["std"], 0)
        self.assertIsNone(result["base_frame_residual"])

    def test_too_few_samples_or_too_short_span_does_not_claim_complete_baseline(self):
        for records in ([row(0)]*100, [row(0), row(1)], [row(i) for i in range(3)]):
            selected = settings()
            selected["baseline_duration_s"] = .2
            result = diagnostic.analyze_records(records, selected)
            self.assertEqual(result["status"], "insufficient_samples")
            self.assertIsNone(result["baseline"])
            self.assertIsNone(result["base_frame_residual"])

    def test_base_residual_rotates_signed_force_and_torque_without_moving_moment_origin(self):
        selected = settings()
        T = np.eye(4)
        T[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        T[:3, 3] = [1., 2., 3.]  # Must not create an invented lever-arm torque.
        selected.update(T_F_S=T, wrench_sign=-1)
        records = [row(i, trace=True) for i in range(4)]
        records[-1]["raw_wrench"] = [11., 20., 30., 1., 4., 3.]
        result = diagnostic.analyze_records(records, selected)
        output = result["base_frame_residual"]
        np.testing.assert_allclose(output["last_wrench"], [0., -1., 0., 2., 0., 0.])
        self.assertEqual(output["moment_origin"], "force_sensor_origin")
        self.assertFalse(output["moment_shift_applied"])
        self.assertTrue(any("two-hole midpoint" in text for text in result["limitations"]))

    def test_configured_pose_motion_withholds_static_compensation(self):
        selected = settings()
        selected.update(T_F_S=np.eye(4), wrench_sign=1,
                        baseline_position_tolerance_m=.001, orientation_tolerance_rad=.01)
        records = [row(i) for i in range(4)]
        records[-1]["T_B_F"][0][3] = .005
        result = diagnostic.analyze_records(records, selected)
        self.assertEqual(result["status"], "motion_detected")
        self.assertIsNone(result["base_frame_residual"])
        self.assertAlmostEqual(result["stationarity"]["max_position_excursion_m"], .005)

    def test_old_trace_accepts_host_timestamp_fallback_but_exposes_missing_evidence(self):
        records = [row(i, trace=True) for i in range(3)]
        for record in records:
            record.pop("timestamp_ns")
            record.pop("right_sequence")
        records.insert(1, dict(records[0]))
        result = diagnostic.analyze_records(records, settings())
        self.assertEqual(result["status"], "sampled")
        self.assertEqual(result["counts"]["valid_samples"], 3)
        self.assertEqual(result["counts"]["freshness_unverifiable"], 3)
        self.assertEqual(result["counts"]["source_sequence_unavailable"], 3)
        self.assertEqual(result["counts"]["duplicate_or_reordered_timestamp"], 1)

    def test_source_sequence_wrap_is_valid_but_reset_is_rejected(self):
        records = [row(i, right_sequence=seq) for i, seq in enumerate([999998, 999999, 0, 999998, 1])]
        result = diagnostic.analyze_records(records, settings())
        self.assertEqual(result["counts"]["valid_samples"], 4)
        self.assertEqual(result["counts"]["reordered_source_sequence"], 1)

    def test_replay_cli_works_with_unset_calibration_and_no_device_or_vision_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, replay, output = root/"receive.yaml", root/"events.jsonl", root/"report.json"
            config.write_text("insertion:\n  baseline_duration_s: 0.1\n  baseline_min_samples: 3\n")
            replay.write_text("broken JSON\n" + "\n".join(json.dumps(row(i)) for i in range(3)))
            original_import = builtins.__import__
            def restricted_import(name, *args, **kwargs):
                if any(word in name for word in ("wuji", "foundationpose", "devices.tianji", "receive.perception")):
                    raise AssertionError(f"Replay tried to import hardware/vision: {name}")
                return original_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=restricted_import), redirect_stdout(io.StringIO()):
                code = diagnostic.main(["--config", str(config), "--replay", str(replay), "--output", str(output)])
            self.assertEqual(code, 0)
            report = json.loads(output.read_text())
            self.assertFalse(report["motion_enabled"])
            self.assertEqual(report["counts"]["malformed_json"], 1)

    def test_live_reader_only_starts_reads_and_closes_arm_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            arm_config = Path(directory)/"arms.yaml"
            arm_config.write_text("controller_ip: 127.0.0.1\n")
            selected = settings(arm_config=str(arm_config))
            sample = NS(header=NS(received_monotonic_ns=1_000_000_000),
                        payload=NS(arms={"right": NS(source_sequence=1, wrench=(0.,)*6,
                                                    joints=NS(position_rad=(0.,)*7))}))
            driver = Mock(spec=["start", "get_latest", "health", "close"])
            driver.get_latest.return_value = sample
            driver.health.return_value = NS(ready=True)
            factory = Mock(return_value=driver)
            pose = NS(position_m=(0.,)*3, orientation_xyzw=(0., 0., 0., 1.))
            kinematics = Mock()
            kinematics.fk.return_value = pose
            times = iter([1_000_000_000, 1_001_000_000, 1_002_000_000, 2_000_000_000])
            records = list(diagnostic.live_records(selected, .1, driver_factory=factory,
                kinematics_factory=lambda: kinematics, clock_ns=lambda: next(times), sleep=lambda _: None))
            factory.assert_called_once_with("127.0.0.1", record_force=True)
            driver.start.assert_called_once_with()
            driver.health.assert_called_once_with(sides=("right",))
            driver.close.assert_called_once_with()
            self.assertEqual(records[0]["right_sequence"], 1)
            self.assertEqual([call[0] for call in driver.mock_calls], ["start", "get_latest", "health", "close"])

    def test_reader_exception_still_closes_feedback_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            arm_config = Path(directory)/"arms.yaml"
            arm_config.write_text("controller_ip: 127.0.0.1\n")
            driver = Mock()
            driver.get_latest.side_effect = RuntimeError("feedback failed")
            with self.assertRaisesRegex(RuntimeError, "feedback failed"):
                list(diagnostic.live_records(settings(arm_config=str(arm_config)), .1,
                    driver_factory=lambda *a, **kw: driver, kinematics_factory=Mock))
            driver.close.assert_called_once_with()
            driver.configure.assert_not_called()
            driver.engage.assert_not_called()


if __name__ == "__main__":
    unittest.main()
