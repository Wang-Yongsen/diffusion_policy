"""Fresh visual retries never authorize motion by themselves; no hardware I/O."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import test_above


class AboveReverificationTests(unittest.TestCase):
    fixture = test_above.AboveTests.fixture
    record = test_above.AboveTests.record

    def unverified(self, directory):
        transfer, hardware = self.fixture(directory)
        target = test_above.T([.04, .03, .6], [.02, -.01, .04])
        state = {"T_B_F": target.copy(), "joints_rad": np.zeros(7)}
        transfer.report = {"T_R_F_above": target.tolist()}
        transfer.status = "above_target_unverified"
        transfer.not_before_ns = time.monotonic_ns()-100_000_000
        transfer.cached_grasp = np.eye(4)
        transfer.visual_verification = {
            "passed": False, "position_error_m": .02, "rotation_error_rad": 0.,
            "position_tolerance_m": transfer.settings["visual_position_tolerance_m"],
            "rotation_tolerance_rad": transfer.settings["visual_rotation_tolerance_rad"],
            "translation_error_base_m": [.02, 0., 0.]}
        old_stamp = time.monotonic_ns()
        transfer.records = [self.record(transfer, old_stamp,
            np.linalg.inv(transfer.geometry.T_B_C) @ transfer.above)]
        hardware.position_move_finished.return_value = True
        hardware.history = []
        return transfer, hardware, state

    def fresh_record(self, transfer, hardware, actual):
        stamp = time.monotonic_ns()
        hardware.history = [(stamp-1_000_000, np.zeros(7)),
                            (stamp+1_000_000, np.zeros(7))]
        return self.record(transfer, stamp,
            np.linalg.inv(transfer.geometry.T_B_C) @ actual)

    def verify(self, transfer, hardware, state, actual):
        for _ in range(transfer.settings["samples"]):
            record = self.fresh_record(transfer, hardware, actual)
            transfer.advance(hardware, state, record)

    def assert_no_motion(self, transfer, hardware):
        hardware.move_to_position.assert_not_called()
        hardware.queue_insertion.assert_not_called()
        hardware.begin_hand.assert_not_called()
        self.assertIsNone(transfer.lowering)
        self.assertFalse(transfer.insertion_started)

    def test_visual_diagnostic_starts_empty_and_begin_resets_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, _ = self.fixture(directory)
            self.assertIsNone(transfer.visual_verification)
            transfer.visual_verification = {"passed": False}
            transfer.begin()
            self.assertIsNone(transfer.visual_verification)

    def test_n_restarts_only_sampling_keeps_diagnostic_and_rejects_old_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.unverified(directory)
            old_record = transfer.records[-1]
            old_diagnostic = deepcopy(transfer.visual_verification)
            retry_time = time.monotonic_ns()
            status = transfer.advance(hardware, state, old_record, lower_requested=True)
            self.assertEqual(status, "verifying_above")
            self.assertGreaterEqual(transfer.not_before_ns, retry_time)
            self.assertEqual(transfer.records, [])
            self.assertEqual(transfer.visual_verification, old_diagnostic)
            self.assertIn("仅重新", transfer.manual_lower_message)
            self.assertIn("再次按 N", transfer.manual_lower_message)
            self.assertIn("20.00 mm", transfer.manual_lower_message)
            self.assert_no_motion(transfer, hardware)

            # Even an old frame that is exactly on target cannot satisfy retry.
            stamp = old_record["timestamp_ns"]
            hardware.history = [(stamp-1_000_000, np.zeros(7)),
                                (stamp+1_000_000, np.zeros(7))]
            transfer.advance(hardware, state, old_record)
            self.assertEqual(transfer.status, "verifying_above")
            self.assertEqual(transfer.records, [])
            self.assertEqual(transfer.visual_verification, old_diagnostic)
            self.assert_no_motion(transfer, hardware)

    def test_retry_requires_unpaused_completed_motion_and_measured_endpoint(self):
        cases = ("paused", "moving", "position_error", "rotation_error")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                transfer, hardware, state = self.unverified(directory)
                if case == "paused":
                    hardware.paused = True
                elif case == "moving":
                    hardware.position_move_finished.return_value = False
                elif case == "position_error":
                    state["T_B_F"][0, 3] += transfer.settings["position_tolerance_m"]*2
                else:
                    state["T_B_F"][:3, :3] = state["T_B_F"][:3, :3] @ Rotation.from_rotvec(
                        [0., 0., transfer.settings["rotation_tolerance_rad"]*2]).as_matrix()
                old_records = deepcopy(transfer.records)
                old_diagnostic = deepcopy(transfer.visual_verification)
                old_cutoff = transfer.not_before_ns
                transfer.advance(hardware, state, None, lower_requested=True)
                self.assertEqual(transfer.status, "above_target_unverified")
                self.assertEqual(transfer.records, old_records)
                self.assertEqual(transfer.not_before_ns, old_cutoff)
                self.assertEqual(transfer.visual_verification, old_diagnostic)
                expected = "已暂停" if case == "paused" else "等待" if case == "moving" else "偏离"
                self.assertIn(expected, transfer.manual_lower_message)
                self.assert_no_motion(transfer, hardware)

    def test_n_during_verification_does_not_clear_samples_or_start_insertion(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.unverified(directory)
            transfer.advance(hardware, state, None, lower_requested=True)
            first = self.fresh_record(transfer, hardware, transfer.above)
            transfer.advance(hardware, state, first)
            self.assertEqual(len(transfer.records), 1)
            old_records = deepcopy(transfer.records)
            cutoff = transfer.not_before_ns
            final_record = self.fresh_record(transfer, hardware, transfer.above)
            transfer.advance(hardware, state, final_record, lower_requested=True)
            self.assertEqual(transfer.status, "above_target")
            self.assertEqual(transfer.records[:1], old_records)
            self.assertEqual(len(transfer.records), 2)
            self.assertEqual(transfer.records[-1]["timestamp_ns"], final_record["timestamp_ns"])
            self.assertEqual(transfer.not_before_ns, cutoff)
            self.assertIn("等待", transfer.manual_lower_message)
            self.assert_no_motion(transfer, hardware)

    def test_repeated_n_does_not_suppress_verification_timeout_even_with_fresh_frame(self):
        for frame_available in (False, True):
            with self.subTest(frame_available=frame_available), tempfile.TemporaryDirectory() as directory:
                transfer, hardware, state = self.unverified(directory)
                transfer.advance(hardware, state, None, lower_requested=True)
                first = self.fresh_record(transfer, hardware, transfer.above)
                transfer.advance(hardware, state, first)
                for _ in range(3):
                    transfer.advance(hardware, state, None, lower_requested=True)
                transfer.config["perception_timeout_s"] = .05
                transfer.not_before_ns = time.monotonic_ns()-100_000_000
                record = self.fresh_record(transfer, hardware, transfer.above) if frame_available else None
                previous_records = deepcopy(transfer.records)
                previous_result = deepcopy(transfer.visual_verification)
                with self.assertRaisesRegex(TimeoutError, "verification timed out"):
                    transfer.advance(hardware, state, record, lower_requested=True)
                self.assertEqual(transfer.records, previous_records)
                self.assertEqual(transfer.visual_verification, previous_result)
                self.assert_no_motion(transfer, hardware)

    def test_motion_completion_and_measured_endpoint_are_checked_during_sampling(self):
        for case in ("moving", "position_error", "rotation_error", "hardware_fault"):
            for lower_requested in (False, True):
                with self.subTest(case=case, n=lower_requested), tempfile.TemporaryDirectory() as directory:
                    transfer, hardware, state = self.unverified(directory)
                    transfer.advance(hardware, state, None, lower_requested=True)
                    first = self.fresh_record(transfer, hardware, transfer.above)
                    transfer.advance(hardware, state, first)
                    previous_result = deepcopy(transfer.visual_verification)
                    if case == "moving":
                        hardware.position_move_finished.return_value = False
                    elif case == "position_error":
                        state["T_B_F"][0, 3] += transfer.settings["position_tolerance_m"]*2
                    elif case == "rotation_error":
                        state["T_B_F"][:3, :3] = state["T_B_F"][:3, :3] @ Rotation.from_rotvec(
                            [0., 0., transfer.settings["rotation_tolerance_rad"]*2]).as_matrix()
                    else:
                        hardware.raise_if_fault.side_effect = RuntimeError("test hardware fault")
                    with self.assertRaises(RuntimeError):
                        transfer.advance(hardware, state, None, lower_requested=lower_requested)
                    self.assertEqual(transfer.visual_verification, previous_result)
                    self.assert_no_motion(transfer, hardware)

    def test_failed_retry_preserves_failed_result_and_can_retry_again(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.unverified(directory)
            transfer.advance(hardware, state, None, lower_requested=True)
            actual = transfer.above.copy()
            actual[0, 3] += .018
            self.verify(transfer, hardware, state, actual)
            self.assertEqual(transfer.status, "above_target_unverified")
            self.assertFalse(transfer.visual_verification["passed"])
            self.assertAlmostEqual(transfer.visual_verification["position_error_m"], .018)
            diagnostic = deepcopy(transfer.visual_verification)
            stale_record = transfer.records[-1]
            transfer.advance(hardware, state, stale_record, lower_requested=True)
            self.assertEqual(transfer.status, "verifying_above")
            self.assertEqual(transfer.records, [])
            self.assertEqual(transfer.visual_verification, diagnostic)
            self.verify(transfer, hardware, state, transfer.above)
            self.assertEqual(transfer.status, "above_target")
            self.assertTrue(transfer.visual_verification["passed"])
            self.assert_no_motion(transfer, hardware)

    def test_passed_retry_needs_a_separate_n_to_start_visual_insertion_modes(self):
        for mode in ("force_spiral", "position_steps"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                transfer, hardware, state = self.unverified(directory)
                transfer.config["lowering"]["mode"] = mode
                with patch.object(transfer, "_start_insertion") as start_force, \
                        patch("rail_receive.lowering.Lowering") as lowering:
                    lowering.return_value.status = "planning_lowering"
                    transfer.advance(hardware, state, None, lower_requested=True)
                    self.verify(transfer, hardware, state, transfer.above)
                    self.assertEqual(transfer.status, "above_target")
                    transfer.advance(hardware, state, None)
                    start_force.assert_not_called()
                    lowering.assert_not_called()
                    self.assert_no_motion(transfer, hardware)
                    transfer.advance(hardware, state, None, lower_requested=True)
                    if mode == "force_spiral":
                        start_force.assert_called_once_with(hardware)
                        lowering.assert_not_called()
                    else:
                        lowering.assert_called_once()
                        start_force.assert_not_called()
                        self.assertEqual(transfer.status, "planning_lowering")

    def test_diagnostics_persist_base_delta_actual_errors_and_limits_with_units(self):
        cases = (([0., 0., 0.], 0., True), ([.012, -.016, 0.], 3., False))
        for delta, angle_deg, passed in cases:
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as directory:
                transfer, hardware, state = self.unverified(directory)
                transfer.advance(hardware, state, None, lower_requested=True)
                actual = transfer.above.copy()
                actual[:3, 3] += delta
                actual[:3, :3] = actual[:3, :3] @ Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
                with redirect_stdout(io.StringIO()) as console:
                    self.verify(transfer, hardware, state, actual)
                diagnostic = transfer.visual_verification
                self.assertEqual(diagnostic["passed"], passed)
                self.assertAlmostEqual(diagnostic["position_error_m"], np.linalg.norm(delta))
                self.assertAlmostEqual(diagnostic["rotation_error_rad"], np.radians(angle_deg))
                self.assertEqual(diagnostic["position_tolerance_m"], transfer.settings["visual_position_tolerance_m"])
                self.assertEqual(diagnostic["rotation_tolerance_rad"], transfer.settings["visual_rotation_tolerance_rad"])
                np.testing.assert_allclose(diagnostic["translation_error_base_m"], delta, atol=1e-12)
                result = json.loads((Path(directory)/"result.json").read_text())
                self.assertEqual(result["visual_verification"], diagnostic)
                self.assertEqual(result["status"], "above_target" if passed else "above_target_unverified")
                self.assertEqual(result["hand_action"], "keep_grasp")
                self.assertFalse(result["insertion_executed"])
                saved = json.loads((Path(directory)/"above_verification.json").read_text())
                self.assertEqual(saved["visual_verification"], diagnostic)
                output = console.getvalue()
                self.assertIn(f"{np.linalg.norm(delta)*1000:.2f} mm", output)
                self.assertIn(f"限值 {transfer.settings['visual_position_tolerance_m']*1000:.2f} mm", output)
                self.assertIn(f"{angle_deg:.2f}°", output)
                self.assertIn(f"限值 {np.degrees(transfer.settings['visual_rotation_tolerance_rad']):.2f}°", output)
                self.assertIn("右基座 XYZ", output)
                self.assert_no_motion(transfer, hardware)


if __name__ == "__main__":
    unittest.main()
