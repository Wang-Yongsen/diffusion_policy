"""N routing, telemetry persistence and configuration gates; no device access."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from queue import Queue
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import receive
import test_above
from rail_receive.insertion import DEFAULT_INSERTION
from rail_receive.logic import load_config


class InsertionRoutingTests(unittest.TestCase):
    def fixture(self, directory):
        transfer, hardware = test_above.AboveTests().fixture(directory)
        transfer.config["lowering"]["mode"] = "force_spiral"
        transfer.status = "above_target"
        transfer.cached_grasp = transfer.above.copy()
        transfer.report = {"T_R_F_above": np.eye(4).tolist()}
        hardware.position_move_finished.return_value = True
        hardware.insertion_snapshot.return_value = {"status": "switching_to_insertion",
            "insertion_executed": False, "revision": 1, "trace": []}
        state = {"T_B_F": np.eye(4), "joints_rad": np.zeros(7)}
        return transfer, hardware, state

    def start(self, transfer, hardware, state):
        # Routing tests isolate calibration validity; controller tests exercise
        # physical parameter validation with synthetic complete fixtures.
        with patch("rail_receive.insertion.validate_settings", return_value=[]):
            return transfer.advance(hardware, state, None, lower_requested=True)

    def test_default_force_mode_loads_with_explicit_missing_parameters(self):
        config = load_config(HERE / "receive.yaml")
        self.assertEqual(config["lowering"]["mode"], "force_spiral")
        self.assertIn("T_F_S", config["insertion_missing"])
        self.assertIn("impedance.stiffness", config["insertion_missing"])
        self.assertTrue(config["insertion_missing"])
        self.assertEqual(set(config["insertion"]), set(DEFAULT_INSERTION))

    def test_missing_parameters_reject_only_n_and_do_not_fall_back_to_position(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.fixture(directory)
            status = transfer.advance(hardware, state, None, lower_requested=True)
            self.assertEqual(status, "above_target")
            self.assertFalse(transfer.insertion_started)
            self.assertIsNone(transfer.lowering)
            self.assertIn("T_F_S", transfer.manual_lower_message)
            self.assertIn("impedance.stiffness", transfer.manual_lower_message)
            hardware.queue_insertion.assert_not_called()
            hardware.move_to_position.assert_not_called()

    def test_gate_requires_verified_start_and_completed_position_motion(self):
        for cause in ("unverified", "moving", "displaced", "paused"):
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory:
                transfer, hardware, state = self.fixture(directory)
                if cause == "unverified":
                    transfer.status = "above_target_unverified"
                elif cause == "moving":
                    hardware.position_move_finished.return_value = False
                elif cause == "displaced":
                    state["T_B_F"][0, 3] = .01
                else:
                    hardware.paused = True
                self.start(transfer, hardware, state)
                hardware.queue_insertion.assert_not_called()
                if cause == "unverified":
                    self.assertEqual(transfer.status, "verifying_above")
                    self.assertIn("本次不下放", transfer.manual_lower_message)
                else:
                    self.assertIn("N 未执行", transfer.manual_lower_message)

    def test_n_queues_frozen_geometry_once_and_stays_latched_after_terminal_state(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.fixture(directory)
            self.assertEqual(self.start(transfer, hardware, state), "switching_to_insertion")
            request = hardware.queue_insertion.call_args.args[0]
            self.assertEqual(set(request), {"settings", "grasp", "goal", "above"})
            np.testing.assert_allclose(request["grasp"], transfer.cached_grasp)
            saved = json.loads((Path(directory) / "insertion_plan.json").read_text())
            self.assertEqual(saved["search_plane"], "XZ")
            self.assertEqual(saved["insertion_axis"], "-Y")
            original = deepcopy(request)
            transfer.config["insertion"]["impedance"]["stiffness"] = [1.] * 6
            transfer.cached_grasp[0, 3] += .1
            self.assertEqual(request, original)
            for status in ("baseline", "searching", "capture", "awaiting_confirmation", "retracted", "paused"):
                hardware.insertion_snapshot.return_value = {"status": status, "trace": [],
                    "insertion_executed": status != "baseline"}
                transfer.advance(hardware, state, None, lower_requested=True)
                self.assertIn("不能重复启动", transfer.manual_lower_message)
                self.assertEqual(transfer.status, status)
            hardware.queue_insertion.assert_called_once()
            hardware.begin_hand.assert_not_called()

    def test_trace_cursor_deduplicates_and_reached_does_not_claim_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.fixture(directory)
            self.start(transfer, hardware, state)
            sample = {"trace_seq": 1, "status": "searching", "wrench": [0.] * 6}
            hardware.insertion_snapshot.return_value = {"status": "searching", "trace": [sample],
                "insertion_executed": True, "revision": 2}
            transfer.poll_insertion(hardware)
            transfer.poll_insertion(hardware)
            hardware.insertion_snapshot.assert_called_with(after_seq=1)
            records = (Path(directory) / "insertion_trace.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 1)
            hardware.insertion_snapshot.return_value = {"status": "awaiting_confirmation", "trace": [],
                "insertion_executed": True, "revision": 3, "depth_m": .05}
            transfer.poll_insertion(hardware)
            result = json.loads((Path(directory) / "result.json").read_text())
            self.assertTrue(result["insertion_executed"])
            self.assertFalse(result["assembly_verified"])
            self.assertFalse(result["automatic_release"])
            self.assertEqual(result["hand_action"], "keep_grasp")
            self.assertNotIn("trace", result)

    def test_rejected_request_remains_at_above_without_latching_or_pausing(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.fixture(directory)
            hardware.queue_insertion.side_effect = ValueError("invalid insertion goal")
            self.assertEqual(self.start(transfer, hardware, state), "above_target")
            self.assertFalse(transfer.insertion_started)
            self.assertIn("invalid insertion goal", transfer.manual_lower_message)
            hardware.pause.assert_not_called()
            hardware.queue_insertion.side_effect = RuntimeError("feedback failed")
            with self.assertRaisesRegex(RuntimeError, "feedback failed"):
                self.start(transfer, hardware, state)

    def test_ring_overrun_is_recorded_as_telemetry_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.fixture(directory)
            self.start(transfer, hardware, state)
            hardware.insertion_snapshot.return_value = {"status": "searching", "trace_first_seq": 5,
                "trace": [{"trace_seq": 5, "status": "searching"}], "insertion_executed": True}
            transfer.poll_insertion(hardware)
            transfer.poll_insertion(hardware)
            records = [json.loads(line) for line in (Path(directory) / "insertion_trace.jsonl").read_text().splitlines()]
            self.assertEqual(records[0], {"event": "telemetry_gap", "expected_trace_seq": 1,
                                          "first_available_trace_seq": 5})
            self.assertEqual(len(records), 2)

    def test_pause_preserves_execution_evidence_and_previous_status(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.fixture(directory)
            self.start(transfer, hardware, state)
            hardware.error = "Operator pause"
            hardware.insertion_snapshot.return_value = {"status": "paused", "previous_status": "inserting",
                "reason": hardware.error, "insertion_executed": True, "trace": []}
            result = receive.paused_result(hardware, transfer)
            self.assertEqual(result["status"], "paused")
            self.assertEqual(result["previous_status"], "inserting")
            self.assertTrue(result["insertion_executed"])
            self.assertFalse(result["assembly_verified"])
            self.assertEqual(result["hand_action"], "retain_last_target")

    def test_keyboard_cannot_queue_jog_during_search_or_held_terminal_state(self):
        for stage_name in ("baseline", "searching", "capture", "inserting", "retracting",
                           "awaiting_confirmation", "retracted"):
            with self.subTest(stage=stage_name):
                terminal, hardware = Mock(), Mock(paused=False)
                end, commands = threading.Event(), Queue(maxsize=1)
                keys = iter(("w", "n", "x"))
                terminal.read.side_effect = lambda _: next(keys)
                receive.keyboard(terminal, hardware, [stage_name], commands, end)
                self.assertEqual(commands.get_nowait(), "n")
                self.assertTrue(commands.empty())

    def test_insertion_preflight_cli_does_not_open_hardware(self):
        report = {"ready": False, "problems": ["missing"], "insertion": {"ready": False}}
        with patch.object(receive, "preflight", return_value=report) as preflight, \
                patch.object(receive, "RightHardware") as hardware, redirect_stdout(io.StringIO()):
            self.assertEqual(receive.main(["check", "--insertion"]), 1)
            self.assertTrue(preflight.call_args.kwargs["insertion"])
            hardware.assert_not_called()


if __name__ == "__main__":
    unittest.main()
