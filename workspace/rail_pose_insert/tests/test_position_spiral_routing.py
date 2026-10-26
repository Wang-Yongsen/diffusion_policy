"""No-force mode selection and operator routing, without opening devices."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from queue import Queue
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import receive
import test_above
from rail_receive.logic import load_config


class PositionSpiralRoutingTests(unittest.TestCase):
    def moving_above(self, directory):
        transfer, hardware = test_above.AboveTests().fixture(directory)
        transfer.config["lowering"]["mode"] = "position_spiral"
        transfer.config["preplace_alignment"]["enabled"] = False
        transfer.settings["settle_s"] = .1
        transfer.status = "moving_above"
        transfer.motion_started = transfer.last_motion_report = 10.
        transfer.controller_completed_at = transfer.settled_since = None
        transfer.report = {"T_R_F_above": np.eye(4).tolist()}
        transfer.cached_grasp = transfer.above.copy()
        hardware.position_move_finished.return_value = True
        state = {"T_B_F": np.eye(4), "joints_rad": np.zeros(7)}
        return transfer, hardware, state

    def advance_at(self, transfer, hardware, state, now, record=None, **kwargs):
        with patch("rail_receive.above.time.monotonic", return_value=now), \
                redirect_stdout(io.StringIO()):
            return transfer.advance(hardware, state, record, **kwargs)

    def test_mechanical_arrival_does_not_require_or_read_new_visual_pose(self):
        for visual in ("missing", "invalid", "large_error"):
            with self.subTest(visual=visual), tempfile.TemporaryDirectory() as directory:
                transfer, hardware, state = self.moving_above(directory)
                actual = transfer.above.copy()
                actual[0, 3] += .03
                actual[:3, :3] = actual[:3, :3] @ Rotation.from_euler("z", 10., degrees=True).as_matrix()
                record = test_above.AboveTests().record(transfer, time.monotonic_ns(),
                    np.linalg.inv(transfer.geometry.T_B_C) @ actual)
                if visual == "missing":
                    record = None
                elif visual == "invalid":
                    record["valid"] = False
                with patch.object(transfer, "_observation", side_effect=AssertionError("must not reverify")):
                    self.assertEqual(self.advance_at(transfer, hardware, state, 11., record), "moving_above")
                    self.assertEqual(self.advance_at(transfer, hardware, state, 11.2, record), "above_target")
                    self.assertEqual(self.advance_at(transfer, hardware, state, 12., record), "above_target")
                self.assertIsNone(transfer.visual_verification)
                self.assertIsNone(transfer.lowering)
                self.assertFalse(transfer.insertion_started)
                hardware.move_to_position.assert_not_called()
                hardware.queue_insertion.assert_not_called()
                hardware.begin_hand.assert_not_called()
                result = json.loads((Path(directory) / "result.json").read_text())
                self.assertEqual(result, json.loads((Path(directory) / "above_arrival.json").read_text()))
                for key, expected in {"status": "above_target", "control_mode": "position",
                        "arrival_check": "sdk_and_fk", "visual_verification_required": False,
                        "visual_verified": False, "insertion_executed": False,
                        "assembly_verified": False, "hand_action": "keep_grasp"}.items():
                    self.assertEqual(result[key], expected, key)
                self.assertFalse((Path(directory) / "above_verification.json").exists())

    def test_mechanical_arrival_still_requires_sdk_fk_and_settle(self):
        for gate in ("sdk_moving", "position_error", "rotation_error", "settling"):
            with self.subTest(gate=gate), tempfile.TemporaryDirectory() as directory:
                transfer, hardware, state = self.moving_above(directory)
                if gate == "sdk_moving":
                    hardware.position_move_finished.return_value = False
                elif gate == "position_error":
                    state["T_B_F"][0, 3] = transfer.settings["position_tolerance_m"] * 2
                elif gate == "rotation_error":
                    state["T_B_F"][:3, :3] = Rotation.from_rotvec(
                        [0., 0., transfer.settings["rotation_tolerance_rad"] * 2]).as_matrix()
                with patch.object(transfer, "_observation", side_effect=AssertionError("must not reverify")), \
                        patch("rail_receive.position_spiral.PositionSpiral") as spiral:
                    self.advance_at(transfer, hardware, state, 11.)
                    self.advance_at(transfer, hardware, state, 11.05)
                    self.assertEqual(transfer.status, "moving_above")
                    self.advance_at(transfer, hardware, state, 11.05, lower_requested=True)
                    self.assertEqual(transfer.status, "moving_above")
                    spiral.assert_not_called()
                self.assertIsNone(transfer.lowering)
                hardware.move_to_position.assert_not_called()
                hardware.queue_insertion.assert_not_called()
                self.assertFalse((Path(directory) / "above_arrival.json").exists())

    def test_n_after_unverified_arrival_keeps_pause_and_current_endpoint_gates(self):
        for gate in ("paused", "sdk_moving", "position_error", "rotation_error"):
            with self.subTest(gate=gate), tempfile.TemporaryDirectory() as directory:
                transfer, hardware, state = self.moving_above(directory)
                self.advance_at(transfer, hardware, state, 11.)
                self.advance_at(transfer, hardware, state, 11.2)
                self.assertEqual(transfer.status, "above_target")
                if gate == "paused":
                    hardware.paused = True
                elif gate == "sdk_moving":
                    hardware.position_move_finished.return_value = False
                elif gate == "position_error":
                    state["T_B_F"][0, 3] = transfer.settings["position_tolerance_m"] * 2
                else:
                    state["T_B_F"][:3, :3] = Rotation.from_rotvec(
                        [0., 0., transfer.settings["rotation_tolerance_rad"] * 2]).as_matrix()
                with patch("rail_receive.position_spiral.PositionSpiral") as spiral:
                    self.advance_at(transfer, hardware, state, 12., lower_requested=True)
                    spiral.assert_not_called()
                self.assertEqual(transfer.status, "above_target")
                self.assertIsNone(transfer.lowering)
                self.assertIn("N 未执行", transfer.manual_lower_message)
                hardware.move_to_position.assert_not_called()
                hardware.queue_insertion.assert_not_called()

    def test_n_during_arrival_is_not_cached_and_insertion_requires_a_fresh_n(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.moving_above(directory)
            with patch("rail_receive.position_spiral.PositionSpiral") as spiral:
                spiral.return_value.status = "planning_position_spiral"
                hardware.position_move_finished.return_value = False
                self.advance_at(transfer, hardware, state, 10.5, lower_requested=True)
                hardware.position_move_finished.return_value = True
                self.advance_at(transfer, hardware, state, 11.)
                # A key arriving on the settling boundary must not start insertion.
                self.advance_at(transfer, hardware, state, 11.2, lower_requested=True)
                self.advance_at(transfer, hardware, state, 11.3)
                self.assertEqual(transfer.status, "above_target")
                self.advance_at(transfer, hardware, state, 12.)
                spiral.assert_not_called()
                self.assertIsNone(transfer.lowering)
                self.advance_at(transfer, hardware, state, 12.1, lower_requested=True)
                spiral.assert_called_once()
                self.assertEqual(transfer.status, "planning_position_spiral")
            hardware.queue_insertion.assert_not_called()
            hardware.begin_hand.assert_not_called()

    def test_selection_does_not_validate_force_or_impedance_parameters(self):
        data = yaml.safe_load((HERE / "receive.yaml").read_text())
        data["insertion"] = {"wrench_sign": 0, "impedance": "unavailable"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(yaml.safe_dump(data))
            config = load_config(path, insertion_mode="position_spiral")
            self.assertEqual(config["lowering"]["mode"], "position_spiral")
            self.assertEqual(config["insertion_missing"], [])
            with self.assertRaises(ValueError):
                load_config(path, insertion_mode="force_spiral")

    def test_preflight_position_mode_has_no_force_requirements(self):
        config = load_config(HERE / "receive.yaml", insertion_mode="position_spiral")
        with patch("rail_receive.insertion.validate_settings", side_effect=AssertionError("force read")):
            report = receive.preflight(config, probe_gpu=False, insertion=True)
        self.assertEqual(report["insertion"], {"mode": "position_spiral", "ready": True,
                         "missing": [], "force_sensor_used": False})

    def test_cli_selects_mode_before_loading_and_preserves_read_only_check(self):
        with patch.object(receive, "preflight", return_value={"ready": True}) as preflight, \
                patch.object(receive, "RightHardware") as hardware, redirect_stdout(io.StringIO()):
            self.assertEqual(receive.main(["check", "--insertion", "--insertion-mode", "position_spiral"]), 0)
        self.assertEqual(preflight.call_args.args[0]["lowering"]["mode"], "position_spiral")
        hardware.assert_not_called()

    def test_n_starts_position_spiral_once_without_wrench_or_force_handover(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hardware, state = self.moving_above(directory)
            self.advance_at(transfer, hardware, state, 11.)
            self.advance_at(transfer, hardware, state, 11.2)
            self.assertEqual(transfer.status, "above_target")
            transfer.advance(hardware, state, None, lower_requested=True)
            self.assertEqual(transfer.status, "planning_position_spiral")
            spiral = transfer.lowering
            self.assertTrue(spiral.done.wait(2.))
            self.assertIsNone(spiral.error)
            with redirect_stdout(io.StringIO()):
                transfer.advance(hardware, state, None, lower_requested=True)
            self.assertIs(transfer.lowering, spiral)
            self.assertEqual(transfer.status, "position_spiral")
            hardware.move_to_position.assert_called_once()
            hardware.queue_insertion.assert_not_called()
            hardware.insertion_snapshot.assert_not_called()
            hardware.begin_hand.assert_not_called()
            # Repeated N must still run deviation checks on the active path.
            state["T_B_F"][0, 3] += .02
            with self.assertRaises(RuntimeError):
                transfer.advance(hardware, state, None, lower_requested=True)
            hardware.move_to_position.assert_called_once()
            hardware.error = "operator pause"
            result = receive.paused_result(hardware, transfer)
            self.assertEqual(result["status"], "paused")
            self.assertFalse(result["force_controlled"])
            self.assertFalse(result["force_sensor_used"])
            self.assertFalse(result["assembly_verified"])
            self.assertEqual(result["hand_action"], "retain_last_target")
            self.assertEqual(json.loads((Path(directory) / "insertion_result.json").read_text()), result)

    def test_keyboard_cannot_jog_or_change_hand_during_or_after_spiral(self):
        for stage in ("planning_position_spiral", "position_spiral", "awaiting_confirmation"):
            with self.subTest(stage=stage):
                terminal, hardware = Mock(), Mock(paused=False)
                end, commands = threading.Event(), Queue(maxsize=1)
                keys = iter(("w", "g", "n", "x"))
                terminal.read.side_effect = lambda _: next(keys)
                receive.keyboard(terminal, hardware, [stage], commands, end)
                self.assertEqual(commands.get_nowait(), "n")
                self.assertTrue(commands.empty())


if __name__ == "__main__":
    unittest.main()
