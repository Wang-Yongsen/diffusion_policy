import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import numpy.testing
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import receive
from rail_preinsert.core import digest, difference, matrix, write_json
from rail_receive.hardware import JointTrackingGuard, RightHardware
from rail_receive.logic import KeyboardJog, ReceiveGeometry, DetectionGate, hand_targets, load_config
from rail_receive.perception import ReceivePerception
from diffusion_policy.third_party.bimanual_teleop.types import Pose, Submission


def config():
    return load_config(HERE / "receive.yaml")


def geometry():
    return NS(T_B_C=np.eye(4), T_F_G=np.eye(4), calibration={"camera_serial": "test"}, mesh_digest="mesh")


def observation(seconds, xyz=(0, 0, 0), angle=0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", angle).as_matrix()
    # Position the actual CAD midpoint, independently of the end-based origin.
    T[:3, 3] = np.asarray(xyz) - T[:3, :3] @ np.asarray(config()["rail_midpoint_cad_m"])
    return {"valid": True, "timestamp_ns": round(seconds * 1e9), "T_C_CAD": T.tolist(),
            "context_epoch": 0,
            "camera_serial": "test", "mesh_sha256": digest(config()["mesh"]), "replay_only": False,
            "detector": "cnos", "detected_in_roi": True, "roi_xyxy": config()["receive_roi_xyxy"],
            "timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic"}


class GateTests(unittest.TestCase):
    def gate(self):
        return DetectionGate(config(), "test", digest(config()["mesh"]))

    def feed(self, gate, stamp, **changes):
        record = dict(observation(stamp), **changes)
        # The detection gate does not consume a pose or an arm-still flag.
        record.pop("T_C_CAD", None)
        return gate.update(record, round(stamp*1e9)+10_000_000)

    def test_detection_for_one_second_closes_once(self):
        gate = self.gate()
        for stamp in (10, 10.25, 10.5, 10.75):
            self.assertFalse(self.feed(gate, stamp)["ready"])
        self.assertTrue(self.feed(gate, 11)["ready"])
        self.assertFalse(self.feed(gate, 11.25)["ready"])
        gate.reset()
        self.assertFalse(self.feed(gate, 12)["ready"])

    def test_negative_frame_restarts_duration(self):
        gate = self.gate()
        self.feed(gate, 10)
        self.feed(gate, 10.5)
        self.feed(gate, 10.75, detected_in_roi=False)
        self.assertFalse(self.feed(gate, 11)["ready"])
        self.assertFalse(self.feed(gate, 11.5)["ready"])
        self.assertTrue(self.feed(gate, 12)["ready"])

    def test_duplicate_detection_does_not_advance_duration(self):
        gate = self.gate()
        for _ in range(100):
            self.assertFalse(self.feed(gate, 10)["ready"])
        self.assertFalse(self.feed(gate, 10.5)["ready"])

    def test_identity_and_replay_checks_reset_count(self):
        for change in ({"valid": False}, {"detector": "other"}, {"roi_xyxy": [0, 0, 640, 480]},
                       {"replay_only": True}, {"camera_serial": "other"}, {"mesh_sha256": "other"},
                       {"timestamp_source": "host_guess"}):
            gate = self.gate()
            self.feed(gate, 10)
            self.assertFalse(self.feed(gate, 10.5, **change)["ready"])
            self.assertIsNone(gate.first_stamp)

    def test_stale_future_and_large_gap_restart_count(self):
        for now in (9_000_000_000, 14_000_000_000):
            gate = self.gate()
            self.feed(gate, 10)
            self.assertFalse(gate.update(observation(10.5), now)["ready"])
            self.assertIsNone(gate.first_stamp)
        gate = self.gate()
        self.feed(gate, 10)
        self.assertFalse(self.feed(gate, 13)["ready"])
        self.assertEqual(gate.first_stamp, 13_000_000_000)


class JogTests(unittest.TestCase):
    def test_right_base_translation_and_fixed_base_rotation(self):
        c = config()
        jog = KeyboardJog(c, np.eye(4))
        target = jog.propose("r")
        np.testing.assert_allclose(target[:3, 3], [0, .001, 0])
        target = jog.propose("u")
        np.testing.assert_allclose(target[:3, :3], Rotation.from_euler("z", c["jog_step_rad"]).as_matrix())

    def test_jog_can_pass_the_removed_extra_workspace_and_lead_limits(self):
        c = config()
        jog = KeyboardJog(c, np.eye(4))
        for _ in range(120):
            jog.propose("w")
        for _ in range(80):
            jog.propose("i")
        self.assertAlmostEqual(jog.target[0, 3], .12)
        self.assertGreater(Rotation.from_matrix(jog.target[:3, :3]).magnitude(), np.radians(30))

    def test_step_scale_controls_remain_available(self):
        jog = KeyboardJog(config(), np.eye(4))
        for _ in range(20):
            jog.propose("+")
        self.assertEqual(jog.scale, 4)
        for _ in range(20):
            jog.propose("-")
        self.assertEqual(jog.scale, .25)


class JointTrackingGuardTests(unittest.TestCase):
    def test_brief_overrun_can_recover_and_later_overrun_starts_a_new_timer(self):
        guard = JointTrackingGuard(config())
        for stamp in (10., 10.2, 10.49):
            self.assertIsNone(guard.update(.04, stamp)["reason"])
        recovered = guard.update(.034, 10.495)
        self.assertFalse(recovered["exceeded"])
        later = guard.update(.04, 20.)
        self.assertIsNone(later["reason"])
        self.assertEqual(later["exceeded_for_s"], 0.)

    def test_sustained_overrun_stops_at_timeout_without_requiring_large_error(self):
        guard = JointTrackingGuard(config())
        guard.update(.0350001, 0.)
        self.assertIsNone(guard.update(.0350001, .499)["reason"])
        self.assertEqual(guard.update(.0350001, .5)["reason"], "persistent")

    def test_large_error_stops_on_first_sample(self):
        result = JointTrackingGuard(config()).update(.070001, 0.)
        self.assertEqual(result["reason"], "hard_limit")
        self.assertEqual(result["exceeded_for_s"], 0.)

    def test_limit_equality_is_not_treated_as_exceeding(self):
        guard = JointTrackingGuard(config())
        self.assertFalse(guard.update(.035, 0.)["exceeded"])
        self.assertIsNone(guard.update(.07, 0.)["reason"])
        self.assertEqual(guard.update(.07, .5)["reason"], "persistent")

    def test_invalid_tracking_feedback_is_rejected(self):
        for error in (None, float("nan"), float("inf"), -.01):
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, "Invalid"):
                JointTrackingGuard(config()).update(error, 0.)

    def test_new_path_resets_previous_overrun_duration(self):
        guard = JointTrackingGuard(config())
        guard.update(.04, 0.)
        guard.reset()
        self.assertIsNone(guard.update(.04, 10.)["reason"])

    def test_tracking_config_rejects_invalid_time_and_hard_limits(self):
        import yaml
        for change in ({"arm_tracking_error_timeout_s": 0},
                       {"arm_tracking_error_timeout_s": float("nan")},
                       {"arm_tracking_hard_error_rad": .035},
                       {"arm_tracking_hard_error_rad": float("inf")}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"receive.yaml"
                path.write_text(yaml.safe_dump(dict(config(), **change)))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_old_config_gets_bounded_tracking_defaults(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            data = config()
            del data["arm_tracking_error_timeout_s"], data["arm_tracking_hard_error_rad"]
            path = Path(directory)/"receive.yaml"
            path.write_text(yaml.safe_dump(data))
            loaded = load_config(path)
            self.assertEqual(loaded["arm_tracking_error_timeout_s"], .5)
            self.assertEqual(loaded["arm_tracking_hard_error_rad"], 2*loaded["arm_tracking_error_rad"])


class HardwareTests(unittest.TestCase):
    def hardware(self, motion=True):
        c = config()
        driver, hand, executor = Mock(), Mock(), Mock()
        driver.health.return_value = hand.health.return_value = NS(ready=True)
        hand.device_id = "wuji_right_hand"
        hand.last_target = NS(position_rad=(.2,) * 20)
        hand.submit.side_effect = lambda command: Submission(command.command_id, True)
        executor.engagement_poses = {"right": Pose("tianji_right_base", "tianji_right_flange", (0, 0, 0), (0, 0, 0, 1))}
        executor.applied_poses = dict(executor.engagement_poses)
        k = NS(fk=lambda side, q: Pose(f"tianji_{side}_base", f"tianji_{side}_flange", (0, 0, 0), (0, 0, 0, 1)))
        hw = RightHardware(c, k, Mock(), motion=motion, devices=(Mock(), driver, hand), executor=executor)
        hw.snapshot = lambda: {"T_B_F": np.eye(4), "joints_rad": np.zeros(7), "hand_position_rad": np.full(20, .2)}
        return hw, driver, hand, executor

    def test_read_only_cannot_configure_or_enable_either_actuator(self):
        hw, driver, hand, executor = self.hardware(motion=False)
        with self.assertRaises(RuntimeError):
            hw.engage()
        executor.configure.assert_not_called()
        driver.engage.assert_not_called()
        hand.engage.assert_not_called()

    def test_only_right_arm_is_configured_and_pause_keeps_right_hand_commands(self):
        hw, _, hand, executor = self.hardware()
        # Test the actual hand thread; arm routing is checked through the configured executor.
        with patch.object(hw, "_arm_loop"):
            hw.engage()
        try:
            profile = executor.configure.call_args.args[0]
            self.assertEqual(profile.parameters["active_arms"], ["right"])
            self.assertEqual(set(profile.parameters["arms"]), {"right"})
            hw.begin_hand(np.full(20, .3))
            time.sleep(.03)
            hw.pause("test")
            frozen = hw.hand_target.copy()
            count = hand.submit.call_count
            time.sleep(.04)
            self.assertGreater(hand.submit.call_count, count)
            np.testing.assert_allclose(hand.submit.call_args.args[0].payload.position_rad, frozen)
            self.assertEqual(hand.submit.call_args.args[0].device_id, "wuji_right_hand")
            hand.disable.assert_not_called()
            executor.request_hold.assert_called_once()
            with self.assertRaises(RuntimeError):
                hw.begin_hand(np.full(20, .4))
        finally:
            hw.close()

    def test_close_attempts_all_resources_even_when_hand_close_fails(self):
        hw, driver, hand, _ = self.hardware()
        hand.close.side_effect = RuntimeError("failure")
        with self.assertRaises(RuntimeError):
            hw.close()
        driver.close.assert_called_once()
        hw.session.close.assert_called_once()

    def test_arm_loop_submits_only_a_right_flange_target(self):
        hw, _, _, executor = self.hardware()
        hw.jogger = KeyboardJog(hw.config, np.eye(4))
        executor.tracking_status = {"right": {"max_joint_tracking_error_rad": 0.0}}
        def submit(target, **kwargs):
            kwargs["before_submit"]()
            hw.stop.set()
            return Submission(target.command_id, True)
        executor.submit.side_effect = submit
        hw._arm_loop()
        target = executor.submit.call_args.args[0]
        self.assertEqual(set(target.tool_poses), {"right"})
        self.assertEqual(target.tool_poses["right"].parent_frame, "tianji_right_base")
        self.assertEqual(target.tool_poses["right"].child_frame, "tianji_right_flange")

    def test_repeated_jog_steps_use_measured_pose_instead_of_accumulating_ahead(self):
        hw, _, _, _ = self.hardware()
        hw.jogger = KeyboardJog(hw.config, np.eye(4))
        for _ in range(30):
            target = hw.jog("w")
        self.assertAlmostEqual(target[0, 3], .001)
        moved = np.eye(4); moved[0, 3] = .012
        hw.snapshot = lambda: {"T_B_F": moved}
        self.assertAlmostEqual(hw.jog("w")[0, 3], .013)
        before = hw.jogger.target.copy()
        hw.jog("+")
        np.testing.assert_array_equal(hw.jogger.target, before)

    def test_large_joint_following_error_still_stops_transport_immediately(self):
        for planned, startup in ((False, False), (True, False), (True, True)):
            with self.subTest(planned=planned, startup=startup):
                hw, _, _, executor = self.hardware()
                hw.arm_enabled = True
                hw.initial_move_active = startup
                hw.jogger = KeyboardJog(hw.config, np.eye(4))
                if planned:
                    hw.arm_path = (NS(at=lambda t: np.zeros(7)), time.monotonic())
                executor.tracking_status = {"right": {"max_joint_tracking_error_rad": .08}}
                def submit(target, **kwargs):
                    # Completion may clear startup mode while a submitted
                    # startup frame is still being handled by the arm loop.
                    if startup:
                        hw.initial_move_active = False
                        hw.arm_path = None
                    hw.stop.set()
                    return Submission(target.command_id, True)
                executor.submit.side_effect = submit
                hw._arm_loop()
                self.assertEqual(hw.paused, planned and not startup)
                if planned and not startup:
                    executor.request_hold.assert_called_once()
                else:
                    executor.request_hold.assert_not_called()

    def test_single_borderline_tracking_sample_does_not_abort_transport(self):
        hw, _, hand, executor = self.hardware()
        hw.arm_enabled = True
        hw.jogger = KeyboardJog(hw.config, np.eye(4))
        hw.arm_path = (NS(at=lambda t: np.zeros(7)), time.monotonic())
        executor.tracking_status = {"right": {"max_joint_tracking_error_rad": .0350001}}
        def submit(target, **kwargs):
            hw.stop.set()
            return Submission(target.command_id, True)
        executor.submit.side_effect = submit
        hw._arm_loop()
        self.assertFalse(hw.paused)
        executor.request_hold.assert_not_called()
        hand.disable.assert_not_called()
        states = [call.kwargs for call in hw.emit.call_args_list if call.args[0] == "right_state"]
        self.assertEqual(states[0]["tracking"]["joint_guard"]["exceeded_for_s"], 0.)
        self.assertIn("applied_position_error_m", states[0]["tracking"])

    def test_persistent_tracking_error_records_exact_fault_then_holds(self):
        hw, _, hand, executor = self.hardware()
        hw.arm_enabled = True
        hw.jogger = KeyboardJog(hw.config, np.eye(4))
        hw.arm_path = (NS(at=lambda t: np.zeros(7)), 0.)
        executor.tracking_status = {"right": {"max_joint_tracking_error_rad": .0350001}}
        clock = [0.]
        def submit(target, **kwargs):
            clock[0] += .25
            if executor.submit.call_count == 3:
                hw.stop.set()
            return Submission(target.command_id, True)
        executor.submit.side_effect = submit
        with patch("rail_receive.hardware.time.monotonic", side_effect=lambda: clock[0]):
            hw._arm_loop()
        self.assertTrue(hw.paused)
        self.assertIn("continuously for 0.500 s", hw.error)
        self.assertIn("excess 1e-07 rad", hw.error)
        self.assertIn("flange-to-command error", hw.error)
        faults = [call.kwargs for call in hw.emit.call_args_list if call.args[0] == "right_tracking_fault"]
        self.assertEqual(faults[0]["error_rad"], .0350001)
        self.assertEqual(faults[0]["reason"], "persistent")
        executor.request_hold.assert_called_once()
        hand.disable.assert_not_called()

    def test_controller_rejection_still_pauses_manual_jog(self):
        hw, _, _, executor = self.hardware()
        hw.jogger = KeyboardJog(hw.config, np.eye(4))
        def submit(target, **kwargs):
            hw.stop.set()
            return Submission(target.command_id, False, "controller fault")
        executor.submit.side_effect = submit
        hw._arm_loop()
        self.assertTrue(hw.paused)
        self.assertEqual(hw.error, "controller fault")

    def test_actual_right_base_registration_agrees_with_sdk_at_nonzero_pose(self):
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics
        from rail_preinsert.core import digest
        k = TianjiKinematics()
        c = config()
        with tempfile.TemporaryDirectory() as directory:
            c["calibration"] = str(Path(directory) / "calibration.json")
            write_json(c["calibration"], {"parent_frame": "tianji_left_base", "child_frame": "head_color_optical",
                "unit": "m", "camera_serial": "test", "sdk_model_sha256": k.model.digest, "T_B_C": np.eye(4).tolist()})
            geo = ReceiveGeometry(c, k)
            from rail_preinsert.geometry import RobotGeometry
            left = RobotGeometry(c, k)
            q = np.zeros(54)
            q[7:14] = np.radians([-30, -60, 34, -52, -30, 12, -4])
            expected = matrix(k.fk("right", tuple(q[7:14]))) @ np.linalg.inv(left.robot.flange("right", q, world=True))
            distance, angle = difference(geo.T_B_C, expected @ np.linalg.inv(left.T_B_W))
            self.assertLess(distance, .0001)
            self.assertLess(angle, .0002)
            self.assertEqual(geo.mesh_digest, digest(c["mesh"]))


class BridgeTests(unittest.TestCase):
    def test_startup_automatically_requests_detection_and_publishes_measured_palm(self):
        with tempfile.TemporaryDirectory() as directory, patch("rail_receive.perception.subprocess.Popen") as start:
            start.return_value.poll.return_value = 0
            c = config()
            bridge = ReceivePerception(c, Path(directory) / "perception", "test")
            try:
                request = json.loads((bridge.directory / "request.json").read_text())
                self.assertEqual(request, {"command": "initialize", "id": bridge.generation})
                measured = np.eye(4); measured[0, 3] = .4
                bridge.publish_context({"timestamp_ns": time.monotonic_ns()}, measured,
                                       enabled=True, epoch=123)
                context = json.loads((bridge.directory / "context.json").read_text())
                np.testing.assert_equal(context["T_C_G"], measured)
                self.assertEqual(context["epoch"], 123)
                self.assertTrue(context["enabled"])
            finally:
                bridge.close()

    def test_previous_initialization_and_worker_exit_cannot_supply_a_grasp_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = ReceivePerception.__new__(ReceivePerception)
            bridge.config, bridge.directory = config(), Path(directory)
            bridge.generation = "new"
            bridge.process = Mock()
            bridge.process.poll.return_value = None
            write_json(bridge.directory / "ready.json", {"ready": True})
            write_json(bridge.directory / "latest.json", dict(observation(10), generation="old"))
            self.assertIsNone(bridge.latest())
            bridge.process.poll.return_value = 1
            with self.assertRaises(RuntimeError):
                bridge.latest()


class StartupPreviewTests(unittest.TestCase):
    def test_setup_refreshes_preview_on_main_thread_while_sdk_initializes(self):
        hw = Mock(paused=False)
        setup_threads, preview_threads = [], []
        def start():
            setup_threads.append(threading.get_ident())
            time.sleep(.07)
        hw.start.side_effect = start
        preview = Mock()
        def update(*args, **kwargs):
            preview_threads.append(threading.get_ident())
        preview.update.side_effect = update
        receive.start_with_preview(hw, preview, Path("unused/preview.png"), config(), threading.Event())
        self.assertGreaterEqual(len(preview_threads), 2)
        self.assertEqual(set(preview_threads), {threading.get_ident()})
        self.assertNotEqual(setup_threads[0], threading.get_ident())
        hw.engage.assert_called_once()

    def test_window_exit_during_setup_prevents_engagement(self):
        hw = Mock(paused=False)
        hw.start.side_effect = lambda: time.sleep(.05)
        hw.pause.side_effect = lambda reason: setattr(hw, "paused", True)
        preview = Mock()
        preview.update.return_value = "x"
        end = threading.Event()
        receive.start_with_preview(hw, preview, Path("unused/preview.png"), config(), end)
        self.assertTrue(end.is_set())
        hw.engage.assert_not_called()


class RunTests(unittest.TestCase):
    def run_case(self, *, source="auto", vision="valid", arm_still=True, during_open=False, refresh_pose=False,
                 startup_delay=0., t_during_wait=False, prompt_saved=False, prompt_during_wait=False):
        c = config()
        c.update(grasp_delay_s=.08 if source == "auto" else 2., post_grasp_pose_delay_s=.12)
        grasp_times, contexts, enabled_times, opened_times, closed_times = [], [], [], [], []
        prompt_saved_times = []
        hw = Mock()
        hw.paused, hw.hand_target = False, np.full(20, .4)
        hw.jogger = NS(scale=1.)
        hw.begin_hand.side_effect = lambda target: grasp_times.append(time.monotonic())
        def startup():
            self.assertTrue(window.update.called)  # Visible placeholder precedes SDK initialization.
            self.assertEqual(window.update.call_args_list[0].kwargs["receiving"]["stage"], "starting")
            time.sleep(startup_delay)
        hw.start.side_effect = startup
        hw.snapshot.side_effect = lambda: {"T_B_F": np.eye(4), "hand_position_rad": np.full(20, .4),
                                          "timestamp_ns": time.monotonic_ns()}
        def transition(*args, require_target=False):
            if require_target:
                if during_open and time.monotonic()-grasp_times[0] < .15:
                    return False
                opened_times.append(time.monotonic())
            else:
                closed_times.append(time.monotonic())
            return True
        hw.hand_transition_finished.side_effect = transition
        hw.arm_still_since.return_value = arm_still
        def pause(reason):
            hw.paused, hw.error = True, reason
        hw.pause.side_effect = pause
        class Perception:
            generation = "automatic"
            seen = False
            pose_reads = 0
            initialized = 0
            def __init__(self, *args):
                Perception.seen = True
                if vision == "startup_error":
                    Perception.seen = True
                    raise RuntimeError("camera unavailable")
                self.epoch = 0
            def initialize(self):
                if vision == "not_ready":
                    raise ValueError("FoundationPose is still initializing")
                self.generation = "new"
                Perception.initialized += 1
            def publish_context(self, state, palm, **kwargs):
                contexts.append(kwargs)
                if kwargs["enabled"]:
                    enabled_times.append(time.monotonic())
                self.epoch = kwargs["epoch"]
                self.mode = kwargs.get("mode", "receive")
            def latest(self):
                Perception.seen = True
                if vision == "runtime_error":
                    raise RuntimeError("vision worker failed")
                if vision in ("none", "not_ready"):
                    return None
                record = dict(observation(time.monotonic()), context_epoch=self.epoch,
                              valid=vision in ("valid", "held_error"))
                if self.mode == "held_once":
                    record.pop("detector")
                    Perception.pose_reads += 1
                    if vision == "held_error":
                        record.update(valid=False, reason="held_pose_unavailable", detail="RGB-D unavailable")
                return record
            def stop_tracking(self):
                pass
            def close(self):
                pass
        class Terminal:
            def __enter__(self):
                self.g_sent, self.jog_sent, self.t_sent = False, False, False
                self.deadline = time.monotonic()+3
                return self
            def __exit__(self, *args):
                pass
            def read(self, timeout):
                time.sleep(.025)
                if time.monotonic() > self.deadline:
                    return "x"
                if source == "g" and hw.begin_hand.call_count == 1 and not self.g_sent:
                    if during_open or Perception.seen:
                        if vision == "not_ready" and not self.t_sent:
                            self.t_sent = True
                            return "t"
                        self.g_sent = True
                        return "g"
                if (output / "held_state.json").is_file() and not self.jog_sent:
                    if t_during_wait and not self.t_sent:
                        self.t_sent = True
                        return "t"
                    if vision in ("valid", "held_error") and Perception.pose_reads == 0:
                        return None
                    if refresh_pose:
                        if not (output / "held_pose.json").is_file():
                            return None
                        if not self.t_sent:
                            self.t_sent = True
                            return "t"
                        if Perception.pose_reads < 2:
                            return None
                    if prompt_saved and (not prompt_saved_times or Perception.pose_reads < 2):
                        return None
                    self.jog_sent = True
                    return "d"
                if self.jog_sent:
                    time.sleep(.05)
                    return "x"
                return None
        window = Mock()
        def preview_update(path, receiving=None):
            stage = (receiving or {}).get("stage")
            if (not prompt_saved_times
                    and ((prompt_saved and stage == "pose_done")
                         or (prompt_during_wait and stage == "waiting_pose"))):
                prompt_saved_times.append(time.monotonic())
                return "prompt_saved"
            return None
        window.update.side_effect = preview_update
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(receive, "preflight", return_value={"ready": True}), \
                patch.object(receive, "kinematics", return_value=Mock()), \
                patch.object(receive, "ReceiveGeometry", return_value=geometry()), \
                patch.object(receive, "RightHardware", return_value=hw), \
                patch.object(receive, "ReceivePerception", Perception), \
                patch("rail_receive.visualization.PreviewWindow", return_value=window) as make_window, \
                patch("diffusion_policy.real_world.tianji.terminal.MotionTerminal", Terminal):
            output = Path(directory) / "run"
            c["calibration"] = str(Path(directory) / "calibration.json")
            write_json(c["calibration"], {"camera_serial": "test"})
            run_started = time.monotonic()
            receive.run(c, NS(motion=True, output=output, no_view=True, move_above=False))
            self.assertEqual(hw.begin_hand.call_count, 2)
            make_window.assert_called_once()
            window.close.assert_called_once()
            hw.jog.assert_called_once_with("d")
            targets = hand_targets(c)
            np.testing.assert_allclose(hw.begin_hand.call_args_list[0].args[0], targets["open"])
            np.testing.assert_allclose(hw.begin_hand.call_args_list[1].args[0], targets["grasp"])
            result = json.loads((output / "result.json").read_text())
            self.assertEqual(result["status"], "holding")
            self.assertEqual(result["grasp_source"], "keyboard" if source == "g" else "timer")
            self.assertFalse(result["grasp_verified"])
            self.assertFalse((output / "receive_observation.json").exists())
            self.assertTrue(all(item["mode"] == "held_once" for item in contexts if item["enabled"]))
            if source == "auto":
                self.assertGreaterEqual(grasp_times[1]-opened_times[0], c["grasp_delay_s"])
            if enabled_times:
                self.assertGreaterEqual(enabled_times[0]-closed_times[0], c["post_grasp_pose_delay_s"])
            if prompt_saved or prompt_during_wait:
                self.assertEqual(len(prompt_saved_times), 1)
                self.assertGreaterEqual(prompt_saved_times[0], closed_times[0])
            if prompt_during_wait:
                self.assertLess(prompt_saved_times[0], enabled_times[0])
            self.assertEqual(json.loads((output / "exit_state.json").read_text())["hand_action"], "disabled_on_exit")
            if source == "g":
                self.assertIsNone(result["trigger_observation"])
            if vision == "valid":
                self.assertTrue((output / "held_pose.json").is_file())
                self.assertEqual(Perception.pose_reads, 2 if refresh_pose or prompt_saved else 1)
                stages = [call.kwargs["receiving"]["stage"] for call in window.update.call_args_list]
                self.assertIn("pose_done", stages)
                self.assertIn("waiting_pose", stages)
                if refresh_pose or prompt_saved:
                    self.assertEqual(Perception.initialized, 1)
                if t_during_wait or prompt_during_wait:
                    self.assertEqual(Perception.initialized, 0)
            if vision == "held_error":
                self.assertFalse((output / "held_pose.json").exists())
                stages = [call.kwargs["receiving"]["stage"] for call in window.update.call_args_list]
                self.assertIn("pose_error", stages)
            return hw

    def test_default_auto_grasp_and_keyboard_jog_after_holding(self):
        self.assertEqual(config()["grasp_mode"], "auto")
        self.assertEqual(config()["grasp_delay_s"], 5.)
        self.assertEqual(config()["post_grasp_pose_delay_s"], 8.)
        self.run_case()

    def test_auto_detection_does_not_require_arm_rest(self):
        self.run_case(arm_still=False)

    def test_timer_waits_for_open_hand_and_closes_without_vision(self):
        self.run_case(vision="startup_error", during_open=True)

    def test_hardware_startup_does_not_consume_grasp_countdown(self):
        self.run_case(startup_delay=.15, during_open=True)

    def test_t_during_postgrasp_wait_does_not_skip_the_delay(self):
        self.run_case(t_during_wait=True)

    def test_t_refreshes_pose_in_the_same_window_without_regrasping(self):
        self.run_case(refresh_pose=True)

    def test_saved_prompt_refreshes_pose_without_regrasping(self):
        self.run_case(prompt_saved=True)

    def test_saved_prompt_during_postgrasp_wait_does_not_skip_the_delay(self):
        self.run_case(prompt_during_wait=True)

    def test_postgrasp_pose_failure_keeps_window_grasp_and_keyboard_jog(self):
        self.run_case(vision="held_error")

    def test_g_closes_without_any_detected_rail_and_without_arm_rest(self):
        self.run_case(source="g", vision="none", arm_still=False)

    def test_g_ignores_invalid_observation(self):
        self.run_case(source="g", vision="invalid", arm_still=False)

    def test_g_can_interrupt_opening(self):
        self.run_case(source="g", vision="none", arm_still=False, during_open=True)

    def test_g_still_works_after_vision_startup_failure(self):
        self.run_case(source="g", vision="startup_error", arm_still=False)

    def test_g_still_works_after_vision_runtime_failure(self):
        self.run_case(source="g", vision="runtime_error", arm_still=False)

    def test_g_still_works_when_t_is_pressed_during_vision_initialization(self):
        self.run_case(source="g", vision="not_ready", arm_still=False)


if __name__ == "__main__":
    unittest.main()
