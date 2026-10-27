"""Core offline regression coverage; all actuator and camera owners are mocked."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import receive
import support as test_above
from rail_common.core import (
    digest,
    matrix,
    write_json,
    read_json,
)
from rail_receive.logic import load_config


def config():
    return load_config(HERE / "receive.yaml")


class RuntimeBboxTests(unittest.TestCase):

    def fixture(self, directory, *, draw=True, initial=True, preplace=True):
        t, hw = test_above.AboveTests().fixture(directory)

        t.draw_bboxes = draw
        t.config["sam_prompt"] = str(Path(directory) / "initial.json")
        t.config["sam_prompt_preplace"] = str(Path(directory) / "preplace.json")
        for stage, present in (("initial", initial), ("preplace", preplace)):
            if present:
                self.save_prompt(t, stage)
        state = {"T_B_F": np.eye(4), "joints_rad": np.zeros(7)}
        hw.position_move_finished.return_value = True
        return (t, hw, state)

    def save_prompt(self, t, stage, *, serial="test"):
        key = "sam_prompt" if stage == "initial" else "sam_prompt_preplace"
        box = [20, 30, 300, 180] if stage == "initial" else [100, 210, 550, 390]
        write_json(
            t.config[key],
            dict(
                camera_serial=serial,
                image_size=[640, 480],
                box_xyxy=box,
                point_coords=[],
                point_labels=[],
            ),
        )

    def arrive(self, t, state):
        t.cached_grasp = t.above.copy()
        t.report = {"T_R_F_above": np.eye(4).tolist()}
        t._finish_position_arrival(state, (0.0, 0.0))

    def test_capture_both_stages_separately_then_next_run_reuses_without_wait(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            t, hw, state = self.fixture(directory, initial=False, preplace=False)
            t.begin()
            self.assertEqual(t.status, "waiting_initial_bbox")
            self.assertFalse(t.needs_reinitialize)
            self.assertIsNone(t.preplace)
            self.save_prompt(t, "initial")
            self.assertTrue(receive.accept_runtime_prompt("prompt_saved", t, hw, state))
            self.assertEqual(t.status, "estimating_held_pose")
            self.assertTrue(t.needs_reinitialize)
            initial_bytes = Path(t.config["sam_prompt"]).read_bytes()
            self.arrive(t, state)
            self.assertEqual(t.status, "waiting_preplace_bbox")
            self.assertIsNone(t.preplace)
            self.save_prompt(t, "preplace")
            self.assertTrue(receive.accept_runtime_prompt("preplace_prompt_saved", t, hw, state))
            self.assertEqual(t.status, "reestimating_preplace")
            self.assertTrue(t.needs_reinitialize)
            self.assertIsNone(t.preplace.expected_generation)
            self.assertEqual(Path(t.config["sam_prompt"]).read_bytes(), initial_bytes)
            t.draw_bboxes = False
            t.begin()
            self.assertEqual(t.status, "estimating_held_pose")
            self.arrive(t, state)
            self.assertEqual(t.status, "reestimating_preplace")
            self.assertIsNone(t.waiting_prompt_stage)
            hw.move_to_position.assert_not_called()
            hw.begin_hand.assert_not_called()

    def test_missing_or_wrong_camera_prompt_waits_in_default_reuse_mode(self):
        for stage in ("initial", "preplace"):
            for invalid in ("missing", "wrong_camera"):
                with self.subTest(
                    stage=stage, invalid=invalid
                ), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                    t, hw, state = self.fixture(directory, draw=False)
                    key = "sam_prompt" if stage == "initial" else "sam_prompt_preplace"
                    if invalid == "missing":
                        Path(t.config[key]).unlink()
                    else:
                        self.save_prompt(t, stage, serial="other_camera")
                    t.begin()
                    if stage == "preplace":
                        self.arrive(t, state)
                    self.assertEqual(t.waiting_prompt_stage, stage)
                    self.assertFalse(
                        receive.accept_runtime_prompt(f"prompt_reuse_{stage}", t, hw, state)
                    )
                    self.assertEqual(t.waiting_prompt_stage, stage)
                    hw.move_to_position.assert_not_called()

    def test_waiting_has_no_perception_deadline_and_m_n_do_not_start_or_queue_motion(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            t, hw, state = self.fixture(directory)
            for stage in ("initial", "preplace"):
                if stage == "initial":
                    t.begin()
                else:
                    self.arrive(t, state)
                with patch(
                    "rail_receive.above.time.monotonic", return_value=time.monotonic() + 36000
                ):
                    for _ in range(3):
                        self.assertEqual(
                            t.advance(
                                hw, state, {"valid": True}, force_move=True, lower_requested=True
                            ),
                            f"waiting_{stage}_bbox",
                        )
                self.assertFalse(t.needs_reinitialize)
                self.assertIn("未执行", t.manual_lower_message)
                self.assertTrue(
                    receive.accept_runtime_prompt(f"prompt_reuse_{stage}", t, hw, state)
                )
                self.assertNotEqual(t.status, "planning_above")
                self.assertIsNone(t.spiral)
            self.assertLess(time.monotonic() - t.preplace.started, 1.0)
            hw.move_to_position.assert_not_called()

    def test_wrong_slot_and_paused_or_moved_arm_cannot_confirm(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            t, hw, state = self.fixture(directory)
            self.arrive(t, state)
            self.assertFalse(receive.accept_runtime_prompt("prompt_saved", t, hw, state))
            self.assertEqual(t.status, "waiting_preplace_bbox")
            hw.paused = True
            with self.assertRaisesRegex(RuntimeError, "paused"):
                receive.accept_runtime_prompt("preplace_prompt_saved", t, hw, state)
            hw.paused = False
            state["T_B_F"][0, 3] += 0.01
            with self.assertRaisesRegex(RuntimeError, "moved"):
                t.advance(hw, state, None)
            with self.assertRaisesRegex(RuntimeError, "moved"):
                receive.accept_runtime_prompt("preplace_prompt_saved", t, hw, state)
            hw.move_to_position.assert_not_called()

    def test_save_clears_old_samples_and_only_one_preplace_wait_per_run(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            t, hw, state = self.fixture(directory)
            self.arrive(t, state)
            old_stamp = time.monotonic_ns()
            t.records.append({"timestamp_ns": old_stamp})
            t.grasps.append(np.eye(4))
            self.assertTrue(receive.accept_runtime_prompt("preplace_prompt_saved", t, hw, state))
            self.assertGreater(t.not_before_ns, old_stamp)
            self.assertEqual(t.records, [])
            self.assertEqual(t.grasps, [])
            t.perception_initialized("new_generation")
            self.assertEqual(t.preplace.expected_generation, "new_generation")
            self.assertFalse(t.needs_reinitialize)
            self.assertIsNone(t.waiting_prompt_stage)
            self.assertFalse(t.prompt_required("preplace"))

    def test_running_already_held_waits_at_both_actual_positions_and_reinitializes_correct_slot(
        self,
    ):
        self.run_already_held_case()

    def test_running_through_insertion_automatically_releases_lifts_and_returns(self):
        self.run_already_held_case(auto_return=True)

    def test_pause_during_release_lift_preserves_insert_result_and_stops_return(self):
        self.run_already_held_case(auto_return=True, pause_stage="lifting_after_release")

    def test_timed_receive_runs_through_grasp_insertion_and_return(self):
        self.run_already_held_case(auto_return=True, already_held=False)

    def run_already_held_case(self, *, auto_return=False, pause_stage=None, already_held=True):
        """Exercise real run-loop event routing with fake arm/camera/window."""
        c = load_config(test_above.HERE / "receive.yaml")
        c.update(grasp_delay_s=0.001, post_grasp_pose_delay_s=0.001)
        c["move_above"].update(samples=2, settle_s=0.001)
        c["position_spiral"]["settle_s"] = 0.001
        kin = test_above.ToyKinematics()
        q = np.array([0.0, 0.2, 0.5, 0.0, 0.0, 0.0, 0.0])
        grasp = test_above.T([0.0, -0.24, 0.004])
        hw = Mock(paused=False, error=None, hand_target=np.full(20, 0.4))
        hw.lock, hw.history, hw.jogger = (threading.RLock(), [], NS(scale=1.0))

        def snapshot():
            stamp = time.monotonic_ns()
            hw.history.append((stamp, q.copy()))
            return dict(
                timestamp_ns=stamp,
                joints_rad=q.copy(),
                T_B_F=matrix(kin.fk("right", q)),
                hand_position_rad=hw.hand_target.copy(),
            )

        hw.snapshot.side_effect = snapshot
        hw.position_move_finished.return_value = True
        hw.move_to_position.side_effect = lambda initial, target, **kwargs: q.__setitem__(
            slice(None), target
        )
        hw.begin_hand.side_effect = lambda target: setattr(hw, "hand_target", target.copy())
        hw.hand_transition_finished.return_value = True
        hw.hand_command_finished.return_value = True
        hw.pause.side_effect = lambda reason: (
            setattr(hw, "paused", True),
            setattr(hw, "error", reason),
        )
        geometry = NS(
            T_B_C=np.eye(4), calibration={"camera_serial": "test"}, mesh_digest=digest(c["mesh"])
        )
        contexts, initializations, seen = ([], [], [])
        wait_counts = {"initial": 0, "preplace": 0}

        class Perception:

            def __init__(self, *args):
                self.generation = "old"

            def ready(self):
                return True

            def publish_context(self, state, **kwargs):
                self.state, self.context = (state, kwargs)
                contexts.append(dict(kwargs))

            def initialize(self, *, predicted_pose=None):
                initializations.append(None if predicted_pose is None else predicted_pose.copy())
                self.generation = f"generation_{len(initializations)}"
                return self.generation

            def latest(self):
                if not self.context["enabled"]:
                    raise AssertionError("Waiting must not read stale perception records")
                return dict(
                    valid=True,
                    camera_serial="test",
                    mesh_sha256=geometry.mesh_digest,
                    timestamp_source="sdk_global_timestamp_mapped_to_host_monotonic",
                    timestamp_ns=self.state["timestamp_ns"],
                    context_epoch=self.context["epoch"],
                    generation=self.generation,
                    T_C_CAD=(self.state["T_B_F"] @ grasp).tolist(),
                )

            def close(self):
                pass

        window = Mock(enabled=True)

        def preview(*args, receiving=None):
            stage = receiving["stage"]
            seen.append(stage)
            if stage == pause_stage:
                return "q"
            if stage == "paused":
                return "x"
            if stage in ("waiting_initial_bbox", "waiting_preplace_bbox"):
                slot = receiving["sam_prompt_stage"]
                self.assertTrue(receiving["prompt_editable"] and receiving["prompt_stage_locked"])
                self.assertFalse(receiving["live_pose_preview"])
                self.assertIsNone(receiving.get("pose_record"))
                wait_counts[slot] += 1
                self.assertEqual(len(initializations), 0 if slot == "initial" else 1)
                if wait_counts[slot] == 3:
                    key = "sam_prompt" if slot == "initial" else "sam_prompt_preplace"
                    write_json(
                        c[key],
                        dict(
                            camera_serial="test",
                            image_size=[640, 480],
                            box_xyxy=[20, 30, 500, 400],
                            point_coords=[],
                            point_labels=[],
                        ),
                    )
                    return "prompt_saved" if slot == "initial" else "preplace_prompt_saved"
                return "n"
            if stage == "waiting_move_key":
                return "m"
            if stage == "above_target":
                return "n" if auto_return else "x"
            if stage == "returned_initial":
                return "x"
            return None

        window.update.side_effect = preview

        def keyboard(terminal, hardware, stages, commands, end):
            if not end.wait(12.0):
                hardware.pause("test run-loop timeout")
                end.set()

        terminal = Mock()
        terminal.__enter__ = Mock()
        terminal.__exit__ = Mock()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(
            io.StringIO()
        ), patch.object(receive, "preflight", return_value={"ready": True}), patch.object(
            receive, "kinematics", return_value=kin
        ), patch.object(
            receive, "ReceiveGeometry", return_value=geometry
        ), patch.object(
            receive, "RightHardware", return_value=hw
        ), patch.object(
            receive, "ReceivePerception", Perception
        ), patch.object(
            receive, "keyboard", keyboard
        ), patch(
            "rail_receive.visualization.PreviewWindow", return_value=window
        ), patch(
            "diffusion_policy.real_world.tianji.terminal.MotionTerminal", return_value=terminal
        ):
            root = Path(directory)
            c.update(
                sam_prompt=str(root / "first.json"), sam_prompt_preplace=str(root / "second.json")
            )
            c["move_above"]["target"] = str(root / "target.json")
            write_json(
                c["move_above"]["target"],
                dict(
                    unit="m",
                    camera_serial="test",
                    mesh_sha256=geometry.mesh_digest,
                    T_C_O_goal=test_above.T([0.02, 0.1, 0.7]).tolist(),
                ),
            )
            receive.run(
                c,
                NS(
                    already_held=already_held,
                    draw_bboxes=True,
                    no_view=False,
                    output=root / "run",
                ),
            )
            self.assertIn("above_target", seen, hw.error)
            self.assertEqual(wait_counts, dict(initial=3, preplace=3))
            self.assertEqual(len(initializations), 2)
            self.assertIsNone(initializations[0])
            self.assertEqual(initializations[1].shape, (4, 4))
            self.assertGreaterEqual(sum((not v["enabled"] for v in contexts)), 4)
            self.assertTrue((root / "first.json").is_file() and (root / "second.json").is_file())
            if auto_return:
                from rail_receive.hardware import initial_receiving_pose
                from rail_receive.logic import hand_targets

                for stage in (
                    "position_spiral",
                    "planning_return",
                    "releasing_hand",
                    "lifting_after_release",
                ):
                    self.assertIn(stage, seen, hw.error)
                self.assertEqual(
                    read_json(root / "run/result.json")["insertion"]["status"], "insertion_reached"
                )
                self.assertEqual(
                    {p.name for p in (root / "run").iterdir()}, {"events.jsonl", "result.json"}
                )
                result = read_json(root / "run/result.json")
                self.assertIn("joints_rad", result["last_feedback"])
                self.assertIn("T_B_F", result["last_feedback"])
                self.assertTrue(result["release_sequence_completed"])
                self.assertFalse(result["hand_open_confirmed"])
                self.assertEqual(hw.begin_hand.call_count, 1 if already_held else 3)
                np.testing.assert_allclose(hw.begin_hand.call_args.args[0], hand_targets(c)["open"])
                if pause_stage:
                    self.assertEqual(result["status"], "paused")
                    self.assertNotIn("returning_initial", seen)
                    self.assertFalse(result["returned_to_initial"])
                else:
                    self.assertIn("returning_initial", seen)
                    self.assertEqual(result["status"], "returned_initial", hw.error)
                    self.assertTrue(result["returned_to_initial"])
                    np.testing.assert_allclose(
                        matrix(kin.fk("right", q)), initial_receiving_pose(c, kin)
                    )
            else:
                hw.move_to_position.assert_called_once()
                hw.begin_hand.assert_not_called()


class PreplaceTransferTests(unittest.TestCase):

    def fixture(self, directory, *, position=0.012, degrees=2.0):
        t, hw = test_above.AboveTests().fixture(directory)
        t.config["preplace_alignment"]["settle_s"] = 1e-05
        q = np.array([0.2, 0.1, 0.5, 0.05, -0.05, 0.03, 0.0])
        flange = matrix(t.kinematics.fk("right", q))
        state = dict(T_B_F=flange, joints_rad=q)
        t.cached_grasp = np.linalg.inv(flange) @ t.above
        observed = t.above.copy()
        observed[0, 3] += position
        observed[:3, :3] = (
            Rotation.from_euler("y", degrees, degrees=True).as_matrix() @ observed[:3, :3]
        )
        grasp = np.linalg.inv(flange) @ observed
        t.report = dict(T_R_F_above=flange.tolist(), joint_waypoints_rad=[q.tolist(), q.tolist()])
        hw.position_move_finished.return_value = True
        t._finish_position_arrival(state, (0.0, 0.0))
        return (t, hw, state, grasp)

    def record(self, t, hw, state, grasp, *, generation="fresh", stamp=None):
        stamp = time.monotonic_ns() if stamp is None else stamp
        hw.history = [
            (stamp - 1000000, state["joints_rad"].copy()),
            (stamp + 1000000, state["joints_rad"].copy()),
        ]
        record = test_above.AboveTests().record(
            t, stamp, np.linalg.inv(t.geometry.T_B_C) @ state["T_B_F"] @ grasp
        )
        record.update(
            generation=generation,
            reestimate_request=True,
            pose_source="preplace_registration",
            body_orientation_prior_enabled=False,
        )
        return record

    def advance(self, t, hw, state, record=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return t.advance(hw, state, record, **kwargs)

    def sample(self, t, hw, state, grasp, *, generation="fresh", **kwargs):
        t.perception_initialized(generation)
        for _ in range(t.config["preplace_alignment"]["samples"]):
            self.advance(
                t, hw, state, self.record(t, hw, state, grasp, generation=generation), **kwargs
            )

    def execute_plan(self, t, hw, state):
        self.assertEqual(t.status, "planning_preplace")
        self.assertTrue(t.preplace.done.wait(10), "mock IK planner did not finish")
        self.advance(t, hw, state)
        self.assertEqual(t.status, "aligning_preplace")
        for _ in range(500):
            state["joints_rad"] = hw.move_to_position.call_args.args[1].copy()
            state["T_B_F"] = matrix(t.kinematics.fk("right", state["joints_rad"]))
            self.advance(t, hw, state, lower_requested=True)
            if t.status != "aligning_preplace":
                break
            t.preplace.settled_since = time.monotonic() - 1
            self.advance(t, hw, state, lower_requested=True)
            if t.status != "aligning_preplace":
                break
        else:
            self.fail("position correction did not complete")

    def test_old_generation_old_frames_duplicates_and_missing_history_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            t, hw, state, grasp = self.fixture(directory)
            record = self.record(t, hw, state, grasp)
            self.advance(t, hw, state, record)
            self.assertEqual(len(t.records), 0)
            t.perception_initialized("fresh")
            self.advance(t, hw, state, dict(record, generation="old"))
            self.advance(t, hw, state, dict(record, timestamp_ns=t.not_before_ns - 1))
            self.advance(t, hw, state, dict(record, timestamp_ns=time.monotonic_ns() - 2000000000))
            hw.history = []
            self.advance(t, hw, state, record)
            self.assertEqual(len(t.records), 0)
            record = self.record(t, hw, state, grasp)
            self.advance(t, hw, state, record)
            self.advance(t, hw, state, record)
            self.assertEqual(len(t.records), 1)
            hw.move_to_position.assert_not_called()

    def test_default_single_correction_holds_without_resampling_until_a_fresh_n(self):
        for position, degrees in ((0.03409, 4.22), (0.001, 0.9), (0.05, 11.0)):
            with self.subTest(
                position=position, degrees=degrees
            ), tempfile.TemporaryDirectory() as directory:
                t, hw, state, grasp = self.fixture(directory, position=position, degrees=degrees)
                original_goal, original_above = (t.goal.copy(), t.above.copy())
                self.sample(t, hw, state, grasp, lower_requested=True)
                self.execute_plan(t, hw, state)
                self.assertEqual(t.status, "above_target")
                self.assertTrue(t.preplace.completed)
                self.assertFalse(t.needs_reinitialize)
                self.assertEqual(t.preplace.plans, 1)
                self.assertEqual(len(t.preplace.measurements), 1)
                self.assertIsNone(t.spiral)
                self.assertTrue(all((e["endpoint_verified"] for e in t.preplace.evidence)))
                commands = hw.move_to_position.call_count
                shifted_grasp = grasp.copy()
                shifted_grasp[0, 3] += 0.02
                for _ in range(4):
                    self.advance(t, hw, state, self.record(t, hw, state, shifted_grasp))
                self.assertEqual(hw.move_to_position.call_count, commands)
                self.assertIsNone(t.spiral)
                np.testing.assert_allclose(t.cached_grasp, grasp, atol=1e-10)
                np.testing.assert_allclose(t.goal, original_goal)
                np.testing.assert_allclose(t.above, original_above)
                saved = read_json(Path(directory) / "result.json")["preplace_alignment"]
                self.assertTrue(saved["completed"])
                self.assertEqual(saved["completion_basis"], "single_pass_sdk_fk")
                self.assertFalse((Path(directory) / "preplace_alignment_plan_2.json").exists())
                with patch("rail_receive.position_spiral.PositionSpiral") as spiral:
                    spiral.return_value.status = "planning_position_spiral"
                    self.advance(t, hw, state, lower_requested=True)
                    spiral.assert_called_once()
                    np.testing.assert_allclose(spiral.call_args.args[4], grasp, atol=1e-10)
                    np.testing.assert_allclose(spiral.call_args.args[5], original_goal)
                    self.assertTrue(spiral.return_value.preplace_alignment["completed"])
                hw.begin_hand.assert_not_called()

    def test_already_aligned_single_pass_waits_for_n_without_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            t, hw, state, grasp = self.fixture(directory, position=0.0, degrees=0.0)
            self.sample(t, hw, state, grasp, lower_requested=True)
            self.assertEqual(t.status, "above_target")
            self.assertTrue(t.preplace.completed)
            self.assertEqual(t.preplace.plans, 0)
            self.assertFalse(t.needs_reinitialize)
            self.assertIsNone(t.spiral)
            hw.move_to_position.assert_not_called()

    def test_pause_and_timeouts_never_dispatch(self):
        for case in ("paused", "perception", "total", "drift"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                t, hw, state, _ = self.fixture(directory)
                if case == "paused":
                    hw.paused = True
                elif case == "perception":
                    t.preplace.sample_started -= 181
                elif case == "total":
                    t.preplace.started -= 901
                else:
                    state["T_B_F"][0, 3] += 0.003
                with self.assertRaises((RuntimeError, TimeoutError)):
                    self.advance(t, hw, state, lower_requested=True)
                hw.move_to_position.assert_not_called()

    def test_sdk_and_fk_both_required_and_stopped_summary_preserves_queued_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            t, hw, state, grasp = self.fixture(directory)
            self.sample(t, hw, state, grasp)
            self.assertTrue(t.preplace.done.wait(10))
            self.advance(t, hw, state)
            hw.move_to_position.assert_called_once()
            self.advance(t, hw, state)
            hw.move_to_position.assert_called_once()
            state["joints_rad"] = hw.move_to_position.call_args.args[1]
            state["T_B_F"] = matrix(t.kinematics.fk("right", state["joints_rad"]))
            hw.position_move_finished.return_value = False
            self.advance(t, hw, state)
            hw.move_to_position.assert_called_once()
            result = t.preplace.stopped("Receiver exit")
            self.assertEqual(result["status"], "paused")
            self.assertEqual(result["previous_status"], "aligning_preplace")
            self.assertTrue(result["execution_summary"]["last_step"]["command_queued"])
            self.assertFalse(result["execution_summary"]["last_step"]["endpoint_verified"])


class PerceptionWorkerTests(unittest.TestCase):
    def run_worker(self, *, preplace=False, rejected=False):
        from dataclasses import make_dataclass
        import receive_perception_worker as worker
        from rail_common.pose_alignment import RailPoseAlignment

        c = load_config(HERE / "receive.yaml")
        c["pose_preview"] = {"T_C_O_goal": np.eye(4).tolist()}
        actual = test_above.T(rpy=np.radians([46, 0, 0])) if rejected else np.eye(4)
        raw = actual @ RailPoseAlignment.from_config(c).candidates[1]
        Frame = make_dataclass("Frame", ["rgb", "K", "timestamp_ns", "mask", "metadata"])
        frame = Frame(
            np.zeros((10, 10, 3), np.uint8),
            np.eye(3),
            time.monotonic_ns(),
            None,
            {"timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic"},
        )
        camera, detector, engine = Mock(), Mock(), Mock()
        camera.frame.return_value = frame
        engine.infer.return_value = (raw, {"valid": True, "T_C_CAD": raw.tolist()})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = dict(id="fresh", command="initialize")
            if preplace:
                request["T_C_O_predicted"] = np.eye(4).tolist()
            write_json(root / "request.json", request)
            write_json(root / "context.json", dict(enabled=True, epoch=7))

            def detect(frame):
                write_json(root / "request.json", dict(id="exit", command="shutdown"))
                return np.ones((10, 10), bool), {"valid": True}

            detector.detect.side_effect = detect
            args = NS(
                serve=root,
                camera_serial="test",
                mesh=Path(c["mesh"]),
                foundationpose_root=Path(c["foundationpose_root"]),
            )
            with patch.object(worker, "Camera", return_value=camera), patch(
                "rail_receive.sam_prompt.SamPromptDetector", return_value=detector
            ) as sam, patch.object(worker, "pose_engine", return_value=engine) as factory, patch(
                "rail_receive.visualization.save_preview"
            ) as preview:
                worker.serve(args, c)
            latest = read_json(root / "latest.json")
            self.assertEqual(latest["generation"], "fresh")
            self.assertEqual(latest["context_epoch"], 7)
            self.assertEqual(
                {p.name for p in root.iterdir()},
                {"request.json", "context.json", "ready.json", "latest.json"},
            )
            if preplace:
                self.assertEqual(sam.call_args.kwargs["stage"], "preplace")
                self.assertFalse(factory.call_args.args[2]["body_orientation_prior"]["enabled"])
            else:
                self.assertNotIn("stage", sam.call_args.kwargs)
            camera.close.assert_called()
            if rejected:
                self.assertFalse(latest["valid"])
                self.assertIn("per-axis", latest["detail"])
                engine.reset.assert_called_once()
                self.assertTrue(all(call.args[3] is None for call in preview.call_args_list))
            else:
                np.testing.assert_allclose(latest["T_C_CAD"], actual, atol=1e-12)
                np.testing.assert_allclose(preview.call_args.args[3], actual, atol=1e-12)

    def test_both_bbox_stages_publish_the_aligned_pose_without_diagnostic_exports(self):
        for preplace in (False, True):
            with self.subTest(preplace=preplace):
                self.run_worker(preplace=preplace)

    def test_out_of_bounds_pose_is_rejected_and_never_drawn(self):
        self.run_worker(rejected=True)
