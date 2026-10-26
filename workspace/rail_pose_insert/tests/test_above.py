"""Held-rail/right-arm geometry and motion lifecycle, without opening hardware."""
from contextlib import redirect_stdout
import io
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
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import receive
from rail_preinsert.core import digest, difference, matrix, write_json
from rail_preinsert.planner import JointPath
from rail_receive.above import AboveTransfer, above_targets, plan_above
from rail_receive.logic import load_config, ReceiveGeometry, KeyboardJog
from rail_receive.visualization import pose_overlay, save_preview
import test_receive
from diffusion_policy.third_party.bimanual_teleop.types import Pose, Submission


def T(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    result = np.eye(4)
    result[:3, 3] = xyz
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    return result


class ToyKinematics:
    """Full SE(3) toy for coordinate tests; every call asserts right-arm routing."""
    def __init__(self):
        arm = NS(lower_rad=(-4.,)*7, upper_rad=(4.,)*7,
                 max_joint_velocity_rad_s=(1.,)*7, limits_native=[(0, 0, 0, 60)]*7)
        self.model = NS(arm=lambda side: arm, digest="toy")

    def fk(self, side, q):
        assert side == "right"
        return Pose("tianji_right_base", "tianji_right_flange", tuple(q[:3]),
                    tuple(Rotation.from_rotvec(q[3:6]).as_quat()))

    def ik(self, side, target, seed):
        assert side == "right" and target.parent_frame == "tianji_right_base"
        return (*target.position_m, *Rotation.from_quat(target.orientation_xyzw).as_rotvec(), 0.)


class AboveTests(unittest.TestCase):
    def test_above_uses_right_base_up_with_rotated_camera_and_cad(self):
        X = T([.4, -.2, .1], [.3, -.8, .4])
        goal = T([.2, .1, .8], [-.5, .3, 1.])
        placed, above = above_targets(X, goal, .05)
        np.testing.assert_allclose(placed, X @ goal)
        np.testing.assert_allclose(above[:3, 3]-placed[:3, 3], [0, .05, 0], atol=1e-12)
        np.testing.assert_equal(above[:3, :3], placed[:3, :3])
        for height in (0, -.1, float("nan"), True):
            with self.assertRaises(ValueError):
                above_targets(X, goal, height)

    def test_right_plan_preserves_grasp_and_ends_above_goal_with_joint_limits(self):
        c, k = load_config(HERE / "receive.yaml"), ToyKinematics()
        grasp = T([.02, -.2, .04], [.1, -.2, .3])
        target = T([.12, .05, .6], [.2, .1, -.1])
        q = np.array([.2, -.02, .5, 0., 0., 0., 0.])
        path, report = plan_above(k, c, q, grasp, target)
        actual = matrix(k.fk("right", path.points[-1])) @ grasp
        np.testing.assert_allclose(actual, target, atol=1e-9)
        self.assertEqual(len(path.points), 2)
        self.assertEqual(report["control_mode"], "position")
        self.assertEqual(report["intermediate_targets"], 0)
        self.assertEqual(report["interpolation_owner"], "controller")
        self.assertEqual(report["side"], "right")
        self.assertFalse(report["collision_checked"])
        self.assertEqual(report["velocity_ratio"], 3)

    def fixture(self, directory):
        c = load_config(HERE / "receive.yaml")
        c["move_above"].update(target=str(Path(directory)/"target.json"), samples=2, settle_s=.001, start_mode="auto")
        geo = NS(T_B_C=T([.1, -.2, .3], [.2, -.1, .3]),
                 calibration={"camera_serial": "test"}, mesh_digest=digest(c["mesh"]))
        goal = T([.1, .2, .4], [.1, .1, .1])
        write_json(c["move_above"]["target"], {"unit": "m", "camera_serial": "test",
                   "mesh_sha256": geo.mesh_digest, "T_C_O_goal": goal.tolist()})
        transfer = AboveTransfer(c, geo, ToyKinematics(), directory)
        hw = Mock(paused=False); hw.lock = threading.RLock()
        return transfer, hw

    def record(self, transfer, stamp, camera_pose):
        return {"valid": True, "timestamp_ns": stamp, "T_C_CAD": camera_pose.tolist(),
                "camera_serial": "test", "mesh_sha256": transfer.geometry.mesh_digest,
                "timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic"}

    def test_sampling_uses_configured_position_limit_and_keeps_angle_limit(self):
        for stage in ("estimating_held_pose", "verifying_above"):
            for spread, angle, accepted in ((.007, 0., True), (.012, 0., False), (0., .05, False)):
                with self.subTest(stage=stage, spread=spread, angle=angle), tempfile.TemporaryDirectory() as directory:
                    transfer, hw = self.fixture(directory)
                    transfer.begin()
                    transfer.needs_reinitialize = False
                    transfer.settings["samples"] = 3
                    transfer.status = stage
                    transfer.report = {"T_R_F_above": np.eye(4)}
                    state = {"joints_rad": np.zeros(7), "T_B_F": np.eye(4)}
                    poses = [T([offset, 0, 0], [0, 0, rotation])
                             for offset, rotation in ((-spread, -angle), (0., 0.), (spread, angle))]
                    observations = [(pose, pose, {"T_C_CAD": pose.tolist(), "valid": True,
                                     "timestamp_ns": time.monotonic_ns()+i}) for i, pose in enumerate(poses)]
                    with patch.object(transfer, "_observation", side_effect=observations), \
                            patch("rail_receive.above.threading.Thread"):
                        for _ in range(2):
                            transfer.advance(hw, state, {})
                        if accepted:
                            transfer.advance(hw, state, {})
                            self.assertIn(transfer.status, ("planning_above", "above_target_unverified"))
                        elif stage == "verifying_above":
                            with self.assertRaisesRegex(ValueError, "Stationary"):
                                transfer.advance(hw, state, {})
                        else:
                            transfer.advance(hw, state, {})
                            self.assertEqual(transfer.status, "waiting_pose_override")

    def unstable_transfer(self, directory):
        transfer, hw = self.fixture(directory)
        transfer.settings["samples"] = 3
        transfer.begin(); transfer.needs_reinitialize = False
        state = {"joints_rad": np.zeros(7)}
        for offset in (-.02, 0., .02):
            stamp = time.monotonic_ns()
            hw.history = [(stamp-1_000_000, np.zeros(7)), (stamp+1_000_000, np.zeros(7))]
            camera = np.array(transfer.preview_poses["T_C_O_goal"])
            camera[0, 3] += offset
            record = self.record(transfer, stamp, camera)
            transfer.advance(hw, state, record)
        self.assertEqual(transfer.status, "waiting_pose_override")
        hw.move_to_position.assert_not_called()
        return transfer, hw, state, record

    def test_m_uses_saved_mean_records_override_and_starts_only_one_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hw, state, record = self.unstable_transfer(directory)
            expected = np.mean([pose[:3, 3] for pose in transfer.grasps], axis=0)
            with patch("rail_receive.above.threading.Thread") as thread:
                transfer.advance(hw, state, record, force_move=True)
                self.assertEqual(transfer.status, "planning_above")
                thread.assert_called_once(); thread.return_value.start.assert_called_once()
                transfer.advance(hw, state, record, force_move=True)
                thread.assert_called_once()
            saved = json.loads((Path(directory)/"held_observation.json").read_text())
            self.assertTrue(saved["pose_sampling"]["stability_overridden"])
            self.assertEqual(saved["pose_sampling"]["request_key"], "M")
            np.testing.assert_allclose(np.array(saved["T_F_O"])[:3, 3], expected)
            hw.move_to_position.assert_not_called()  # Planning is a separate step.

    def test_m_uses_saved_mean_without_requiring_any_current_camera_data(self):
        for case in ("stale", "missing", "invalid", "replay"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                transfer, hw, state, record = self.unstable_transfer(directory)
                expected = transfer.cached_grasp.copy()
                saved = json.loads((Path(directory)/"held_observation.json").read_text())
                if case == "stale":
                    for stored in transfer.records:
                        stored["timestamp_ns"] -= 2_000_000_000
                    record = dict(record, timestamp_ns=record["timestamp_ns"]-2_000_000_000)
                elif case == "missing": record = None
                elif case == "invalid": record = dict(record, valid=False)
                elif case == "replay": record = dict(record, replay_only=True)
                with patch("rail_receive.above.threading.Thread") as thread:
                    with patch.object(transfer, "_observation", side_effect=AssertionError("must not resample")):
                        transfer.advance(hw, state, record, force_move=True)
                    self.assertEqual(transfer.status, "planning_above")
                    thread.assert_called_once()
                    np.testing.assert_array_equal(thread.call_args.kwargs["args"][1], expected)
                    np.testing.assert_array_equal(transfer.cached_grasp, np.asarray(saved["T_F_O"]))
                hw.move_to_position.assert_not_called()

    def test_m_does_not_create_a_mean_from_invalid_samples_or_resume_a_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hw = self.fixture(directory)
            transfer.settings["start_mode"] = "manual"
            transfer.begin(); transfer.needs_reinitialize = False
            state = {"joints_rad": np.zeros(7)}
            with patch("rail_receive.above.threading.Thread") as thread:
                for record in (None, {"valid": False}):
                    transfer.advance(hw, state, record, force_move=True)
                    self.assertIsNone(transfer.cached_grasp)
                    self.assertIn("还没有保存", transfer.manual_move_message)
                thread.assert_not_called()
            transfer, hw, state, record = self.unstable_transfer(directory)
            hw.paused = True
            with patch("rail_receive.above.threading.Thread") as thread:
                transfer.advance(hw, state, record, force_move=True)
                self.assertIn("已暂停", transfer.manual_move_message)
                thread.assert_not_called()

    def test_waiting_for_m_does_not_start_automatically_when_pose_later_stabilizes(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hw, state, record = self.unstable_transfer(directory)
            with patch("rail_receive.above.threading.Thread") as thread:
                for _ in range(4):
                    stamp = time.monotonic_ns()
                    hw.history = [(stamp-1_000_000, np.zeros(7)), (stamp+1_000_000, np.zeros(7))]
                    record = self.record(transfer, stamp, np.array(transfer.preview_poses["T_C_O_goal"]))
                    transfer.advance(hw, state, record)
                self.assertFalse(transfer.pose_sampling["stable"])  # Saved estimate remains frozen.
                self.assertEqual(transfer.status, "waiting_pose_override")
                thread.assert_not_called()

    def test_terminal_m_always_reaches_status_feedback_and_never_resumes_pause(self):
        from queue import Queue
        for stage, pressed, expected in (("waiting_pose_override", ["M"], True),
                                        ("estimating_held_pose", ["m"], True),
                                        ("holding", ["m"], True),
                                        ("moving_above", ["m"], True),
                                        ("waiting_pose_override", ["mm"], False),
                                        ("waiting_pose_override", ["q", "m"], True)):
            with self.subTest(stage=stage, keys=pressed):
                terminal, hw = Mock(), Mock(paused=False)
                hw.pause.side_effect = lambda _: setattr(hw, "paused", True)
                end, commands = threading.Event(), Queue(maxsize=1)
                keys = iter(pressed)
                def read(_):
                    try: return next(keys)
                    except StopIteration: end.set(); return None
                terminal.read.side_effect = read
                receive.keyboard(terminal, hw, [stage], commands, end)
                self.assertEqual(not commands.empty(), expected)
                if expected: self.assertEqual(commands.get_nowait(), "m")
                if "q" in pressed: self.assertTrue(hw.paused)

    def test_grasp_uses_acquisition_time_right_joints_and_rejects_stale_or_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hw = self.fixture(directory)
            transfer.begin(); transfer.needs_reinitialize = False
            stamp = time.monotonic_ns()
            q0, q1 = np.zeros(7), np.array([.02, .01, 0, 0, .02, 0, 0])
            hw.history = [(stamp-10_000_000, q0), (stamp+10_000_000, q1)]
            grasp = T([.03, -.1, .05], [.2, .1, -.1])
            camera_pose = np.linalg.inv(transfer.geometry.T_B_C) @ matrix(transfer.kinematics.fk("right", (q0+q1)/2)) @ grasp
            record = self.record(transfer, stamp, camera_pose)
            np.testing.assert_allclose(transfer._observation(record, hw)[1], grasp, atol=1e-10)
            # Even a fresh, otherwise valid pose is unusable if acquired before grasp completion.
            self.assertIsNone(transfer._observation(dict(record, timestamp_ns=transfer.not_before_ns-1), hw))
            self.assertIsNone(transfer._observation(dict(record, timestamp_ns=stamp-2_000_000_000), hw))
            with self.assertRaises(ValueError):
                transfer._observation(dict(record, replay_only=True), hw)
            hw.history = [(stamp+1, q0), (stamp+2, q1)]
            self.assertIsNone(transfer._observation(record, hw))

    def test_verified_above_requires_fresh_post_arrival_vision_and_keeps_hand(self):
        for offset, expected in ((0., "above_target"), (.02, "above_target_unverified")):
            with self.subTest(offset=offset), tempfile.TemporaryDirectory() as directory:
                transfer, hw = self.fixture(directory)
                transfer.status = "verifying_above"
                transfer.report = {"T_R_F_above": np.eye(4)}
                transfer.not_before_ns = time.monotonic_ns()-100_000_000
                actual = transfer.above.copy(); actual[0, 3] += offset
                for _ in range(2):
                    stamp = time.monotonic_ns()
                    hw.history = [(stamp-1_000_000, np.zeros(7)), (stamp+1_000_000, np.zeros(7))]
                    record = self.record(transfer, stamp, np.linalg.inv(transfer.geometry.T_B_C) @ actual)
                    transfer.advance(hw, {"T_B_F": np.eye(4)}, record)
                result = json.loads((Path(directory)/"result.json").read_text())
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["hand_action"], "keep_grasp")
                self.assertFalse(result["insertion_executed"])
                hw.begin_hand.assert_not_called()

    def test_planning_failure_never_starts_arm_path(self):
        with tempfile.TemporaryDirectory() as directory:
            transfer, hw = self.fixture(directory)
            transfer.status, transfer.planning_started = "planning_above", time.monotonic()
            transfer.planning_error = ValueError("unreachable")
            transfer.planning_done.set()
            with self.assertRaisesRegex(ValueError, "unreachable"):
                transfer.advance(hw, {}, None)
            hw.move_to_position.assert_not_called()

    def test_real_sdk_ik_fk_reaches_constructed_above_goal_in_right_base(self):
        from diffusion_policy.third_party.bimanual_teleop.devices.tianji.model import TianjiKinematics
        c, k = load_config(HERE/"receive.yaml"), TianjiKinematics()
        geo = ReceiveGeometry(c, k)
        transfer = AboveTransfer(c, geo, k, Path(tempfile.gettempdir()))
        q = np.radians([-30, -60, 34, -52, -30, 12, -4])
        q_goal = q+np.array([.015, -.01, .005, 0, .005, 0, 0])
        # Constructed grasp, actual taught goal/calibration, official IK/FK. No device I/O.
        grasp = np.linalg.inv(matrix(k.fk("right", tuple(q_goal)))) @ transfer.above
        path, _ = plan_above(k, c, q, grasp, transfer.above)
        distance, angle = difference(matrix(k.fk("right", tuple(path.points[-1]))) @ grasp, transfer.above)
        self.assertLess(distance, .0001)
        self.assertLess(angle, .0002)

    def test_visualization_draws_all_three_poses_and_saves_atomic_preview(self):
        mesh = NS(bounds=np.array([[-.02, -.1, -.01], [.02, .1, .01]]))
        rgb = np.zeros((240, 320, 3), np.uint8)
        K = np.array([[300., 0, 160], [0, 300, 120], [0, 0, 1]])
        current = T([-.15, .1, .7])
        refs = dict(T_C_O_goal=T([0, .1, .7]).tolist(), T_C_O_above=T([.15, .1, .7]).tolist())
        image = pose_overlay(rgb, mesh, current, K, refs)
        for bgr in ((70, 230, 70), (0, 150, 255), (255, 220, 0)):
            self.assertTrue(np.any(np.all(image[80:]==bgr, axis=-1)))
        with tempfile.TemporaryDirectory() as directory:
            save_preview(directory, NS(rgb=rgb, K=K), mesh, current, refs)
            self.assertTrue((Path(directory)/"preview.png").is_file())
            self.assertFalse((Path(directory)/"preview.tmp").exists())


class MotionRoutingTests(unittest.TestCase):
    def run_transfer_case(self, *, source="g", failure=None, override=None, start_mode="auto", noisy=True,
                          no_frames_until_move=False, paused_m_source=None, lower_source=None):
        c = load_config(HERE/"receive.yaml")
        c["post_grasp_pose_delay_s"] = .05
        c["grasp_delay_s"] = .05
        if failure in ("invalid_pose", "camera_after_mean"):
            c["perception_timeout_s"] = .15
        c["move_above"].update(samples=2, settle_s=.001, start_mode=start_mode)
        c["lowering"]["settle_s"] = .001
        c["lowering"]["mode"] = "position_steps"  # This fixture exercises legacy lowering.
        kin = ToyKinematics()
        hw = Mock(paused=False, hand_target=np.full(20, .4))
        hw.lock, hw.history, hw.jogger = threading.RLock(), [], NS(scale=1.)
        q = np.array([0., 0., .7, 0., 0., 0., 0.])
        grasp = T([0., -.24, .004])
        def snapshot():
            stamp = time.monotonic_ns()
            hw.history.append((stamp, q.copy()))
            return {"timestamp_ns": stamp, "joints_rad": q.copy(), "hand_position_rad": hw.hand_target,
                    "T_B_F": matrix(kin.fk("right", q))}
        hw.snapshot.side_effect = snapshot
        hw.hand_transition_finished.return_value = True
        hw.arm_still_since.return_value = True
        def follow(initial, target, *, timeout_s, **ratios):
            q[:] = target
        hw.position_move_finished.return_value = True
        hw.move_to_position.side_effect = follow
        def pause(reason):
            hw.paused, hw.error = True, reason
        hw.pause.side_effect = pause
        geo = NS(T_B_C=np.eye(4), T_F_G=np.eye(4), calibration={"camera_serial": "test"},
                 mesh_digest=digest(c["mesh"]))
        class Perception:
            initialized, stopped = 0, 0
            modes = []
            frames = 0
            def __init__(self, *args):
                pass
            def ready(self):
                return True
            def initialize(self):
                Perception.initialized += 1
            def publish_context(self, state, palm, **kwargs):
                self.state, self.epoch = state, kwargs["epoch"]
                if kwargs["enabled"]:
                    Perception.modes.append(kwargs["mode"])
            def latest(self):
                state = self.state
                if (output/"held_observation.json").is_file():
                    if failure == "camera_after_mean":
                        raise RuntimeError("camera disconnected after saving mean")
                    if no_frames_until_move and not hw.move_to_position.called:
                        return None
                pose = state["T_B_F"] @ grasp
                Perception.frames += 1
                if override and noisy and not hw.move_to_position.called:
                    pose[0, 3] += .02 if Perception.frames % 2 else -.02
                return {"valid": failure != "invalid_pose", "camera_serial": "test", "mesh_sha256": geo.mesh_digest,
                    "timestamp_source": "sdk_global_timestamp_mapped_to_host_monotonic",
                    "timestamp_ns": state["timestamp_ns"], "T_C_CAD": pose.tolist(),
                    "context_epoch": self.epoch}
            def stop_tracking(self):
                Perception.stopped += 1
            def close(self):
                pass
        class Terminal:
            def __enter__(self):
                self.sent, self.deadline = False, time.monotonic()+3
                self.move_sent = self.lower_sent = False
                self.paused_m_sent = False
                return self
            def __exit__(self, *args):
                pass
            def read(self, timeout):
                time.sleep(.01)
                if source == "g" and hw.begin_hand.call_count == 1 and not self.sent:
                    self.sent = True
                    return "g"
                if override == "terminal" and not self.move_sent and (output/"pose_sampling_wait.json").is_file():
                    self.move_sent = True
                    return "m"
                if hw.paused and paused_m_source == "terminal" and not self.paused_m_sent:
                    self.paused_m_sent = True
                    return "m"
                if (output/"result.json").is_file():
                    result = json.loads((output/"result.json").read_text())
                    if result["status"] == "above_target" and lower_source == "terminal" and not self.lower_sent:
                        self.lower_sent = True
                        return "n"
                    if result["status"] in (("placement_reached", "paused") if lower_source else ("above_target", "paused")):
                        if (result["status"] == "placement_reached" and not any(
                                call.kwargs["receiving"]["stage"] == "placement_reached"
                                for call in window.update.call_args_list)):
                            return None
                        if (not paused_m_source
                                or "manual_move_rejected" in (output/"events.jsonl").read_text()):
                            return "x"
                if time.monotonic()>self.deadline:
                    return "x"
        real_plan = plan_above
        def quick_plan(*args):
            if failure == "planning":
                raise ValueError("unreachable test target")
            path, report = real_plan(*args)
            return path, report
        window = Mock()
        def preview(*args, receiving=None):
            if lower_source == "preview" and receiving["stage"] == "above_target":
                return "n"
            if override == "preview" and receiving["stage"] in ("waiting_pose_override", "waiting_move_key"):
                return "m"
            if failure == "pause" and receiving["stage"] == "planning_above":
                return "q"
            if receiving["stage"] == "paused" and paused_m_source == "preview":
                return "m"
            return None
        window.update.side_effect = preview
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as console, \
                patch.object(receive, "preflight", return_value={"ready": True}), \
                patch.object(receive, "load_config", return_value=c), \
                patch.object(receive, "RightHardware", return_value=hw), \
                patch.object(receive, "ReceiveGeometry", return_value=geo), \
                patch.object(receive, "ReceivePerception", Perception), patch.object(receive, "kinematics", return_value=kin), \
                patch("rail_receive.above.plan_above", side_effect=quick_plan), \
                patch("rail_receive.visualization.PreviewWindow", return_value=window), \
                patch("diffusion_policy.real_world.tianji.terminal.MotionTerminal", Terminal):
            output = Path(directory)/"run"
            c["move_above"]["target"] = str(Path(directory)/"target.json")
            goal = T([.03, -.23, .704])
            write_json(c["move_above"]["target"], {"unit": "m", "camera_serial": "test",
                       "mesh_sha256": geo.mesh_digest, "T_C_O_goal": goal.tolist()})
            # Exercise the actual default motion CLI, without --move-above.
            self.assertEqual(receive.main(["run", "--motion", "--output", str(output), "--no-view"]), 0)
            result = json.loads((output/"result.json").read_text())
            self.assertEqual(hw.begin_hand.call_count, 2)  # Open once, grasp once; no release.
            self.assertEqual(Perception.initialized, 1)  # Fresh registration after grasp.
            self.assertEqual(Perception.stopped, 0)
            self.assertTrue(Perception.modes)
            self.assertEqual(set(Perception.modes), {"held"})
            views = [call.kwargs["receiving"] for call in window.update.call_args_list]
            self.assertIn("estimating_held_pose", [view["stage"] for view in views])
            self.assertTrue(all(not view["prompt_editable"] for view in views if view.get("live_pose_preview")))
            if override:
                self.assertIn("waiting_move_key" if start_mode == "manual" else "waiting_pose_override", [view["stage"] for view in views])
                sampling = json.loads((output/"held_observation.json").read_text())["pose_sampling"]
                self.assertEqual(sampling["stability_overridden"], noisy)
                self.assertEqual(sampling["request_key"], "M")
            if failure:
                self.assertEqual(result["status"], "paused", result)
                reason = {"invalid_pose": "No fresh held-rail pose", "planning": "unreachable test target",
                          "pause": "Operator preview pause", "camera_after_mean": "visual verification timed out"}[failure]
                self.assertIn(reason, result["reason"])
                if failure == "camera_after_mean": hw.move_to_position.assert_called_once()
                else: hw.move_to_position.assert_not_called()
                self.assertEqual(result["hand_action"], "retain_last_target")
                if paused_m_source:
                    events = [json.loads(line) for line in (output/"events.jsonl").read_text().splitlines()]
                    rejected = [event for event in events if event.get("event") == "manual_move_rejected"]
                    self.assertTrue(rejected, events)
                    self.assertEqual(rejected[-1]["reason"], result["reason"])
                    self.assertIn("M 已收到，但机械臂因以下原因暂停", console.getvalue())
                return
            self.assertEqual(result["status"], "placement_reached" if lower_source else "above_target", result)
            self.assertEqual(result["hand_action"], "keep_grasp")
            self.assertFalse(result["assembly_verified"] if lower_source else result["insertion_executed"])
            for stage in ("planning_above", "moving_above", "verifying_above"):
                self.assertIn(stage, [view["stage"] for view in views])
            if lower_source:
                self.assertEqual(hw.move_to_position.call_count, 4)  # M + three 5 mm lowering steps.
                for call in hw.move_to_position.call_args_list[1:]:
                    self.assertEqual(call.kwargs["velocity_ratio"], 1)
                for stage in ("planning_lowering", "lowering", "placement_reached"):
                    self.assertIn(stage, [view["stage"] for view in views])
                self.assertIn("N 已收到", console.getvalue())
                self.assertTrue((output/"lowering_plan.json").is_file())
            else:
                hw.move_to_position.assert_called_once()
            self.assertTrue((output/"held_observation.json").is_file())
            self.assertTrue((output/"above_plan.json").is_file())

    def test_n_from_terminal_and_preview_lowers_after_m_without_releasing(self):
        for source in ("terminal", "preview"):
            with self.subTest(source=source):
                self.run_transfer_case(override=source, start_mode="manual", noisy=False, lower_source=source)

    def test_auto_mode_g_grasp_reestimates_plans_moves_and_verifies_without_releasing(self):
        self.run_transfer_case()

    def test_auto_mode_timed_grasp_moves_after_stable_pose(self):
        self.run_transfer_case(source="timer")

    def test_default_invalid_pose_never_starts_transfer(self):
        self.run_transfer_case(failure="invalid_pose")

    def test_default_planning_failure_keeps_grasp_without_motion(self):
        self.run_transfer_case(failure="planning")

    def test_preview_pause_before_execution_never_starts_transfer(self):
        self.run_transfer_case(failure="pause")

    def test_m_in_paused_state_reports_reason_from_both_inputs_without_resuming(self):
        for source in ("terminal", "preview"):
            with self.subTest(source=source):
                self.run_transfer_case(failure="pause", paused_m_source=source)

    def test_m_from_terminal_and_preview_plans_moves_and_verifies(self):
        for source in ("terminal", "preview"):
            with self.subTest(source=source):
                self.run_transfer_case(override=source)

    def test_m_does_not_bypass_planning_failure_or_operator_pause(self):
        for failure in ("planning", "pause"):
            with self.subTest(failure=failure):
                self.run_transfer_case(override="preview", failure=failure)

    def test_default_manual_mode_waits_for_m_even_when_estimates_are_stable(self):
        self.assertEqual(load_config(HERE/"receive.yaml")["move_above"]["start_mode"], "manual")
        for source in ("terminal", "preview"):
            with self.subTest(source=source):
                self.run_transfer_case(override=source, start_mode="manual", noisy=False, no_frames_until_move=True)

    def test_saved_mean_can_start_motion_after_camera_disconnect_but_cannot_fake_verification(self):
        self.run_transfer_case(override="terminal", start_mode="manual", failure="camera_after_mean")

    def test_startup_trajectory_uses_existing_cartesian_owner_and_keeps_hand(self):
        hw, _, hand, executor = test_receive.HardwareTests().hardware()
        hw.arm_enabled = True
        hw.jogger = KeyboardJog(hw.config, np.eye(4))
        path = JointPath([np.zeros(7), np.full(7, .01)], .1, .2)
        hw.follow_path(path)
        executor.tracking_status = {"right": {"max_joint_tracking_error_rad": 0.}}
        def submit(target, **kwargs):
            hw.stop.set()
            return Submission(target.command_id, True)
        executor.submit.side_effect = submit
        hw._arm_loop()
        self.assertEqual(set(executor.submit.call_args.args[0].tool_poses), {"right"})
        hand.submit.assert_not_called()
        hand.disable.assert_not_called()

    def test_already_held_start_does_not_open_or_close_hand(self):
        c = load_config(HERE/"receive.yaml")
        hw = Mock(paused=False, hand_target=np.full(20, .4))
        hw.snapshot.return_value = {"hand_position_rad": np.full(20, .4), "T_B_F": np.eye(4),
                                    "timestamp_ns": time.monotonic_ns()}
        transfer = Mock(status="idle", preview_poses={}, pose_sampling=None, cached_grasp=None)
        transfer.begin.side_effect = lambda: setattr(transfer, "status", "estimating_held_pose")
        transfer.advance.side_effect = lambda *args, **kwargs: transfer.status
        def keyboard(terminal, hardware, stage, commands, end):
            while not transfer.begin.called:
                time.sleep(.005)
            end.set()
        terminal = Mock(); terminal.__enter__ = Mock(); terminal.__exit__ = Mock()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
                patch.object(receive, "preflight", return_value={"ready": True}), \
                patch.object(receive, "RightHardware", return_value=hw), \
                patch.object(receive, "ReceiveGeometry", return_value=NS(calibration={"camera_serial": "test"},
                    T_B_C=np.eye(4), T_F_G=np.eye(4), mesh_digest="test")), \
                patch.object(receive, "ReceivePerception"), patch.object(receive, "keyboard", keyboard), \
                patch.object(receive, "kinematics"), patch("rail_receive.above.AboveTransfer", return_value=transfer), \
                patch("rail_receive.visualization.PreviewWindow"), \
                patch("diffusion_policy.real_world.tianji.terminal.MotionTerminal", return_value=terminal):
            # End immediately after engagement; test startup branch independently of perception.
            hw.engage.side_effect = lambda: time.sleep(.04)
            receive.run(c, NS(motion=True, output=Path(directory)/"run", move_above=True,
                              already_held=True, no_view=True))
            hw.begin_hand.assert_not_called()
            transfer.begin.assert_called_once()


if __name__ == "__main__":
    unittest.main()
