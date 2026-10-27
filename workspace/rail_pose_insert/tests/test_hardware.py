"""Core offline regression coverage; all actuator and camera owners are mocked."""

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
import support as test_above
from support import ToyKinematics
from rail_common.core import (
    matrix,
    pose,
    read_json,
)
from rail_common.planner import JointPath
from rail_receive.logic import load_config, hand_targets, KeyboardJog
from rail_receive.hardware import RightHardware, initial_receiving_pose
from rail_receive.position_spiral import (
    PositionSpiral,
)
from rail_receive.release_return import (
    ReleaseReturn,
)
from diffusion_policy.third_party.bimanual_teleop.types import Pose, Submission


def config():
    return load_config(HERE / "receive.yaml")


class HardwareTests(unittest.TestCase):

    def hardware(self, motion=True):
        c = config()
        driver, hand, executor = (Mock(), Mock(), Mock())
        driver.health.return_value = hand.health.return_value = NS(ready=True)
        hand.device_id = "wuji_right_hand"
        hand.last_target = NS(position_rad=(0.2,) * 20)
        hand.submit.side_effect = lambda command: Submission(command.command_id, True)
        executor.engagement_poses = {
            "right": Pose("tianji_right_base", "tianji_right_flange", (0, 0, 0), (0, 0, 0, 1))
        }
        executor.applied_poses = dict(executor.engagement_poses)
        k = NS(
            fk=lambda side, q: Pose(
                f"tianji_{side}_base", f"tianji_{side}_flange", (0, 0, 0), (0, 0, 0, 1)
            )
        )
        hw = RightHardware(
            c, k, Mock(), motion=motion, devices=(Mock(), driver, hand), executor=executor
        )
        hw.snapshot = lambda: {
            "T_B_F": np.eye(4),
            "joints_rad": np.zeros(7),
            "hand_position_rad": np.full(20, 0.2),
        }
        return (hw, driver, hand, executor)

    def test_read_only_cannot_configure_or_enable_either_actuator(self):
        hw, driver, hand, executor = self.hardware(motion=False)
        with self.assertRaises(RuntimeError):
            hw.engage()
        executor.configure.assert_not_called()
        driver.engage.assert_not_called()
        hand.engage.assert_not_called()

    def test_hand_command_wait_starts_only_after_successful_final_target_submission(self):
        for case in ("early", "final", "rejected", "replaced"):
            with self.subTest(case=case):
                hw, _, hand, _ = self.hardware()
                hw.hand_enabled, hw.hand_target = (True, np.full(20, 0.2))
                with patch("rail_receive.hardware.time.monotonic", return_value=10.0):
                    hw.begin_hand(np.full(20, 0.3))

                def submit(command):
                    if case == "replaced":
                        hw.begin_hand(np.full(20, 0.4))
                    hw.stop.set()
                    return Submission(command.command_id, case != "rejected", "test rejection")

                hand.submit.side_effect = submit
                with patch(
                    "rail_receive.hardware.time.monotonic",
                    return_value=12.0 if case == "early" else 13.0,
                ):
                    hw._hand_loop()
                hw.stop.clear()
                if case == "final":
                    self.assertFalse(hw.hand_command_finished(13.49))
                    self.assertTrue(hw.hand_command_finished(13.5))
                    np.testing.assert_allclose(
                        hand.submit.call_args.args[0].payload.position_rad, 0.3
                    )
                    hw.begin_hand(np.full(20, 0.4))
                    self.assertIsNone(hw.hand_command_completed_at)
                elif case == "rejected":
                    self.assertIsNone(hw.hand_command_completed_at)
                    with self.assertRaises(RuntimeError):
                        hw.hand_command_finished(13.5)
                else:
                    self.assertIsNone(hw.hand_command_completed_at)
                    self.assertFalse(hw.hand_command_finished(13.5))
                    with self.assertRaisesRegex(TimeoutError, "final target was not submitted"):
                        hw.hand_command_finished(30.0)

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


class PositionMoveTests(unittest.TestCase):

    def hardware(self):
        hw, driver, hand, executor = HardwareTests().hardware()
        hw.kinematics = ToyKinematics()
        hw.arm_enabled = hw.hand_enabled = True
        hw.jogger = KeyboardJog(hw.config, np.eye(4))
        hw.hand_target = np.full(20, 0.2)
        state = {"joints_rad": np.zeros(7), "T_B_F": np.eye(4), "arm_control_state": 3}
        hw.snapshot = lambda: dict(state)

        def move(side, target, *, cancel, timeout_s):
            self.assertEqual(side, "right")
            self.assertFalse(cancel.is_set())
            self.assertEqual(timeout_s, 120.0)
            state.update(
                joints_rad=np.array(target),
                T_B_F=matrix(hw.kinematics.fk(side, target)),
                arm_control_state=1,
            )

        driver.move_joints.side_effect = move
        return (hw, driver, hand, executor, state)

    def start(self, hw, *, hand=False):
        for function in [hw._arm_loop, hw._hand_loop] if hand else [hw._arm_loop]:
            thread = threading.Thread(target=function, daemon=True)
            hw.threads.append(thread)
            thread.start()

    def test_only_one_right_position_target_and_no_cartesian_resubmission(self):
        hw, driver, hand, executor, state = self.hardware()
        goal = np.array([0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.0])
        hw.move_to_position(np.zeros(7), goal, timeout_s=120.0)
        driver.move_joints.assert_not_called()
        self.assertFalse(hw.position_move_finished())
        self.start(hw)
        try:
            self.assertTrue(hw.position_done.wait(1.0))
            self.assertTrue(hw.position_move_finished())
            driver.move_joints.assert_called_once_with(
                "right", tuple(goal), cancel=hw.position_cancel, timeout_s=120.0
            )
            executor.request_hold.assert_called_once()
            for _ in range(3):
                hw._position_step()
            executor.submit.assert_not_called()
            executor.engage.assert_not_called()
            np.testing.assert_allclose(state["joints_rad"], goal)
            hand.disable.assert_not_called()
            self.assertEqual(hw.arm_mode, "position")
        finally:
            hw.close()

    def test_pause_cancels_blocking_position_move_and_hand_keeps_holding(self):
        hw, driver, hand, executor, _ = self.hardware()
        entered, cancelled, held_after_pause = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def move(side, target, *, cancel, timeout_s):
            entered.set()
            if not cancel.wait(1.0):
                raise RuntimeError("test cancellation deadline")
            cancelled.set()
            raise RuntimeError("SDK position move cancelled")

        driver.move_joints.side_effect = move
        original_submit = hand.submit.side_effect

        def submit(command):
            if hw.paused:
                held_after_pause.set()
            return original_submit(command)

        hand.submit.side_effect = submit
        hw.move_to_position(np.zeros(7), np.full(7, 0.1), timeout_s=120.0)
        self.start(hw, hand=True)
        try:
            self.assertTrue(entered.wait(1.0))
            self.assertFalse(hw.position_done.is_set())
            with self.assertRaisesRegex(RuntimeError, "already active"):
                hw.move_to_position(np.zeros(7), np.full(7, 0.2), timeout_s=120.0)
            hw.pause("Operator pause")
            self.assertTrue(cancelled.wait(1.0))
            self.assertTrue(held_after_pause.wait(1.0))
            self.assertEqual(hw.error, "Operator pause")
            self.assertFalse(hw.position_done.is_set())
            executor.submit.assert_not_called()
            hand.disable.assert_not_called()
            np.testing.assert_allclose(hand.submit.call_args.args[0].payload.position_rad, 0.2)
            with self.assertRaisesRegex(RuntimeError, "Operator pause"):
                hw.move_to_position(np.zeros(7), np.full(7, 0.2), timeout_s=120.0)
        finally:
            hw.close()

    def test_limits_changed_start_and_read_only_reject_before_any_mode_switch(self):
        for case in ("limits", "moved", "read_only", "paused"):
            with self.subTest(case=case):
                hw, driver, _, executor, state = self.hardware()
                target = np.full(7, 0.1)
                if case == "limits":
                    target[5:7] = [1.0, 1.0]
                elif case == "moved":
                    state["joints_rad"] = np.full(7, 0.04)
                elif case == "read_only":
                    hw.motion = False
                elif case == "paused":
                    hw.paused = True
                with self.assertRaises((RuntimeError, ValueError)):
                    hw.move_to_position(np.zeros(7), target, timeout_s=120.0)
                driver.move_joints.assert_not_called()
                executor.request_hold.assert_not_called()

    def test_sdk_failure_pauses_and_does_not_claim_completion(self):
        hw, driver, _, executor, _ = self.hardware()
        paused = threading.Event()
        hw.emit.side_effect = lambda event, **kw: paused.set() if event == "paused" else None
        driver.move_joints.side_effect = RuntimeError("position mode was not reported")
        hw.move_to_position(np.zeros(7), np.full(7, 0.1), timeout_s=120.0)
        self.start(hw)
        try:
            self.assertTrue(paused.wait(1.0))
            self.assertFalse(hw.position_done.is_set())
            self.assertIn("position mode was not reported", hw.error)
            executor.submit.assert_not_called()
        finally:
            hw.close()


class InitialOffsetTests(unittest.TestCase):

    def hardware(self):
        config = load_config(HERE / "receive.yaml")
        config["move_above"].update(settle_s=0.001, reach_timeout_s=0.1)
        driver, hand, executor = (Mock(), Mock(), Mock())
        hand.last_target.position_rad = (0.2,) * 20
        hw = RightHardware(
            config,
            ToyKinematics(),
            Mock(),
            motion=True,
            devices=(Mock(), driver, hand),
            executor=executor,
        )
        q = np.array([0.2, -0.02, 0.5, 0.0, 0.4, 0.0, 0.0])
        state = {
            "joints_rad": q,
            "T_B_F": matrix(hw.kinematics.fk("right", q)),
            "arm_control_state": 3,
            "arm_low_speed": 0,
        }
        hw.snapshot = lambda: state

        def move(side, target, *, cancel, timeout_s):
            self.assertEqual(side, "right")
            self.assertIs(cancel, hw.position_cancel)
            self.assertFalse(cancel.is_set())
            self.assertEqual(timeout_s, config["move_above"]["position_move_timeout_s"])
            q = np.asarray(target)
            state.update(
                joints_rad=q.copy(),
                T_B_F=matrix(hw.kinematics.fk(side, q)),
                arm_control_state=1,
                arm_low_speed=1,
            )

        driver.move_joints.side_effect = move

        def engage(**kwargs):
            self.assertIs(kwargs["cancel"], hw.position_cancel)
            executor.engagement_poses = {"right": hw.kinematics.fk("right", state["joints_rad"])}
            state["arm_control_state"] = 3

        executor.engage.side_effect = engage
        return (hw, state)

    def test_position_return_sends_one_fixed_endpoint_before_any_impedance_writer(self):
        hw, state = self.hardware()
        reference = matrix(
            hw.kinematics.fk("right", np.radians(hw.config["initial_reference_joints_deg"]))
        )
        with patch.object(
            JointPath, "at", side_effect=AssertionError("must not stream a host path")
        ):
            hw.move_to_initial_pose()
        actual = state["T_B_F"]
        np.testing.assert_allclose(actual[:3, 3] - reference[:3, 3], [0.0, 0.0, -0.15], atol=1e-10)
        np.testing.assert_allclose(actual[:3, :3], reference[:3, :3], atol=1e-10)
        hw.driver.move_joints.assert_called_once()
        hw.driver.configure.assert_called_once_with(hw.arm_profile, cancel=hw.position_cancel)
        profile = hw.driver.configure.call_args.args[0]
        self.assertEqual(profile.parameters["active_arms"], ["right"])
        self.assertEqual(profile.parameters["arms"]["right"]["velocity_ratio"], 3)
        self.assertEqual(profile.parameters["arms"]["right"]["acceleration_ratio"], 3)
        self.assertFalse(hw.initial_move_active)
        self.assertEqual(state["arm_control_state"], 1)
        self.assertFalse(hw.arm_enabled)
        self.assertEqual(hw.threads, [])
        hw.executor.configure.assert_not_called()
        hw.executor.engage.assert_not_called()
        hw.executor.submit.assert_not_called()
        hw.hand.engage.assert_not_called()

    def test_sdk_completion_still_requires_measured_flange_arrival(self):
        hw, state = self.hardware()
        target = initial_receiving_pose(hw.config, hw.kinematics)
        state.update(T_B_F=target.copy(), arm_control_state=1, arm_low_speed=1)
        state["T_B_F"][0, 3] += 0.00827
        hw.driver.move_joints.side_effect = None
        with patch("rail_receive.hardware.time.monotonic", side_effect=[0.0, 1000.0]):
            with self.assertRaisesRegex(TimeoutError, "fixed initial receiving pose"):
                hw.move_to_initial_pose()
        self.assertFalse(hw.initial_move_active)
        hw.driver.request_hold.assert_called_once()
        hw.executor.engage.assert_not_called()
        self.assertNotIn(
            "right_initial_pose_complete", [call.args[0] for call in hw.emit.call_args_list]
        )

    def test_pause_cancels_blocking_startup_move_without_enabling_hand_or_impedance(self):
        hw, _ = self.hardware()
        entered, errors = (threading.Event(), [])

        def move(side, target, *, cancel, timeout_s):
            entered.set()
            if not cancel.wait(1.0):
                raise RuntimeError("test cancellation deadline")
            raise RuntimeError("startup move cancelled")

        hw.driver.move_joints.side_effect = move

        def run():
            try:
                hw.move_to_initial_pose()
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(1.0))
            hw.pause("operator pause")
        finally:
            hw.position_cancel.set()
            thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(str(errors[0]), "startup move cancelled")
        self.assertFalse(hw.initial_move_active)
        hw.driver.request_hold.assert_called()
        hw.executor.engage.assert_not_called()
        hw.hand.engage.assert_not_called()


class BothArmKinematics:

    def __init__(self):
        arm = NS(
            lower_rad=(-4.0,) * 7,
            upper_rad=(4.0,) * 7,
            max_joint_velocity_rad_s=(1.0,) * 7,
            limits_native=[(0, 0, 0, 60)] * 7,
        )
        self.model = NS(arm=lambda side: arm)

    def fk(self, side, q):
        return Pose(
            f"tianji_{side}_base",
            f"tianji_{side}_flange",
            tuple(q[:3]),
            tuple(Rotation.from_rotvec(q[3:6]).as_quat()),
        )

    def ik(self, side, target, seed):
        assert side == "left" and target.parent_frame == "tianji_left_base"
        return (*target.position_m, *Rotation.from_quat(target.orientation_xyzw).as_rotvec(), 0.0)


class LeftInitialPoseTests(unittest.TestCase):

    def hardware(self):
        config = load_config(HERE / "receive.yaml")
        config["move_above"]["settle_s"] = 0.001
        driver = Mock()
        driver.health.return_value = NS(ready=True)
        joints = np.radians([20, -50, -30, -40, 20, 10, 2])

        def sample():
            return NS(
                header=NS(valid=True, received_monotonic_ns=time.monotonic_ns()),
                payload=NS(arms={"left": NS(joints=NS(position_rad=joints.copy()))}),
            )

        def move(side, target, **kwargs):
            self.assertEqual(side, "left")
            joints[:] = target
            return sample()

        driver.get_latest.side_effect, driver.move_joints.side_effect = (sample, move)
        hw = RightHardware(
            config,
            BothArmKinematics(),
            Mock(),
            motion=True,
            devices=(Mock(), driver, Mock()),
            executor=Mock(),
        )
        return (hw, joints)

    def test_left_returns_directly_to_one_fixed_target_without_accumulating(self):
        hw, joints = self.hardware()
        expected = initial_receiving_pose(hw.config, hw.kinematics, side="left")
        for i in range(3):
            if i == 2:
                joints[:] = [0.3, -0.4, 0.2, 0.1, 0.2, 0.1, 0.0]
            hw.move_left_to_initial_pose()
            self.assertEqual(hw.driver.move_joints.call_count, i + 1)
            command = hw.driver.move_joints.call_args
            self.assertEqual(command.args[0], "left")
            np.testing.assert_allclose(
                matrix(hw.kinematics.fk("left", command.args[1])), expected, atol=1e-12
            )
            self.assertIs(command.kwargs["cancel"], hw.position_cancel)
        self.assertFalse(hw.left_initial_move_active)
        profile = hw.driver.configure.call_args.args[0]
        self.assertEqual(profile.parameters["active_arms"], ["left"])
        self.assertEqual(set(profile.parameters["arms"]), {"left"})
        self.assertEqual(profile.parameters["arms"]["left"]["velocity_ratio"], 3)
        self.assertEqual(profile.parameters["arms"]["left"]["acceleration_ratio"], 3)
        hw.executor.configure.assert_not_called()

    def test_already_held_skips_both_returns(self):
        config = load_config(HERE / "receive.yaml")
        hw = Mock(paused=False)
        receive.start_with_preview(
            hw, Mock(), Path("unused.png"), config, threading.Event(), move_initial=False
        )
        hw.move_left_to_initial_pose.assert_not_called()
        hw.move_to_initial_pose.assert_not_called()

    def test_failed_left_return_prevents_right_engagement(self):
        config = load_config(HERE / "receive.yaml")
        hw = Mock(paused=False)
        hw.move_left_to_initial_pose.side_effect = RuntimeError("left failed")
        with self.assertRaisesRegex(RuntimeError, "left failed"):
            receive.start_with_preview(hw, Mock(), Path("unused.png"), config, threading.Event())
        hw.engage.assert_not_called()
        hw.move_to_initial_pose.assert_not_called()


class PositionSpiralTests(unittest.TestCase):

    def geometry(self, start_offset=None):
        config, kinematics = (load_config(HERE / "receive.yaml"), test_above.ToyKinematics())
        grasp = test_above.T([0.02, -0.2, 0.04], [0.1, -0.2, 0.3])
        goal = test_above.T([0.12, 0.05, 0.6], [0.2, 0.1, -0.1])
        above = goal.copy()
        above[1, 3] += 0.05
        actual = above.copy()
        if start_offset is not None:
            actual[:3, 3] += start_offset
        joints = np.asarray(
            kinematics.ik("right", pose(actual @ np.linalg.inv(grasp), "right"), np.zeros(7))
        )
        return (config, kinematics, joints, grasp, goal, above)

    def fixture(self, directory, start_offset=None):
        config, kinematics, joints, grasp, goal, above = self.geometry(start_offset)
        insertion = PositionSpiral(config, kinematics, directory, joints, grasp, goal, above)
        self.assertTrue(insertion.done.wait(3.0))
        self.assertIsNone(insertion.error)
        state = dict(joints_rad=joints.copy(), T_B_F=matrix(kinematics.fk("right", joints)))
        hardware = Mock()
        hardware.position_move_finished.return_value = False
        return (insertion, hardware, state)

    def advance(self, insertion, hardware, state, now):
        with patch(
            "rail_receive.position_spiral.time.monotonic", return_value=now
        ), redirect_stdout(io.StringIO()):
            return insertion.advance(hardware, state)

    def reach(self, insertion, state):
        state.update(
            joints_rad=np.asarray(insertion.report["joint_waypoints_rad"][insertion.index + 1]),
            T_B_F=np.asarray(insertion.report["flange_targets"][insertion.index]),
        )

    def test_sdk_measured_pose_and_continuous_settle_are_all_required(self):
        with tempfile.TemporaryDirectory() as directory:
            insertion, hw, state = self.fixture(directory, [0.002, 0.0, 0.0])
            now = insertion.started
            self.advance(insertion, hw, state, now)
            hw.move_to_position.assert_called_once()
            hw.position_move_finished.return_value = True
            self.advance(insertion, hw, state, now + 0.2)
            hw.move_to_position.assert_called_once()
            self.reach(insertion, state)
            hw.position_move_finished.return_value = False
            self.advance(insertion, hw, state, now + 0.3)
            hw.position_move_finished.return_value = True
            self.advance(insertion, hw, state, now + 0.4)
            self.advance(insertion, hw, state, now + 0.45)
            hw.move_to_position.assert_called_once()
            state["T_B_F"][0, 3] += 0.0003
            self.advance(insertion, hw, state, now + 0.46)
            self.reach(insertion, state)
            self.advance(insertion, hw, state, now + 0.5)
            self.advance(insertion, hw, state, now + 0.55)
            hw.move_to_position.assert_called_once()
            self.advance(insertion, hw, state, now + 0.7)
            self.assertEqual(hw.move_to_position.call_count, 2)
            self.assertTrue(insertion.evidence[0]["endpoint_verified"])
            self.assertEqual(hw.move_to_position.call_args.kwargs["velocity_ratio"], 1)
            self.assertEqual(hw.move_to_position.call_args.kwargs["acceleration_ratio"], 1)

    def test_faults_never_dispatch_another_segment_and_paused_summary_is_durable(self):
        causes = (
            "corridor",
            "rotation",
            "overshoot",
            "upward",
            "step_timeout",
            "total_timeout",
            "hardware",
        )
        for cause in causes:
            with self.subTest(cause=cause), tempfile.TemporaryDirectory() as directory:
                insertion, hw, state = self.fixture(directory)
                now = insertion.started
                self.advance(insertion, hw, state, now)
                if cause == "corridor":
                    state["T_B_F"][0, 3] += 0.004
                elif cause == "rotation":
                    state["T_B_F"][:3, :3] = (
                        test_above.T(rpy=[0.1, 0.0, 0.0])[:3, :3] @ state["T_B_F"][:3, :3]
                    )
                elif cause == "overshoot":
                    state["T_B_F"][1, 3] -= 0.002
                elif cause == "upward":
                    state["T_B_F"][1, 3] += 0.0015
                elif cause == "step_timeout":
                    self.reach(insertion, state)
                    hw.position_move_finished.return_value = True
                    now += 21.0
                elif cause == "total_timeout":
                    now += 601.0
                elif cause == "hardware":
                    hw.raise_if_fault.side_effect = RuntimeError("paused")
                with self.assertRaises((RuntimeError, TimeoutError)) as caught:
                    self.advance(insertion, hw, state, now + 0.1)
                result = insertion.paused_summary(state, str(caught.exception))
                self.assertEqual(result["status"], "paused")
                self.assertEqual(result["completed_steps"], 0)
                self.assertEqual(result["queued_steps"], 1)
                self.assertEqual(result["hand_action"], "retain_last_target")
                self.assertEqual(result["previous_status"], "position_spiral")
                self.assertIsNone(result["insertion_executed"])
                self.assertTrue(result["reason"])
                self.advance(insertion, hw, state, now + 1.0)
                hw.move_to_position.assert_called_once()
                self.assertEqual(json.loads((Path(directory) / "result.json").read_text()), result)


class ReleaseReturnTests(unittest.TestCase):

    def fixture(self, directory):
        config = load_config(HERE / "receive.yaml")
        kinematics = test_above.ToyKinematics()
        joints = np.array([0.2, 0.1, 0.5, 0.05, -0.05, 0.03, 0.0])
        state = dict(
            joints_rad=joints,
            T_B_F=matrix(kinematics.fk("right", joints)),
            hand_position_rad=hand_targets(config)["grasp"].copy(),
        )
        insertion = dict(
            status="insertion_reached",
            insertion_executed=True,
            T_R_O_from_fk=np.eye(4).tolist(),
            assembly_verified=False,
        )
        motion = ReleaseReturn(config, kinematics, directory, state, insertion)
        self.assertTrue(motion.done.wait(3))
        hardware = Mock(paused=False)
        hardware.position_move_finished.return_value = True
        hardware.hand_command_finished.return_value = False
        return (motion, hardware, state)

    def begin_lift(self, motion, hardware, state):
        motion.advance(hardware, state)
        self.assertEqual(motion.status, "releasing_hand")
        hardware.begin_hand.assert_called_once()
        hardware.move_to_position.assert_not_called()
        state["hand_position_rad"] = motion.open_target.copy()
        hardware.hand_command_finished.return_value = True
        motion.advance(hardware, state)
        self.assertEqual(motion.status, "lifting_after_release")

    def reach(self, motion, hardware, state):
        index = motion.index + 1
        state.update(
            joints_rad=np.array(motion.report["joint_waypoints_rad"][index]),
            T_B_F=np.array(motion.report["flange_waypoints"][index]),
        )
        motion.advance(hardware, state)
        motion.settled_since = time.monotonic() - 1.0
        motion.advance(hardware, state)

    def test_full_lifecycle_opens_then_lifts_then_returns_once_and_freezes_insertion_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            motion, hw, state = self.fixture(directory)
            self.assertIsNone(motion.error)
            hw.begin_hand.assert_not_called()
            self.begin_lift(motion, hw, state)
            for _ in range(motion.report["lift_steps"]):
                self.assertEqual(motion.status, "lifting_after_release")
                self.reach(motion, hw, state)
            self.assertEqual(motion.status, "returning_initial")
            self.reach(motion, hw, state)
            self.assertEqual(motion.status, "returned_initial")
            commands = hw.move_to_position.call_count
            motion.advance(hw, state)
            self.assertEqual(hw.move_to_position.call_count, commands)
            self.assertEqual(commands, motion.report["lift_steps"] + 1)
            for call in hw.move_to_position.call_args_list:
                self.assertEqual(call.kwargs["velocity_ratio"], 1)
            hw.begin_hand.assert_called_once_with(motion.open_target)
            result = read_json(Path(directory) / "result.json")
            self.assertTrue(result["returned_to_initial"] and result["release_sequence_completed"])
            self.assertFalse(result["hand_open_confirmed"])
            self.assertEqual(result["release_completion_basis"], "open_command_then_wait")
            self.assertTrue(
                result["execution_summary"]["verified_steps"]
                == result["execution_summary"]["attempted_steps"]
            )
            self.assertFalse(result["assembly_verified"])
            self.assertEqual(result["hand_action"], "keep_open")
            np.testing.assert_array_equal(result["insertion"]["T_R_O_from_fk"], np.eye(4))

    def test_unreachable_home_is_detected_before_hand_opening(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "rail_receive.release_return.plan_release_return",
            side_effect=ValueError("home IK failed"),
        ):
            motion, hw, state = self.fixture(directory)
            with self.assertRaisesRegex(ValueError, "home IK failed"):
                motion.advance(hw, state)
            hw.begin_hand.assert_not_called()
            hw.move_to_position.assert_not_called()

    def test_sdk_completion_alone_and_fk_alone_do_not_advance_lift(self):
        with tempfile.TemporaryDirectory() as directory:
            motion, hw, state = self.fixture(directory)
            self.begin_lift(motion, hw, state)
            motion.advance(hw, state)
            self.assertEqual(hw.move_to_position.call_count, 1)
            state.update(
                T_B_F=np.array(motion.report["flange_waypoints"][1]),
                joints_rad=np.array(motion.report["joint_waypoints_rad"][1]),
            )
            hw.position_move_finished.return_value = False
            motion.advance(hw, state)
            self.assertEqual(hw.move_to_position.call_count, 1)
            hw.position_move_finished.return_value = True
            self.reach(motion, hw, state)
            self.assertEqual(hw.move_to_position.call_count, 2)

    def test_real_command_gate_waits_after_submission_and_ignores_measured_pose_error(self):
        for submitted in (False, True):
            with self.subTest(submitted=submitted), tempfile.TemporaryDirectory() as directory:
                motion, hw, state = self.fixture(directory)
                began = time.monotonic()
                hw.config, hw.lock = (motion.config, threading.RLock())
                hw.hand_enabled = True
                hw.hand_blend = (
                    state["hand_position_rad"].copy(),
                    motion.open_target.copy(),
                    began,
                    3.0,
                )
                hw.hand_command_completed_at = began + 3.0 if submitted else None
                hw.hand_command_finished.side_effect = (
                    lambda now: RightHardware.hand_command_finished(hw, now)
                )
                state["hand_current_a"] = np.zeros(20)
                with patch("rail_receive.release_return.time.monotonic", return_value=began):
                    motion.advance(hw, state)
                state["hand_position_rad"] = motion.open_target.copy()
                state["hand_position_rad"][1] += 0.25
                for elapsed in (2.9, 3.0, 3.49):
                    with patch(
                        "rail_receive.release_return.time.monotonic", return_value=began + elapsed
                    ):
                        motion.advance(hw, state)
                    hw.move_to_position.assert_not_called()
                if submitted:
                    with patch(
                        "rail_receive.release_return.time.monotonic", return_value=began + 3.5
                    ):
                        motion.advance(hw, state)
                    self.assertEqual(motion.status, "lifting_after_release")
                    hw.move_to_position.assert_called_once()
                    self.assertTrue(motion.release_sequence_completed)
                    self.reach(motion, hw, state)
                    self.assertEqual(hw.move_to_position.call_count, 2)
                else:
                    with patch(
                        "rail_receive.release_return.time.monotonic", return_value=began + 13.1
                    ):
                        with self.assertRaisesRegex(
                            TimeoutError, "final target was not submitted"
                        ) as caught:
                            motion.advance(hw, state)
                        result = motion.paused_summary(caught.exception)
                    hw.move_to_position.assert_not_called()
                    self.assertFalse(result["release_sequence_completed"])
                    self.assertAlmostEqual(result["hand_feedback"]["target_error_rad"][1], 0.25)
                    np.testing.assert_allclose(result["hand_feedback"]["current_a"], 0.0)
                self.assertIsNotNone(motion.hand_feedback)
                hw.hand_transition_finished.assert_not_called()

    def test_lift_deviation_or_device_fault_stops_before_next_command(self):
        for case in ("lateral", "down", "device_fault"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                motion, hw, state = self.fixture(directory)
                self.begin_lift(motion, hw, state)
                if case == "lateral":
                    state["T_B_F"][0, 3] += 0.01
                elif case == "down":
                    state["T_B_F"][1, 3] -= 0.01
                else:
                    hw.raise_if_fault.side_effect = RuntimeError("hand feedback is stale")
                with self.assertRaises(RuntimeError):
                    motion.advance(hw, state)
                self.assertEqual(hw.move_to_position.call_count, 1)

    def test_pause_and_timeouts_preserve_release_and_insertion_evidence(self):
        for case in ("pause", "step_timeout", "total_timeout"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                motion, hw, state = self.fixture(directory)
                self.begin_lift(motion, hw, state)
                if case == "pause":
                    hw.paused = True
                elif case == "step_timeout":
                    motion.step_started -= 121
                else:
                    motion.started -= 601
                with self.assertRaises((RuntimeError, TimeoutError)):
                    motion.advance(hw, state)
                transfer = Mock(release_return=motion)
                hw.error = case
                result = receive.paused_result(hw, transfer)
                self.assertEqual(result["status"], "paused")
                self.assertTrue(
                    result["release_command_sent"] and result["release_sequence_completed"]
                )
                self.assertFalse(result["hand_open_confirmed"])
                self.assertTrue(result["insertion_executed"])
                self.assertEqual(result["hand_action"], "retain_last_target")
                self.assertEqual(hw.move_to_position.call_count, 1)
                transfer.spiral.paused_summary.assert_not_called()
