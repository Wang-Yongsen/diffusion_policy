"""Decision trace buffering, command association and offline summaries, without hardware."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from diffusion_policy.common.paths import PROJECT_ROOT
from diffusion_policy.diagnostics.trace import (
    alignment_series,
    assess_trace,
    compare_action_pair,
    inference_motion_mm,
    load_trace,
    replay_follow_mm,
    request_to_command_mm,
    write_alignment_plot,
)
from diffusion_policy.real_world.tianji.env import TianjiRailEnv
from diffusion_policy.real_world.tianji.trace import (
    SessionTrace,
    file_sha256,
    finite_joints,
    snapshot_arm_feedback,
    write_provenance,
)
from diffusion_policy.third_party.bimanual_teleop.types import Submission
from tests.rail_devices import config


def observation(fill=0):
    obs = {f"camera_{index}": np.full((2, 4, 5, 3), fill, np.uint8) for index in range(3)}
    obs["robot_eef_pose"] = np.arange(24, dtype=np.float32).reshape(2, 12)
    obs["hand_joint"] = np.ones((2, 40), np.float32)
    return obs


def trace():
    return SessionTrace(
        decision_capacity=2,
        feedback_capacity=4,
        command_capacity=4,
        hand_capacity=4,
        status_capacity=4,
    )


def arm_sample(packet, target):
    def arm():
        return SimpleNamespace(
            joints=SimpleNamespace(position_rad=(0.2,) * 7),
            controller_target_rad=tuple(target),
            source_sequence=packet,
            input_sequence=packet + 10,
        )

    payload = SimpleNamespace(packet_index=packet, arms={"left": arm(), "right": arm()})
    header = SimpleNamespace(
        received_monotonic_ns=1_000_000_000 + packet,
        ref=SimpleNamespace(sequence=packet, stream="tianji.feedback"),
        source_sequence=packet,
    )
    return SimpleNamespace(payload=payload, header=header)


class Pose:
    def __init__(self, xyz):
        self.position_m = tuple(xyz)
        self.orientation_xyzw = (0.0, 0.0, 0.0, 1.0)


class DecisionTraceTests(unittest.TestCase):
    def test_missing_joint_is_not_zero(self):
        self.assertIsNone(finite_joints((None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 7))
        self.assertFalse(snapshot_arm_feedback(None)["present"])
        snap = snapshot_arm_feedback(arm_sample(4, (None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        self.assertIsNone(snap["arms"]["left"]["controller_target"])
        self.assertIsNotNone(snap["arms"]["left"]["q"])

    def test_overflow_keeps_the_first_decision_and_marks_the_trace_invalid(self):
        item = SessionTrace(
            decision_capacity=1,
            feedback_capacity=2,
            command_capacity=2,
            hand_capacity=2,
            status_capacity=2,
        )
        item.allocate_decisions(observation(), 2)
        stored = item.store_decision(
            obs=observation(3),
            alignment={"query_ns": [1]},
            anchor_ns=10,
            actions=np.zeros((2, 52)),
            started_ns=1,
            observed_ns=2,
            preprocessed_ns=3,
            inputs_ready_ns=4,
            returned_ns=5,
        )
        rejected = item.store_decision(
            obs=observation(9),
            alignment={},
            anchor_ns=20,
            actions=np.ones((2, 52)),
            started_ns=6,
            observed_ns=7,
            preprocessed_ns=8,
            inputs_ready_ns=9,
            returned_ns=10,
        )
        self.assertEqual(stored, 0)
        self.assertIsNone(rejected)
        self.assertFalse(item.valid)
        self.assertEqual(item.dropped["decisions"], 1)
        np.testing.assert_array_equal(item.images["camera_0"][0], observation(3)["camera_0"])

    def test_sink_failure_is_accepted_and_invalidates_the_trace(self):
        item = trace()
        item.append_sample = lambda sample: (_ for _ in ()).throw(RuntimeError("boom"))
        self.assertTrue(item.sink.try_publish(object()))
        self.assertTrue(
            item.sink.try_event(SimpleNamespace(kind="other", observed_monotonic_ns=1, details={}))
        )
        self.assertFalse(item.valid)
        self.assertIn("boom", item.invalid_reason)

    def test_submit_records_the_feedback_read_inside_submit(self):
        item = trace()
        executor = SimpleNamespace(
            samples=[arm_sample(7, (0.3,) * 7)],
            applied_poses={},
            tracking_status={},
            timing_status={},
        )
        executor.driver = SimpleNamespace(get_latest=lambda: executor.samples[-1])

        def submit(target):
            sample = executor.driver.get_latest()
            executor.used = sample.payload.packet_index
            pose = Pose((0.25, 0.01, 0.02))
            executor.applied_poses = {"left": pose, "right": pose}
            executor.tracking_status = {
                side: {"limited": True, "progress": 0.4, "position_error_m": 0.003}
                for side in ("left", "right")
            }
            executor.timing_status = {"elapsed_ns": 12}
            return Submission(target.command_id, True)

        executor.submit = submit
        item.install_on(executor)
        executor.samples.append(arm_sample(8, (0.3,) * 7))
        executor.submit(SimpleNamespace(command_id="cmd-7"))
        executor.samples.append(arm_sample(99, (0.9,) * 7))
        self.assertEqual(executor.used, 8)
        self.assertEqual(int(item.cmd_feedback_packet[0]), 8)
        self.assertTrue(item.cmd_fk_present[0, 0])
        self.assertAlmostEqual(item.cmd_fk_xyz[0, 0, 0], 0.25)
        self.assertTrue(item.cmd_limited[0, 0])
        self.assertAlmostEqual(item.cmd_position_error_m[0, 0], 0.003)

    def test_rejected_submit_does_not_reuse_the_previous_command_fk(self):
        item = trace()
        executor = SimpleNamespace(
            samples=[arm_sample(1, (0.3,) * 7)],
            applied_poses={"left": Pose((9.0, 9.0, 9.0)), "right": Pose((9.0, 9.0, 9.0))},
            tracking_status={},
            timing_status={},
        )
        executor.driver = SimpleNamespace(get_latest=lambda: executor.samples[-1])
        executor.submit = lambda target: Submission(target.command_id, False, "rejected")
        item.install_on(executor)
        executor.submit(SimpleNamespace(command_id="cmd-reject"))
        self.assertFalse(item.cmd_accepted[0])
        self.assertFalse(item.cmd_fk_present[0].any())
        self.assertTrue(np.isnan(item.cmd_fk_xyz[0]).all())

    def test_missing_controller_target_stays_missing_after_flush(self):
        item = trace()
        item.append_sample(arm_sample(5, (None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        header = SimpleNamespace(
            received_monotonic_ns=5,
            source_sequence=0,
            ref=SimpleNamespace(stream="wuji_left_hand/joints", sequence=1),
        )
        joints = (None,) + (0.2,) * 19
        item.append_sample(
            SimpleNamespace(
                payload=SimpleNamespace(position_rad=joints, joint_names=("j",) * 20), header=header
            )
        )
        with tempfile.TemporaryDirectory() as root:
            item.flush(root)
            loaded = load_trace(root)
        self.assertFalse(loaded["feedback"]["controller_present"][0].any())
        self.assertTrue(np.isnan(loaded["feedback"]["controller_target"][0]).all())
        self.assertFalse(loaded["hands"]["feedback_present"][0])
        self.assertTrue(np.isnan(loaded["hands"]["feedback_q"][0]).all())
        self.assertEqual(int(loaded["hands"]["feedback_source_sequence"][0]), 0)

    def test_second_flush_does_not_keep_the_first_command(self):
        item = trace()
        command = SimpleNamespace(
            command_id="cmd-1",
            payload=SimpleNamespace(
                targets={"left": (0.2,) * 7, "right": (0.3,) * 7},
                cartesian_targets={"left": Pose((0.1, 0.0, 0.0)), "right": Pose((0.0, 0.1, 0.0))},
                requested_cartesian_targets={
                    "left": Pose((0.11, 0.0, 0.0)),
                    "right": Pose((0.0, 0.11, 0.0)),
                },
            ),
        )
        item.record_driver_command(
            SimpleNamespace(
                kind="tianji_command_submitted",
                observed_monotonic_ns=20,
                details={"command": command},
            )
        )
        with tempfile.TemporaryDirectory() as root:
            first = Path(root) / "first"
            second = Path(root) / "second"
            self.assertEqual(item.flush(first)["counts"]["commands"], 1)
            self.assertEqual(item.flush(second)["counts"]["commands"], 0)
            self.assertEqual(len(load_trace(second)["commands"]["command_id"]), 0)
            loaded = load_trace(first)
        self.assertAlmostEqual(loaded["commands"]["requested_xyz"][0, 0, 0], 0.11)
        gaps = request_to_command_mm(loaded["commands"])
        self.assertAlmostEqual(gaps["left"]["max"], 10.0, places=5)

    def test_summaries_leave_absent_measurements_missing(self):
        demo = np.zeros((2, 2, 12))
        real = np.zeros((2, 2, 12))
        real[0, -1, 0] = 0.02
        follow = replay_follow_mm(demo, real, np.array([True, False]))
        self.assertAlmostEqual(follow[0], 20.0)
        self.assertTrue(np.isnan(follow[1]))
        stamps = np.array([10, 20, 30, 40])
        points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]])
        motion = inference_motion_mm(stamps, points, np.array([10, 30]), np.array([20, 40]))
        np.testing.assert_allclose(motion, [10.0, 10.0])
        self.assertIsNone(
            assess_trace({"valid": False, "invalid_reason": "full", "dropped": {}}, {})[
                "root_cause"
            ]
        )
        demo_actions = np.zeros((2, 4, 52))
        demo_actions[:, -1, 2] = 0.03
        real_actions = demo_actions.copy()
        real_actions[:, -1, 2] = 0.01
        summary = compare_action_pair(demo_actions, real_actions)
        self.assertAlmostEqual(summary["left_position_mm"]["max"], 20.0, places=4)
        self.assertAlmostEqual(summary["left_position_mm"]["mean"], 5.0, places=4)
        self.assertEqual(summary["left_direction_cosine"], [1.0, 1.0])

    def test_alignment_uses_command_fk_and_not_the_servo_residual(self):
        commands = {name: np.zeros((1, 2, 3)) for name in ("requested_xyz", "command_fk_xyz")}
        commands["requested_xyz"][0, 0] = (0.1, 0.0, 0.0)
        commands["command_fk_xyz"][0, 0] = (0.12, 0.0, 0.0)
        commands["requested_present"] = np.ones((1, 2), bool)
        commands["command_fk_present"] = np.ones((1, 2), bool)
        commands["started_ns"] = np.array([1_000_000_000])
        commands["sink_received_ns"] = np.array([-1])
        commands["position_error_m"] = np.array([[0.5, 0.5]])
        feedback = {
            "received_ns": np.array([1_000_000_000]),
            "q": np.zeros((1, 2, 7)),
            "q_present": np.ones((1, 2), bool),
            "controller_target": np.full((1, 2, 7), np.nan),
            "controller_present": np.zeros((1, 2), bool),
        }
        series = alignment_series(commands, feedback, fk=lambda side, joints: joints[:3])
        self.assertAlmostEqual(series["command_fk"]["xyz_mm"][0, 0], 120.0)
        self.assertEqual(len(series["controller_fk"]["time_s"]), 0)
        hands = {
            side: {
                "time_s": np.array([0.0]),
                "error_deg": np.array([1.0 if side == "left" else np.nan]),
            }
            for side in ("left", "right")
        }
        with tempfile.TemporaryDirectory() as root:
            write_alignment_plot(root, series, hands, np.array([1.5, np.nan]))
            self.assertTrue((Path(root) / "alignment_left.png").is_file())
            self.assertTrue((Path(root) / "inference_motion.png").is_file())

    def test_provenance_hashes_the_imported_file(self):
        with tempfile.TemporaryDirectory() as root:
            files = {"decision_trace": Path(file_sha256.__code__.co_filename)}
            payload = write_provenance(root, files, PROJECT_ROOT)
            self.assertEqual(len(payload["commit"]), 40)
            self.assertEqual(
                payload["files"]["decision_trace"]["sha256"], file_sha256(files["decision_trace"])
            )
            self.assertTrue((Path(root) / "source.diff").is_file())

    def test_env_start_passes_the_trace_sink(self):
        module = "diffusion_policy.real_world.tianji.env"
        item = trace()
        with (
            patch(f"{module}.TianjiKinematics") as kine_type,
            patch(f"{module}.TianjiDriver") as arm_type,
            patch(f"{module}.WujiSdkSession"),
            patch(f"{module}.WujiHandDriver") as hand_type,
            patch(f"{module}.CameraRig") as camera_type,
        ):
            kine_type.return_value.model.digest = config()["expected_model_sha256"]
            arm_type.return_value.get_latest.return_value = None
            hand_type.return_value.get_latest.return_value = None
            camera_type.return_value.poll.return_value = []
            env = TianjiRailEnv(config(), 2)
            env.enable_trace(item)
            try:
                env.start()
            finally:
                env.close()
            env.driver.start.assert_called_once_with(item.sink)
            self.assertTrue(
                all(call.args == (item.sink,) for call in env.hands["left"].start.call_args_list)
            )
            self.assertGreaterEqual(env.hands["left"].start.call_count, 1)


if __name__ == "__main__":
    unittest.main()
