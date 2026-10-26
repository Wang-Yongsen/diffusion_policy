"""Sequential synchronous dispatch and direct hand targets, without hardware."""

import threading
import unittest
from unittest.mock import patch

import numpy as np

from diffusion_policy.real_world.tianji_policy_runtime import (
    TianjiPolicyRuntime,
    validate_actions,
)
from tests.test_tianji_rail_real_eval import FakeArm, FakeHand, config


PERIOD_NS = round(1e9 / 30)


class SyncConsumptionTests(unittest.TestCase):
    def runtime(self, mode="sync"):
        self.now, self.events = 1_000_000_000, []
        cfg = config()
        cfg["inference_mode"] = mode
        runtime = TianjiPolicyRuntime(
            FakeArm(),
            {side: FakeHand(side) for side in ("left", "right")},
            cfg,
            lambda: None,
            lambda event, **values: self.events.append((event, values)),
            clock_ns=lambda: self.now,
        )
        runtime.seed(np.zeros(52), arm_ready=True)
        self.addCleanup(runtime.close)
        return runtime

    def actions(self):
        actions = np.zeros((8, 52))
        actions[:, 0] = np.arange(8) * 0.001
        actions[:, 12] = np.arange(1, 9) * 0.1
        actions[:, 32] = -np.arange(1, 9) * 0.1
        return actions

    def tick_all(self, runtime):
        runtime.tick("arm")
        runtime.tick("hand")

    def test_late_ticks_consume_every_step_before_observation_is_requested(self):
        runtime = self.runtime()
        actions = self.actions()
        runtime.submit(actions, 0, prediction_id=9)
        schedule = self.events[-1][1]
        planned_end = int(schedule["times_ns"][-1]) + PERIOD_NS
        self.assertNotIn("next_observation_query_ns", schedule)
        self.now = int(schedule["times_ns"][0]) - 1
        self.tick_all(runtime)
        self.assertTrue(
            all(
                values["mode"] == "hold"
                for event, values in self.events
                if event == "submission"
            )
        )

        # Even the first tick is two policy periods late; later ticks also miss deadlines.
        self.now += 70_000_001
        for _ in actions:
            self.tick_all(runtime)
            self.assertTrue(runtime.active)
            self.assertIsNone(runtime.sync_observation_window())
            self.now += 70_000_000
        self.tick_all(runtime)
        window = runtime.sync_observation_window()
        self.assertNotIn("query_ns", window)
        self.assertEqual(window["completed_ns"], self.now)
        self.assertGreater(self.now, planned_end)
        self.assertEqual(window["source_prediction_id"], 9)
        for kind, side in (("arm", None), ("hand", "left"), ("hand", "right")):
            steps = [
                v["step_id"]
                for e, v in self.events
                if e == "submission"
                and v["mode"] == "policy"
                and v["kind"] == kind
                and v["side"] == side
            ]
            self.assertEqual(steps, list(range(8)))
        np.testing.assert_allclose(runtime.reference_action, actions[-1])
        complete = [v for e, v in self.events if e == "chunk_complete"]
        self.assertEqual(len(complete), 1)
        self.assertEqual(complete[0]["consumed_steps"], 8)
        self.now += 500_000_000
        self.tick_all(runtime)
        self.assertEqual(runtime.sync_observation_window(), window)
        self.assertAlmostEqual(
            runtime.hands["left"].last_target.position_rad[0], actions[-1, 12]
        )
        self.assertAlmostEqual(
            runtime.hands["right"].last_target.position_rad[0], actions[-1, 32]
        )

    def test_current_step_waits_for_every_group_and_its_policy_period(self):
        runtime = self.runtime()
        runtime.submit(self.actions(), 0)
        self.now = int(runtime._times[0])
        runtime.tick("arm")
        runtime.tick("hand", "left")
        self.now += 100_000_000
        runtime.tick("arm")
        runtime.tick("hand", "left")
        steps = [v["step_id"] for e, v in self.events if e == "submission"]
        self.assertEqual(steps, [0, 0, 0, 0])
        runtime.tick("hand", "right")
        runtime.tick("arm")
        self.assertEqual(self.events[-1][1]["step_id"], 1)
        self.tick_all(runtime)
        self.now += PERIOD_NS - 1
        self.tick_all(runtime)
        self.assertEqual(self.events[-1][1]["step_id"], 1)
        self.now += 1
        runtime.tick("arm")
        self.assertEqual(self.events[-1][1]["step_id"], 2)

    def test_delayed_chunk_reads_observation_near_actual_completion(self):
        from tests.test_tianji_fixed_observation import append_window, memory_env

        runtime = self.runtime()
        runtime.submit(self.actions(), 0)
        times = runtime._times.copy()
        planned_end = int(times[-1]) + PERIOD_NS
        self.now = int(times[0])
        for _ in times:
            self.tick_all(runtime)
            self.now += 70_000_000
        self.tick_all(runtime)
        self.assertFalse(runtime.active)
        anchor = self.now - 1_000_000
        for steps in (1, 2):
            with self.subTest(steps=steps):
                env = memory_env(n_obs_steps=steps)
                env.motion, env.runtime = True, runtime
                append_window(env, anchor)
                with patch(
                    "diffusion_policy.real_world.tianji.env.time.monotonic_ns",
                    return_value=self.now,
                ):
                    obs, selected, alignment = env.get_obs()
                self.assertEqual(selected, anchor)
                self.assertEqual(
                    alignment["query_ns"],
                    [anchor] if steps == 1 else [anchor - PERIOD_NS, anchor],
                )
                self.assertTrue(all(len(values) == steps for values in obs.values()))
                self.assertGreater(alignment["query_ns"][0], planned_end)
                self.assertEqual(env.sync_observation_window()["completed_ns"], self.now)

    def test_completion_waits_for_last_hand_submission(self):
        runtime = self.runtime()
        actions = self.actions()
        runtime.submit(actions, 0)
        times = runtime._times.copy()
        for stamp in times[:-1]:
            self.now = int(stamp)
            self.tick_all(runtime)
        self.now = int(times[-1])
        runtime.tick("arm")
        runtime.tick("hand", "left")
        self.now += 100_000_000
        runtime.tick("arm")
        runtime.tick("hand", "left")
        self.assertTrue(runtime.active)
        self.assertIsNone(runtime.sync_observation_window())
        runtime.tick("hand", "right")
        runtime.tick("arm")
        self.assertFalse(runtime.active)
        np.testing.assert_allclose(runtime.reference_action, actions[-1])

    def test_inflight_arm_submission_cannot_be_consumed_by_hand_threads(self):
        runtime = self.runtime()
        runtime.submit(self.actions(), 0)
        entered, release, failures = threading.Event(), threading.Event(), []
        original = runtime.arm.submit

        def submit(target):
            entered.set()
            release.wait(2)
            return original(target)

        def tick_arm():
            try:
                runtime.tick("arm")
            except Exception as error:
                failures.append(error)

        runtime.arm.submit = submit
        self.now = int(runtime._times[0])
        worker = threading.Thread(target=tick_arm)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.now += 60_000_000
            runtime.tick("hand")
            runtime.tick("hand")
            self.assertTrue(all(v["step_id"] == 0 for e, v in self.events if e == "submission"))
            self.assertIsNone(runtime.sync_observation_window())
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        runtime.tick("hand", "left")
        self.assertEqual(self.events[-1][1]["step_id"], 1)

    def test_rejected_hand_does_not_advance_or_update_reference(self):
        runtime = self.runtime()
        actions = self.actions()
        runtime.submit(actions, 0)
        self.now = int(runtime._times[0])
        runtime.tick("arm")
        np.testing.assert_allclose(runtime.reference_action[12:], 0)
        runtime.hands["left"].accepted = False
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            runtime.tick("hand", "left")
        self.assertFalse(self.events[-1][1]["accepted"])
        self.now += 70_000_000
        runtime.tick("arm")
        self.assertEqual(self.events[-1][1]["step_id"], 0)
        np.testing.assert_allclose(runtime.reference_action[12:], 0)

    def test_old_inflight_receipt_cannot_consume_a_restarted_chunk(self):
        runtime = self.runtime()
        runtime.submit(self.actions(), 0)
        entered, release, failures = threading.Event(), threading.Event(), []
        original = runtime.arm.submit

        def submit(target):
            entered.set()
            release.wait(2)
            return original(target)

        def tick_arm():
            try:
                runtime.tick("arm")
            except Exception as error:
                failures.append(error)

        runtime.arm.submit = submit
        self.now = int(runtime._times[0])
        worker = threading.Thread(target=tick_arm)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            # Reseeding invalidates the in-flight command's generation.
            seed = np.zeros(52)
            seed[0] = 0.02
            runtime.seed(seed, arm_ready=True)
            runtime.submit(self.actions(), 0)
            self.now = int(runtime._times[0])
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        np.testing.assert_allclose(runtime.reference_action, seed)
        # Hands cannot advance the new step using the old arm's receipt.
        runtime.tick("hand")
        self.now += 70_000_000
        runtime.tick("hand", "left")
        self.assertEqual(self.events[-1][1]["step_id"], 0)

    def test_pause_and_seed_reset_partial_consumption(self):
        runtime = self.runtime()
        runtime.submit(self.actions(), 0)
        self.now = int(runtime._times[0])
        runtime.tick("hand")
        runtime.pause("operator")
        self.assertIsNone(runtime.sync_observation_window())
        runtime.arm.driver.engaged = True
        runtime.seed(np.zeros(52), arm_ready=True)
        runtime.submit(self.actions(), 0)
        self.now = int(runtime._times[0])
        runtime.tick("arm")
        self.now += 70_000_000
        runtime.tick("arm")
        self.assertEqual(self.events[-1][1]["step_id"], 0)

    def test_hand_speed_checks_and_clipping_are_absent_in_both_modes(self):
        for mode in ("sync", "async"):
            with self.subTest(mode=mode):
                runtime = self.runtime(mode)
                # A legacy config value must not reinstate either speed gate.
                runtime.config["max_hand_speed_rad_s"] = 0.001
                actions = self.actions()
                actions[:, 12] = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
                actions[:, 32] = -actions[:, 12]
                runtime.submit(actions, self.now)
                expected = runtime._actions[0].copy()
                self.now = int(runtime._times[0])
                runtime.tick("hand")
                np.testing.assert_allclose(
                    runtime.hands["left"].last_target.position_rad, expected[12:32]
                )
                np.testing.assert_allclose(
                    runtime.hands["right"].last_target.position_rad, expected[32:52]
                )

    def test_hand_joint_ranges_and_jump_checks_still_apply(self):
        cfg = config()
        actions = np.zeros((2, 52))
        times = np.array([1_000_000_000, 1_033_333_333])
        actions[1, 12] = 0.9  # 27 rad/s passes: hand speed is not constrained.
        validate_actions(actions, times, np.zeros(52), cfg)
        actions[1, 12] = 3
        with self.assertRaisesRegex(ValueError, "joint limits"):
            validate_actions(actions, times, np.zeros(52), cfg)
        cfg["max_hand_jump_rad"] = 0.5
        actions[1, 12] = 0.9
        with self.assertRaisesRegex(ValueError, "jumps too far"):
            validate_actions(actions, times, np.zeros(52), cfg)


if __name__ == "__main__":
    unittest.main()
