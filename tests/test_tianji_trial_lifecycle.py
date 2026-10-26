"""Optional trial duration and operator-controlled endings, without hardware."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from diffusion_policy.env_runner.tianji_rail_real_runner import TianjiRailRealRunner
from diffusion_policy.real_world.tianji_rail_env import PROJECT_ROOT, load_deployment_config
from tests.test_tianji_rail_real_eval import config


RUNNER_MODULE = "diffusion_policy.env_runner.tianji_rail_real_runner"


class TrialLifecycleTests(unittest.TestCase):
    def test_deployment_defaults_to_no_time_limit_and_accepts_explicit_limit(self):
        for filename in ("tianji_rail_real_eval.yaml", "tianji_rail_real_eval_loose.yaml"):
            path = PROJECT_ROOT / "diffusion_policy/config" / filename
            with self.subTest(config=filename):
                self.assertIsNone(load_deployment_config(path)["trial_timeout_s"])
                self.assertIsNone(
                    load_deployment_config(path, ["trial_timeout_s=null"])["trial_timeout_s"]
                )
                self.assertEqual(
                    load_deployment_config(path, ["trial_timeout_s=60"])["trial_timeout_s"], 60
                )
                for value in ("0", "-1", "true", ".inf", ".nan", "invalid"):
                    with self.assertRaisesRegex(ValueError, "trial_timeout_s"):
                        load_deployment_config(path, [f"trial_timeout_s={value}"])
                with self.assertRaisesRegex(ValueError, "max_state_age_s"):
                    load_deployment_config(path, ["max_state_age_s=null"])

    def run_motion_trial(self, command, *, mode="sync", fault=None):
        pauses = []
        cfg = config()
        cfg["inference_mode"] = mode
        env = SimpleNamespace(
            motion=True,
            config=cfg,
            recorder=None,
            error=None,
            runtime=SimpleNamespace(fault=fault),
            engage=lambda: None,
            pause=pauses.append,
        )
        commands = iter([f"{command}\n", "q\n"])
        terminal = SimpleNamespace(isatty=lambda: True, readline=lambda: next(commands))
        with tempfile.TemporaryDirectory() as root:
            runner = TianjiRailRealRunner(
                env, SimpleNamespace(n_action_steps=8), {}, {}, Path(root) / "session"
            )
            runner._begin()
            # Exercise the real operator loop after more than the old 30-second limit.
            runner._started -= 120
            with patch.object(runner, "_infer", side_effect=runner._stop.wait), patch(
                f"{RUNNER_MODULE}.sys.stdin", terminal
            ), patch(
                f"{RUNNER_MODULE}.select.select", return_value=([terminal], [], [])
            ), patch(
                "builtins.print"
            ):
                report = runner.run()
            self.assertEqual(report["motion_trials"], 1)
            return report["trials"][0], pauses

    def test_trial_continues_past_30_seconds_until_operator_pauses_or_ends(self):
        for mode in ("sync", "async"):
            for command, outcome in (
                ("p", "takeover"),
                ("y", "success"),
                ("n", "failure"),
                ("q", "takeover"),
            ):
                with self.subTest(mode=mode, command=command):
                    trial, pauses = self.run_motion_trial(command, mode=mode)
                    reason = "Runner exit" if command == "q" else f"Operator {command}"
                    self.assertEqual((trial["outcome"], trial["reason"]), (outcome, reason))
                    self.assertEqual(pauses[0], reason)
                    self.assertNotIn("Trial time limit", pauses)

    def test_device_fault_still_aborts_trial_without_time_limit(self):
        trial, pauses = self.run_motion_trial("q", fault="fake disconnect")
        self.assertEqual(
            (trial["outcome"], trial["reason"]), ("device_abort", "fake disconnect")
        )
        self.assertEqual(pauses[0], "fake disconnect")


if __name__ == "__main__":
    unittest.main()
