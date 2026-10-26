"""Check operator entrypoints without connecting cameras or actuators."""
import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import receive
import run
from rail_preinsert.perception import python_command


class EntrypointTests(unittest.TestCase):
    def test_direct_launch_enables_motion_and_cleans_temporary_files(self):
        for module, operation in ((receive, "run"), (run, "live")):
            with self.subTest(module=module.__name__):
                seen = []
                def execute(config, args):
                    self.assertTrue(args.motion)
                    if module is receive:
                        self.assertTrue(args.move_above)
                    target = Path(args.output)
                    self.assertFalse(target.exists())
                    target.mkdir()
                    (target / "feedback.json").write_text("{}")
                    seen.append(target)
                with patch.object(module, "load_config", return_value={}), patch.object(module, operation, execute):
                    self.assertEqual(module.main([]), 0)
                self.assertFalse(seen[0].parent.exists())

    def test_legacy_read_only_remains_read_only(self):
        for module, operation in ((receive, "run"), (run, "live")):
            with patch.object(module, "load_config", return_value={}), patch.object(module, operation) as execute:
                self.assertEqual(module.main(["run"]), 0)
                self.assertFalse(execute.call_args.args[1].motion)
                if module is receive:
                    self.assertFalse(execute.call_args.args[1].move_above)

    def test_receive_motion_defaults_to_transfer_and_pose_only_is_explicit(self):
        cases = [(["run", "--motion"], True),
                 (["run", "--motion", "--move-above"], True),
                 (["run", "--motion", "--pose-only"], False),
                 (["run", "--motion", "--already-held"], True),
                 (["run", "--motion", "--above-height", "0.08"], True)]
        for argv, expected in cases:
            config = {"move_above": {"height_m": .05}}
            with self.subTest(argv=argv), patch.object(receive, "load_config", return_value=config), \
                    patch.object(receive, "run") as execute:
                self.assertEqual(receive.main(argv), 0)
                self.assertEqual(execute.call_args.args[1].move_above, expected)
                if "--above-height" in argv:
                    self.assertEqual(execute.call_args.args[0]["move_above"]["height_m"], .08)

    def test_invalid_motion_options_never_open_devices(self):
        cases = [["run", "--move-above"], ["run", "--already-held"],
                 ["run", "--motion", "--pose-only", "--already-held"],
                 ["run", "--motion", "--pose-only", "--above-height", "0.08"],
                 ["run", "--motion", "--pose-only", "--move-above"]]
        for argv in cases:
            with self.subTest(argv=argv), patch.object(receive, "run") as execute, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    receive.main(argv)
                self.assertEqual(error.exception.code, 2)
                execute.assert_not_called()

    def test_startup_error_cleans_temporary_files(self):
        for module, operation in ((receive, "run"), (run, "live")):
            seen = []
            def fail(config, args):
                seen.append(Path(args.output))
                seen[-1].mkdir()
                raise RuntimeError("simulated connection failure")
            with patch.object(module, "load_config", return_value={}), patch.object(module, operation, fail), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(module.main([]), 1)
            self.assertFalse(seen[0].parent.exists())

    def test_explicit_diagnostics_are_retained(self):
        for module, operation in ((receive, "run"), (run, "live")):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "saved"
                def execute(config, args):
                    Path(args.output).mkdir()
                with patch.object(module, "load_config", return_value={}), patch.object(module, operation, execute):
                    self.assertEqual(module.main(["run", "--output", str(output)]), 0)
                self.assertTrue(output.is_dir())

    def test_named_scripts_select_the_intended_task(self):
        cases = [
            ("move_above.py", receive, ["run", "--motion", "--move-above"]),
            ("move_above_held.py", receive, ["run", "--motion", "--move-above", "--already-held"]),
            ("insert_spiral.py", receive, ["run", "--motion", "--insertion-mode", "position_spiral"]),
            ("teach_target.py", run, ["teach-target"]),
            ("preinsert.py", run, ["run", "--motion"]),
            ("read_state.py", receive, ["run"]),
            ("read_preinsert_state.py", run, ["run"]),
        ]
        for script, module, expected in cases:
            with self.subTest(script=script), patch.object(sys, "argv", [script]), \
                    patch.object(module, "main", return_value=0) as main:
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(HERE / script), run_name="__main__")
                self.assertEqual(result.exception.code, 0)
                main.assert_called_once_with(expected)

    def test_check_runs_both_checks_and_reports_failure(self):
        with patch.object(receive, "main", return_value=1) as right, \
                patch.object(run, "main", return_value=0) as left:
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(HERE / "check.py"), run_name="__main__")
            self.assertEqual(result.exception.code, 1)
            right.assert_called_once_with(["check", "--move-above"])
            left.assert_called_once_with(["check"])

    def test_perception_uses_active_python_even_with_old_env_setting(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(python_command({"foundationpose_env": "obsolete"}), [sys.executable])
            self.assertEqual(python_command({"foundationpose_python": "/explicit/python"}), ["/explicit/python"])


if __name__ == "__main__":
    unittest.main()
