"""Exercise a relocated checkout while denying reads from the source workspace."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

from rail_preinsert.perception import python_command
from tools.verify_bundle import verify

# -I disables PYTHONPATH; remove editable-install finders as well so an existing
# development environment cannot silently supply code from the source checkout.
BOOTSTRAP = r'''
import pathlib, runpy, sys
root, forbidden = pathlib.Path(sys.argv.pop(1)), pathlib.Path(sys.argv.pop(1))
sys.path[:] = [str(root)] + [p for p in sys.path if p and not pathlib.Path(p).is_relative_to(forbidden)]
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, "__module__", "").startswith("__editable__")]
def guard(event, args):
    if event == "import" and args and args[0].split(".")[0] == "fcl":
        raise ModuleNotFoundError("fcl is unavailable in the independent checkout")
    if event in ("open", "os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        if path.is_absolute() and path.is_relative_to(forbidden):
            raise RuntimeError("Unexpected source-workspace access: " + str(path))
sys.addaudithook(guard)
script = sys.argv[1]
sys.argv = sys.argv[1:]
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exc:
    code = exc.code or 0
for name, module in list(sys.modules.items()):
    if name.startswith(("diffusion_policy.", "rail_preinsert.", "rail_pose.")) and getattr(module, "__file__", None):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(root), (name, module.__file__)
assert "fcl" not in sys.modules
sys.exit(code)
'''


class StandaloneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="rail-standalone-")
        cls.clone = Path(cls.temporary.name) / "单项目 with spaces"
        # Exercise a fresh checkout, independent of locally imported measurements.
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "outputs", "build", ".pytest_cache",
                                        "calibration_runs", "head_to_left_base.json", "rail_goal.json")
        for name in ("diffusion_policy", "third_party"):
            shutil.copytree(ROOT / name, cls.clone / name, ignore=ignore)
        shutil.copytree(HERE, cls.clone / "workspace/rail_pose_insert", ignore=ignore)
        cls.experiment = cls.clone / "workspace/rail_pose_insert"

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_script(self, relative, *args, expected=0):
        result = subprocess.run(
            [sys.executable, "-I", "-c", BOOTSTRAP, str(self.clone), str(ROOT.parent),
             str(self.clone / relative), *map(str, args)],
            cwd=self.temporary.name, capture_output=True, text=True, timeout=120, check=False,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_snapshot_hashes_and_geometry_references_survive_relocation(self):
        report = verify(self.clone, source_only=True)
        self.assertTrue(report["ready"], report["problems"])

    def test_offline_planning_and_replay_without_source_workspaces(self):
        case = self.clone / "offline_case"
        self.run_script("workspace/rail_pose_insert/tools/make_offline_case.py", "--output", case)
        self.run_script("workspace/rail_pose_insert/run.py", "--config", case / "config.yaml", "plan",
                        "--state", case / "state.json", "--observation", case / "observation.json",
                        "--output", case / "plan.json", "--allow-test-data")
        self.run_script("workspace/rail_pose_insert/run.py", "--config", case / "config.yaml", "replay",
                        "--plan", case / "plan.json", "--output", case / "replay")
        report = json.loads((case / "replay/result.json").read_text())
        self.assertEqual(report["status"], "offline_path_verified")
        self.assertFalse(report["hardware_connected"])
        self.assertFalse(report["collision_checked"])
        self.assertEqual(len(json.loads((case / "plan.json").read_text())["joint_waypoints_rad"]), 2)

    def test_live_check_reports_only_missing_measured_inputs(self):
        result = self.run_script("workspace/rail_pose_insert/run.py", "check", expected=1)
        report = json.loads(result.stdout)
        self.assertEqual(len(report["problems"]), 2, report)
        self.assertFalse(report["ready_for_live_configuration"])
        self.assertNotIn(str(ROOT.parent), result.stdout)

    def test_calibration_loads_its_own_urdf_and_sdk_without_devices(self):
        probe = self.clone / "calibration_probe.py"
        probe.write_text('''from pathlib import Path
from diffusion_policy.third_party.camera_calibration.robot import load_model, world_flange, URDF
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.sdk import sdk_root
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import DEFAULT_CONFIG
model, digest = load_model()
assert URDF.is_relative_to(Path(__file__).parent)
assert sdk_root().is_relative_to(Path(__file__).parent)
assert DEFAULT_CONFIG.is_file() and DEFAULT_CONFIG.is_relative_to(Path(__file__).parent)
assert len(digest) == 64
assert world_flange(model, "left", [0.] * 7).shape == (4, 4)
''')
        self.run_script("calibration_probe.py")
        self.run_script("third_party/camera_calibration/run.py", "--help")
        self.run_script("third_party/camera_calibration/run.py", "external", "collect", "--help")

    def test_explicit_gpu_python_remains_configurable(self):
        self.assertEqual(python_command({"foundationpose_python": "/example/python"}), ["/example/python"])


if __name__ == "__main__":
    unittest.main()
