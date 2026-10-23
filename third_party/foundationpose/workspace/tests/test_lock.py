import importlib.util
from pathlib import Path


def test_runtime_lock_does_not_depend_on_build_machine_paths(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/write_lock.py"
    spec = importlib.util.spec_from_file_location("write_lock", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "version", lambda name: {"packaging": "26.0"}[name])
    frozen = ("packaging @ file:///private/build-machine/work\n"
              "numpy==2.4.6\ntorch==2.7.1+cu128\nnvdiffrast==0.4.0\n")
    result = module.runtime_requirements(frozen)
    assert result == "packaging==26.0\nnumpy==2.4.6\n"
