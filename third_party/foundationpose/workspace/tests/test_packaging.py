import json
import tarfile
from pathlib import Path
from rail_pose import packaging


def test_deployment_is_portable_without_git_and_omits_compiled_extensions(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    workspace = root / "workspace"
    (workspace / "vendor").mkdir(parents=True)
    (workspace / "vendor/manifest.json").write_text("{}")
    (workspace / "run.py").write_text("# entry")
    (root / "estimater.py").write_text("# core")
    (workspace / "vendor/pytorch3d/build").mkdir(parents=True)
    (workspace / "vendor/pytorch3d/build/compiled.so").write_bytes(b"machine-specific")
    (workspace / "weights").mkdir()
    (workspace / "weights/test.pth").write_bytes(b"weights")
    monkeypatch.setattr(packaging, "REPOSITORY", root)
    monkeypatch.setattr(packaging, "WORKSPACE", workspace)
    output = workspace / "deployment/test.tar.gz"
    packaging.package(output)
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert "FoundationPose/estimater.py" in names
        assert "FoundationPose/workspace/weights/test.pth" in names
        assert not any(name.endswith(".so") for name in names)
        assert all(name.startswith("FoundationPose/") for name in names)
    assert json.loads(output.with_suffix(".sha256.json").read_text())["sha256"]
