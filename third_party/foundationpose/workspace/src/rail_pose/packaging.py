"""Package this local FoundationPose snapshot without consulting Git repositories."""
from pathlib import Path
import tarfile
from .common import REPOSITORY, WORKSPACE, save_json, sha256

def package(output: Path) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded = {".git", "__pycache__", "build", ".pytest_cache", "outputs", "deployment"}
    selected = [p for p in REPOSITORY.rglob("*") if p.is_file() and p.resolve() != output
                and not excluded.intersection(p.relative_to(REPOSITORY).parts)
                and not p.is_relative_to(WORKSPACE / "outputs") and p.suffix not in {".pyc", ".so"}]
    temporary = output.with_suffix(output.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(selected):
            archive.add(path, arcname=str(Path("FoundationPose") / path.relative_to(REPOSITORY)), recursive=False)
    temporary.replace(output)
    save_json(output.with_suffix(".sha256.json"), {"archive": output.name, "sha256": sha256(output), "files": len(selected)})
    return output
