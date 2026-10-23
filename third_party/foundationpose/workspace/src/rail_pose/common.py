from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

WORKSPACE = Path(__file__).resolve().parents[2]
REPOSITORY = WORKSPACE.parent


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (WORKSPACE / value).resolve()


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path | None = None) -> dict:
    data = json.loads(resolve(path or "configs/default.json").read_text(encoding="utf-8"))
    if int(data["pose_batch_size"]) < 1:
        raise ValueError("pose_batch_size must be positive")
    if not 0 < data["max_depth_m"] < 100:
        raise ValueError("max_depth_m must be in metres and below 100")
    return data


def new_run(label: str, directory: str | Path | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = resolve(directory) if directory else WORKSPACE / "outputs" / f"{label}-{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path
