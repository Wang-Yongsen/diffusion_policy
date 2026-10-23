#!/usr/bin/env python3
from pathlib import Path
from importlib.metadata import version
import re
import subprocess
import sys


def runtime_requirements(frozen: str) -> str:
    excluded = ("torch==", "torchvision==", "triton==", "nvidia-", "nvdiffrast", "pytorch3d")
    runtime = []
    for line in frozen.splitlines():
        if not line or line.lower().startswith(excluded):
            continue
        # Conda's pip metadata can retain build-machine file:// URLs. Pin the
        # installed version for deployment; keep original URLs in the audit lock.
        direct = re.match(r"^([A-Za-z0-9_.-]+)\s*@\s*", line)
        if direct:
            name = direct.group(1)
            line = f"{name}=={version(name)}"
        runtime.append(line)
    return "\n".join(runtime) + "\n"


def main():
    workspace = Path(__file__).resolve().parents[1]
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True, check=True).stdout
    (workspace / "requirements-lock.txt").write_text(frozen)
    (workspace / "requirements-runtime.txt").write_text(runtime_requirements(frozen))


if __name__ == "__main__":
    main()
