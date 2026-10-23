#!/usr/bin/env python3
"""Portable workspace entrypoint, independent of the shell working directory."""
from pathlib import Path
import sys

workspace = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace.parent))
sys.path.insert(0, str(workspace / "src"))

from rail_pose.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
