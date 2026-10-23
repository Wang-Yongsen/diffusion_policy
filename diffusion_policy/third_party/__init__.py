"""Resolve bundled dependencies when running directly from a source checkout."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / "third_party")]
