"""Shared geometry, planning and perception for the rail insertion workflow."""

from pathlib import Path
import sys

# Direct scripts must also work when launched outside the repository root.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
