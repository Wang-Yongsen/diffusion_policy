from pathlib import Path
import sys

workspace = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(workspace / "src"), str(workspace.parent)]
