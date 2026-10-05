"""Default configuration resources bundled with the host package."""

from pathlib import Path
import diffusion_policy

PROJECT_ROOT = Path(next(iter(diffusion_policy.__path__))).resolve()
