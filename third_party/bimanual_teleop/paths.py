"""Default configuration resources bundled with the host package."""

from importlib import import_module
from pathlib import Path

# Resolve the config namespace itself: the outer checkout directory can also
# appear in diffusion_policy.__path__ when Python starts in its parent.
PROJECT_ROOT = Path(next(iter(import_module("diffusion_policy.config").__path__))).resolve().parent
