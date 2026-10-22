"""Compatibility entry; use python -m diffusion_policy.cli.tianji home."""

from diffusion_policy.real_world.tianji.home import MotionTerminal
from diffusion_policy.cli.tianji.home import main

if __name__ == "__main__":
    raise SystemExit(main())
