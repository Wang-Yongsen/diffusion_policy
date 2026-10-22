"""Compatibility entry for python -m diffusion_policy.cli.tianji probe."""

from diffusion_policy.cli.tianji.probe import main

if __name__ == "__main__":
    raise SystemExit(main())
