"""Compatibility entry for python -m diffusion_policy.cli.tianji analyze."""

from diffusion_policy.cli.tianji.analyze import main

if __name__ == "__main__":
    raise SystemExit(main())
