"""Compatibility entry; use python -m diffusion_policy.cli.tianji replay."""

from diffusion_policy.cli.tianji.replay import main

if __name__ == "__main__":
    raise SystemExit(main())
