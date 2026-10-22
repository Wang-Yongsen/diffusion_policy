"""Compatibility entry; use python -m diffusion_policy.cli.tianji eval."""

from diffusion_policy.cli.tianji.eval import main

if __name__ == "__main__":
    raise SystemExit(main())
