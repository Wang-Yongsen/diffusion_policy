"""从本项目入口生成训练 zarr；repair 必须显式传配置。"""

from diffusion_policy.third_party.bimanual_teleop.cli.convert_recording import main

if __name__ == "__main__":
    raise SystemExit(main())
