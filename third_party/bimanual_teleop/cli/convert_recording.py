"""将遥操作记录按严格主相机时刻或修复时间网格导出为 Diffusion Policy 数据集。"""

import argparse
from pathlib import Path
import sys

from diffusion_policy.third_party.bimanual_teleop.common.console import EpisodeProgress
from diffusion_policy.third_party.bimanual_teleop.recording.convert import convert_recordings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="原始记录根目录、session 或单个 episode")
    parser.add_argument("--output", required=True, type=Path, help="新建的 .zarr 数据集，禁止覆盖")
    parser.add_argument("--action-space", required=True, choices=("eef", "joint"),
                        help="eef: 双臂目标末端+双手（52维）；joint: 双臂目标关节+双手（54维）")
    parser.add_argument("--include-depth", action="store_true",
                        help="导出主相机深度并检查深度匹配；默认只导出 RGB，不因深度缺帧剔除样本")
    parser.add_argument("--conversion-config", type=Path,
                        help="带注释的转换 YAML；不传则保留原严格模式")
    args = parser.parse_args(argv)
    try:
        options = {} if args.conversion_config is None else {"conversion_config": args.conversion_config}
        with EpisodeProgress("转换条目") as progress:
            report = convert_recordings(args.input, args.output, action_space=args.action_space,
                                        include_depth=args.include_depth, progress=progress, **options)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
        print(f"转换失败：{error}", file=sys.stderr)
        return 1
    print(f"已导出 {report['output_episodes']} 条原始演示、{report['output_segments']} 个连续片段、"
          f"{report['output_frames']} 帧：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
