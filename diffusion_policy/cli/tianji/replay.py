"""示范观测回放；运动需要 --motion 和手动开始。"""

import argparse
from pathlib import Path

from diffusion_policy.common.paths import PROJECT_ROOT
from diffusion_policy.real_world.tianji.replay import run


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--seed",
        "--batch-seed",
        dest="seed",
        type=int,
        help="初始化结束后、正式第0次预测前重置种子",
    )
    parser.add_argument("--skip-approach", action="store_true", help="跳过起始姿态接近段")
    parser.add_argument(
        "--dataset", required=True, type=Path, help="convert_recording.py 导出的 eef zarr"
    )
    parser.add_argument(
        "--episode", default="0", help="来源条目名，或从 0 开始的条目序号；默认最长连续片段"
    )
    parser.add_argument("--segment", type=int, help="改为指定全局片段序号")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval.yaml",
    )
    parser.add_argument("--device", help="覆盖配置中的模型设备，例如 cpu 或 cuda:0")
    parser.add_argument("--max-windows", type=int, help="只回放前若干个执行块")
    parser.add_argument("--output", type=Path, help="新建会话目录，不覆盖已有目录")
    parser.add_argument(
        "--motion", action="store_true", help="把预测动作下发到真机；须在交互终端手动输入 s"
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="运动时保存示范观测，并另存当时的真实观测；真实观测不参与控制",
    )
    parser.add_argument("--trace-capacity", type=int, default=48, help="完整记录最多保存多少次决策")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
