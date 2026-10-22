"""离线分析一次决策追踪。"""

import argparse
from pathlib import Path

from diffusion_policy.diagnostics.analyze import run


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial", required=True, type=Path, help="含 trace_meta.json 的 trial 目录"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="有示范/真实成对观测时，用这份权重做配对推理"
    )
    parser.add_argument(
        "--device", help="配对推理设备；缺省时用会话里的 device。CPU 与 CUDA 不能当成同一噪声"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
