"""离线对比数据集观测与策略预测。"""

import argparse
from pathlib import Path

from diffusion_policy.diagnostics.probe import run


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-zarr", type=Path, required=True)
    parser.add_argument("--val-zarr", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, help="可选的原始观测窗口，RGB 为 uint8 NHWC")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0, help="起始有效窗口序号")
    parser.add_argument("--chunks", type=int, default=10, help="最多比较的连续窗口数量")
    return parser


def main(argv=None):
    return run(build_parser().parse_args(argv))
