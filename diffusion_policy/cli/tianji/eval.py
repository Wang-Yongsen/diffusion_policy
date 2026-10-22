"""天机导轨评估：默认影子模式，--check-only 仅检查模型。"""

import argparse
from pathlib import Path

from diffusion_policy.common.paths import PROJECT_ROOT
from diffusion_policy.real_world.tianji.evaluation import run
from diffusion_policy.real_world.tianji.runtime import INFERENCE_MODES


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--seed",
        "--batch-seed",
        dest="seed",
        type=int,
        help="正式第0次预测前重置采样种子；预热不占正式采样噪声",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "diffusion_policy/config/tianji_rail_real_eval.yaml",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖部署配置中已有的参数，可重复；值按 YAML 解析，嵌套参数用点号，如 workspace.left.min=[0,0,0]",
    )
    parser.add_argument("--device", help="覆盖配置中的模型设备，例如 cpu 或 cuda:0")
    parser.add_argument(
        "--inference-mode",
        choices=INFERENCE_MODES,
        help="覆盖配置中的推理模式：async 推理与执行并行；sync 推理期间保持，动作块执行完再观测",
    )
    parser.add_argument("--output", type=Path, help="新建会话目录，不覆盖已有目录")
    parser.add_argument(
        "--check-only", action="store_true", help="只重载模型与模拟输入推理，不连接设备"
    )
    parser.add_argument("--motion", action="store_true", help="允许手动接合；必须已填写工作空间")
    parser.add_argument(
        "--record-observations",
        action="store_true",
        help="试验中额外记录模型输入的低维状态，以及少量图像窗口和 30 Hz 实测位姿",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="把每次决策的原始输入、伺服命令和反馈留在内存，运动结束后再落盘",
    )
    parser.add_argument("--trace-capacity", type=int, default=48, help="完整记录最多保存多少次决策")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
