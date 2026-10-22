"""机械臂与灵巧手回位。"""

import argparse
import sys
from pathlib import Path

from diffusion_policy.real_world.tianji.home import Cancelled, run
from diffusion_policy.third_party.bimanual_teleop.devices.tianji.config import (
    DEFAULT_CONFIG as ARM_CONFIG,
)
from diffusion_policy.third_party.bimanual_teleop.devices.wuji.config import (
    DEFAULT_CONFIG as HAND_CONFIG,
)


def build_parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--part",
        choices=("both", "arms", "hands"),
        default="both",
        help="回位部位，默认机械臂和灵巧手都回",
    )
    result.add_argument("--side", choices=("both", "left", "right"), default="both")
    result.add_argument(
        "--tianji-config",
        type=Path,
        default=ARM_CONFIG,
        help="天机配置，默认 diffusion_policy/config/tianji_teleop.yaml",
    )
    result.add_argument(
        "--wuji-config",
        type=Path,
        default=HAND_CONFIG,
        help="灵巧手配置，默认 diffusion_policy/config/wuji_teleop.yaml",
    )
    result.add_argument("--ip", help="临时覆盖天机控制器 IP")
    result.add_argument(
        "--duration-s", type=float, default=3.0, help="灵巧手平滑移动时间，默认 3 秒"
    )
    result.add_argument("--tolerance-deg", type=float, default=5.0, help="灵巧手到位容差，默认 5°")
    result.add_argument(
        "--settle-timeout-s", type=float, default=2.0, help="平滑结束后的到位等待，默认 2 秒"
    )
    result.add_argument("--inspect", action="store_true", help="只读取当前关节，不运动")
    return result


def main(argv=None):
    try:
        return run(build_parser().parse_args(argv))
    except Cancelled:
        print("已取消回位。")
        return 0
    except KeyboardInterrupt:
        print("已取消回位。")
        return 0
    except Exception as exc:
        print(f"回位失败：{exc}", file=sys.stderr)
        return 1
