"""只读并打印右臂和右手反馈，不使能动作。"""

import argparse
from pathlib import Path
from rail_common.core import HERE
from rail_receive.logic import load_config
from receive import read_state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "receive.yaml")
    args = parser.parse_args(argv)
    read_state(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
