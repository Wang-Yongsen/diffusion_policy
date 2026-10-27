"""检查当前插入配置、现场文件和 GPU 依赖，不连接机器人。"""

import argparse
import json
from pathlib import Path
from rail_common.core import HERE
from rail_receive.logic import load_config
from receive import preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "receive.yaml")
    args = parser.parse_args(argv)
    try:
        report = preflight(load_config(args.config))
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        report = dict(ready=False, problems=[str(error)])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
