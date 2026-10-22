"""Dispatch rail commands without importing training or device code for help."""

import argparse
import sys
from importlib import import_module


def main(argv=None):
    parser = argparse.ArgumentParser(description="天机导轨训练配套工具")
    parser.add_argument("command", choices=("eval", "replay", "home", "analyze", "probe"))
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    module = import_module(f".{command}", __package__)
    return module.main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
