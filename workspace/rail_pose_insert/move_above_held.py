"""保持当前手型估姿并保存均值，默认按 M 移到目标上方，不先张手。"""
import sys
from receive import main

if __name__ == "__main__":
    raise SystemExit(main(["run", "--motion", "--move-above", "--already-held", *sys.argv[1:]]))
