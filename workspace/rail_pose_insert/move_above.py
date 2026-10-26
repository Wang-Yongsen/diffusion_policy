"""接料抓握后保存位姿均值，默认按 M 移到目标上方并保持。"""
import sys
from receive import main

if __name__ == "__main__":
    raise SystemExit(main(["run", "--motion", "--move-above", *sys.argv[1:]]))
