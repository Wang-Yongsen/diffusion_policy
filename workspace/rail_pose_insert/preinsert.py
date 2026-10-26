"""左臂接料与预插入；启动后按 O/G/M/E 推进，不执行最终插入。"""
import sys
from run import main

if __name__ == "__main__":
    raise SystemExit(main(["run", "--motion", *sys.argv[1:]]))
