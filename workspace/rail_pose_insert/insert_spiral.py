"""右臂接料、搬运与预放置视觉纠偏；对准后按 N 位置螺旋插入，不采集六维力。"""
import sys

from receive import main


if __name__ == "__main__":
    raise SystemExit(main(["run", "--motion", "--insertion-mode", "position_spiral", *sys.argv[1:]]))
