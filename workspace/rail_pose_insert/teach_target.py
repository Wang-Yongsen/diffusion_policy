"""导轨正确就位后，用相机记录 targets/rail_goal.json。"""
import sys
from run import main

if __name__ == "__main__":
    raise SystemExit(main(["teach-target", *sys.argv[1:]]))
