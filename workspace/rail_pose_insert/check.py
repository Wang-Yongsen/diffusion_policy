"""检查右手接料、目标上方和左臂预插入配置；不连接设备。"""
from receive import main as check_receive
from run import main as check_preinsert

if __name__ == "__main__":
    right = check_receive(["check", "--move-above"])
    left = check_preinsert(["check"])
    raise SystemExit(right or left)
