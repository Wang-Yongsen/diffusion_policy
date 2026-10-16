"""同步推理与异步推理的讲解用时间线对比；推理耗时为假设，不连接设备。"""
from render_runtime_timing import A, B, DROP, GRID, HOLD, STEP, TEXT, Timeline

INFER_MS = 100
CHUNK = 12


def axis(p, y, bottom):
    for t in range(0, p.high+1, 100):
        x = p.x(t)
        p.draw.line((x, y, x, bottom), fill=GRID, width=1)
        p.text(x, y-38, str(t), 24, center=True)
    p.draw.line((p.left, y, p.right, y), fill=TEXT, width=2)


def main():
    p = Timeline(1400, "同步推理 vs 异步推理：同一秒内机器人在做什么",
                 f"示例：每次推理（预处理 + GPU）假设{INFER_MS} ms；每次预测{CHUNK}步、30 Hz，覆盖400 ms。非硬件实测。",
                 low=-20, high=1000)

    p.text(35, 160, "A · 同步：观测 → 推理 → 执行完全部动作 → 再观测", 30, bold=True)
    axis(p, 245, 560)
    p.text(35, 205, "时间 / ms", 25)
    p.lane(300, "观测")
    for i, t in enumerate((0, 500)):
        p.dot(t, 300, f"O{i+1}")
    p.lane(400, "推理")
    p.bar(0, INFER_MS, 400, "推理P1", A, 24)
    p.bar(500, 500+INFER_MS, 400, "推理P2", B, 24)
    p.lane(500, "机器人执行")
    p.bar(0, INFER_MS, 500, "停住", HOLD, 24)
    p.bar(INFER_MS, 500, 500, "执行P1的12步（依据O1）", A)
    p.bar(500, 500+INFER_MS, 500, "停住", HOLD, 24)
    p.bar(500+INFER_MS, 1000, 500, "执行P2的12步（依据O2）", B)
    for t in (INFER_MS, 500+INFER_MS):
        p.dashed(t, 420, 480)
    p.text(p.left, 545, "推理期间机器人等待；P1最后一步执行时，O1已过去约470 ms。", 25)

    p.text(35, 640, "B · 异步：推理线程连续推理，控制线程按时间执行最新预测", 30, bold=True)
    axis(p, 725, 1160)
    p.text(35, 685, "时间 / ms", 25)
    p.lane(780, "观测")
    for k in range(10):
        p.dot(k*INFER_MS, 780, f"O{k+1}")
    p.lane(890, "推理线程", "独立进程做GPU推理")
    for k in range(10):
        p.bar(k*INFER_MS, (k+1)*INFER_MS, 890, f"P{k+1}", A if k % 2 == 0 else B, 24)
    p.lane(1000, "P2的12步", "锚点100 ms")
    returned, replaced = 2*INFER_MS, 3*INFER_MS
    for i in range(CHUNK):
        t = INFER_MS + i*STEP
        if t <= returned + 10:
            p.dot(t, 1000, "×", fill=DROP)
        elif t < replaced:
            p.dot(t, 1000, fill="#35618B", radius=10)
        else:
            p.dot(t, 1000, "†", fill="#7D8792")
    p.dashed(returned, 920, 1130)
    p.dashed(replaced, 920, 1130)
    p.lane(1110, "机器人执行", "臂200 Hz / 手各120 Hz")
    p.bar(0, INFER_MS, 1110, "保持", HOLD, 24)
    for k in range(1, 10):
        p.bar(k*INFER_MS, (k+1)*INFER_MS, 1110, f"用P{k}", A if k % 2 == 1 else B, 24)

    p.text(35, 1195, "P2在200 ms返回：×为推理期间已过期的步骤，直接丢弃；蓝色大点为实际执行；"
                     "†为300 ms时被P3整体替换。", 26)
    p.text(35, 1240, "异步时机器人不等待，执行的动作所依据的观测始终在约100–200 ms以内；"
                     "代价是每轮都有步骤被丢弃或替换。", 26)
    p.text(35, 1285, "图中简化：P2返回到首个保留步骤（233 ms）之间保持上一个目标；"
                     "真实系统两次观测间隔至少 replanning_steps/30 ≈ 67 ms。", 26)
    p.save("08-sync-vs-async.png")


if __name__ == "__main__":
    main()
    print("已生成同步与异步对比时间线PNG。")
