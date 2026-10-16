"""同一模型（推理耗时相同）在不同动作频率下的异步执行对比；按Runtime调度规则计算，非硬件实测。"""
from PIL import ImageFont

from render_runtime_timing import DROP, FONT, GRID, HOLD, TEXT, Timeline

INFER_MS = 100
CHUNK = 12
MARGIN_MS = 10
END_MS = 400
RATES = (15, 30, 60)
NAMES = "ABC"
DOT = ("#35618B", "#43734C", "#9A6A2F")
FILL = ("#DCE8F5", "#DDEFE1", "#F5E7D3")
CARRY = ("#F0F5FB", "#F0F8F2", "#FBF5EC")
GREY = "#9AA3AD"


def predictions(hz):
    """第k次预测：锚点100(k-1) ms，100k ms返回；返回时只保留晚于返回+余量的步骤。"""
    period = 1000/hz
    preds = []
    for k in range(4):
        anchor, returned = k*INFER_MS, (k+1)*INFER_MS
        times = [anchor + i*period for i in range(CHUNK)]
        preds.append(dict(anchor=anchor, returned=returned, times=times,
                          kept=[i for i, t in enumerate(times) if t > returned + MARGIN_MS]))
    return preds


def targets(preds, period):
    """当前目标随时间的分段：(开始, 结束, 预测序号, 步序号, 是否为沿用段)。"""
    def at(t):
        live = [k for k, p in enumerate(preds) if p["returned"] <= t and p["kept"]]
        if not live:
            return None
        p = preds[live[-1]]
        due = [i for i in p["kept"] if p["times"][i] <= t]
        return (live[-1], due[-1]) if due else None

    cuts = sorted({0, END_MS, *(p["returned"] for p in preds),
                   *(p["times"][i] for p in preds for i in p["kept"] if p["times"][i] < END_MS)})
    merged, current = [], None
    for start, end in zip(cuts, cuts[1:]):
        current = at(start + 1e-6) or current
        if merged and merged[-1][2] == current:
            merged[-1][1] = end
        else:
            merged.append([start, end, current])
    out = []
    for start, end, key in merged:
        if key is None:
            out.append((start, end, None, None, False))
            continue
        k, i = key
        solid = min(end, preds[k]["times"][i] + period)
        out.append((start, solid, k, i, False))
        if solid < end:
            out.append((solid, end, k, i, True))
    return out


def cell(p, start, end, y, fill, label, dashed=False):
    x1, x2 = p.x(start), p.x(end)
    p.draw.rectangle((x1, y-24, x2, y+24), fill=fill, outline=GREY if dashed else "#7D8792", width=1)
    if p.draw.textlength(label, font=ImageFont.truetype(FONT, 21)) < x2-x1-6:
        p.text((x1+x2)/2, y-15, label, 21, center=True, fill="#5B6673" if dashed else TEXT)


def panel(p, top, hz):
    period = 1000/hz
    preds = predictions(hz)
    p.text(35, top, f"{hz} Hz：每步 {period:.1f} ms，12步覆盖 {CHUNK*period:.0f} ms", 30, bold=True)
    rows = [top+90, top+175, top+260]
    robot_y = top+355
    for t in range(0, END_MS+1, 50):
        p.draw.line((p.x(t), top+45, p.x(t), robot_y+28), fill=GRID, width=1)
    for p_ in preds[:3]:
        p.dashed(p_["returned"], top+45, robot_y-26)

    for k, y in enumerate(rows):
        pred = preds[k]
        p.lane(y, f"P{k+1} 的12步", f"锚点 {pred['anchor']} ms")
        replaced_at = preds[k+1]["returned"]
        for i, t in enumerate(pred["times"]):
            if t > END_MS:
                break
            x, name = p.x(t), f"{NAMES[k]}{i}"
            if i not in pred["kept"]:
                p.draw.line((x-7, y-7, x+7, y+7), fill=DROP, width=3)
                p.draw.line((x-7, y+7, x+7, y-7), fill=DROP, width=3)
                p.text(x, y+14, name, 19, fill=DROP, center=True)
            elif t < replaced_at:
                p.draw.ellipse((x-10, y-10, x+10, y+10), fill=DOT[k])
                p.text(x, y+14, name, 20, bold=True, fill=DOT[k], center=True)
            else:
                p.draw.ellipse((x-7, y-7, x+7, y+7), outline=GREY, width=2)
                p.text(x, y+14, name, 19, fill=GREY, center=True)

    p.lane(robot_y, "机器人目标", "控制线程发给设备")
    for start, end, k, i, carry in targets(preds, period):
        if k is None:
            cell(p, start, end, robot_y, HOLD, "初始目标")
        elif carry:
            cell(p, start, end, robot_y, CARRY[k], "沿用", dashed=True)
        else:
            cell(p, start, end, robot_y, FILL[k], f"{NAMES[k]}{i}")

    p2 = preds[1]
    dropped = CHUNK - len(p2["kept"])
    used = sum(p2["times"][i] < preds[2]["returned"] for i in p2["kept"])
    remaining = p2["anchor"] + CHUNK*period - p2["returned"]
    p.text(p.left, robot_y+36, f"每份预测：丢弃 {dropped} 步，执行 {used} 步，"
           f"被替换 {CHUNK-dropped-used} 步；返回后剩余覆盖 {remaining:.0f} ms。", 25)


def main():
    p = Timeline(1890, "同一模型、不同动作频率下的异步推理",
                 f"三种频率推理耗时相同：每 {INFER_MS} ms 返回一份 {CHUNK} 步预测。"
                 "按Runtime调度规则计算，非硬件实测；只显示前400 ms。",
                 low=-10, high=END_MS)
    p.text(35, 162, "时间 / ms", 25)
    for t in range(0, END_MS+1, 50):
        p.text(p.x(t), 162, str(t), 24, center=True)
    p.draw.line((p.left, 200, p.right, 200), fill=TEXT, width=2)
    p.lane(255, "推理线程", "三种频率相同")
    for k in range(4):
        cell(p, k*INFER_MS, (k+1)*INFER_MS, 255, FILL[k % 3] if k < 3 else HOLD,
             f"推理 P{k+1}，{(k+1)*INFER_MS} ms 返回")

    for n, hz in enumerate(RATES):
        panel(p, 330 + n*460, hz)

    p.text(35, 1730, "●：执行　×：返回时已过期，丢弃　○：被下一份预测替换，未执行"
                     "　虚线：预测返回时刻　浅色“沿用”：新预测的首个可用步骤未到，继续发上一个目标", 25)
    p.text(35, 1775, "频率越高：目标切换越细，但每份预测有更多步骤被推理延迟浪费，剩余覆盖越少，推理稍慢就要保持。", 25)
    p.text(35, 1820, "频率越低：剩余覆盖大，对推理耗时容忍度高；但每份预测只用上一两步，目标切换粗。"
                     "前提：每个模型都按对应频率的数据训练。", 25)
    p.save("09-async-action-rates.png")


if __name__ == "__main__":
    main()
    print("已生成不同动作频率的异步推理对比PNG。")
