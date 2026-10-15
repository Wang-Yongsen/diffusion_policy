"""按当前调度语义绘制讲解用时间轴；所有推理耗时均为假设，不连接设备。"""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
TEXT, GRID = "#263445", "#D9DFE6"
A, B, HOLD, DROP = "#E5EEF7", "#E7F1E9", "#EFF0F2", "#AA4545"
STEP = 1000 / 30


class Timeline:
    def __init__(self, height, title, subtitle, low=-40, high=320):
        self.image = Image.new("RGB", (2000, height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.low, self.high = low, high
        self.left, self.right = 330, 1930
        self.text(35, 28, title, 42, bold=True)
        self.text(35, 92, subtitle, 26)

    def x(self, time_ms):
        return self.left + (time_ms-self.low)/(self.high-self.low)*(self.right-self.left)

    def text(self, x, y, value, size=27, bold=False, fill=TEXT, center=False):
        font = ImageFont.truetype(BOLD if bold else FONT, size)
        if center:
            x -= self.draw.textlength(value, font=font)/2
        assert x >= 0 and x+self.draw.textlength(value, font=font) <= 2000, value
        self.draw.text((x, y), value, font=font, fill=fill)

    def axis(self, bottom):
        self.text(35, 160, "相对时间 / ms", 27, bold=True)
        for i in range(-1, 10):
            t = i*STEP
            if not self.low <= t <= self.high:
                continue
            x = self.x(t)
            self.draw.line((x, 200, x, bottom), fill=GRID, width=1)
            label = "0" if i == 0 else f"{t:.1f}"
            self.text(x, 157, label, 24, center=True)
        self.draw.line((self.left, 200, self.right, 200), fill=TEXT, width=2)

    def lane(self, y, title, detail=""):
        self.text(35, y-24, title, 29, bold=True)
        if detail:
            self.text(35, y+17, detail, 23)
        self.draw.line((self.left, y, self.right, y), fill=GRID, width=2)

    def dot(self, t, y, label="", fill=TEXT, radius=7):
        x = self.x(t)
        self.draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill)
        if label:
            self.text(x, y+17, label, 26, fill=fill, center=True)

    def bar(self, start, end, y, label, color, size=27):
        x1, x2 = self.x(start), self.x(end)
        self.draw.rectangle((x1, y-23, x2, y+23), fill=color, outline="#7D8792", width=1)
        font = ImageFont.truetype(FONT, size)
        assert self.draw.textlength(label, font=font) < x2-x1-6, label
        self.text((x1+x2)/2, y-19, label, size, center=True)

    def tick(self, time_ms, y, height=11, fill="#7D8792"):
        x = self.x(time_ms)
        self.draw.line((x, y-height, x, y+height), fill=fill, width=2)

    def dashed(self, t, y1, y2, fill="#7D8792"):
        x = self.x(t)
        for y in range(y1, y2, 16):
            self.draw.line((x, y, x, min(y+9, y2)), fill=fill, width=2)

    def save(self, name):
        self.image.save(ROOT/name)


def normal():
    p = Timeline(1390, "真机闭环时间序列：两轮预测如何衔接",
                 "示例：每轮预处理与推理合计40 ms；发布时间近似为返回时间。非硬件实测，所有目标假设通过校验。")
    p.axis(1240)
    p.lane(275, "相机 / 状态历史", "持续采集，保存时间戳")
    for i in range(-1, 10):
        p.dot(i*STEP, 275)
    p.text(p.x(0), 224, "t0(1)=0", 25, bold=True, center=True)
    p.text(p.x(2*STEP), 224, "t0(2)=66.7", 25, bold=True, center=True)
    p.lane(400, "对齐的两步观测", "Env：图像匹配 / 状态插值")
    p.bar(-STEP, 0, 400, "O1", A)
    p.bar(STEP, 2*STEP, 400, "O2", B)
    for t in (-STEP, 0, STEP, 2*STEP):
        p.dot(t, 400, radius=5)
    p.lane(530, "预处理 + GPU推理", "Runner：线程独立运行")
    p.bar(10, 50, 530, "P1: 40 ms", A, 24)
    p.bar(80, 120, 530, "P2: 40 ms", B, 24)
    p.text(p.x(50), 565, "50：提交P1", 25, center=True)
    p.text(p.x(120), 565, "120：提交P2", 25, center=True)
    p.lane(690, "P1名义动作时刻", "A0…A7；锚点0 ms")
    for i in range(8):
        fill = DROP if i < 2 else "#35618B" if i < 4 else "#7D8792"
        p.dot(i*STEP, 690, f"A{i}"+("×" if i < 2 else "†" if i >= 4 else ""), fill=fill)
    p.text(p.x(50), 636, "仅保留 >60 ms 的目标", 25, center=True)
    p.lane(830, "P2名义动作时刻", "B0…B7；锚点66.7 ms")
    for i in range(8):
        p.dot((i+2)*STEP, 830, f"B{i}"+("×" if i < 2 else ""), fill=DROP if i < 2 else "#43734C")
    p.text(p.x(120), 776, "仅保留 >130 ms 的目标", 25, center=True)
    p.lane(990, "Runtime当前目标", "实际SDK提交发生在各控制tick")
    p.bar(0, 2*STEP, 990, "接合参考", HOLD)
    p.bar(2*STEP, 3*STEP, 990, "A2", A)
    p.bar(3*STEP, 4*STEP, 990, "A3", A)
    for i in range(2, 7):
        p.bar((i+2)*STEP, (i+3)*STEP, 990, f"B{i}", B)
    p.bar(9*STEP, 320, 990, "B7", B, 24)
    p.dashed(50, 610, 1030)
    p.dashed(120, 610, 1030)
    p.text(p.x(120)+18, 1040, "120 ms替换队列；133.3 ms前保持A3", 25)
    p.lane(1130, "双臂控制线程", "200 Hz；名义周期5 ms")
    p.lane(1230, "左右手控制线程", "各120 Hz；名义周期8.33 ms")
    for t in range(0, 321, 5):
        p.tick(t, 1130)
    for i in range(math.floor(320/(1000/120))+1):
        p.tick(i*1000/120, 1230)
    p.text(35, 1295, "×：过期或不足10 ms提交余量，丢弃；†：P2发布后，旧队列中的未来A4…A7被替换。", 26)
    p.text(35, 1338, "实际反馈持续进入历史缓存，供后续观测使用；名义目标时间 ≠ SDK提交时间 ≠ 物理到位时间。", 26)
    p.save("06-real-runtime-timeline.png")


def expired():
    p = Timeline(960, "推理过慢时：控制线程独立处理动作覆盖耗尽",
                 "示例：P1最后目标233.3 ms，覆盖到266.7 ms；P2从80 ms推理到310 ms。非硬件实测。",
                 low=0, high=340)
    p.axis(750)
    p.lane(300, "P1当前可用覆盖", "最后目标 + 一个30 Hz间隔")
    p.bar(50, 8*STEP, 300, "等待首目标 → 按时推进A2…A7", A)
    p.dot(7*STEP, 300, "A7@233.3", fill="#35618B")
    p.lane(455, "P2预处理 / 推理", "GPU线程尚未返回")
    p.bar(80, 310, 455, "P2仍在计算", B)
    p.text(p.x(310), 490, "310 ms返回", 25, center=True)
    p.lane(610, "控制线程 / 停止", "不等待GPU推理结束")
    p.bar(50, 8*STEP, 610, "正常策略目标", A)
    p.bar(8*STEP, 340, 610, "请求暂停 / 保持", HOLD, 24)
    p.dashed(8*STEP, 220, 770, fill=DROP)
    p.text(p.x(8*STEP), 220, "266.7 ms：覆盖耗尽", 25, fill=DROP, center=True)
    p.text(350, 705, "首个检测到过期的控制tick：清空未来目标，双臂请求保持，健康手维持最后接受目标。", 26)
    p.text(35, 815, "Runner观察到fault后结束试验；已停止试验的迟到预测不会自动恢复运动。", 27)
    p.text(35, 865, "这里表示停止请求与软件状态切换；物理停止完成时刻必须查实际反馈，不由图中时间直接保证。", 26)
    p.save("07-real-runtime-expiry.png")


def mermaid():
    source = """%% 讲解用两轮时序；40ms为假设，实际时间需查日志。
sequenceDiagram
    participant H as 相机与实际状态历史
    participant E as Env：对齐观测
    participant P as Runner：预处理/Policy
    participant R as Runtime：未来队列
    participant C as 双臂/左右手控制线程
    participant D as 真机与SDK
    H->>E: 历史覆盖t0₁=0；查询[-33.3,0]ms
    E->>P: O1 + t0₁；10ms开始计算
    P->>R: 50ms返回8步A；名义时刻t0₁+i/30
    Note over R: 仅保留>60ms；A0/A1丢弃
    R->>C: 保持接合参考，66.7ms起使用A2
    C->>D: 200Hz双臂/各120Hz手按tick提交
    D-->>H: 实际图像与状态持续反馈
    H->>E: 历史覆盖t0₂=66.7；查询[33.3,66.7]ms
    E->>P: O2 + t0₂；80ms开始计算
    Note over C,D: P2计算期间继续推进P1的当前目标
    P->>R: 120ms返回8步B；保留>130ms
    Note over R: 丢弃B0/B1；未来队列整体替换
    R->>C: 133.3ms前保持A3，此后使用B2...
    C->>D: 连续控制；实际反馈进入下一轮观测
    Note over R,C: 若覆盖耗尽：独立清队列、请求暂停，不等待GPU
"""
    (ROOT/"06-real-runtime-timeline.mmd").write_text(source, encoding="utf-8")


if __name__ == "__main__":
    normal()
    expired()
    mermaid()
    print("已生成正常闭环与覆盖耗尽时间轴PNG，以及两轮时序Mermaid。")
