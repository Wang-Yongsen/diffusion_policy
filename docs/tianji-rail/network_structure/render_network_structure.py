"""根据当前导轨任务配置绘制网络结构；仅生成文档图，不连接设备。
主图依据 2026.10.19/14.08.04_tianji_rail_joint/.hydra/config.yaml。
核对入口：DiffusionTransformerHybridImagePolicy / TransformerForDiffusion。
"""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
INK, LINE, BORDER = "#233244", "#4F6173", "#9BA8B5"
BLUE, GREEN, GRAY = "#EDF3F9", "#EFF5F0", "#F5F6F8"

class Figure:
    def __init__(self, height, title, subtitle):
        self.im = Image.new("RGB", (2000, height), "white")
        self.d = ImageDraw.Draw(self.im)
        self.text(45, 25, title, 43, bold=True)
        self.text(45, 90, subtitle, 27)
        self.boxes = {}

    def text(self, x, y, text, size=28, bold=False, center=False, color=INK):
        font = ImageFont.truetype(BOLD if bold else FONT, size)
        width = self.d.textlength(text, font=font)
        if center:
            x -= width / 2
        assert 0 <= x and x + width <= 2000, (text, x, width)
        self.d.text((x, y), text, font=font, fill=color)

    def group(self, rect, title):
        self.d.rectangle(rect, outline="#CBD3DC", width=2)
        self.text(rect[0]+22, rect[1]+16, title, 31, bold=True)

    def box(self, key, rect, title, *lines, fill=BLUE, size=27):
        self.boxes[key] = rect
        x1, y1, x2, y2 = rect
        self.d.rectangle(rect, fill=fill, outline=BORDER, width=2)
        title_font = ImageFont.truetype(BOLD, 30)
        body_font = ImageFont.truetype(FONT, size)
        assert self.d.textlength(title, font=title_font) < x2-x1-20, title
        for line in lines:
            assert self.d.textlength(line, font=body_font) < x2-x1-20, line
        line_height = size+12
        total = 39 + line_height * len(lines)
        assert total <= y2-y1, (key, total, y2-y1)
        yy = y1 + (y2-y1-total)/2
        self.text((x1+x2)/2, yy, title, 30, bold=True, center=True)
        for i, line in enumerate(lines):
            self.text((x1+x2)/2, yy+42+i*line_height, line, size, center=True)

    def anchor(self, key, side):
        x1, y1, x2, y2 = self.boxes[key]
        return {"top": ((x1+x2)/2, y1), "bottom": ((x1+x2)/2, y2),
                "left": (x1, (y1+y2)/2), "right": (x2, (y1+y2)/2)}[side]

    def arrow(self, points):
        self.d.line(points, fill=LINE, width=3, joint="curve")
        x1, y1 = points[-2]
        x2, y2 = points[-1]
        length = math.hypot(x2-x1, y2-y1)
        ux, uy = (x2-x1)/length, (y2-y1)/length
        self.d.polygon([(x2, y2), (x2-13*ux+6*uy, y2-13*uy-6*ux),
                        (x2-13*ux-6*uy, y2-13*uy+6*ux)], fill=LINE)

    def connect(self, a, b, sa="bottom", sb="top", via=()):
        self.arrow([self.anchor(a, sa), *via, self.anchor(b, sb)])

    def save(self, name):
        target = ROOT/name
        self.im.save(target, optimize=True)
        print(f"{target} ({self.im.width} x {self.im.height})")

def overview():
    p = Figure(2230, "导轨上下料任务：Diffusion Policy 网络结构",
               "主图：最新关节版实验（10月19日 14:08）｜1帧观测｜30 Hz｜动作54维")
    p.group((40, 145, 1960, 715), "A  观测编码器：三路视觉特征 + 双臂双手状态")
    for i, x in enumerate((75, 550, 1025)):
        p.box(f"cam{i}", (x, 220, x+410, 320), f"camera_{i}：RGB",
              "3 × 480 × 640")
        p.box(f"enc{i}", (x, 365, x+410, 515), "独立 ResNet18 + GN",
              "SpatialSoftmax：32个点", "Linear → 64维视觉特征", size=25)
        p.connect(f"cam{i}", f"enc{i}")
    p.box("state", (1500, 220, 1925, 320), "实测机器人状态", "双臂14维 + 双手40维")
    p.box("norm", (1500, 365, 1925, 515), "状态归一化", "保留54维低维状态",
          "图像：整幅输入，无缩放裁剪", fill=GRAY, size=25)
    p.connect("state", "norm")
    p.box("concat", (390, 585, 1610, 685), "拼接为一帧观测特征 cond",
          "64 + 64 + 64 + 14 + 40 = 246维", fill=GREEN)
    for i in range(3):
        cx = p.anchor(f"enc{i}", "bottom")[0]
        p.connect(f"enc{i}", "concat", via=((cx, 550), (1000, 550)))
    p.connect("norm", "concat", via=((1712.5, 550), (1000, 550)))

    p.group((40, 755, 1960, 1630), "B  条件扩散去噪：Transformer 预测噪声，DDIM 更新动作")
    p.box("condition", (80, 835, 880, 990), "条件分支",
          "观测246 → 256；扩散步 k → 256",
          "拼成2个token + 位置嵌入 → 条件MLP", size=26)
    p.connect("concat", "condition", sb="left", via=((1000, 735), (55, 735), (55, 912.5)))
    p.box("noise", (1120, 835, 1920, 935), "含噪动作序列 x_k：16 × 54",
          "首次从高斯噪声采样", fill=GRAY)
    p.box("action_emb", (1120, 995, 1920, 1095), "动作嵌入",
          "逐步 Linear 54 → 256 + 动作位置嵌入")
    p.connect("noise", "action_emb")
    p.box("decoder", (470, 1165, 1530, 1320), "Transformer Decoder × 8层",
          "因果自注意力 → 条件交叉注意力 → 前馈网络",
          "隐藏维度256｜4个注意力头｜前馈维度1024")
    p.connect("condition", "decoder", sb="left", via=((480, 1125), (390, 1125), (390, 1242.5)))
    p.connect("action_emb", "decoder", via=((1520, 1130), (1000, 1130)))
    p.box("head", (470, 1375, 1530, 1475), "输出头：LayerNorm → Linear 256 → 54",
          "预测噪声 εθ：16 × 54")
    p.connect("decoder", "head")
    p.box("ddim", (470, 1525, 1530, 1625), "DDIM 调度器",
          "结合当前 x_k 和预测噪声更新动作；共迭代8次", fill=GREEN)
    p.connect("head", "ddim")
    p.connect("ddim", "action_emb", "right", "right", via=((1940, 1575), (1940, 1045)))
    p.text(1580, 1390, "更新 x_k 后", 27)
    p.text(1580, 1430, "再次嵌入并去噪", 27)
    p.text(85, 1410, "观测特征复用", 26)
    p.text(85, 1450, "扩散步 k 逐次变化", 26)

    p.group((40, 1680, 1960, 2055), "C  动作输出与执行")
    p.box("out", (470, 1760, 1530, 1870), "反归一化 → 完整16步动作 → 取前8步",
          "action_pred：16 × 54；action = action_pred[:, 0:8]", fill=GREEN)
    p.connect("ddim", "out")
    p.box("runtime", (470, 1920, 1530, 2020), "双臂双手的绝对关节目标",
          "左臂7 + 右臂7 + 左手20 + 右手20 → 执行器", fill=GRAY)
    p.connect("out", "runtime")
    p.text(45, 2090, "末端版 eef：双臂状态12维，观测特征244维，动作52维；使用相同的网络骨干。", 27)
    p.text(45, 2135, "10月19日 12:56 的末端版实验使用2帧观测，执行段为[:, 1:9]；当前关节版为[:, 0:8]。", 26)
    p.text(45, 2180, "图中省略批维 B。扩散迭代8次、预测16步动作、取出8步执行，是三个不同的参数。", 26)
    p.save("01-network-overview.png")

def transformer_detail():
    p = Figure(2070, "去噪骨干内部：TransformerForDiffusion",
               "当前关节版｜n_emb=256｜n_head=4｜n_layer=8｜n_cond_layers=0")
    p.box("obs", (70, 175, 650, 285), "观测特征 cond", "(B, 1, 246)")
    p.box("time", (730, 175, 1310, 285), "扩散时间步 k", "训练：0～99；推理：DDIM选取")
    p.box("noisy", (1390, 175, 1930, 285), "含噪动作 x_k", "(B, 16, 54)", fill=GRAY)
    p.box("obsproj", (70, 345, 650, 445), "Linear：246 → 256", "(B, 1, 256)")
    p.box("timeproj", (730, 345, 1310, 445), "正弦时间步嵌入", "(B, 1, 256)")
    p.box("actionproj", (1390, 345, 1930, 445), "Linear：54 → 256", "(B, 16, 256)")
    for a,b in (("obs","obsproj"), ("time","timeproj"), ("noisy","actionproj")):
        p.connect(a,b)
    p.box("condmlp", (70, 520, 1310, 680), "条件序列：[扩散时间token，观测token] + 可学习位置嵌入",
          "逐token条件MLP：256 → 1024 → Mish → 256",
          "memory：(B, 2, 256)；当前条件分支使用MLP", size=26)
    p.connect("obsproj", "condmlp", via=((360, 480), (690, 480)))
    p.connect("timeproj", "condmlp", via=((1020, 490), (690, 490)))
    p.box("actpos", (1390, 520, 1930, 680), "加可学习动作位置嵌入",
          "动作序列：(B, 16, 256)", "输入Dropout：0", size=25)
    p.connect("actionproj", "actpos")
    p.group((420, 790, 1580, 1410), "Transformer Decoder × 8层")
    p.box("self", (470, 870, 1530, 1000), "LayerNorm → 因果自注意力 → 残差相加",
          "Q / K / V 来自动作token；屏蔽后续动作位置")
    p.box("cross", (470, 1060, 1530, 1190), "LayerNorm → 条件交叉注意力 → 残差相加",
          "Q 来自动作token；K / V 来自条件 memory")
    p.box("ffn", (470, 1250, 1530, 1380), "LayerNorm → 前馈网络 → 残差相加",
          "Linear 256 → 1024 → GELU → Linear 1024 → 256")
    p.connect("actpos", "self", via=((1660, 745), (1000, 745)))
    p.connect("self", "cross")
    p.connect("cross", "ffn")
    p.box("memory", (70, 870, 350, 1190), "条件 memory",
          "2个token", "每个256维", "作为每一层", "交叉注意力的", "K 和 V", fill=GREEN, size=25)
    p.connect("condmlp", "memory", via=((690, 730), (210, 730)))
    p.connect("memory", "cross", "right", "left", via=((385, 1030), (385, 1125)))
    p.box("params", (1635, 870, 1930, 1190), "层内参数",
          "注意力头：4", "每头维度：64", "隐藏维度：256",
          "前馈维度：1024", "Dropout：0.3", fill=GRAY, size=24)
    p.box("finalhead", (470, 1480, 1530, 1580), "8层之后：LayerNorm → Linear 256 → 54",
          "每个动作位置输出一组54维噪声预测")
    p.connect("ffn", "finalhead")
    p.box("eps", (470, 1650, 1530, 1750), "预测噪声 εθ(x_k, k, cond)",
          "(B, 16, 54) → 交给DDIM调度器更新动作", fill=GREEN)
    p.connect("finalhead", "eps")
    p.group((40, 1820, 1960, 2020), "同一套网络如何训练")
    p.text(70, 1890, "示范动作归一化 → 随机选扩散步 k → 加高斯噪声得到 x_k → 网络预测 εθ。", 28)
    p.text(70, 1945, "损失：MSE(预测噪声, 实际添加的噪声)；观测编码器与Transformer一起更新。", 28)
    p.save("02-transformer-detail.png")

if __name__ == "__main__":
    overview()
    transformer_detail()
