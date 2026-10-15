"""导轨实现说明：同一规格生成 PNG 与带组件边界的 Mermaid 源码。"""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def node(key, box, title, *lines, group=None):
    return dict(key=key, box=box, title=title, lines=lines, group=group)


def group(key, box, title):
    return dict(key=key, box=box, title=title)


def edge(src, dst, sides=("bottom", "top"), via=(), label="", dashed=False):
    return dict(src=src, dst=dst, sides=sides, via=via, label=label, dashed=dashed)


def anchor(box, side):
    x, y, w, h = box
    return {"top": (x+w/2, y), "bottom": (x+w/2, y+h),
            "left": (x, y+h/2), "right": (x+w, y+h/2)}[side]


FIGURES = [
    dict(name="01-overview", size=(1800, 1710), title="导轨任务：从示范采集到模型闭环",
         subtitle="框线表示组件职责；文件名对应当前实现。BT = bimanual-teleop，DP = diffusion_policy。",
         groups=[
             group("data", (40, 140, 1720, 435), "A · 数据处理 〔bimanual-teleop项目；bimanual-teleop环境〕"),
             group("train", (40, 625, 1720, 575), "B · 数据接口与训练 〔DP；train.py + Task / Train YAML 组织调用〕"),
             group("offline", (40, 1370, 820, 245), "C · 离线评估 〔DP；读取留出数据〕"),
             group("real", (940, 1370, 820, 245), "D · 真机闭环评估 〔DP；本地设备接口快照〕"),
         ], nodes=[
             node("capture", (75, 220, 500, 250), "1  示范采集", "Quest / 手套 → 双臂双手", "三路 RGB + 实测状态 + 已提交指令", "cli/teleop_quest_tianji.py", "recording/recorder.py + sink.py", "recording/camera.py + spool.py", group="data"),
             node("finalize", (650, 220, 500, 250), "2  原始记录整理", "raw_spool → MP4 + raw.zarr", "检查视频、重建实测末端位姿", "recording/finalize.py", "recording/storage.py", "BT入口：scripts/finalize_recording.py", group="data"),
             node("convert", (1225, 220, 500, 250), "3  对齐、修复与转换", "30 Hz；图像匹配 / 状态插值", "动作保持过去目标；无效点切段", "recording/convert.py", "configs/recording_conversion.yaml", "BT入口：scripts/convert_recording.py", group="data"),
             node("zarr", (390, 510, 1020, 50), "输出：训练 zarr · 三路 RGB / 12维 EEF / 40维手关节 / 52维动作 / 边界及追溯", group="data"),
             node("dataset", (75, 715, 500, 265), "4  Dataset + DataLoader", "按原始演示划分训练 / 验证", "片段内采样：2步观测 + 16步动作", "原图480×640；训练来源拟合归一化", "dataset/tianji_rail_image_dataset.py", "model/common/normalizer.py", group="train"),
             node("workspace", (650, 715, 500, 265), "5  Workspace", "训练循环、反向传播、优化器", "学习率、EMA、验证、日志与保存", "workspace/", "train_diffusion_transformer_", "hybrid_workspace.py", group="train"),
             node("policy", (1225, 715, 500, 265), "6  Policy + 扩散模型", "观测编码 → Transformer 条件去噪", "训练：compute_loss", "推理：predict_action → 8步动作", "policy/diffusion_transformer_", "hybrid_image_policy.py", group="train"),
             node("ckpt", (550, 1060, 700, 110), "7  checkpoint", "模型 + EMA + 归一化器 + 配置 + 训练状态", "workspace/base_workspace.py", group="train"),
             node("off", (75, 1450, 750, 145), "eval.py → TianjiRailOfflineRunner", "env_runner/tianji_rail_offline_runner.py", "预测 vs 标签[1:9] → 位置mm / 旋转° / 手关节°", "输出：eval_log.json + offline_report.json", group="offline"),
             node("live", (975, 1450, 750, 145), "eval_tianji_rail.py → Env / Runner / Runtime", "对齐实测观测 → 推理 → 定时目标 → 设备反馈", "tianji_rail_env.py / tianji_rail_real_runner.py", "tianji_policy_runtime.py → 日志、视频、成功率", group="real"),
         ], edges=[
             edge("capture", "finalize", ("right", "left")), edge("finalize", "convert", ("right", "left")),
             edge("convert", "zarr", ("bottom", "right"), [(1475, 535)]),
             edge("zarr", "dataset", ("left", "left"), [(55, 535), (55, 847)]),
             edge("dataset", "workspace", ("right", "left")),
             edge("workspace", "policy", ("right", "left"), label="调用"),
             edge("workspace", "ckpt"),
             edge("ckpt", "off", ("bottom", "right"), [(900, 1280), (900, 1522)], label="加载 EMA / 模型"),
             edge("ckpt", "live", ("bottom", "left"), [(900, 1280), (900, 1522)]),
             edge("dataset", "off", ("left", "left"), [(20, 847), (20, 1522)], label="独立验证窗口", dashed=True),
         ], notes=["当前架构选用 Transformer；U-Net 是可替换的去噪骨干，未接入本导轨训练配置。",
                    "训练与评估软件已实现；正式训练完成及模型选择、真实设备闭环验收仍须分别确认。"]),
    dict(name="02-policy", size=(1800, 1550), title="Policy 内部：观测条件怎样变成动作",
         subtitle="同一个 Transformer 被训练和推理调用；扩散迭代次数与动作时间步数是两个概念。",
         groups=[
             group("obs", (40, 145, 1720, 300), "A · 观测编码 〔DiffusionTransformerHybridImagePolicy 内部的 obs_encoder〕"),
             group("loss", (40, 530, 820, 810), "B · compute_loss：学习预测噪声 〔训练〕"),
             group("sample", (940, 530, 820, 810), "C · predict_action / conditional_sample 〔推理〕"),
         ], nodes=[
             node("obs_in", (75, 235, 500, 180), "两步观测", "每步：三路RGB + EEF12维 + 手40维", "图像：(B,2,3,480,640)，每路各一份", "低维：(B,2,12) / (B,2,40)", group="obs"),
             node("encode", (650, 235, 500, 180), "归一化 + robomimic观测编码器", "图像特征与低维特征组合", "Policy 构造 obs_encoder；可训练", "common/robomimic_config_util.py", group="obs"),
             node("cond", (1225, 235, 500, 180), "条件特征 cond", "形状：(B,2,Do)", "Do = 观测编码器输出维度", "提供给下方两种 Transformer 调用", group="obs"),
             node("gt", (75, 615, 750, 120), "示范动作 (B,16,52)", "归一化 → 随机抽取扩散步 k", "正常动作标签来自过去已成功提交的目标", group="loss"),
             node("noise", (75, 805, 750, 120), "scheduler.add_noise", "干净动作 + 高斯噪声 → 含噪动作 a_k", "训练扩散步：100；k 表示噪声级别", group="loss"),
             node("train_model", (75, 995, 750, 130), "TransformerForDiffusion(a_k, k, cond)", "动作 token 自注意力 + 观测条件交叉注意力", "model/diffusion/transformer_for_diffusion.py", group="loss"),
             node("mse", (75, 1200, 750, 115), "预测噪声 vs 添加噪声 → MSE", "Workspace 反向传播；更新编码器和 Transformer", "prediction_type: epsilon", group="loss"),
             node("random", (975, 615, 750, 120), "随机动作噪声 (B,16,52)", "观测条件在本次动作生成中保持固定", "输入来自 Dataset 或真机 Env", group="sample"),
             node("denoise", (975, 805, 750, 205), "Transformer + DDIM 调度器：循环8次", "Transformer 预测噪声", "scheduler.step 更新含噪动作", "直到得到归一化动作序列", "Policy：conditional_sample", group="sample"),
             node("output", (975, 1080, 750, 235), "反归一化 → 16步完整预测 → 8步执行段", "action_pred：(B,16,52)", "action = action_pred[:,1:9]：(B,8,52)", "最新观测对应原始窗口索引1", "顺序：左臂6 / 右臂6 / 左手20 / 右手20", group="sample"),
         ], edges=[
             edge("obs_in", "encode", ("right", "left")), edge("encode", "cond", ("right", "left")),
             edge("gt", "noise"), edge("noise", "train_model"), edge("train_model", "mse"),
             edge("random", "denoise"), edge("denoise", "output"),
             edge("cond", "train_model", ("bottom", "right"), [(1475, 485), (900, 485), (900, 1060)], label="cond", dashed=True),
             edge("cond", "denoise", ("right", "right"), [(1780, 325), (1780, 907)], label="cond", dashed=True),
         ], notes=["Policy核心：policy/diffusion_transformer_hybrid_image_policy.py；归一化：model/common/normalizer.py。",
                    "U-Net替代实现：model/diffusion/conditional_unet1d.py；当前导轨配置使用上面的Transformer。"]),
    dict(name="03-real-loop", size=(1800, 1730), title="真机闭环：观测、推理与设备控制各负责什么",
         subtitle="30 Hz 是观测与动作的时间网格；推理独立运行，双臂200 Hz、左右手各120 Hz推进目标。",
         groups=[
             group("entry", (40, 140, 1720, 215), "A · 评估入口与模型加载"),
             group("env", (40, 410, 820, 475), "B · TianjiRailEnv 〔实时观测与设备生命周期〕"),
             group("runner", (940, 410, 820, 475), "C · TianjiRailRealRunner 〔试验管理与独立推理〕"),
             group("runtime", (940, 990, 820, 360), "D · TianjiPolicyRuntime 〔定时队列、校验与独立控制〕"),
             group("device", (40, 990, 820, 360), "E · BT 设备接口 〔运动学、伺服、驱动及SDK〕"),
             group("record", (40, 1410, 1720, 225), "F · 试验记录与汇总 〔Runner 中的 TrialRecorder〕"),
         ], nodes=[
             node("entry_n", (75, 215, 1650, 115), "eval_tianji_rail.py + config/tianji_rail_real_eval.yaml", "读取checkpoint的配置 / EMA / 归一化器；检查模型与设备配置，创建Env和Runner", "--check-only：模拟输入检查；默认：真实观测影子模式；--motion + 手动开始：运动试验", group="entry"),
             node("history", (75, 495, 750, 145), "七路带时间戳历史缓存", "三路RGB + 左右臂实际关节FK + 左右手关节", "real_world/tianji_rail_env.py", "复用BT：recording/camera.py、devices/tianji/model.py", group="env"),
             node("align", (75, 705, 750, 155), "get_obs → align_observation", "两步30 Hz历史：图像最近匹配；状态线性 / SLERP", "检查数据年龄、图像偏差、三路跨度、状态缺口", "preprocess_observation：与checkpoint图像设置一致", group="env"),
             node("infer", (975, 495, 750, 145), "_infer → policy.predict_action", "独立推理线程，读对齐观测；输出8步52维目标", "env_runner/tianji_rail_real_runner.py", "按replanning_steps检查新观测；不保证推理30 Hz", group="runner"),
             node("trial", (975, 705, 750, 155), "run / _begin / _end", "s开始；p接管；y成功；n失败；q退出", "操作员定义结果，记录超时、设备中止与接管", "切换试验时阻止旧推理结果发布", group="runner"),
             node("schedule", (975, 1070, 750, 250), "exec_actions → runtime.submit → 控制线程", "名义时刻 t0 + i/30 秒；丢弃已过期动作", "校验工作空间 / 跳变 / 速度；替换未来队列", "双臂线程200 Hz；左手、右手各120 Hz", "real_world/tianji_policy_runtime.py", "故障或覆盖耗尽：清队列并请求暂停", group="runtime"),
             node("devices", (75, 1070, 750, 250), "绝对EEF / 手关节目标 → 设备执行", "control/arm/cartesian.py + arm/servo.py", "devices/tianji/model.py：FK / IK", "devices/tianji/driver.py + sdk.py", "devices/wuji/adapter.py：左右手驱动", "实际图像 / 状态反馈返回Env", group="device"),
             node("log", (75, 1490, 1650, 120), "TrialRecorder + summarize_trials 〔env_runner/tianji_rail_real_runner.py〕", "逐试验：trial.json / 三路MP4 / events.jsonl / timing.json；会话：session.json / summary.json", "成功率 = 成功运动试验 / 所有已开始运动试验；影子试验不进入分母", group="record"),
         ], edges=[
             edge("entry_n", "history", ("bottom", "right"), [(900, 385), (900, 567)]),
             edge("entry_n", "infer", ("bottom", "left"), [(900, 385), (900, 567)]),
             edge("history", "align"),
             edge("align", "infer", ("right", "left"), [(900, 782), (900, 567)], label="obs + t0"),
             edge("infer", "schedule", ("right", "right"), [(1780, 567), (1780, 1195)], label="预测动作"),
             edge("trial", "schedule", ("bottom", "top"), [(1350, 935), (1660, 935), (1660, 1040), (1350, 1040)], label="开始 / 暂停"),
             edge("schedule", "devices", ("left", "right"), label="设备目标"),
             edge("devices", "history", ("left", "left"), [(20, 1195), (20, 567)], label="实际反馈闭环"),
             edge("devices", "log", via=[(450, 1380), (900, 1380)], dashed=True),
             edge("schedule", "log", via=[(1350, 1380), (900, 1380)], dashed=True),
         ], notes=["真机评估使用tianji-rail-diffusion-policy环境和本地设备接口快照。虚线表示记录事件。",
                    "软件已实现；现场观测、停止、短时闭环与成功率仍待验收。统一时间域不等于延迟已标定。"]),
]


FIGURES.extend([
    dict(name="04-workspace-setup", size=(1800, 1620), title="Workspace：组织训练对象与运行资源",
         subtitle="核心：workspace/train_diffusion_transformer_hybrid_workspace.py；保存加载由BaseWorkspace提供。",
         groups=[
             group("entry", (40, 145, 1720, 175), "A · train.py：解析配置，按cfg._target_创建Workspace"),
             group("state", (40, 385, 1720, 300), "B · __init__：创建Workspace持有的训练状态"),
             group("resources", (40, 775, 1720, 660), "C · run()：检查恢复点，创建本次运行的资源，再进入训练循环"),
         ], nodes=[
             node("entry", (75, 220, 1650, 75), "Task + Train YAML + 命令行覆盖 → cfg → Workspace(cfg) → workspace.run()", group="entry"),
             node("main", (75, 475, 500, 185), "self.model：主Policy", "hydra.instantiate(cfg.policy)", "内部含编码器 / Transformer / 调度器", "训练梯度更新作用于这个模型", group="state"),
             node("ema", (650, 475, 500, 185), "self.ema_model：EMA Policy", "deepcopy(self.model)", "use_ema=true时创建", "由EMA更新器平滑更新参数", group="state"),
             node("optim", (1225, 475, 500, 185), "self.optimizer + 计数器", "self.model.get_optimizer(...)", "global_step=0；epoch=0", "优化器持有主Policy的可训练参数", group="state"),
             node("resume", (75, 870, 500, 175), "1  检查是否恢复训练", "resume=true且latest.ckpt存在", "load_checkpoint → 主 / EMA / 优化器", "恢复epoch、global_step", group="resources"),
             node("data", (650, 870, 500, 175), "2  Dataset + 两个DataLoader", "cfg.task.dataset实例化Dataset", "训练loader + 验证loader", "get_validation_dataset()获取验证集", group="resources"),
             node("norm", (1225, 870, 500, 175), "3  归一化器", "dataset.get_normalizer()", "仅训练来源拟合", "set_normalizer复制给主模型与EMA", group="resources"),
             node("lr", (75, 1130, 500, 180), "4  学习率调度器 + EMA更新器", "warmup / cosine；关联optimizer", "EMA更新器关联self.ema_model", "恢复时按epoch重建调度进度", group="resources"),
             node("runner", (650, 1130, 500, 180), "5  TianjiRailOfflineRunner", "cfg.task.env_runner实例化", "output_dir来自Workspace", "本导轨配置执行留出动作评估", group="resources"),
             node("logging", (1225, 1130, 500, 180), "6  日志、TopK与设备迁移", "logs.json.txt + 日志接口", "TopKCheckpointManager", "模型 / EMA / optimizer迁移到GPU", group="resources"),
             node("loop", (450, 1360, 900, 55), "进入epoch循环：从self.epoch运行到num_epochs总目标", group="resources"),
         ], edges=[
             edge("entry", "main", ("bottom", "left"), [(900, 350), (20, 350), (20, 567)]),
             edge("main", "ema", ("right", "left")),
             edge("main", "optim", ("top", "top"), [(325, 450), (1475, 450)]),
             edge("main", "resume", ("bottom", "left"), [(325, 725), (20, 725), (20, 957)]),
             edge("resume", "data", ("right", "left")), edge("data", "norm", ("right", "left")),
             edge("norm", "lr", ("bottom", "top"), [(1475, 1090), (325, 1090)]),
             edge("lr", "runner", ("right", "left")), edge("runner", "logging", ("right", "left")),
             edge("logging", "loop", ("bottom", "right"), [(1475, 1387)]),
         ], notes=["__init__持有的model、ema_model、optimizer及epoch/global_step进入checkpoint；cfg也保存。",
                    "Dataset、DataLoader、Runner、LR调度器、EMA更新器、TopK管理器在run()中重新创建。"]),
    dict(name="05-workspace-epoch", size=(1800, 1620), title="Workspace：一个epoch如何训练、评估与保存",
         subtitle="左侧为训练batch循环；右侧为epoch末尾任务。每项按配置间隔触发，然后继续下一个epoch。",
         groups=[
             group("train", (40, 145, 820, 1170), "A · 主Policy训练与参数更新"),
             group("eval", (940, 145, 820, 1170), "B · 评估、checkpoint与日志"),
         ], nodes=[
             node("batch", (75, 240, 750, 120), "DataLoader → batch → GPU", "三路图像各(B,2,3,480,640)", "EEF(B,2,12) / 手(B,2,40) / 动作(B,16,52)", group="train"),
             node("loss", (75, 430, 750, 120), "self.model.compute_loss(batch)", "Policy负责归一化、加噪、预测噪声与MSE", "Workspace检查损失有限；按累积组缩放loss", group="train"),
             node("backward", (75, 620, 750, 120), "loss.backward()", "计算并累积主Policy的梯度", "编码器与Transformer一起学习", group="train"),
             node("update", (75, 810, 750, 150), "满足累积步数，或到本epoch最后一批", "optimizer.step → zero_grad → lr_scheduler.step", "ema.step(self.model)：同步平滑更新EMA", "默认累积步数1，每个batch更新一次", group="train"),
             node("batchlog", (75, 1045, 750, 150), "记录batch损失、学习率与global_step", "还有batch：继续读取", "本epoch结束：train_loss取batch均值", "global_step记录已完成batch数", group="train"),
             node("select", (975, 240, 750, 120), "选择生成动作的策略并policy.eval()", "use_ema=true：self.ema_model", "否则：self.model", group="eval"),
             node("offline", (975, 430, 750, 120), "rollout_every：env_runner.run(policy)", "本任务是OfflineRunner，比较执行段与标签", "输出位置 / 旋转 / 手关节误差；空验证则跳过", group="eval"),
             node("val", (975, 620, 750, 120), "val_every：计算val_loss", "torch.no_grad + self.model.compute_loss", "使用主模型计算验证噪声MSE；不反向传播", group="eval"),
             node("sample", (975, 810, 750, 150), "sample_every：训练batch动作采样检查", "policy.predict_action → action_pred(B,16,52)", "完整预测与训练标签比较MSE", "启用EMA时，这一步使用EMA模型", group="eval"),
             node("save", (975, 1045, 750, 245), "checkpoint_every：保存latest与候选TopK", "保存前epoch前进到下一轮，global_step加1", "BaseWorkspace保存cfg / state_dicts / pickles", "policy.train()；合并本轮最后一批与评估日志", "继续下一epoch；结束时等待保存线程", group="eval"),
         ], edges=[
             edge("batch", "loss"), edge("loss", "backward"), edge("backward", "update"), edge("update", "batchlog"),
             edge("batchlog", "batch", ("left", "left"), [(20, 1120), (20, 300)], label="下一batch"),
             edge("batchlog", "select", ("right", "left"), [(900, 1120), (900, 300)], label="epoch训练结束"),
             edge("select", "offline"), edge("offline", "val"), edge("val", "sample"), edge("sample", "save"),
             edge("save", "batch", ("right", "top"), [(1780, 1167), (1780, 1340), (5, 1340), (5, 205), (450, 205)], dashed=True),
         ], notes=["间隔条件使用epoch从0开始的取模；第1轮结束也触发评估与保存，命令行可覆盖间隔。",
                    "保存模型、优化器、归一化器和计数器；恢复从最近保存的epoch边界继续，不从中断batch精确续跑。"]),
])


def render(fig):
    image = Image.new("RGB", fig["size"], "#FFFFFF")
    draw = ImageDraw.Draw(image)
    def text(x, y, value, size=26, bold=False, fill="#263445"):
        font = ImageFont.truetype(BOLD if bold else FONT, size)
        draw.text((x, y), value, font=font, fill=fill)
    text(40, 28, fig["title"], 42, True)
    text(40, 88, fig["subtitle"], 25)
    for g in fig["groups"]:
        x, y, w, h = g["box"]
        draw.rectangle((x, y, x+w, y+h), outline="#7D8792", width=2)
        text(x+20, y+16, g["title"], 28, True)
    nodes = {n["key"]: n for n in fig["nodes"]}
    for e in fig["edges"]:
        points = [anchor(nodes[e["src"]]["box"], e["sides"][0]), *e["via"],
                  anchor(nodes[e["dst"]]["box"], e["sides"][1])]
        for p, q in zip(points, points[1:]):
            length = math.dist(p, q)
            if not length:
                continue
            if e["dashed"]:
                for k in range(0, int(length), 18):
                    u, v = k/length, min(k+10, length)/length
                    draw.line((p[0]+(q[0]-p[0])*u, p[1]+(q[1]-p[1])*u,
                               p[0]+(q[0]-p[0])*v, p[1]+(q[1]-p[1])*v), fill="#748091", width=2)
            else:
                draw.line((p, q), fill="#43566B", width=3)
        p, q = points[-2:]
        length = math.dist(p, q)
        if length:
            ux, uy = (q[0]-p[0])/length, (q[1]-p[1])/length
            draw.polygon((q, (q[0]-13*ux+6*uy, q[1]-13*uy-6*ux),
                          (q[0]-13*ux-6*uy, q[1]-13*uy+6*ux)), fill="#43566B")
        if e["label"]:
            _, p, q = max((math.dist(p, q), p, q) for p, q in zip(points, points[1:]))
            font = ImageFont.truetype(FONT, 22)
            width = draw.textlength(e["label"], font=font)
            if abs(q[0]-p[0]) >= abs(q[1]-p[1]):
                x, y = (p[0]+q[0]-width)/2, (p[1]+q[1])/2-29
            else:
                x, y = (p[0]+q[0])/2+5, (p[1]+q[1])/2-15
                if x > fig["size"][0]-40:
                    x -= 42
                # 将长边的标签竖排，避免侵入相邻组件。
                tag = Image.new("RGB", (int(width)+8, 30), "white")
                ImageDraw.Draw(tag).text((4, 0), e["label"], font=font, fill="#43566B")
                tag = tag.rotate(90, expand=True)
                image.paste(tag, (int(x), int((p[1]+q[1]-tag.height)/2)))
                continue
            draw.rectangle((x-3, y, x+width+3, y+30), fill="white")
            text(x, y, e["label"], 22)
    for n in fig["nodes"]:
        x, y, w, h = n["box"]
        draw.rectangle((x, y, x+w, y+h), fill="#EEF3F7", outline="#708194", width=2)
        lines = [n["title"], *n["lines"]]
        size = 27
        while size >= 19:
            if all(draw.textlength(line, font=ImageFont.truetype(BOLD if i == 0 else FONT, size)) < w-30
                   for i, line in enumerate(lines)) and (size+10)*len(lines) < h-8:
                break
            size -= 1
        if size < 19:
            raise ValueError(f"节点过密：{n['key']}")
        for i, line in enumerate(lines):
            text(x+15, y+(h-(size+10)*len(lines))/2+i*(size+10), line, size, i == 0)
    for i, note in enumerate(fig["notes"]):
        text(40, fig["size"][1]-80+i*34, note, 24)
    image.save(ROOT / (fig["name"]+".png"))
    lines = ["%% 节点、组件边界与PNG由同一份规格生成；PNG布局单独指定。", "flowchart TD"]
    for g in fig["groups"]:
        lines += [f'  subgraph {g["key"]}["{g["title"]}"]']
        for n in fig["nodes"]:
            if n["group"] == g["key"]:
                label = "<br/>".join([n["title"], *n["lines"]]).replace('"', '&quot;')
                lines += [f'    {n["key"]}["{label}"]']
        lines += ["  end"]
    for e in fig["edges"]:
        arrow = "-.->" if e["dashed"] else "-->"
        label = f'|"{e["label"]}"|' if e["label"] else ""
        lines += [f'  {e["src"]} {arrow}{label} {e["dst"]}']
    (ROOT / (fig["name"]+".mmd")).write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    for fig in FIGURES:
        render(fig)
    print(f"已生成{len(FIGURES)}张PNG及Mermaid源文件：{ROOT}")
