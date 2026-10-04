# 汽车导轨任务分阶段构建计划

版本：v1.7（阶段 0—5、7 已验收，阶段 6 待独立真实演示）  
更新日期：2026-10-03  
状态：转换修复、Dataset、任务配置、OfflineRunner、GPU 短训练和 checkpoint 重载已验收，33 项测试通过；正式留出训练需补充独立真实演示。

## 目标与运行环境

完成 **缺口修复与 zarr 生成 → Dataset → Transformer GPU 训练 → checkpoint 重载 → 离线评估 → 快速使用说明**。

任务为固定目标区域的双臂双手导轨抓取与放置。使用三路 RGB、双臂末端位姿和双手关节，预测 52 维末端与手关节目标。当前范围止于训练和离线评估；真实放置成功率需要后续实机闭环验证。

NVIDIA RTX PRO 6000 Blackwell 已在沙箱外完成真实数据及合成数据的 GPU 短训练；`robodiff` 环境使用 PyTorch 2.7.1 / CUDA 12.9，设备为 `cuda:0`。采集侧转换与测试使用 Conda `bimanual-teleop`；模型、Dataset 和 Runner 使用 `robodiff`。测试采用 unittest，没有新增 pytest 依赖。GPU 在当前沙箱内不可见，实际 GPU 验收在已授权的沙箱外执行。

## 当前基线与阶段状态

实施顺序、缺口修复流程和模块连接见 [实施图示说明](/home/jiyuchen/project/WYS/diffusion_policy/wys/2026-10-03-rail-build-implementation/implementation.md)。

多相机、多模态缺口的论文与作者实现依据见 [文献核查说明](/home/jiyuchen/project/WYS/diffusion_policy/wys/2026-10-03-rail-build-implementation/multimodal-gap-literature.md)。时间对齐、图像复用、低维插值和失效标签分段有代表性工作支持；本任务的具体阈值及暂停条件合并属于工程选择，不作为论文统一标准。

- 已完成：RGB 默认不依赖深度匹配；原始演示边界和连续片段边界分别保存；现有采集侧 Dataset 兼容两种边界。
- 当前严格转换数据：1 条原始演示、1882 个有效帧、7 个连续片段。拆片段不会增加独立演示数。
- 修复版真实数据：1 条原始演示、1887 帧、2 个连续片段，16 步窗口从 1786 增至 1857；唯一剩余内部边界为已复核保留的暂停接缝。
- 转换、Dataset、Transformer GPU 短训练、checkpoint 重载及空／非空验证接口均已验收。合成双演示数据仅用于接口测试，不计入真实演示数量。
- 转换、短训练、恢复及评估命令已执行；完整证据见 [构建验收记录](/home/jiyuchen/project/WYS/diffusion_policy/wys/2026-10-03-rail-build-implementation/build-validation.md)。

| 阶段 | 状态 | 构建内容 | 产物与通过条件 |
| --- | --- | --- | --- |
| 0．确认基线 | 已验收 | 核对字段、单位、来源、缺口、暂停位置和运行依赖 | baseline.json 及接缝图像已保存；主相机第 1341 帧前的暂停明确复核保留边界 |
| 1．改进转换器 | 测试通过 | 配置、30 Hz 网格、RGB 复用、状态插值、暂停检查与条件合并 | 转换相关 24 项测试通过，旧严格模式兼容，配置有中文注释 |
| 2．生成修复版 zarr | 已验收 | 从原始记录重新转换，比较旧版数据，抽查修复图像和低维轨迹 | 1887 帧、2 片段；13 个图像逐像素来源检查、低维数值与 30 Hz 间隔核验通过 |
| 3．实现 Dataset | 已验收 | 按原始演示划分、按修复后片段采样；配置 RGB 缩放开关、尺寸及插值方式，拟合训练集归一化参数 | 来源划分、短片段、训练集归一化、图像配置及旧格式测试通过，并用于实际 GPU 训练 |
| 4．配置与 GPU 冒烟 | 已验收 | 接入 Transformer Policy、Workspace 及专用优化器；配置裁剪及评估行为；短训练，保存并重载 checkpoint | 2轮×2批 GPU 训练完成；3组优化器参数，模型／EMA／归一化器重载一致，输出 `(2,8,52)` |
| 5．离线评估 | 已验收 | 新增 OfflineRunner，分别统计左右臂和左右手误差 | 指标与对齐测试通过；eval.py 在真实单演示下跳过验证，在合成双演示下输出完整报告 |
| 6．多演示正式训练 | 待独立真实演示 | 补充独立真实演示，冻结修复参数和来源划分，GPU 训练及留出评估 | 当前录制目录仅一条完整 episode；未启动正式训练，不以合成数据代替 |
| 7．快速说明验收 | 已验收 | 随阶段补齐并验证转换、质检、训练、恢复和评估命令 | 快速说明已精简并指向实际配置与输出；恢复命令确认不重复已完成轮次 |

阶段 1—5 可使用合成多演示数据验证接口。阶段 6 必须使用真实独立演示，不能把当前七个片段当成七条演示。暂停区间归属不明确的数据先隔离再训练。

## zarr 缺口修复

### 配置与入口

已新增 [通用转换配置](/home/jiyuchen/project/WYS/bimanual-teleop/configs/recording_conversion.yaml)及 CLI 参数 `--conversion-config`；Python `convert_recordings` 支持可选转换配置路径或映射，不传时保留严格行为。当前记录使用 [任务转换配置](/home/jiyuchen/project/WYS/diffusion_policy/docs/tianji-rail/recording_conversion.yaml)，其中保存了本次暂停的明确复核结论；新记录使用通用配置并重新复核。

| 参数 | 默认值 | 注释需要说明的含义 |
| --- | --- | --- |
| `mode` | `repair` | 固定时间网格与短缺口修复 |
| `target_fps` | `30` | 输出时间网格频率 |
| `max_missing_camera_frames` | `2` | 每路相机最多连续修复两个未正常匹配的采样点；30 Hz 下两帧缺失通常对应约 100 ms 的真实帧间隔 |
| `camera_match_tolerance_ms` | `20` | 正常图像匹配的最大时间差 |
| `max_state_interp_gap_ms` | `100` | 状态插值需要的两端实测数据最大间隔 |
| `max_command_age_ms` | `50` | 最近一次成功提交指令的最大允许年龄 |
| `pause_policy` | `checked_compress` | 暂停区间独立对齐；仅对通过连续性复核的接缝压缩时间并合并。未复核或未通过时保留边界 |
| `pause_offset_tolerance_ms` | `100` | 来源时间与压缩时间偏移突变的识别阈值；应高于时钟抖动、低于目标暂停长度，三路归属不一致时报错 |
| `pause_reviews` | `{}` | 按来源演示和暂停后主相机原始帧索引记录 merge 和 reason；未复核保持边界，结论不可直接复用到新记录 |

配置写明单位和默认值；提高相机修复上限会延长旧画面复用时间，不会恢复实际缺失图像。相机、状态与命令阈值分别控制各自的数据质量，不能互相替代。

### 时间轴与修复规则

1. 先确定同一 episode 内的持续录制区间。在每个区间内以第一帧主相机时间为锚点建立 30 Hz 网格，时间戳由索引直接计算到纳秒，避免逐步累加产生漂移。流匹配和插值不跨 episode、原始暂停或明确的时间戳异常；暂停两侧在独立对齐后按下述规则检查是否可拼接。
2. RGB 优先采用与网格时刻相差不超过 20 ms 的最近真实帧。内部未匹配区间长度不超过配置上限时，复用该时刻之前最近的真实图像，且图像年龄不得超过 `(max_missing_camera_frames+1)/target_fps` 秒。超过上限的整段未匹配区间不修复；不在超长缺口起点先保留两帧伪连续数据。
3. 每路相机分别判断缺口；填补必须处于有效真实数据之间，不延伸首尾覆盖范围。三路图像、全部必需状态和动作同时可用，该网格时刻才能发布。
4. 状态使用原始流的直接相邻有效测量插值；位置、关节和力线性插值，四元数姿态用 SLERP。两端间隔超过配置值、数值无效或跨暂停时拒绝，不跳过异常测量寻找更远端点，不向观测范围外推。
5. 动作仍采用网格时刻之前最近一次成功提交的指令，检查指令年龄和有效性。动作不做线性插值，也不移动标签索引。
6. 首尾缺少覆盖时裁剪；长缺口、状态异常、指令过期及未通过连续性复核的暂停仍切段。相邻有效网格点属于同一连续片段，成功填补的内部短缺口不再产生边界。

历史暂停通过相机 `source_time_ms` 与压缩后 `time_ns` 的偏移突变识别，结合可用事件核对。三路相机的暂停前后归属共同确定允许匹配和插值的区间；归属不明确的边缘数据裁剪。时间倒退或时钟重置不得当成普通缺帧处理。

### 脱合／接合暂停的最终策略

采用“先识别和复核，满足条件则压缩暂停并拼接”的策略，不因遥操作参考重建而一律分段。保存的机械臂／手关节目标为绝对关节角；当前 EEF 模式的机械臂动作是各自基座系中的绝对末端位姿。重建操作者映射参考不会改变这些标签的坐标语义，但目标值可能重设，需与实测状态分开检查。

- 普通短采样缺失：仍按最多两帧的配置补齐，并在有效性检查通过后保持连续。
- 无完整暂停记录的历史数据：检查暂停两侧的实测末端、机械臂关节、手关节及绝对目标变化，并复核三路图像中的导轨、抓持和场景衔接；确认没有回位、故障或人工改变物体状态造成的中断。首版不凭端点数值接近自动认定整个暂停静止；阶段 0 为各接缝保存明确的合并／保留边界决定和依据，未经复核默认保留边界。
- 通过复核的接缝：保留两侧各自已对齐的真实来源样本，移除暂停等待时长，并将后侧首个样本安排在前侧末个样本的下一训练步，使训练时间保持 30 Hz；删除这一接缝的 `segment_ends` 边界。不是为数秒暂停复制数十帧图像，也不在线性插值中跨越没有原始记录的暂停。真实已提交目标若发生重设，保留这个变化，不生成中间控制指令。
- 未通过复核或证据不足：保留片段边界。若未来暂停期间有完整有效记录，可按原始状态和指令重采样，但不能据目前只有两端的数据推定期间轨迹。

保留逐样本来源时间和训练时间映射、暂停时长、拼接位置、复核结论及目标切换信息。指令新鲜度在原始来源区间内判断，不能通过压缩时间把过期指令变成有效指令。所有接缝均通过时，同一 episode 可成为一个连续训练片段；否则仍是一条原始演示下的多个连续片段。

### 格式、追溯与发布

- 保持 schema v2 的边界约定：`meta/episode_ends` 为原始演示，`meta/segment_ends` 为修复后连续训练区间；现有字段及 52 维动作布局不变。
- 在元数据中新增逐网格点图像来源帧索引、匹配图像时间、复用标记和扩展状态插值标记；保存修复参数、网格频率、暂停识别／拼接决定、来源到训练时间的映射及剩余分段原因。`timestamp` 属性更新为实际的训练网格时间语义。
- 正常 RGB 仍来自真实视频帧，修复标记明确区分正常匹配与超出正常容差的复用；这些字段默认不进入模型观测。
- 从原始视频和 `raw.zarr` 生成 `/home/jiyuchen/project/WYS/diffusion_policy/data/tianji/episodes_eef_rgb_repaired.zarr`。保留旧数据，不覆盖已有输出；维持现有转换器的临时生成和完成后发布机制。
- 报告比较输出帧数、片段数、有效窗口数、各路修复比例、最长图像年龄和未修复原因。固定时间网格会重新对齐状态与动作，不能要求与旧版逐行相同；应按来源时间与原始流验证。

## Transformer 训练接入

### Policy、Workspace 与训练配置

采用 [DiffusionTransformerHybridImagePolicy](/home/jiyuchen/project/WYS/diffusion_policy/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py) 与 [TrainDiffusionTransformerHybridWorkspace](/home/jiyuchen/project/WYS/diffusion_policy/diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py)，以 [实机 Transformer Hybrid 配置](/home/jiyuchen/project/WYS/diffusion_policy/diffusion_policy/config/train_diffusion_transformer_real_hybrid_workspace.yaml) 为模板。

已实现：

- `task/tianji_rail_image.yaml`：Dataset、OfflineRunner、观测和动作定义。
- `train_diffusion_transformer_tianji_rail_workspace.yaml`：Transformer 专用训练配置。
- `TianjiRailImageDataset` 及 `TianjiRailOfflineRunner`，分别位于现有 Dataset、env_runner 目录；语义测试集中在 `tests/test_tianji_rail.py`，另有 Workspace 累积与恢复回归测试。

配置中的 `_target_` 分别为：

```yaml
_target_: diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace.TrainDiffusionTransformerHybridWorkspace
policy:
  _target_: diffusion_policy.policy.diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy
```

Transformer 参数默认采用参考配置起始值：`n_layer=8`、`n_cond_layers=0`、`n_head=4`、`n_emb=256`、`p_drop_emb=0.0`、`p_drop_attn=0.3`、`causal_attn=true`、`time_as_cond=true`、`obs_as_cond=true`。参数可在配置文件修改；显式设置 `pred_action_steps_only=false`，预测完整 16 步动作，再截取执行段。

使用 Transformer Policy 的 `get_optimizer` 接口：`transformer_weight_decay=1e-3`、`obs_encoder_weight_decay=1e-6`、`learning_rate=1e-4`、`betas=[0.9,0.95]`。不带入 U-Net 架构参数或其优化器实例化参数。正式训练保留 500 步学习率预热、600 个 epoch、EMA、100 个扩散训练时间步和 8 次 DDIM 推理步骤。

### 图像处理配置与参数注释

图像处理参数已接入实际 [Task YAML](/home/jiyuchen/project/WYS/diffusion_policy/diffusion_policy/config/task/tianji_rail_image.yaml)及 [Train YAML](/home/jiyuchen/project/WYS/diffusion_policy/diffusion_policy/config/train_diffusion_transformer_tianji_rail_workspace.yaml)，均有中文注释；[图像参数摘录](/home/jiyuchen/project/WYS/diffusion_policy/docs/tianji-rail/image_preprocessing.example.yaml)便于对照，不是完整可执行训练配置。

| 配置项 | 默认值 | 功能与修改建议 |
| --- | --- | --- |
| `task.image_shape` | `[3,240,320]` | Dataset 输出通道、高、宽；显存不足可降低高宽，须检查操作细节和原图宽高比。Dataset、Runner 与 RGB shape_meta 共用此尺寸 |
| `task.dataset.resize_enabled` | `true` | 启用或关闭缩放；关闭时保留原图并将输出尺寸配置为原图尺寸。首版三路输出尺寸共用，原图尺寸不一致时需开启缩放 |
| `task.dataset.resize_interpolation` | `area` | 支持 area、bilinear、nearest；缩小优先 area，放大可用 bilinear，RGB 通常不推荐 nearest。关闭缩放时不执行 |
| `policy.crop_shape` | `null` | null 关闭裁剪；[高,宽] 启用训练随机裁剪，例如 [216,288]。当前裁剪实现要求高宽分别严格小于输入，先检查是否裁掉导轨或手 |
| `policy.eval_fixed_crop` | `true` | 启用裁剪时，验证和离线评估使用固定中心裁剪；建议保持 true 以便复现，关闭裁剪时此项不产生裁剪 |
| `policy.obs_encoder_group_norm` | `true` | 使用 GroupNorm；小 batch 且使用 EMA 时建议保持 true |

处理顺序为 Dataset 可选缩放 → HWC uint8 转 CHW float32 / 255 → Policy 可选裁剪。尺寸来自配置，240×320 仅为默认值；裁剪只在视觉编码器执行一次。训练、验证和 Runner 共用确定性的图像预处理，随机裁剪仅按 Policy 的训练／评估模式执行。启动时校验参数类型、允许值、正整数尺寸、裁剪范围以及实际图像与 shape_meta 是否一致，错误直接说明配置项和原因。

转换、任务和训练 YAML 中的每个可调参数都要就近写中文注释：说明作用、单位／尺寸顺序、默认值或关闭方式、修改建议及关联参数。缺口阈值、来源划分、时间窗口、模型结构、优化器、batch、worker、学习率、扩散步骤、checkpoint 等也遵守此要求，不只注释图像项。注释保存在实际配置文件中，不能仅由 README 或本计划替代。评估重用 checkpoint 保存的配置，改变输入尺寸或预处理须创建并验收新实验。

### Dataset 与样本契约

- 继承 `BaseImageDataset`，实现样本读取、长度、验证集、归一化和动作读取接口；样本返回 `{"obs": ..., "action": ...}`。
- 三路 RGB 按需读取，按图像配置选择是否缩放，默认输出 `240×320`，HWC uint8 转 CHW float32 / 255；避免将全部原图解压进内存。状态选择 `robot_eef_pose` 和 `hand_joint`，不自动加入深度、力或机械臂关节观测。
- 原始末端为米和旋转向量弧度，手关节为弧度。动作顺序为左臂、右臂、左手、右手，共 52 维，首版不改为 rotation_6d。
- `horizon=16`、`n_obs_steps=2`、`n_action_steps=8`、`n_latency_steps=0`；`crop_shape=null`、视觉编码器使用 GroupNorm。
- 按原始来源划分训练和验证；按修复后的连续片段采样，并兼容旧格式边界和来源分组。Dataset 不复制首尾凑窗口，剩余不足 16 帧的片段不产生样本。转换器修复缺口与 Dataset 端点补齐是两个不同环节。
- 只用训练来源拟合状态和动作归一化参数；验证和独立评估使用训练保存的参数。缺口修复标记保留用于质检，不传入 Policy 的训练 batch。

### 训练与评估默认值

- 使用 GPU `cuda:0`，`seed=42`，关闭在线日志。冒烟 batch size 2、零 worker、关闭 persistent_workers；两个 epoch，每个最多两个训练 batch，学习率预热为 0，checkpoint 和采样检查每个 epoch 执行。Workspace 启动会实例化 Runner，因此阶段 4 先提供 OfflineRunner 的可实例化接口与空验证集跳过路径，阶段 5 再补全动作预测和指标计算。
- 正式训练 batch size 8，初始仍使用零 worker 以保持读取方式一致；学习率预热及其他训练参数采用上面的 Transformer 默认值。新实验使用独立输出目录；恢复训练只使用同一配置和数据版本。
- 单演示 `val_ratio=0`，checkpoint 自动监控 `train_loss`；多演示 `val_ratio=0.1`，至少各保留一条训练和验证演示，自动监控 `val_loss`。文件只按 epoch 命名，不需随监控项手动修改。单来源请求留出验证时报错，空验证窗口不能用于监控 val_loss。
- Runner 继承 `BaseImageRunner`，提供 `run(policy) -> dict`；与训练使用相同来源划分和预处理，不重新拟合归一化器。按每个窗口执行预测，比较输出 `action` 与标签 `[1:9]`。
- 报告左右臂位置欧氏误差（毫米）、SO(3) 旋转测地误差（度）、左右手关节平均绝对误差（度），以及窗口数、执行步数和来源。汇总按有效预测步统计，并另给逐演示统计。
- 空验证集返回明确的跳过状态和原因，不伪造验证误差。通过现有 `eval.py` 生成 `eval_log.json`，Runner 另保存逐演示报告。

## 测试与验收

转换测试在 `bimanual-teleop` 环境运行，覆盖真实编码小视频：一／两帧相机缺口可修复，三帧缺口在默认配置下保留边界；参数修改后结果随之变化；单路、多路与主相机缺帧都需覆盖。检查图像来源索引、像素复用、30 Hz 间隔、首尾裁剪、暂停隔离、超限状态间隔、无效四元数、指令过期、旧严格模式兼容和失败时不发布半成品。

暂停测试另覆盖：已复核连续接缝压缩后合并且维持 30 Hz，未经复核或回位／物体改变的接缝仍分段；目标重设保留真实绝对标签；暂停期间不生成伪状态或控制指令；原始时长及拼接位置可追溯，压缩时间不绕过指令新鲜度检查。

Dataset 和 Runner 测试在 `robodiff` 环境运行。用合成多演示数据验证分组、短片段、剩余缺口、RGB 通道和范围、训练集归一化及旧格式兼容；以带时间索引的动作检查标签对齐，以已知位置、旋转及关节偏差检查指标。图像配置验收覆盖缩放开／关、非默认尺寸及插值方式生效、crop_shape 为 null／有效尺寸、固定评估裁剪，以及非法尺寸或 shape_meta 不一致时报错；同一输入在 Dataset 和 Runner 的确定性预处理结果一致。

Transformer 冒烟核对 GPU 上损失有限、有效梯度及优化器参数分组，训练标签 `(B,16,52)`、完整预测 `(B,16,52)`、执行段 `(B,8,52)`。checkpoint 保存并重载后核对模型、EMA 和归一化参数；固定随机种子检查同一评估输入可复现。

正式阶段报告来源划分、修复参数、配置、checkpoint、指标和样本量。报告不含 NaN 或 Infinity，不将离线误差写成真实任务成功率。

## 文档与同步约定

快速说明见 [导轨任务 README](/home/jiyuchen/project/WYS/diffusion_policy/docs/tianji-rail/README.md)。保持环境、转换、短训练／恢复和评估四步，参数说明直接指向实际配置；命令已验收并移除待实现提示。正式训练仍明确需要独立真实演示。详细验证见 [构建验收记录](/home/jiyuchen/project/WYS/diffusion_policy/wys/2026-10-03-rail-build-implementation/build-validation.md)。

后续本任务中修订范围、接口、参数或验收条件时，在回复前同步更新本文件，记录版本、日期和变更。涉及实际使用方式的变化同时更新快速说明，阶段完成时更新状态和验收证据。

## 变更记录

- 2026-10-02 / v1.0：首次保存分阶段计划；确认 GPU 可用；加入 docs 快速说明；采用 30 Hz、默认两帧的缺口修复配置并保留暂停边界；按用户要求改用 Transformer Hybrid Image Policy、配套 Workspace 和优化器。
- 2026-10-02 / v1.1：快速说明精简为环境、转换、训练和评估四步；后续持续保持该格式，详细设计与参数在计划中维护。
- 2026-10-03 / v1.2：补充实施顺序、缺口修复及模块连接的 PNG 和 Mermaid 图源，明确各阶段验收、多演示数据依赖及冒烟前的 Runner 接口准备；业务阶段仍待实施。
- 2026-10-03 / v1.3：将暂停一律保留边界改为连续性复核后的条件压缩拼接，澄清绝对动作标签与映射参考的区别，同步更新快速说明和图示；转换代码与数据尚未改动。
- 2026-10-03 / v1.4：补充多模态缺口文献依据，区分论文／作者实现与本任务的阈值及暂停工程选择。
- 2026-10-03 / v1.5：明确缩放开关、图像尺寸、插值方式、裁剪开关及评估行为可配置；新增带中文注释的 YAML 设计示例，要求实际配置每个可调参数说明作用和修改建议，补充配置一致性验收。
- 2026-10-03 / v1.6：开始实际构建；转换相关 24 项及 Dataset/Runner 7 项测试通过，真实数据生成和 GPU 验收进行中。暂不启动缺少独立真实演示的阶段 6。
- 2026-10-03 / v1.7：阶段 0—5、7 验收完成；真实数据修复为1887帧、2片段、1857个16步窗口；33项测试通过，两类 GPU 短训练、恢复命令、checkpoint 重载与空／非空评估入口通过。保持暂停复核边界；阶段6待补独立真实演示，未启动600轮正式训练。
