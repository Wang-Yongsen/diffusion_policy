# 导轨任务真实观测输入诊断方案：交接本地 Agent

2026-10-18统一执行说明：后续以[一次完成的最终实验批次](tianji-rail-one-pass-final-batch-20261018.md)为唯一入口；本文件保留为历史依据，不重复执行。


日期：2026-10-17（UTC+8）  
本地项目：`/home/yuchen/project/diffusion_policy`  
服务器项目：`/home/jiyuchen/project/WYS/diffusion_policy`

## 1. 任务与当前结论

目标：找出同一 epoch 50 diffusion policy 在“逐时刻输入示教观测”时可以执行轨迹，但接入真实机器人观测后左臂只外伸少量、随后抖动的原因。交付能复核的输入、实验代码、数值结果和结论；不要只写“可能是训练不好”或“还需更多数据”。

用户新增的关键实测：每个时刻都用训练数据中的完整观测替代真实观测，物理机器人能够按预期轨迹执行。因此，当前故障重点是实际输入内容及其与示教、反馈状态之间的差异，不能直接认定模型没有学会轨迹，也不能认定执行器无法完成外伸。

本轮默认执行离线推理与只读观测采集。交叉组合图像和状态的动作仅用于离线诊断，不能下发给机器人。不要为了获取输入先启动运动。确需采集运动中的故障输入时，给既有正常测试增加可选记录，在用户原有的人工开始/停止流程中取样；不通过本方案自动发起新的运动试验。

先阅读本机适用的 `AGENTS.md`，保留已有修改，不自动拉取或覆盖本机部署代码。以下“已完成核验”是服务器代码与所列样本的结果，不能替代本机核验。

## 2. 已知事实、反证与不能再使用的推断

### 2.1 固定实验对象

| 项目 | 值 |
|---|---|
| 训练运行 | `2026.10.16/23.09.56_tianji_rail` |
| 权重 | epoch 50，使用 `ema_model` |
| 已核验 checkpoint SHA256 | `575b901e0aa801437046789d3ae626969943043f0d8a2307acb88d4b158286d3` |
| 本机已有权重位置 | `/home/yuchen/project/diffusion_policy/data/outputs/2026.10.16/23.09.56_tianji_rail/checkpoints/latest.ckpt` |
| 观测/动作采样率 | 30 Hz；两帧观测间隔约 33.33 ms |
| 模型时间参数 | `n_obs_steps=2`，`horizon=16`，`n_action_steps=8` |
| 去噪 | DDIM，`num_inference_steps=8` |
| 观测 | 三路 RGB 480×640；双臂 EEF 12D；双手关节 40D |
| 动作 | 52D 绝对目标：左臂6、右臂6、左手20、右手20；米、rotvec弧度、关节弧度 |
| 左臂原生基座轴 | +X 向前、+Y 向下、+Z 向外；不能把原生轴直接当统一世界坐标 |

`latest.ckpt` 可能变化。必须按 SHA256 固定实验权重；不要按文件名猜 epoch，也不要下载服务器当前 latest 替换这份权重。本机若只有 latest、没有 epoch=0050 文件，直接使用已校验的 latest。

### 2.2 真实失败记录 motion-sync-028

- 本次实际使用 30 Hz、sync、上述权重与 EMA。
- 三路相机序列号依次为 `405622076394`、`408322072104`、`408322070938`，都是 D436；相应旋转为 0°、180°、0°，与纳入训练的41条示教元数据一致。不要再把服务器默认文件中的 D435 注释当作本次实际输入。
- 59 次预测中58块被调度，464个计划目标均获得 arm submit；这证明发生了下发调用，不证明真实关节到位。
- 已调度左臂目标 Z 仅约225–295 mm，首个目标到末个目标净变化约+8.2 mm；示教前5秒 Z 中位约447 mm。图中目标不是实测轨迹。
- 最后约26秒处，第59次预测因左手某关节速度8.17 rad/s超过8 rad/s被整块拒绝。它解释最后中止，不能解释此前的外伸缺失。
- 旧日志缺失原始观测数组、真实末端轨迹和伺服跟踪状态，不能从这些文件复原故障时的完整输入。

### 2.3 已完成的核验与反证

1. 相同保存的原始观测经过训练 Dataset 与部署预处理，15个窗口、5个输入字段逐元素一致；3组配对预测差异为0。
2. 当前部署 FK 对126个原始实测姿态重算，与录制EEF一致；259个训练时间点的当前live状态插值结果与训练Zarr一致。未发现单位、左右顺序或rotvec转换错位。
3. 相同同步节奏下，给伺服正确示教目标能外伸至约0.50 m；给失败记录中的目标会停在约0.25 m附近。同步停顿会拖慢，但不会单独阻止正确目标外伸。
4. 保持示教图像逐时刻播放，20块理想伺服干预结果：全示教观测最终左Z447.81 mm；接回左EEF反馈450.22 mm；接回双臂与手理想反馈448.40 mm。这里没有真实图像反馈、接触或真实动力学。不能再声称“只要接回状态反馈就会卡住”。
5. 左腕相机历史确有重复帧：56/59次两帧选到同一图像时间戳；部分匹配偏差超过训练转换的20 ms容忍度。但8次配对消融中，单独重复左腕帧的最大预测XYZ影响约1.05 mm，不支持它单独解释约20 cm缺失。
6. 早晚示教图像/状态交叉替换表明模型强烈依赖左EEF状态，但这种敏感性不等于导致本次失败。此前据此直接归因的结论已撤回。

## 3. 交接包及环境冻结

交接包在服务器：

```text
/home/jiyuchen/.codex/visualizations/2026/10/17/01a14808-7071-71e0-8322-331ee70f25ce/tianji-rail-input-diagnosis-20261017.zip
```

包内结构：

```text
tianji-rail-input-diagnosis-20261017/
  PLAN.md
  manifest.json
  tools/capture_tianji_inputs.py
  tools/test_capture_tianji_inputs.py
  reference/index.json
  reference/episode_000000_f0000.npz  ... 共10份观测
  reference/episode_000034_f0150.npz
  reference/motion_sync_028_deployment.yaml
  reference/training_config.yaml
  reference/server_source_provenance.json
  reference/expert_replay_summary.json
  evidence/ ... 已完成对照的JSON与原始motion-sync-028.zip
```

本机没有完整训练集时，使用这10份参考开始：两个来源，各取第0、8、30、90、150帧，即约0、0.267、1、3、5秒。它们覆盖初始外伸段的部分状态，不覆盖全部任务，也不是任意真实姿态的动作真值。NPZ包含两帧原始RGB/EEF/手状态、30 Hz元数据及对应示教动作；详细定义见index。

推荐在本机现有部署Conda环境执行，不重装环境。可从本机用已有SSH别名下载并解压：

```bash
DIAG_PROJECT=/home/yuchen/project/diffusion_policy
mkdir -p "$DIAG_PROJECT/data/input_diagnosis"
scp raids-pro6000:/home/jiyuchen/.codex/visualizations/2026/10/17/01a14808-7071-71e0-8322-331ee70f25ce/tianji-rail-input-diagnosis-20261017.zip "$DIAG_PROJECT/data/input_diagnosis/"
python -m zipfile -e "$DIAG_PROJECT/data/input_diagnosis/tianji-rail-input-diagnosis-20261017.zip" "$DIAG_PROJECT/data/input_diagnosis/"

DIAG_BUNDLE="$DIAG_PROJECT/data/input_diagnosis/tianji-rail-input-diagnosis-20261017"
DIAG_CKPT="$DIAG_PROJECT/data/outputs/2026.10.16/23.09.56_tianji_rail/checkpoints/latest.ckpt"
DIAG_CFG="$DIAG_BUNDLE/reference/motion_sync_028_deployment.yaml"
DIAG_OUT="$DIAG_PROJECT/data/input_diagnosis/results_20261017_01"
sha256sum "$DIAG_CKPT"
```

后续命令假定 `python` 已是 `tianji-rail-diffusion-policy` 环境。变量定义只在当前shell有效；跨终端需重新定义。输出目录应全新，不能覆盖前次结果。

`DIAG_CFG` 是失败会话保存的30 Hz部署参数，仍引用本机的设备配置文件。先检查文件存在并与本次实际配置相符；若不符，使用本机实测的30 Hz YAML并记录差异。服务器当前默认部署YAML为10 Hz，不能直接套用。频率不一致时定位配置来源，不删除校验或更换模型频率来绕过。

记录：checkpoint hash、EMA、依赖版本、git commit/dirty及关键实际import文件hash、部署配置、相机身份/旋转/分辨率。明确本机变更与服务器核验代码的区别。

## 4. 实验A：本机离线基线与处理链一致性

先运行工具测试，再用包内初始示教观测重放：

```bash
python -m unittest discover -s "$DIAG_BUNDLE/tools" -p 'test_capture_tianji_inputs.py' -v
python "$DIAG_BUNDLE/tools/capture_tianji_inputs.py" \
  --project-root "$DIAG_PROJECT" \
  --checkpoint "$DIAG_CKPT" \
  --replay "$DIAG_BUNDLE/reference/episode_000000_f0000.npz" \
  --seed 42 --cpu-threads 2 \
  --output "$DIAG_OUT/A_expert_initial"
```

预期参考：同一epoch50、CPU、seed42时，示教000000初始观测的左臂第8步XYZ约 `[612.4521,117.7634,251.0147] mm`。以包内reference summary为准。先确认形状、输入值、EMA和采样设置；不同设备/依赖版本不要求与服务器逐bit一致，量化差异后解释。

随后在本机对包内10份参考执行：

- 将同一原始数组分别按本机训练Dataset规则和实际部署preprocess构造输入；比较每字段shape、dtype、通道顺序、min/max及元素最大差。
- 若完整数据集可用，直接与Dataset取样核对；若不可用，使用已保存参考数组与已审计的Dataset实现，明确不能称作本机完整Dataset读盘回归。
- RGB期望为原始`uint8 [2,480,640,3]`，准备后`float32 [2,3,480,640]`且范围[0,1]；EEF `[2,12]`、手 `[2,40]`。
- 参考图像已在采集上游完成定向，不再旋转camera_1。不要另外做BGR交换、裁剪、缩放或ImageNet归一化。
- 同一机器、同一policy、同一原始数组、同一seed的两条构造路径应一致。若不一致，先定位第一个分歧字段，暂停后续根因解释；确认bug后做最小修复，并在相同输入上回归。

## 5. 实验B：采集真正进入模型的真实输入

### B1. 只读当前观测

结束占用设备的旧评估进程后，使用工具获取完整两帧观测。不要与另一个SDK控制程序同时运行。

```bash
python "$DIAG_BUNDLE/tools/capture_tianji_inputs.py" \
  --project-root "$DIAG_PROJECT" \
  --checkpoint "$DIAG_CKPT" --config "$DIAG_CFG" \
  --seed 42 --cpu-threads 2 \
  --output "$DIAG_OUT/B_real_initial_01"
```

工具以`motion=False`启动读取，取得一组完整观测即关闭设备，再保存文件并执行CPU预测；不调用engage、exec_actions或submit。已通过3项mock测试与真实权重的离线replay；尚未在本机真实设备上验收。若采集失败，保留error.json与终端输出，定位具体错误，不改动动作限制作为修复。

先在实际测试准备好的初始场景采3组独立快照，输出名用`01/02/03`，记录姿态/场景是否改变。有已留在失败位置的机器人时，可额外只读采集该位置，但标注为“失败后静态快照”。本工具取得的是当前观测，不能声称重建了motion-sync-028中的历史故障帧。

每组应有：

- `raw_observation.npz`：实际输入的两帧三路RGB、两帧12D EEF、两帧40D手状态；
- `camera_0_last.png`、`camera_1_last.png`、`camera_2_last.png`；
- `metadata.json`：对齐信息、设备信息、配置、权重与源码hash；
- `prediction.npz`：8步action与完整action_pred；
- `summary.json`：实际EEF、首/末目标与seed。

### B2. 若静态快照不能重现原先的失败特征

不能把一个新静态快照当作故障时输入。此时需要本地agent给既有推理路径增加可选、有限数量的观测记录：

1. 在原始obs已组装、尚未预处理时保存它的独立快照，连同prediction_id、anchor、query/各相机/状态括号时间戳。
2. 记录模型**实际使用**的输入、返回action、对应schedule和模式；保存首个预测、明显失去推进前后及终止前的少量窗口。建议最多3–6份，不录整段视频。
3. 记录同一时刻的实测关节/EEF/手反馈和采样时间；如果现有驱动提供tracking_status，可一并记录。其position_error是请求目标相对内部命令FK的误差，不等于实测FK误差，字段说明必须区分。
4. 在推理子进程内、采样前保存CPU/CUDA RNG状态或初始采样噪声，并保存实际返回action。session中的seed42只表示初始种子，预热和此前推理已消耗随机数，不能当作每次预测的seed。
5. 不在200 Hz控制线程压缩、写图或额外计算FK；有界保存原始数组到内存，结束后落盘。记录功能不得改变索引推进、seed、动作选择、阈值或同步模式。

优先先在只读/影子模式验证记录功能。需要运动中的输入时，随用户原有正常测试流程取样；不将下面的混合输入用于运动。若当前阶段没有运动采样条件，交付B1结果与明确缺失项，不把它写成已复现原故障。

## 6. 实验C：先确认RR真实输入的预测特征

用B中保存的真实NPZ离线重放；同一组条件先用CPU，控制线程数2，避免CPU/GPU随机数差异混入因果比较。

```bash
python "$DIAG_BUNDLE/tools/capture_tianji_inputs.py" \
  --project-root "$DIAG_PROJECT" --checkpoint "$DIAG_CKPT" \
  --replay "$DIAG_OUT/B_real_initial_01/raw_observation.npz" \
  --seed 42 --cpu-threads 2 \
  --output "$DIAG_OUT/C_real_replay_01_seed42"
```

检查：左臂绝对目标是否仍在错误区域；相对该真实当前EEF的向外/向下位移是多少；姿态是否发展；8步是否频繁换向。没有真实动作标签时，不给这组预测虚构“准确率”。

若保存了确切现场RNG状态，在同设备/同版本环境复跑实际预测；否则使用多个固定seed检查失败特征是否稳定，不以某次坐标不完全相同判定输入正确或错误。CPU与CUDA同一个数值seed不保证相同初始随机噪声。

如RR不能稳定呈现原来特征，先排查快照取得的阶段、实际权重/版本、预处理及真实记录是否对应。此时2×2仍可做一般敏感性诊断，但不能解释为已找到历史故障的原因。

## 7. 实验D：配对真实输入与示教参考，做2×2离线干预

### D1. 参考选择

在查看模型干预结果前，选择并锁定参考：

- 先根据导轨/孔口位置、是否接触或抓住、双手任务状态确定粗阶段，再寻找该阶段中EEF/手状态接近的参考。
- 初始快照可用包内两个来源的0、8帧；外伸中间状态可用30、90、150帧。不要按“真实第5秒=示教第5秒”硬配。
- 每个真实样本先固定3个参考，尽量来自两个来源；一组按任务阶段接近，一组按姿态接近，分别解释。包内000034属于已使用的留出来源，000000属于训练来源。
- 记录XYZ距离、SO(3)姿态距离、手关节差，以及三路图像对照。参考稀疏、无近邻时标为“参考支持不足”，不能挑输出最好的一帧代替正确参考；必要时只补下载少量指定训练窗口。
- 两帧必须保持原时间顺序，三路图像一起替换；EEF和手先整体替换。真实与示教的绝对monotonic时间没有共同原点，只比较间隔与对齐残差。

### D2. 四组条件

用`E`表示示教，`R`表示真实。第一个字母是三路图像，第二个字母是全部低维状态（12D EEF+40D手）：

| 条件 | 图像 | 状态 | 用途 |
|---|---|---|---|
| EE | 示教 | 示教 | 对应示教的基线 |
| RE | 真实 | 示教 | 固定示教状态，仅换图片 |
| ER | 示教 | 真实 | 固定真实状态，仅换图片 |
| RR | 真实 | 真实 | 实际输入基线 |

由本地agent实现`compare_inputs_2x2.py`，保留实现及运行命令。要求：

- 四组使用同一checkpoint SHA、EMA、policy.eval()、预处理、DDIM8、horizon16、obs2、action8和设备。
- 每组预测前调用policy.reset()并重设同一个seed；CUDA时也设相应CUDA seed，或显式使用共同初始噪声。不要按条件使用不同seed。
- 第一遍最多3个真实样本×3个参考×4组×seed42，共36次。若效应接近采样波动或需要验证稳定性，再在少量关键配对上加固定seed `[42,43,44,45,46]`；不把实验无边界扩大。
- 从原始uint8/低维数组组合后，统一调用实际部署preprocess。不能把一组用Zarr张量、另一组用BGR图片文件，或对某组重复定向/归一化。
- 输出所有8步52D动作，不能只挑第8步的一个Z值。

### D3. 数值与图

每条件、参考、seed至少输出：

1. 全部8步左臂绝对XYZ、姿态，首/末/均值Z。
2. 相对**该条件输入的最新EEF**的XYZ位移与SO(3)旋转，用来判断是否真正改变了动作方向/幅度。
3. 另给相对**同一个真实当前EEF**的位移，方便把四组绝对目标放在相同几何基准比较；不能把不同条件基准混在同一列。
4. 8步内部相邻位移中位/p95、Z增量反号次数；有连续真实窗口时再算块间跳变/回拉。不同孤立快照之间不能算真实运动速度或连续回拉率。
5. EE对对应示教标签`expert_action[1:9]`的误差。示教标签不是其他三组的真实正确动作，不能直接当它们的ground truth。
6. 图像副本对照PNG、左臂8步XYZ图、条件差与不同seed范围。区分目标和实测轨迹，位置使用mm，姿态使用SO(3)角度，不直接把rotvec分量差当姿态误差。

在相同参考/seed下计算位置条件效应：

```text
固定真实状态的图片效应：RR - ER
固定示教状态的图片效应：RE - EE
固定真实图片的状态效应：RR - RE
固定示教图片的状态效应：ER - EE
位置交互：RR - ER - RE + EE
```

对相同seed做配对比较，并与同输入不同seed的波动、参考之间的变化比较。不要把数值变化最大的条件直接宣布为根因，不预设“超过某mm就是bug”。初筛可把10 mm位置、3°姿态作为值得检查的量级，但结论要看它是否稳定改变失败特征且大于采样波动；这些是本次工作阈值，不是控制阈值或准确率标准。

## 8. 依据结果继续定位，避免循环猜测

| 实验结果 | 支持的方向 | 下一步 |
|---|---|---|
| 同一原始输入经本机两条路径变成不同张量 | 输入构造/代码差异 | 找到第一个字段与源码行；最小修复后在相同保存输入上回归 |
| RR保持失败特征，ER在真实状态下稳定恢复合理推进；多个参考/seed一致 | 真实图像内容或图像链路更值得优先查 | 逐相机替换，再检查视角、位置、定向、裁剪/缩放、颜色、曝光、场景物体；不能仅凭2×2判定哪项处理有bug |
| RE在示教状态下失败，而EE正常 | 图片效应的另一项证据 | 与RR/ER对照综合，检查是否交互或参考不匹配 |
| 状态替换只改变绝对位置，相对动作近似不变 | 状态作为位置锚，未证明会卡住 | 不据此归因；核对真实状态是否在参考支持范围、实际反馈与FK是否合理 |
| 状态替换稳定改变推进/姿态，图片替换影响小 | 真实低维状态/分布差异更可疑 | 逐块替换左EEF、右EEF、手，再拆位置/姿态/两帧历史，核对单位、坐标、真实起点及反馈，而非只调loss |
| 单独替换均影响小，EE与RR却差异大 | 联合分布或交互 | 报告交互，检查阶段/物理一致性；不能硬选图像或状态一个背锅 |
| 效应随参考窗口翻转或与seed波动同量级 | 证据不稳定 | 改善阶段匹配或补少量参考，不作确定因果结论 |
| 原始预测目标稳定推进，实际EEF却不跟随 | 执行/实际跟踪 | 同时对齐请求位姿、内部应用位姿、实测FK、progress/limited、关节余量和反馈时间；submit accepted不是到位 |

若图片路径成为明确线索，先逐相机替换，仅对有稳定效应的相机展开；若状态成为线索，先区分左EEF/右EEF/手，保留原始两帧和顺序。每次只改一个因素。暂不同时重训、换checkpoint、增加去噪步数、切换sync/async或放宽控制限制，这些会破坏归因。

## 9. 交付、判定与后续修复

在`DIAG_OUT`下交付：

```text
report.md
environment.json
commands.log
samples/                  # 原始真实输入、metadata、PNG、预测
references_used.json       # 参考选择理由、阶段、距离
compare_inputs_2x2.py
paired_metrics.csv
paired_predictions.npz
figures/*.png
tests/                     # 比较脚本的有意义测试
changes.patch              # 仅有代码改动时提供，保留本机原修改
```

比较脚本至少验证：四组替换范围正确、没有修改原始样本、固定seed的配对可重复、RR原样重放一致、旋转误差按SO(3)计算。采集工具现有mock测试不等于本机设备采集已验收。

结果报告必须回答：

1. 真实输入能否复现先前失败特征？若不能，具体缺少什么，不把静态快照写成故障重现。
2. 本机权重/版本/预处理是否与基线一致？用hash和数值支持。
3. 哪一类输入在固定其余因素时稳定改变推进？影响量、seed范围、参考一致性是多少？
4. 有无可指出源码行的部署缺陷？有无明确的起始状态/场景分布差异？严格区分已证实与候选解释。
5. 哪些先前假说被反证？不能再次用“状态依赖强”或“重复帧存在”直接宣布根因。
6. 下一项最小修复是什么，如何用同一保存输入验证它？若是确定的输入转换bug，做最小改动和离线回归；若是分布/恢复能力问题，再提出对应采集或训练实验，不同时改多项。

先完成不运动的验证与归因报告。需要物理运动验证修复时，只用完整真实观测，在既有人工开始/停止流程下验证目标与实测轨迹；混合输入始终只做离线实验。

最终压缩原始观测、报告、结果和新增实验代码，不包含大权重或完整训练集。脚本、输入和配置应足够让服务器agent在不连接机器人时复跑分析。

## 10. 可直接交给本地 Agent 的执行指令

> 请在`/home/yuchen/project/diffusion_policy`按本方案执行。先读取本机AGENTS并保留现有修改，核验epoch50权重hash和30 Hz配置。完成离线基线、只读真实观测采集及配对2×2实验，按阶段和姿态预先选择参考；所有混合输入只做离线推理。需要记录运动中输入时，仅为既有人工启动的正常测试添加可选记录，不自动发起运动，也不改变控制参数或模型设置。用原始输入、同seed对照、源码hash和数值定位差异，不沿用已撤回的根因推断。交付第9节的复现包和明确的证据结论；若当前输入不能复现原故障，明确报告，不编造唯一根因。
