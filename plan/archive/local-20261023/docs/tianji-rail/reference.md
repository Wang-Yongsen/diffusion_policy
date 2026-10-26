# 天机导轨完整参考

核对日期：2026-10-21。现场操作用[四步简洁版](README.md)；本页保留训练、评估、模型切换与参数说明。

## 环境

所有命令在项目根目录执行：

```bash
cd /home/yuchen/project/diffusion_policy
conda activate tianji-rail-diffusion-policy
export PYTHONNOUSERSITE=1
```

仅首次创建环境时执行：

```bash
PYTHONNOUSERSITE=1 PIP_USER=false conda env create -f conda_environment_tianji.yaml
```

`python -c "import sys; print(sys.executable)"` 应指向该 Conda 环境。移动源码后执行 `python -m pip install --no-deps -e .`。设备接口、相机和 SDK 使用项目内的[依赖快照](../../third_party/README.md)；地址、序列号和相机顺序见[设备约定](device-runtime.md)。

## 数据与动作空间

数据采集与转换见 [bimanual-teleop](../../../bimanual-teleop/docs/recording_guide.md)。每次训练显式指定 `.zarr` 路径；模型项目不负责采集或转换。

| 动作空间 | 训练参数 | 低维观测 | 绝对动作排列 |
| --- | --- | --- | --- |
| 末端版 | `task.action_space=eef` | `robot_eef_pose` 12维＋`hand_joint` 40维 | 左臂6、右臂6、左手20、右手20，共52维 |
| 关节版 | `task.action_space=joint` | `robot_joint` 14维＋`hand_joint` 40维 | 左臂7、右臂7、左手20、右手20，共54维 |

两者均输入三路 RGB，当前为640×480原图，不缩放、不裁剪。位置单位为米，旋转向量和关节角为弧度。Zarr 的 `action_space` 必须与训练参数一致；训练窗口不跨连续片段，训练集用于拟合归一化参数。

默认 `task.sample_hz=30`；10 Hz数据须设为10。训练校验 Zarr 的 `target_fps` 与实际时间间隔，部署 `policy_hz` 须与 checkpoint 一致。改变频率需使用对应数据重新训练，不能只修改属性或部署频率。

## 训练

先设置本次数据路径；下面使用关节版数据，末端版改为 `eef` 并换用对应数据。

```bash
data_path=/path/to/rail_joint_30hz.zarr

# UNet：短训练检查，再正式训练。
python train.py --config-name=train_diffusion_unet_tianji_rail_smoke_workspace \
  task.action_space=joint "task.dataset.dataset_path=$data_path"
python train.py --config-name=train_diffusion_unet_tianji_rail_workspace \
  task.action_space=joint "task.dataset.dataset_path=$data_path"

# Transformer：按需选择这一组。
python train.py --config-name=train_diffusion_transformer_tianji_rail_smoke_workspace \
  task.action_space=joint "task.dataset.dataset_path=$data_path"
python train.py --config-name=train_diffusion_transformer_tianji_rail_workspace \
  task.action_space=joint "task.dataset.dataset_path=$data_path"
```

| 参数 | 当前默认值／规则 |
| --- | --- |
| `task.action_space` / `task.sample_hz` | `eef` / `30`；关节版须显式设 `joint` |
| `n_obs_steps` | Transformer为1，UNet为2 |
| `horizon` / `n_action_steps` | 16 / 8；须满足 `n_obs_steps-1+n_action_steps <= horizon` |
| `dataloader.batch_size` / `num_workers` | 64 / 4；显存不足时减小批大小 |
| `dataloader.persistent_workers` | `true`；读取进程数为0时须改为 `false` |
| `training.num_epochs` / `task.dataset.val_ratio` | 600 / 0.05；单条演示的验证比例设0 |
| 学习率 | 均为0.0001；Transformer用 `optimizer.learning_rate`，UNet用 `optimizer.lr` |
| 验证／离线评估间隔 | 每5轮验证，每50轮离线评估 |

短训练为2轮，每轮训练与验证各最多2批。两种网络均采用 DDIM8；UNet 默认宽度为 `[256,512,1024]`，`horizon` 须为4的倍数。命令行 `参数名=值` 覆盖本次运行；长期默认值在[任务配置](../../diffusion_policy/config/task/tianji_rail_image.yaml)、[Transformer配置](../../diffusion_policy/config/train_diffusion_transformer_tianji_rail_workspace.yaml)或[UNet配置](../../diffusion_policy/config/train_diffusion_unet_tianji_rail_workspace.yaml)中修改。

输出目录自动生成在 `data/outputs/日期/时间_tianji_rail_<动作空间>/`；UNet 目录带 `_unet_`。其中保存 `.hydra/` 配置、`logs.json.txt` 和 `checkpoints/`。

### 恢复训练

保持原数据、动作空间、网络及观测参数，沿用原输出目录。示例为 UNet 关节版；恢复其他实验时换成原配置和参数。

```bash
run_dir=/path/to/original-run
test -f "$run_dir/checkpoints/latest.ckpt" && \
python train.py --config-name=train_diffusion_unet_tianji_rail_workspace \
  task.action_space=joint "task.dataset.dataset_path=$data_path" \
  "hydra.run.dir=$run_dir" training.resume=true
```

恢复入口读取原目录的 `checkpoints/latest.ckpt`；`training.num_epochs` 是总目标轮数，不是追加轮数。

## 离线评估

使用数据集中的图像和状态替代真实观测，在电脑上预测动作并比较数据集标签；不连接设备，也不下发动作。`eval.py` 默认只评估验证集，网络、数据划分和预处理从 checkpoint 读取；报告目录使用新路径。

```bash
checkpoint=/path/to/model.ckpt
report_dir="data/offline_eval/$(date +%Y%m%d-%H%M%S)"
python eval.py --checkpoint "$checkpoint" --output_dir "$report_dir" --seed 42
```

数据搬迁时加 `--dataset_path /new/path/data.zarr`，保持数据版本、频率及来源划分一致。默认按 checkpoint 选择 EMA／模型权重，固定评估模式与随机种子。

对比 DDIM 采样截断时，使用相同 checkpoint、数据和种子，分别指定新目录：

```bash
python eval.py --checkpoint "$checkpoint" --output_dir "${report_dir}-clip-on" \
  --seed 42 --clip-sample
python eval.py --checkpoint "$checkpoint" --output_dir "${report_dir}-clip-off" \
  --seed 42 --no-clip-sample
```

数据已搬迁时，两条命令都加 `--dataset_path /new/path/data.zarr`。报告包括 `eval_log.json`、`offline_report.json`、`eval_settings.json`。末端版报告位置与旋转误差，关节版报告机械臂关节误差，两者均报告手关节误差。离线动作误差不代表真机成功率。

## 真机评估

统一入口为 `eval_tianji_rail.py`。网络结构和观测步数从 checkpoint 读取，动作空间字段为 `task.dataset.action_space`；旧 checkpoint 缺失时按末端版处理。当前已核对的本机模型：

| 模型 | checkpoint | 观测步数 | 可用推理模式 |
| --- | --- | --- | --- |
| UNet 关节版 | `data/outputs/2026.10.20/15.30.50_tianji_rail_unet_joint/checkpoints/epoch=0060.ckpt` | 2 | `sync`、`async` |
| Transformer 关节版 | `data/outputs/2026.10.19/14.08.04_tianji_rail_joint/checkpoints/epoch=0150.ckpt` | 1 | `sync`、`async` |
| Transformer 末端版，原混合观测模型 | `data/outputs/2026.10.16/23.09.56_tianji_rail/checkpoints/epoch=0050.ckpt` | 2 | `sync`、`async` |

以上模型均为30 Hz、DDIM8、EMA、8步执行段、`n_latency_steps=0`。

标准预检、影子和运动命令见[简洁版](README.md)。切换模型后重新走这三步；Transformer epoch150 同样支持同步模式：

```bash
checkpoint=data/outputs/2026.10.19/14.08.04_tianji_rail_joint/checkpoints/epoch=0150.ckpt
run_tag="$(date +%Y%m%d-%H%M%S)"

python eval_tianji_rail.py \
  --checkpoint "$checkpoint" --set policy_hz=30 \
  --inference-mode sync --check-only \
  --output "data/real_eval/${run_tag}-check"
python eval_tianji_rail.py \
  --checkpoint "$checkpoint" --set policy_hz=30 \
  --inference-mode sync \
  --output "data/real_eval/${run_tag}-shadow"
python eval_tianji_rail.py \
  --checkpoint "$checkpoint" --set policy_hz=30 \
  --inference-mode sync --motion \
  --output "data/real_eval/${run_tag}-motion"
```

### 推理模式

| 模式 | 观测与动作规则 |
| --- | --- |
| `async`，配置默认值 | 新锚点前进至少 `replanning_steps/policy_hz` 秒后推理；动作保留观测锚点对应时刻，过期步骤丢弃，新预测替换未来队列 |
| `sync` | 按 checkpoint 自动使用单步或两步观测，仍要求至少两步动作；各步经过至少一个策略周期且双臂、左右手均成功下发后才推进，迟到不跳步；完成后取最新可对齐窗口再推理，推理期间保持上一目标 |

同步观测按 checkpoint 的 `n_obs_steps` 查询：单步为 `[t₀]`，三路相机各匹配一张图像；两步为 `[t₀−1/policy_hz, t₀]`，三路相机各匹配两张不同的递增图像。状态均在对应查询时刻插值，图像偏差、三相机跨度和状态插值间隔限制保持不变。锚点由各路缓存共同覆盖，不要求晚于动作块完成时刻。缺少合格窗口时保持并等待，成功读取后固定本轮输入。

异步默认30 Hz、`replanning_steps=1`，最短重规划间隔33.3 ms；原宽松试验用2，约66.7 ms。8步动作在30 Hz覆盖约267 ms，在10 Hz覆盖800 ms；异步推理、观测延迟及提交余量会缩短剩余覆盖。同步模式不使用 `replanning_steps`，控制迟到会延长实际执行。

部署将模型推理放在独立进程，PyTorch 和 BLAS 各限制为1个计算线程。相机30 Hz、状态历史100 Hz、双臂控制200 Hz、双手各120 Hz；这些频率不等于实际推理频率。

## 用训练集或验证集观测部署真机

`eval_tianji_rail.py --dataset` 用 Zarr 中的三路图像、机械臂状态和双手状态替代策略真实观测，预测动作仍由原同步执行器下发真机。现在优先使用本地缓存的单 episode 文件，不需要下载完整处理后数据集；单步／两步、末端版／关节版按 checkpoint 自动适配。同一缓存同时含 `robot_joint`、`robot_eef_pose` 时，两种模型可共用，不再要求缓存动作标签的空间与模型相同；观测字段、形状、采样频率及片段边界仍须符合 checkpoint。

真实反馈继续用于接合参考、动作限制和设备健康检查，相机也仍按原部署环境启动。策略输入使用示教时钟，现场偏离示教时不会依据真实观测纠正轨迹。

### 已下载的单 episode 缓存

2026-10-21 从 `raids-pro6000` 的关节版30 Hz完整数据导出并下载，保留三路640×480 RGB、14维机械臂关节、40维双手关节、54维动作以及时间戳和原连续片段边界。原末端版缓存未覆盖。

| 本地文件 | 帧数 | 原划分 | 合法执行窗口数 |
| --- | --- | --- | --- |
| `data/replay_obs/rail_place3_joint_episode_000000.zarr` | 1549 | train | 192 |
| `data/replay_obs/rail_place3_joint_episode_000003.zarr` | 1597 | val | 198 |
| `data/replay_obs/rail_place3_joint_episode_000034.zarr` | 1253 | val | 155 |

这批约9.1 GiB。表中归属对应当前 Transformer epoch150／UNet epoch60 的 seed42、`val_ratio=0.05`；窗口数对应 `horizon=16`、每块8步。来源清单见 `data/replay_obs/joint_episodes_manifest.json`。

另按录制质量和关键画面筛选了以下三条训练示教。优先用 `000013`，`000030` 作对照，`000033` 用于较快示教的对照；已有的两个验证 episode 继续保留。

| 新增本地文件 | 帧数 | 数据时长（30 Hz） | 合法执行窗口数 | 实测关节速度绝对值 P95 |
| --- | --- | --- | --- | --- |
| `data/replay_obs/rail_place3_joint_episode_000013.zarr` | 1386 | 46.20秒 | 172 | 0.163 rad/s |
| `data/replay_obs/rail_place3_joint_episode_000030.zarr` | 1355 | 45.17秒 | 168 | 0.165 rad/s |
| `data/replay_obs/rail_place3_joint_episode_000033.zarr` | 1187 | 39.57秒 | 147 | 0.206 rad/s |

新增三条共8.11 GiB。转换质量报告均无告警、图像补帧、状态超限修复或内部无效帧；均为一个连续片段。速度按相邻实测关节差乘30计算，P95汇总全部14个关节和时间步，仅用于描述轨迹。抽查三路相机的关键画面，可见取料、搬运、放置和双手撤离；原录制没有独立的任务成功标签，也未逐帧评审，因此不将这些结果记为已标注的成功示教。筛选记录见 `data/replay_obs/tianji-teleop-quality-20261021.json`，预览见 `data/replay_obs/tianji-selected-episodes-20261021.jpg`，导出来源见 `data/replay_obs/good_joint_episodes_manifest.json`。

三条均完成全部文件的 SHA256 汇总核验，与服务器导出的副本一致，低维数组均为有限值。完整性记录见 `data/replay_obs/good_joint_episodes_integrity.json`。观测共用适配后118项相关测试通过；三条分别通过 Transformer joint epoch150（单步）、UNet joint epoch60（两步）及 EEF epoch120（两步）的 CUDA 预检，共九组，均未连接设备。报告目录为 `data/real_eval/good-episode-check-20261021-<transformer|unet|eef>-<episode>/`。

每个文件的 `episode_export` 元数据保存原服务器、原数据路径、原示教来源的完整顺序、原帧范围和元数据摘要。读取单 episode 时不重新划分；按当前 checkpoint 的种子和验证比例在原来源顺序上恢复归属。更换 checkpoint 后会重新计算，指定不相符的 `--dataset-split` 会在连接设备前报错。

### 训练集 episode 部署

在项目根目录、已激活的 Conda 环境中执行，下面路径均已存在：

```bash
checkpoint=data/outputs/2026.10.19/14.08.04_tianji_rail_joint/checkpoints/epoch=0150.ckpt
data_path=data/replay_obs/rail_place3_joint_episode_000013.zarr
run_tag="$(date +%Y%m%d-%H%M%S)"

# 不连接设备：用缓存 episode 的第一个窗口检查实际权重推理。
python eval_tianji_rail.py --checkpoint "$checkpoint" \
  --dataset "$data_path" --dataset-mode episode --dataset-split train \
  --set policy_hz=30 --inference-mode sync --check-only \
  --output "data/real_eval/${run_tag}-episode-train-check"

# 输入 s 后开始实际运动，策略观测来自该 episode。
python eval_tianji_rail.py --checkpoint "$checkpoint" \
  --dataset "$data_path" --dataset-mode episode --dataset-split train \
  --set policy_hz=30 --inference-mode sync --motion \
  --record-observations --batch-seed 42 \
  --output "data/real_eval/${run_tag}-episode-train-motion"
```

### 验证集 episode 部署

切换到验证 episode 文件，同时指定 `--dataset-split val`。不能只对同一条训练 episode 改划分参数：

```bash
python eval_tianji_rail.py --checkpoint "$checkpoint" \
  --dataset data/replay_obs/rail_place3_joint_episode_000034.zarr \
  --dataset-mode episode --dataset-split val \
  --set policy_hz=30 --inference-mode sync --motion \
  --record-observations --batch-seed 42 \
  --output "data/real_eval/${run_tag}-episode-val-motion"
```

另一条验证示教将文件名改为 `rail_place3_joint_episode_000003.zarr`。无设备预检把 `--motion` 换成 `--check-only`，并使用新输出目录。影子测试只去掉 `--motion`：会连接设备并自动开始推理，不下发运动目标。

### 数据模式与执行规则

- `--dataset-mode auto`（默认）：只有一条原始 episode 时按单 episode 读取；多 episode 时按完整数据模式读取。有完整导出来源信息的 episode 会自动标记原 train／val，可省略 `--dataset-split`。
- `--dataset-mode episode`：明确要求单 episode，不在这个文件上重新抽取验证集。多个连续片段可以保留，但不跨缺口执行。旧的无导出来源信息的单 episode 也可直接使用，此时须省略 `--dataset-split`，日志标记 `split=unknown`，不会按文件名猜测归属。
- `--dataset-mode full`：保留完整数据支持，默认选择原训练集；`--dataset-split val` 选择原验证集。应使用原完整数据和原来源顺序，不要拿几条 episode 拼成的新数据集重新划分。
- 默认取所选 episode 最长的合法连续片段。`--dataset-segment 0` 可指定本地文件的连续片段序号；完整数据模式下该序号是全局片段序号。单 episode 文件通常无需 `--dataset-episode`；如使用，该参数接受已保存的原全局示教序号或 `source_episode` 名称。
- `--dataset-max-windows 2` 只执行前两个观测窗口；不指定则执行所选片段的全部合法窗口。每个窗口须满足训练 Dataset 的完整 `horizon`；末尾不足完整训练窗口的部分不执行，不裁切或补齐动作块。
- 只支持 `sync`。每块仍完整执行 checkpoint 的 `n_action_steps` 步，各步等待双臂和左右手成功下发后推进。整块完成后数据窗口前进相同步数：8步动作时，单步输入依次为 `[s]`、`[s+8]`，两步输入为 `[s,s+1]`、`[s+8,s+9]`。
- 窗口耗尽后正常暂停，记录 `playback_complete`，不自动判断任务成功。运动模式下重新按 `s` 从该片段开头开始，`p` 接管、`y` 成功、`n` 失败、`q` 退出。

`--record-observations` 记录数据集模型输入和实际反馈；暂不支持与 `--trace` 同用。元数据的 `observation_source` 保存模式、归属及是否核验、来源和窗口范围；`action_space` 为当前策略空间，`dataset_action_space` 为文件动作标签空间，`dataset_actions_used=false` 表示未使用示教动作。预测 `alignment` 同时记录本地帧号、示教时间戳及可恢复的原数据全局帧号。数据集时间不作为现场观测年龄，时序报告另记 `dataset_read_to_prediction_return_ms`。

### 再下载其他少量 episode

可用新增导出脚本在服务器读取完整数据，只传所选 episode。本地不保存完整数据。示教名称用 `episode_000034` 这种完整名称；纯数字参数表示原数据中的全局 episode 序号，两者可能不同。

```bash
remote_out="/tmp/tianji-joint-episodes-$(date +%Y%m%d-%H%M%S)"
ssh raids-pro6000 \
  env PYTHONNOUSERSITE=1 /opt/miniconda3/envs/tianji-rail-diffusion-policy/bin/python - \
  --dataset /home/jiyuchen/project/WYS/bimanual-teleop/datasets/rail_place3_joint_30hz.zarr \
  --output "$remote_out" --prefix rail_place3_joint \
  --episodes episode_000000 episode_000003 episode_000034 \
  < scripts/export_tianji_episodes.py

rsync -a --checksum --info=progress2 \
  "raids-pro6000:${remote_out}/" data/replay_obs/
```

导出不改原始数据、不重新转换或缩放图像；保留原压缩图像块、动作和状态数值，连续片段边界重置到 episode 文件内。导出未完成时元数据标记 `complete=false`，部署会拒绝读取。导出清单或同名目标文件已存在时，导出脚本拒绝覆盖。

本次无设备验证：116项相关测试通过，包含单 episode 的原划分核验、连续片段保留、逐元素导出一致性，以及单步／两步、末端／关节版的模拟设备执行。三条下载的真实 episode 均完成 Transformer epoch150 CUDA 预检；训练 `000000` 和验证 `000034` 另完成 UNet epoch60 两步观测预检，共五组。报告位于 `data/real_eval/episode-cache-check-20261021-<模型>-<episode>/preflight.json`。未连接设备或执行真机运动。

旧入口 `eval_tianji_rail_recorded_obs.py` 继续用于末端版历史回放和混合观测实验，见[实验说明](../../plan/tianji-rail-mixed-observation-experiments.md)。

### 同一 episode 的 joint／EEF 能否合并

可以共用图像和观测。现有两版 Zarr 均已同时保存 `robot_joint(14)` 与 `robot_eef_pose(12)`，区别主要是 `data/action`：关节版54维，末端版52维。分别下载两版会重复保存三路图像，占用几乎两倍空间。

建议统一缓存按下表保存，仍只保留一套 episode／segment 边界与原数据来源：

| 字段 | 内容 | 读取规则 |
| --- | --- | --- |
| `camera_0`、`camera_1`、`camera_2`、`timestamp` | 共享图像和时间轴 | 两种模型共用 |
| `robot_joint`、`robot_eef_pose`、`hand_joint`、`wrench` | 共享实测状态 | 按 checkpoint 的观测字段选择 |
| `action_joint` | 原54维关节动作指令 | joint 模型的训练／离线对比 |
| `action_eef` | 原52维末端动作指令 | EEF 模型的训练／离线对比 |

合并前必须确认同一示教来源、转换频率与配置、连续片段、逐帧时间戳、共享状态和图像一致，不能仅凭文件名或帧数相同拼接。压缩块的二进制内容可能因压缩器设置不同而不同，应比较解码后的像素。末端动作应来自录制的控制器末端目标，不能用实测 `robot_eef_pose` 或 joint 指令的正运动学结果代替；前者是反馈，后者可能丢失原末端目标语义。

已核验本机 `000000`、`000003`、`000034` 的两版缓存：共4399帧的三路 RGB 解码像素、时间戳、两套机械臂状态、手部状态、力矩、片段边界及手部动作逐元素一致，具备合并条件。完整比对记录为 `data/replay_obs/joint_eef_decoded_alignment_20261021.json`。

**当前部署已支持共用观测，训练的双动作格式尚未实现。** `DatasetObservationSource` 从 checkpoint 选择输入字段，只用文件自身的动作空间校验未使用的标签，不改变 checkpoint、模型维度、归一化参数或真机执行逻辑。这样此次新增的 joint 缓存也可作为 EEF 策略的观测来源；只需换 checkpoint，`--dataset` 路径保持不变，`policy_hz` 仍须匹配模型。

例如用同一份 `000013` 观测检查本机 EEF epoch120（两步观测、52维输出）：

```bash
python eval_tianji_rail.py \
  --checkpoint data/outputs/2026.10.16/23.09.56_tianji_rail/checkpoints/epoch=0120.ckpt \
  --dataset data/replay_obs/rail_place3_joint_episode_000013.zarr \
  --dataset-mode episode --dataset-split train \
  --set policy_hz=30 --inference-mode sync --check-only \
  --output "data/real_eval/${run_tag}-shared-episode-eef-check"
```

训练 Dataset 仍严格检查根属性 `action_space` 和 `data/action`，防止将54维关节目标当作52维末端目标训练。上表的 `action_joint`／`action_eef` 是后续训练格式方案：需要另外保留真实 EEF 动作标签，再让训练加载器按所选空间映射为统一的 `action`。此次新增三条包含两套机械臂观测，尚未包含独立的 EEF 动作标签。

## 部署参数

当前使用[主部署配置](../../diffusion_policy/config/tianji_rail_real_eval.yaml)；旧 `tianji_rail_real_eval_loose.yaml` 仅为本地历史副本。入口支持重复的 `--set KEY=VALUE`，仅覆盖已有参数，值按 YAML 解析，嵌套键用点号。

| 场景 | 参数 |
| --- | --- |
| 10 Hz模型 | `--set policy_hz=10`；须匹配 checkpoint，缺少频率字段的旧模型按30 Hz处理 |
| 调整异步重规划间隔 | `--set replanning_steps=2`；须小于 checkpoint 的 `n_action_steps` |
| 不设试验时限（默认） | `--set trial_timeout_s=null`；正常运行由操作员手动暂停或结束 |
| 显式设置60秒试验时限 | `--set trial_timeout_s=60` |
| 收紧末端跳变 | `--set max_position_jump_m=0.05 --set max_rotation_jump_rad=0.2` |
| 收紧关节跳变与速度 | `--set max_arm_joint_jump_rad=0.3 --set max_arm_joint_speed_rad_s=1.0` |
| 设置末端工作空间 | `--set "workspace.left.min=[0.1,-0.3,0.2]" --set "workspace.left.max=[0.6,0.3,0.7]"`；右臂另填 |
| 保存少量输入窗口及实测状态 | `--record-observations`；两种动作空间均支持 |
| 完整决策记录 | `--trace --trace-capacity 48`；仅末端版支持 |
| CPU模型预检 | `--device cpu --check-only` |

`--device` 和 `--inference-mode` 覆盖配置中的对应值。`trial_timeout_s` 默认为 `null`，表示不自动结束；指定数值时须为正数。其他时间、频率与限值须为正数；`history_s` 须覆盖观测窗口。实际配置与覆盖参数写入 `preflight.json` 或 `session.json` 的 `deployment`、`config_overrides`。

末端版须填写实测工作空间；主配置每轴 ±1000 米近似不限制位置，不能代替现场范围或碰撞检测。关节版不检查末端工作空间。两者都检查手关节范围与跳变，当前不校验或裁剪手指速度。

## 关节版部署

- 观测为每臂7个实测关节及双手关节；末端位姿仍可记录，但不进入关节版模型。
- 每批动作整段检查关节与腕部耦合限位、跳变和速度；任一步失败则拒绝整段并暂停。默认最大关节跳变0.6 rad、相邻目标速度2.0 rad/s。
- 双臂以200 Hz直接提交关节目标，不经逆运动学；实际速度和加速度仍受[机械臂配置](../../diffusion_policy/config/tianji_teleop.yaml)约束。当前比例均为15%，最快关节约0.47 rad/s，可能滞后于快速目标。
- `--trace`、旧入口 `eval_tianji_rail_recorded_obs.py` 和批量混合观测工具仍仅支持末端版；关节版数据集观测部署用本页的主入口 `--dataset`。混合观测操作见[实验说明](../../plan/tianji-rail-mixed-observation-experiments.md)。

## 停止与记录

运动模式从当前接合姿态开始，不自动回位或张手。终端输入后按 Enter：`s` 开始，`p` 接管，`y` 成功，`n` 失败，`q` 退出；默认不设试验时限，数据集模式在窗口耗尽后结束当前试验，重新开始仍需手动输入 `s`。

暂停清空未来目标，双臂请求保持，健康手保持最后接受的目标；退出关闭驱动并禁用双手，先处理抓持物。推理故障后若线程退出，需重新启动入口。SDK 接受回执不代表物理到位，时效、停止和短时运动须现场核验。

| 文件 | 内容 |
| --- | --- |
| `preflight.json` | 模型预检；默认模拟输入，带 `--dataset` 时用数据集窗口，不代表闭环延迟或成功率 |
| `session.json` / `summary.json` | checkpoint、配置、设备元数据／会话结果 |
| `trial_*/trial.json` | 结果、接管、超时及日志丢弃／错误计数 |
| `trial_*/events.jsonl` | 预测、调度、设备提交与暂停；同步含动作块和步号 |
| `trial_*/timing.json` | 观测、预处理、推理、提交耗时；`sync_hold_ms` 统计完成动作块至预测返回的停顿 |

默认不保存视频。`--record-observations` 额外保存少量模型输入窗口和实测状态；末端版 `--trace` 保存完整决策与控制记录。日志队列满时丢弃新事件并计数，写入错误记入 `record_error`，均不自动改变试验结果。

成功率以所有已开始的运动试验为分母，包括失败、超时、接管和设备中止；影子试验不计入，无运动试验时为 `null`。文件职责与流程见[结构说明](real_eval_structure/README.md)，时序示例见[运行时间轴](task_flow_explanation/real_runtime_timing.md)。
