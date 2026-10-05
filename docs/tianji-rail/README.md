# 汽车导轨任务快速使用

模型使用三路 RGB、双臂末端位姿和双手关节，预测52维动作。构建状态见[分阶段计划](../../plan/tianji-rail-build-plan.md)。

## 1．环境

首次在项目根目录创建统一环境；随后只激活一次，整理、转换、训练和评估均使用同一个 Python：

```bash
conda env create -f conda_environment_tianji.yaml
conda activate tianji-rail
```

已有环境无需重复创建。环境名由 YAML 的 `name: tianji-rail` 指定，存放在 Conda 默认环境目录，本机为 `~/miniforge3/envs/tianji-rail`。

用 VS Code／Cursor 打开本项目根目录时，[项目设置](../../.vscode/settings.json)将默认 Python 解释器指向该环境；新建终端通过[项目 Bash 配置](../../.vscode/bashrc)自动激活，不修改全局 shell 配置。这里由终端启动配置负责激活，关闭 Python 扩展重复激活，避免已有解释器选择覆盖终端环境。设置只影响新建终端；若编辑器此前记住其他解释器，执行一次 `Python: Select Interpreter`，选择 `~/miniforge3/envs/tianji-rail/bin/python`。

设备接口、SDK 与整理转换代码采用本项目 `third_party` 的本地快照，只有本项目自身以 editable 安装。项目在本机移动后，重新执行 `python -m pip install --no-deps -e .` 更新项目安装位置；换机器时创建同名环境，并按该机器的 Conda 安装位置调整解释器及 Bash 配置。数据和模型随项目保留，配置及说明使用项目内相对路径。来源及更新方式见[本地依赖](../../third_party/README.md)。

## 2．数据

原始数据格式、整理行为与转换规则见[数据准备](data-preparation.md)。将兼容会话放到本项目的数据目录，在本项目整理和转换（将 `<session>` 换成实际路径，输出目录必须尚不存在）：

```bash
python scripts/finalize_recording.py \
  --input data/recordings/<session> --spool-archive data/raw_spools
python scripts/convert_recording.py \
  --input data/recordings/<session> \
  --output data/<session>/episodes_eef_repaired.zarr --action-space eef \
  --conversion-config diffusion_policy/config/recording_conversion.yaml
```

已 complete 的条目可跳过整理。整理会删除标记为作废的条目，成功后将 `raw_spool` 归档并留下符号链接；`--refinalize` 显式重新生成派生产物。转换未传 `--conversion-config` 时使用严格模式；会话复核配置只适用于其对应会话。两个入口调用本项目内同一份已验收转换实现。已有训练 zarr 可以直接使用，读取路径由 `task.dataset.dataset_path` 决定。

## 3．训练

**训练参数已在 YAML 中设定，命令行只需写本次要覆盖的值。** 命令行参数优先于 YAML，只对本次运行生效。长期使用的参数在配置文件修改：

| 配置文件 | 内容 |
| --- | --- |
| [任务配置](../../diffusion_policy/config/task/tianji_rail_image.yaml) | 数据路径、验证比例、图像预处理；默认使用旧单演示数据，验证比例为0 |
| [正式训练配置](../../diffusion_policy/config/train_diffusion_transformer_tianji_rail_workspace.yaml) | 默认每批32个样本、4个读取进程、600轮、每轮完整训练 |
| [短训练配置](../../diffusion_policy/config/train_diffusion_transformer_tianji_rail_smoke_workspace.yaml) | 继承正式配置，每批2个样本、0个读取进程、每轮2批、共2轮，用于检查流程 |

当前按原图训练：`task.image_shape=[3,480,640]`、`task.dataset.resize_enabled=false`、`policy.crop_shape=null`，保留宽640、高480的整幅图像。改变输入尺寸后新建实验；恢复训练须使用 checkpoint 对应的原配置。

正式配置已采用原图GPU实测推荐值：`batch_size=32`、`num_workers=4`、`persistent_workers=true`，验证读取参数自动同步；短训练保持2／0／false。实测峰值分配显存约26.6GiB，详情见[测试结果](../../wys/2026-10-04-rail-native-resolution-test/test-summary.json)。

**输出目录**由 `hydra.run.dir` 指定，程序自动创建，用于保存 `checkpoints/latest.ckpt`（模型及训练状态）、`logs.json.txt`（日志）和 `.hydra/`（配置记录）。新实验换一个目录名称，恢复训练使用原目录。下方的 `data/outputs/...` 均相对于项目目录。

### 短训练

使用任务配置中的旧数据，只检查训练流程能否跑通：

```bash
python train.py --config-name=train_diffusion_transformer_tianji_rail_smoke_workspace \
  hydra.run.dir=data/outputs/tianji-rail-smoke-002
```

### 正式训练

使用新批修复数据，并按原始演示来源留出验证集。当前7条来源按 `val_ratio=0.1` 划分为6条训练、1条验证，较优模型按 `val_loss` 选择。其余参数使用正式训练配置：

```bash
python train.py --config-name=train_diffusion_transformer_tianji_rail_workspace \
  task.dataset.dataset_path=data/tianji-202610041246/episodes_eef_repaired.zarr \
  task.dataset.val_ratio=0.1 \
  hydra.run.dir=data/outputs/tianji-rail-formal-001
```

### 恢复训练

沿用原实验的数据、配置和输出目录，加 `training.resume=true`。以下对应上面的正式实验，先确认模型文件存在再启动：

```bash
test -f data/outputs/tianji-rail-formal-001/checkpoints/latest.ckpt && \
python train.py --config-name=train_diffusion_transformer_tianji_rail_workspace \
  task.dataset.dataset_path=data/tianji-202610041246/episodes_eef_repaired.zarr \
  task.dataset.val_ratio=0.1 \
  hydra.run.dir=data/outputs/tianji-rail-formal-001 training.resume=true
```

`training.num_epochs` 是总目标轮数：默认600轮，若 checkpoint 已完成100轮，则继续训练500轮。

## 4．离线评估

指定待评估模型和一个尚不存在的报告目录；数据、模型及预处理配置从 checkpoint 读取：

```bash
python eval.py \
  --checkpoint data/outputs/tianji-rail-formal-001/checkpoints/latest.ckpt \
  --output_dir data/outputs/tianji-rail-formal-001-eval-001
```

若迁移后旧 checkpoint 内的数据路径失效，可在同一命令加入 `--dataset_path data/<session>/episodes_eef_repaired.zarr`，只覆盖本次评估的数据位置。

报告为 `eval_log.json`、`offline_report.json`；位置误差用毫米，旋转与手关节误差用度。没有验证窗口时记录跳过原因。离线误差不代表实机成功率。

## 5．真机评估

首先检查模型重载及 batch=1 推理，不连接任何设备：

```bash
python eval_tianji_rail.py \
  --checkpoint data/outputs/tianji-rail-formal-001/checkpoints/latest.ckpt \
  --check-only --output data/real_eval/check-001
```

该检查使用模拟图像，不能作为实时延迟或任务成功率。正式评估应使用正式训练完成后选定的模型。

设备已连接、其他控制程序已退出时，省略 `--check-only` 进入影子模式，默认不配置控制参数、不使能或下发动作：

```bash
python eval_tianji_rail.py \
  --checkpoint data/outputs/tianji-rail-formal-001/checkpoints/latest.ckpt \
  --output data/real_eval/shadow-001
```

[部署配置](../../diffusion_policy/config/tianji_rail_real_eval.yaml)引用项目内机械臂、手和相机配置（[设备约定](device-runtime.md)），并为时间阈值、速度、试验超时等参数提供注释。模型形状、RGB 预处理、EMA 和归一化器以 checkpoint 为准；首版支持 30 Hz、`n_latency_steps=0`、52 维绝对 EEF 动作。相机只采三路 RGB。

现场填写左右臂基座系下法兰的 `workspace.min/max`，完成影子观测、控制周期、停止和短时执行验收后，加入 `--motion`。在终端输入 `s` 后按 Enter 才会开始每次试验；`p` 暂停并记接管，`y` 记成功，`n` 记失败，`q` 退出。影子模式自动记录一条限时试验；运动模式重启后仍须手动开始，不自动回位或恢复旧动作。Ctrl+C 同样进入退出清理。

暂停清空未来目标，双臂请求停止，健康手继续发送最后已接受的目标；退出会关闭驱动并禁用手。退出前应由操作员处理好抓持物。模型推理或观测故障中止后，按提示重新启动入口；程序不会自动恢复运动。

每条试验保存 `trial.json`、三路 MP4、`events.jsonl` 和 `timing.json`；会话保存 `session.json`、`summary.json`。视频为 30 fps 编码，曝光/SDK 时间应查日志中的逐帧记录。日志含原始反馈到达时间、实测状态、预测、计划时刻、提交及 SDK 事件；SDK 接受并不等于实际到位。成功率分母固定为所有开始的运动试验，包括失败、超时、接管和设备中止；影子试验不进入分母。无运动试验时成功率为 null。

结构与时间轴见[图示说明](../../wys/2026-10-04-rail-real-eval-design/explanation.md)。当前硬件时效阈值为待现场验收的初始值，没有开启未经标定的动作提前量或观测延迟补偿。
