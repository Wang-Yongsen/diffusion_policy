# 汽车导轨任务快速使用

本项目负责模型训练、离线评估和真机评估；数据采集、整理和转换在 [bimanual-teleop](../../../bimanual-teleop/README.md) 中完成。模型使用三路 RGB、双臂末端位姿和双手关节，预测52维动作。构建状态见[分阶段计划](../../plan/tianji-rail-build-plan.md)。

## 1．环境

在 `diffusion_policy` 项目根目录操作。训练和评估使用 `tianji-rail-diffusion-policy` 环境：

```bash
conda env create -f conda_environment_tianji.yaml  # 仅首次创建
conda activate tianji-rail-diffusion-policy
```

已有环境无需重复创建。环境名由 YAML 的 `name: tianji-rail-diffusion-policy` 指定，本机路径为 `/opt/miniconda3/envs/tianji-rail-diffusion-policy`。该环境固定NumPy 1.26.4，与pandas 1.5.3兼容；原 `tianji-rail` 环境不作为本项目的默认环境。bimanual-teleop 的数据处理使用其自己的 `bimanual-teleop` 环境。

用 VS Code／Cursor 打开本项目根目录时，[项目设置](../../.vscode/settings.json)将默认Python解释器指向该训练环境；新建终端通过[项目 Bash 配置](../../.vscode/bashrc)自动激活。已打开的终端需手动执行上面的激活命令；编辑器若记住旧解释器，执行 `Python: Select Interpreter`，选择 `/opt/miniconda3/envs/tianji-rail-diffusion-policy/bin/python`。换机器时按实际Conda位置调整解释器设置。

真机评估需要的设备接口、实时相机读取和SDK采用本项目 `third_party` 的本地快照，来源见[本地依赖](../../third_party/README.md)。只有本项目自身以editable安装；移动源码后执行 `python -m pip install --no-deps -e .` 更新位置。

确认当前终端使用的解释器：

```bash
python -c "import sys; print(sys.executable)"
```

本机应输出 `/opt/miniconda3/envs/tianji-rail-diffusion-policy/bin/python`。

## 2．读取训练数据

采集、整理和转换的操作见 bimanual-teleop 的[数据采集操作](../../../bimanual-teleop/docs/recording_guide.md)，格式和转换规则见[数据采集参考](../../../bimanual-teleop/docs/recording_reference.md)。完成后直接指定生成的 `.zarr` 路径，无需把原始记录或数据处理代码复制到本项目。

当前示例读取 `../bimanual-teleop/datasets/rail_place.zarr`。`task.dataset.dataset_path` 决定读取哪个数据集；Dataset负责来源划分、连续片段内采样及训练集归一化。

当前模型的数据契约为三路RGB原图640×480、12维末端位姿、40维手关节和52维绝对EEF动作（左臂6、右臂6、左手20、右手20）；位置用米，旋转向量和关节角用弧度。原始演示边界为 `meta/episode_ends`，连续片段边界为 `meta/segment_ends`，训练窗口不跨片段边界。数据中的其他字段不会自动成为模型输入。

## 3．训练

**训练参数已在 YAML 中设定，命令行只需写本次要覆盖的值。** 命令行参数优先于 YAML，只对本次运行生效。长期使用的参数在配置文件修改：

| 配置文件                                                                                                | 内容                                                                 |
| ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [任务配置](../../diffusion_policy/config/task/tianji_rail_image.yaml)                                    | 数据路径、验证比例、图像预处理；默认使用旧单演示数据，验证比例为0    |
| [正式训练配置](../../diffusion_policy/config/train_diffusion_transformer_tianji_rail_workspace.yaml)     | 默认每批64个样本、4个读取进程、600轮、每轮完整训练                   |
| [短训练配置](../../diffusion_policy/config/train_diffusion_transformer_tianji_rail_smoke_workspace.yaml) | 继承正式配置，每批2个样本、0个读取进程、每轮2批、共2轮，用于检查流程 |

当前按原图训练：`task.image_shape=[3,480,640]`、`task.dataset.resize_enabled=false`、`policy.crop_shape=null`，保留宽640、高480的整幅图像。改变输入尺寸后新建实验；恢复训练须使用 checkpoint 对应的原配置。

当前正式配置为 `batch_size=64`、`num_workers=4`、`persistent_workers=true`，验证读取参数自动同步；短训练保持2／0／false。历史原图GPU测试采用每批32个样本，峰值分配显存约26.6GiB，详情见[测试结果](../../wys/2026-10-04-rail-native-resolution-test/test-summary.json)。

**输出目录默认按启动日期和时间自动创建**，格式为 `data/outputs/年.月.日/时.分.秒_tianji_rail/`，例如 `data/outputs/2026.10.14/18.30.00_tianji_rail/`。新训练命令不需要传 `hydra.run.dir`；需要自定义目录时再用该参数覆盖。目录保存 `checkpoints/latest.ckpt`（模型及训练状态）、`logs.json.txt`（日志）和 `.hydra/`（配置记录），路径均相对于项目目录。恢复训练必须显式指定原输出目录。

### 短训练

使用任务配置中的旧数据，只检查训练流程能否跑通：

```bash
python train.py --config-name=train_diffusion_transformer_tianji_rail_smoke_workspace
```

### 正式训练

以下示例使用真机数据 `rail_place.zarr`，按独立演示来源留出验证集。当前51条来源按 `val_ratio=0.05` 划分为48条训练、3条验证，较优模型按 `val_loss` 选择。其余参数使用正式训练配置，输出目录按启动时间自动命名：

```bash
python train.py --config-name=train_diffusion_transformer_tianji_rail_workspace \
  task.dataset.dataset_path=../bimanual-teleop/datasets/rail_place.zarr \
  task.dataset.val_ratio=0.05
```

### 恢复训练

沿用原实验的数据、配置和输出目录，加 `training.resume=true`。将下面的 `run_dir` 换成实际生成的训练输出目录；示例日期和时间仅说明格式。先确认模型文件存在再启动，不能让恢复训练另建时间目录：

```bash
run_dir=data/outputs/2026.10.14/18.30.00_tianji_rail
test -f "$run_dir/checkpoints/latest.ckpt" && \
python train.py --config-name=train_diffusion_transformer_tianji_rail_workspace \
  task.dataset.dataset_path=../bimanual-teleop/datasets/rail_place.zarr \
  task.dataset.val_ratio=0.05 \
  "hydra.run.dir=$run_dir" training.resume=true
```

`training.num_epochs` 是总目标轮数：默认600轮，若 checkpoint 已完成100轮，则继续训练500轮。

## 4．离线评估

将 `run_dir` 换成实际训练输出目录；报告目录须尚不存在。数据、模型及预处理配置从 checkpoint 读取：

```bash
run_dir=data/outputs/2026.10.14/18.30.00_tianji_rail
python eval.py \
  --checkpoint "$run_dir/checkpoints/latest.ckpt" \
  --output_dir "${run_dir}-eval-001"
```

若旧checkpoint内的数据路径失效，可在同一命令加入 `--dataset_path ../bimanual-teleop/datasets/rail_place.zarr`，只覆盖本次评估的数据位置，须保持数据版本和来源划分一致。

评估使用EMA模型的评估模式，关闭dropout；随机种子默认沿用checkpoint的训练种子，可用 `--seed 42` 指定。训练中的验证损失也使用同一EMA模型，按验证样本数加权计算。

对比采样截断时，沿用上面设置的 `run_dir`，使用同一个checkpoint和随机种子，并为两组结果指定不同目录：

```bash
python eval.py \
  --checkpoint "$run_dir/checkpoints/latest.ckpt" \
  --output_dir "${run_dir}-eval-clip-on" \
  --seed 42 --clip-sample
python eval.py \
  --checkpoint "$run_dir/checkpoints/latest.ckpt" \
  --output_dir "${run_dir}-eval-clip-off" \
  --seed 42 --no-clip-sample
```

采样截断只覆盖本次离线评估，不改变checkpoint或归一化参数；未指定时沿用checkpoint设置。报告为 `eval_log.json`、`offline_report.json`、`eval_settings.json`；位置误差用毫米，旋转与手关节误差用度。没有验证窗口时记录跳过原因。离线误差不代表实机成功率。

## 5．真机评估

当前文件结构、三种运行模式与闭环流程见[真机评估结构与流程图](real_eval_structure/README.md)。

将 `run_dir` 换成实际训练输出目录。首先检查模型重载及 batch=1 推理，不连接任何设备：

```bash
run_dir=data/outputs/2026.10.14/18.30.00_tianji_rail
python eval_tianji_rail.py \
  --checkpoint "$run_dir/checkpoints/latest.ckpt" \
  --check-only --output data/real_eval/check-001
```

该检查使用模拟图像，不能作为实时延迟或任务成功率。正式评估应使用正式训练完成后选定的模型。

设备已连接、其他控制程序已退出时，省略 `--check-only` 进入影子模式，默认不配置控制参数、不使能或下发动作：

```bash
python eval_tianji_rail.py \
  --checkpoint "$run_dir/checkpoints/latest.ckpt" \
  --output data/real_eval/shadow-001
```

[部署配置](../../diffusion_policy/config/tianji_rail_real_eval.yaml)引用项目内机械臂、手和相机配置（[设备约定](device-runtime.md)），并为时间阈值、速度、试验超时等参数提供注释。模型形状、RGB 预处理、EMA 和归一化器以 checkpoint 为准；首版支持 30 Hz、`n_latency_steps=0`、52 维绝对 EEF 动作。相机只采三路 RGB。

现场填写左右臂基座系下法兰的 `workspace.min/max`，完成影子观测、控制周期、停止和短时执行验收后，加入 `--motion`。在终端输入 `s` 后按 Enter 才会开始每次试验；`p` 暂停并记接管，`y` 记成功，`n` 记失败，`q` 退出。影子模式自动记录一条限时试验；运动模式重启后仍须手动开始，不自动回位或恢复旧动作。Ctrl+C 同样进入退出清理。

暂停清空未来目标，双臂请求停止，健康手继续发送最后已接受的目标；退出会关闭驱动并禁用手。退出前应由操作员处理好抓持物。模型推理或观测故障中止后，按提示重新启动入口；程序不会自动恢复运动。

每条试验保存 `trial.json`、三路 MP4、`events.jsonl` 和 `timing.json`；会话保存 `session.json`、`summary.json`。视频为 30 fps 编码，曝光/SDK 时间应查日志中的逐帧记录。日志含原始反馈到达时间、实测状态、预测、计划时刻、提交及 SDK 事件；SDK 接受并不等于实际到位。成功率分母固定为所有开始的运动试验，包括失败、超时、接管和设备中止；影子试验不进入分母。无运动试验时成功率为 null。

结构与时间轴见[图示说明](../../wys/2026-10-04-rail-real-eval-design/explanation.md)。当前硬件时效阈值为待现场验收的初始值，没有开启未经标定的动作提前量或观测延迟补偿。
