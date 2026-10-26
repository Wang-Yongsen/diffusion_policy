# 评估与设备操作

使用 `tianji-rail-diffusion-policy` 环境。以下路径替换为实际位置，输出目录必须是新目录。

## 离线评估

```bash
python eval.py --checkpoint /absolute/path/latest.ckpt \
  --dataset_path /absolute/path/data.zarr --output_dir /tmp/rail-offline --device cpu
```

动作误差写入 `offline_report.json`；它不是任务成功率。
评估默认使用 checkpoint 中的 EMA 和训练种子。
末端版须使用14维四元数观测、58维6D旋转动作的新 checkpoint；旧52维末端模型须重新训练。
现场与Zarr的 rotvec 状态由预处理转换；模型输出先解码，再进入原有控制器和追踪的52维接口。

## 模型检查与影子推理

```bash
python -m diffusion_policy.cli.tianji eval \
  --checkpoint /absolute/path/latest.ckpt --check-only --output /tmp/rail-check

python -m diffusion_policy.cli.tianji eval \
  --checkpoint /absolute/path/latest.ckpt --output /absolute/path/shadow-session
```

`--check-only` 只加载模型并检查输入，不连接设备；默认使用合成输入，指定 `--dataset` 时使用所选数据集窗口。
省略该参数时读取设备和相机，默认影子模式，不使能机器人。

## 真机评估

先核对[部署配置](../../diffusion_policy/config/tianji_rail_real_eval.yaml)的设备地址、相机和工作空间。

```bash
python -m diffusion_policy.cli.tianji eval \
  --checkpoint /absolute/path/latest.ckpt --motion \
  --output /absolute/path/new-session
```

| 参数 | 用途 |
| --- | --- |
| `--config PATH` | 部署配置 |
| `--set KEY=VALUE` | 覆盖已有配置项，可重复 |
| `--device cpu` | 覆盖推理设备 |
| `--inference-mode sync` | 完整动作块执行后重新观测；推理期间保持 |
| `--inference-mode async` | 异步更新未来动作，丢弃过期部分 |
| `--seed N` | 预热后、每次试验首个预测前重置采样种子 |
| `--record-observations` | 保存输入摘要与少量图像窗口 |
| `--trace --trace-capacity N` | 保存完整决策、命令和反馈；当前仅支持 eef |

部署的 `policy_hz` 必须匹配 checkpoint；10 Hz 模型加 `--set policy_hz=10`。
关节版从 checkpoint 自动识别，不需要另一个部署入口。
终端输入后按 Enter：`s` 开始、`p` 暂停并记接管、`y` 成功、`n` 失败、`q` 退出。
输出包含会话、逐次试验、事件和时间统计；完整追踪在停止后写盘。
同步模式支持 checkpoint 的单步或两步观测。动作逐步提交到双臂和双手，全部接受且经过对应策略周期后才推进；调度延迟不会跳过动作。
`trial_timeout_s: null` 表示不设自动时限，使用终端命令手动结束。

## 数据集观测部署

```bash
python -m diffusion_policy.cli.tianji eval \
  --checkpoint /absolute/path/latest.ckpt --dataset /absolute/path/data.zarr \
  --dataset-split val --inference-mode sync --check-only --output /tmp/rail-dataset-check
```

`--dataset-mode auto` 自动识别完整数据集或单条示教；完整数据按 checkpoint 的原划分选择，导出的单条示教保留原来源和划分信息。
可用 `--dataset-episode`、`--dataset-segment` 和 `--dataset-max-windows` 限定范围。
移除 `--check-only` 后默认影子执行；加 `--motion` 才允许手动开始真机执行。真实反馈继续用于设备控制，数据集窗口仅在上一动作块完整执行后推进。
该模式可用 `--record-observations`，暂不支持 `--trace`。

导出单条示教可运行 `python scripts/export_tianji_episodes.py --help`。

## 示范观测回放

```bash
python -m diffusion_policy.cli.tianji replay \
  --checkpoint /absolute/path/eef.ckpt --dataset /absolute/path/eef-data.zarr \
  --episode 0 --max-windows 24 --output /tmp/rail-replay
```

默认只预测示范观测并统计标签误差。加 `--motion` 才会执行预测，仍需手动 `s`。
运动回放使用同步模式和起始接近段；`--trace` 会另存当时真实观测，不将它送入模型。
当前回放支持 30 Hz 的 eef 数据。

`--observation-mode dataset-images-live-state` 使用示教图像和现场状态；
`--observation-mode live-images-dataset-state` 使用现场图像和示教状态。
两种混合模式都需要 `--motion`，每块使用新对齐的现场观测，并跳过首动作接近段；缺少现场输入时等待，不替换为示教数据。
`--full-segment` 执行完整片段并裁切最后一个动作块，不能与 `--max-windows` 同用。
整段追踪将图像存入文件映射，结束后导出 NPZ；导出失败时保留临时图像。

## 回位

```bash
python -m diffusion_policy.cli.tianji home --inspect
python -m diffusion_policy.cli.tianji home --part arms --side both
```

机械臂使用设备配置的 ready pose；手使用既有示范起点。回位过程可按 Q 或 Ctrl+C 取消。

## 诊断

```bash
python -m diffusion_policy.cli.tianji analyze --trial /absolute/path/trial_0001
python -m diffusion_policy.cli.tianji probe --checkpoint /absolute/path/latest.ckpt \
  --train-zarr /absolute/path/train.zarr --val-zarr /absolute/path/val.zarr \
  --output /tmp/rail-probe
```

`analyze` 输出追踪报告和对齐图；`probe` 对比显式输入的观测、预测及标签误差。
`probe --reference-npz PATH` 可额外比较一份原始观测窗口。
更多参数见各子命令的 `--help`。
