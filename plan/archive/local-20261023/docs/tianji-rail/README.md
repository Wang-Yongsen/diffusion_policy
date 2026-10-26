# 天机导轨真机测试（简洁版）

当前示例：UNet 关节版 epoch60，30 Hz，同步执行。模型切换、训练和参数见[完整版](reference.md)。

## 1．准备环境与模型

在自己的终端中执行，设备连接前退出其他控制程序。

```bash
cd /home/yuchen/project/diffusion_policy
conda activate tianji-rail-diffusion-policy
export PYTHONNOUSERSITE=1

checkpoint=data/outputs/2026.10.20/15.30.50_tianji_rail_unet_joint/checkpoints/epoch=0060.ckpt
run_tag="$(date +%Y%m%d-%H%M%S)"
```

## 2．模型预检

不连接设备。通过后继续下一步。

```bash
python eval_tianji_rail.py \
  --checkpoint "$checkpoint" --set policy_hz=30 \
  --inference-mode sync --check-only \
  --output "data/real_eval/${run_tag}-check"
```

## 3．影子测试

连接双臂、双手和三路相机。读取真实观测，不下发运动目标；默认不设时限，手动结束。

```bash
python eval_tianji_rail.py \
  --checkpoint "$checkpoint" --set policy_hz=30 \
  --inference-mode sync \
  --output "data/real_eval/${run_tag}-shadow"
```

## 4．运动测试

确认观测、目标和运动限制后执行。从当前姿态开始，不自动回位。

```bash
python eval_tianji_rail.py \
  --checkpoint "$checkpoint" --set policy_hz=30 \
  --inference-mode sync --motion \
  --output "data/real_eval/${run_tag}-motion"
```

输入后按 Enter：`s` 开始，`p` 暂停并记接管，`y` 成功，`n` 失败，`q` 退出。退出会禁用双手，先处理好抓持物。

默认不设自动结束时限，正常运行由操作员手动暂停或结束。若原启动命令带有 `--set trial_timeout_s=30`，请删除该参数或改为 `--set trial_timeout_s=null`。

每次重跑重新执行第1步，避免输出目录重名。关节版不要加 `--trace`；Transformer epoch150 的单步观测和 UNet 的两步观测均支持 `sync`，观测步数从 checkpoint 自动读取。

## 用本地单 episode 观测部署真机

已从 `raids-pro6000` 缓存六条关节版 episode，无需在本机保存完整处理后数据。新筛选的训练示教为 `000013`、`000030`、`000033`，优先使用 `000013`；原训练示教 `000000` 和验证示教 `000003`、`000034` 继续保留。来源及原划分信息随文件保存。新增三条均无转换质量告警，关键画面可见取料、搬运、放置和撤手，但没有独立的成功标签。

```bash
checkpoint=data/outputs/2026.10.19/14.08.04_tianji_rail_joint/checkpoints/epoch=0150.ckpt
run_tag="$(date +%Y%m%d-%H%M%S)"

# 训练 episode：输入 s 后开始运动。
python eval_tianji_rail.py --checkpoint "$checkpoint" \
  --dataset data/replay_obs/rail_place3_joint_episode_000013.zarr \
  --dataset-mode episode --dataset-split train \
  --set policy_hz=30 --inference-mode sync --motion \
  --record-observations --batch-seed 42 \
  --output "data/real_eval/${run_tag}-episode-train-motion"
```

验证 episode 同时改文件和划分：`--dataset data/replay_obs/rail_place3_joint_episode_000034.zarr --dataset-split val`，并更换输出目录；也可选 `000003`。无设备预检把 `--motion` 换成 `--check-only`，此时读取该 episode 的数据窗口。先检查前两个执行块可加 `--dataset-max-windows 2`。

单 episode 不重新划分，原归属由服务器导出信息和 checkpoint 配置恢复；归属不匹配会报错。完整动作块执行、真实反馈检查和手动接合保持原有逻辑。数据窗口耗尽后暂停，再按 `s` 从片段开头开始。设备仍需连接，策略不会用现场观测纠正偏差。

缓存帧数、筛选依据、模式选择和再次下载命令见[完整部署说明](reference.md#用训练集或验证集观测部署真机)。

这些缓存同时保存 joint 与 EEF 实测观测，部署时可共用同一 `--dataset` 路径：更换 checkpoint 后自动选择输入字段，动作仍由对应模型预测。数据集动作标签不参与部署；训练若要共用两类动作标签，需要另行保存两套目标，见[合并说明](reference.md#同一-episode-的-jointeef-能否合并)。
