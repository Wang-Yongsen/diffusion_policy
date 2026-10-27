# Diffusion Policy · 天机导轨

三路 RGB 与双臂双手状态驱动的导轨策略训练、离线评估和真机评估。
数据采集与转换由相邻的 [bimanual-teleop](../bimanual-teleop/docs/recording_guide.md) 完成。

## 环境

```bash
# 首次创建；已有环境直接激活。
PYTHONNOUSERSITE=1 PIP_USER=false conda env create -f conda_environment_tianji.yaml
conda activate tianji-rail-diffusion-policy
```

移动仓库后执行 `python -m pip install --no-deps -e .`。
真机驱动与 SDK 使用仓库内的 [依赖快照](third_party/README.md)。

## 快速入口

```bash
# 用实际数据路径覆盖预设值；eef 为 58 维末端目标（6D旋转），joint 为 54 维关节目标。
python train.py --config-name=train_diffusion_unet_tianji_rail_smoke_workspace \
  task.dataset.dataset_path=/absolute/path/data.zarr task.action_space=joint

# 检查模型，不连接设备。
python -m diffusion_policy.cli.tianji eval \
  --checkpoint /absolute/path/latest.ckpt --check-only --output /tmp/rail-preflight

python -m diffusion_policy.cli.tianji --help
```

## 文档

| 内容 | 入口 |
| --- | --- |
| 数据要求、训练、恢复 | [训练](docs/tianji-rail/train.md) |
| 离线评估、真机、回放、回位 | [评估](docs/tianji-rail/evaluate.md) |
| 模块职责、测试、旧命令迁移 | [开发](docs/tianji-rail/development.md) |
| 设备单位、控制和停止约定 | [设备运行](docs/tianji-rail/device-runtime.md) |
| 既往实验的有效结论 | [实验摘要](docs/tianji-rail/results.md) |

## 目录

| 目录 | 职责 |
| --- | --- |
| `diffusion_policy/` | 数据、模型、训练、评估与命令入口 |
| `docs/tianji-rail/` | 当前操作说明 |
| `tests/` | 离线回归 |
| `third_party/` | 设备 SDK、FoundationPose 源码、任务模型及标定工具 |
| `workspace/` | [FoundationPose 导轨插入](workspace/rail_pose_insert/README.md)，依赖已内置 |
| `plan/` | 实施计划 |

## 上游

基于 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)。
作者 Cheng Chi、Siyuan Feng、Yilun Du、Zhenjia Xu、Eric Cousineau、Benjamin Burchfiel、Shuran Song。
其他基准任务沿用上游实现，使用方法见[上游项目](https://diffusion-policy.cs.columbia.edu/)。
许可证见 [LICENSE](LICENSE)。
