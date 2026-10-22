# 训练

在项目根目录激活 `tianji-rail-diffusion-policy` 环境。首次安装见[根 README](../../README.md)。

## 数据

每次显式指定 `task.dataset.dataset_path`；路径相对于项目根目录，或使用绝对路径。
数据由 [bimanual-teleop](../../../bimanual-teleop/docs/recording_guide.md) 导出为 Zarr。

| 参数 | 要求 |
| --- | --- |
| `task.action_space=eef` | 三路 RGB、14 维末端位姿（每臂 xyz＋xyzw 四元数）、40 维手关节；58 维动作（每臂 xyz＋6D旋转，再接双手关节） |
| `task.action_space=joint` | 三路 RGB、14 维臂关节、40 维手关节；54 维动作 |
| `task.sample_hz` | 默认 30；必须匹配数据的 `target_fps` 和实际时间间隔 |
| `task.image_shape` | `[通道,高,宽]`；默认 `[3,480,640]` |
| `task.dataset.val_ratio` | 按原始示范来源划分验证集；同源片段不分到两侧 |

动作是绝对目标，双侧顺序为左臂、右臂、左手、右手；位置为米，关节角为弧度。
末端版模型使用四元数观测和6D旋转动作；6D表示按 DP/PyTorch3D 约定取旋转矩阵前两行展开。
四元数、6D旋转维度保持原尺度，位置及手关节按训练集逐维 Min–Max 归一化。
Zarr继续保存12维 xyz＋rotvec 观测和52维动作，Dataset读取时转换，不需重新导出。
部署预处理执行同样的观测转换，预测动作解码为52维控制目标后再进入原有校验、追踪和设备控制。
原有52维末端版 checkpoint 需要重新训练；关节版维度、归一化和部署接口不变。
窗口不跨暂停或缺帧边界。归一化只使用训练来源，验证和部署不执行图像增强。

## 选择配置

| 配置后缀（共同前缀 `train_diffusion_`） | 用途 |
| --- | --- |
| `unet_tianji_rail_workspace` | UNet，默认 2 帧观测 |
| `transformer_tianji_rail_workspace` | Transformer，默认 1 帧观测 |
| `unet_tianji_rail_320_crop_workspace` | 缩放至 320×240，裁剪至 288×216 |
| `unet_tianji_rail_320_crop_aug_workspace` | 缩放裁剪，并加入训练时颜色、模糊、灰度增强 |

上述配置将结尾 `workspace` 换为 `smoke_workspace` 即为短训练。
默认预测 16 步、执行 8 步；必须满足 `n_obs_steps-1+n_action_steps <= horizon`。

## 短训练与正式训练

```bash
DATA=/absolute/path/data.zarr
python train.py --config-name=train_diffusion_unet_tianji_rail_smoke_workspace \
  task.dataset.dataset_path="$DATA" task.action_space=joint

python train.py --config-name=train_diffusion_unet_tianji_rail_workspace \
  task.dataset.dataset_path="$DATA" task.action_space=joint
```

10 Hz 数据加 `task.sample_hz=10`。更换动作空间、频率、数据或模型时使用新的输出目录。
输出位于 `data/outputs/`，包含 checkpoint、配置、训练日志及离线报告。
默认关闭外部 WandB 上传。

## 恢复训练

使用原配置、原数据和原输出目录，`training.num_epochs` 是总轮数：

```bash
python train.py --config-name=train_diffusion_unet_tianji_rail_workspace \
  task.dataset.dataset_path="$DATA" task.action_space=joint \
  training.resume=true hydra.run.dir=/absolute/path/original-run
```

## 常改参数

| 参数 | 含义 |
| --- | --- |
| `dataloader.batch_size` | 每批窗口数；显存不足时减小 |
| `dataloader.num_workers` | 数据进程数；为 0 时同时设置 `persistent_workers=false` |
| `training.num_epochs` | 总训练轮数，默认 600 |
| `training.seed` | 来源划分和训练随机种子 |
| `task.dataset.resize_enabled` | 开启后按 `task.image_shape` 缩放 |
| `policy.crop_shape` | `[高,宽]`；训练随机裁剪，评估固定中心裁剪 |
| `policy.num_inference_steps` | 扩散推理迭代次数 |

增强按同一路相机的整个历史窗口抽取一次随机参数；不同相机独立抽取。
UNet 学习率为 `optimizer.lr`，Transformer 为 `optimizer.learning_rate`。
