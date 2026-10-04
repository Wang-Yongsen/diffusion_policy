# 汽车导轨任务快速使用

模型：`DiffusionTransformerHybridImagePolicy`；三路 RGB＋双臂末端位姿＋双手关节，输出 52 维动作。

转换、GPU 冒烟训练、checkpoint 重载与评估入口已验收。目前仅一条真实演示，正式留出训练需补充独立演示。详见[构建计划](/home/jiyuchen/project/WYS/diffusion_policy/plan/tianji-rail-build-plan.md)。

## 1．环境

转换使用 `bimanual-teleop`，训练和评估使用 `robodiff`。GPU 为 NVIDIA RTX PRO 6000，设备为 `cuda:0`。

## 2．生成 zarr

已有修复数据可跳过此步；重新转换时改用不存在的输出目录。输入须为 `status=complete`，包含视频和 `raw.zarr`。

```bash
conda run -p /home/jiyuchen/miniforge3/envs/bimanual-teleop python /home/jiyuchen/project/WYS/bimanual-teleop/scripts/convert_recording.py \
  --input /home/jiyuchen/project/WYS/bimanual-teleop/recordings/20260924_204339_afdb5a7a \
  --output /home/jiyuchen/project/WYS/diffusion_policy/data/tianji/episodes_eef_rgb_repaired.zarr \
  --action-space eef \
  --conversion-config /home/jiyuchen/project/WYS/diffusion_policy/docs/tianji-rail/recording_conversion.yaml
```

默认 30 Hz、最多修复两帧缺口。本记录为 1887 帧、2 个片段，暂停接缝保留；其他采集记录使用[通用转换配置](/home/jiyuchen/project/WYS/bimanual-teleop/configs/recording_conversion.yaml)并重新复核暂停。

## 3．训练

```bash
cd /home/jiyuchen/project/WYS/diffusion_policy
conda run -p /home/jiyuchen/miniforge3/envs/robodiff python /home/jiyuchen/project/WYS/diffusion_policy/train.py \
  --config-name=train_diffusion_transformer_tianji_rail_workspace \
  task.dataset.dataset_path=/home/jiyuchen/project/WYS/diffusion_policy/data/tianji/episodes_eef_rgb_repaired.zarr \
  dataloader.batch_size=2 training.num_epochs=2 training.max_train_steps=2 \
  training.lr_warmup_steps=0 training.checkpoint_every=1 training.sample_every=1 \
  hydra.run.dir=/home/jiyuchen/project/WYS/diffusion_policy/data/outputs/tianji-rail-transformer-train
```

这是两轮短训练。图像缩放、尺寸、插值和来源划分在[任务配置](/home/jiyuchen/project/WYS/diffusion_policy/diffusion_policy/config/task/tianji_rail_image.yaml)修改；裁剪、模型和训练参数在[训练配置](/home/jiyuchen/project/WYS/diffusion_policy/diffusion_policy/config/train_diffusion_transformer_tianji_rail_workspace.yaml)修改，均有中文注释；`crop_shape: null` 关闭裁剪。

正式训练去掉短训覆盖参数，换新输出目录；有多条独立演示后设 `task.dataset.val_ratio=0.1`，自动监控 `val_loss`。恢复时保持数据、配置及输出目录，追加 `training.resume=true`；`num_epochs` 为总目标轮数，已完成的轮次不会重复。

## 4．离线评估

```bash
conda run -p /home/jiyuchen/miniforge3/envs/robodiff python /home/jiyuchen/project/WYS/diffusion_policy/eval.py \
  --checkpoint /home/jiyuchen/project/WYS/diffusion_policy/data/outputs/tianji-rail-transformer-train/checkpoints/latest.ckpt \
  --output_dir /home/jiyuchen/project/WYS/diffusion_policy/data/outputs/tianji-rail-transformer-eval \
  --device cuda:0
```

评估目录需不存在。输出 `eval_log.json`、`offline_report.json`（汇总、逐演示和逐窗口）：位置误差为毫米，旋转和手关节误差为度。单演示验证会跳过；离线误差不代表实机成功率。
