# 本地权重

权重不提交或上传到 GitHub。将本地已有的两个目录放在这里，每组均需 `config.yml` 和 `model_best.pth`：

- `2023-10-28-18-33-37/`
- `2024-01-11-20-02-45/`

准备后运行 `python workspace/rail_pose_insert/tools/verify_bundle.py`（从 diffusion_policy 根目录）。仅检查公开源码时添加 `--source-only`。不要通过 `git add -f` 提交此处权重或其他模型检查点。
