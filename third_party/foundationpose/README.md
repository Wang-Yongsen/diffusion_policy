# FoundationPose 本地快照

完整保留来源 FoundationPose 项目的源码目录结构，包含官方示例、BundleSDF、Docker 文件、构建脚本、文档/图片、本地 workspace 工具与测试、嵌套依赖源码。两组官方预训练权重保留在本地且不上传，导轨米制网格随源码保存，服务于 `workspace/rail_pose_insert/` 的 RGB-D 位姿估计。运行不读取相邻项目，不下载权重，不创建指向外部的软链接。

排除项仅为 Git 元数据、Python/编译缓存、历史输出、部署归档和工作区的大体积录制数据；内置三帧离线样本位于 `../rail_assets/examples/rgbd/`。完整源码文件映射及全部本地修改摘要记录在 `../rail_insert_manifest.json`。官方基准示例需要各自的数据集；本实验不依赖这些数据集。

- `estimater.py`、`Utils.py`、`learning/`：FoundationPose 推理。
- `weights/`：两组普通权重文件，合计约 247 MiB，不进入 Git。
- `workspace/src/rail_pose/`：导轨适配、分块推理、图像格式与交互。
- `workspace/vendor/`：nvdiffrast、PyTorch3D、SLDPRT 解码器源码。
- `mycpp/native/`：本地生成的扩展，不进入 Git；由安装脚本从 `mycpp/` 源码构建。
- `workspace/scripts/install.sh`：从本目录安装 GPU 扩展，并检查环境。标准 Python 包和 CUDA 编译工具仍需安装。

在 diffusion_policy 项目根目录运行：

```bash
bash third_party/foundationpose/workspace/scripts/install.sh
conda run -n foundationpose --no-capture-output python \
  third_party/foundationpose/workspace/run.py check-env
```

权重不上传 GitHub。运行前按 [本地权重说明](weights/README.md) 准备两组模型文件；源码检查可运行 `python workspace/rail_pose_insert/tools/verify_bundle.py --source-only`。第三方的大型演示/测试数据和视频也不进入 Git，源码、脚本、配置与许可证保持完整。

默认环境名 `foundationpose`；安装时可设置 `FOUNDATIONPOSE_ENV`，运行时对应修改实验配置的 `foundationpose_env`。`FOUNDATIONPOSE_PYTHON` 可覆盖感知解释器。独立感知进程用于隔离 PyTorch/Python 版本，不要求另有项目仓库。

来源、原始与本地文件摘要见 `../rail_insert_manifest.json`。上游许可证保留在 `LICENSE`，嵌套依赖保留各自许可证；本地适配不改变它们的许可。
