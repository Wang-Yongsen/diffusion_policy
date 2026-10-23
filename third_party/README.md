# 本地依赖快照

导轨预插入的完整项目依赖现已内置：`foundationpose/` 保存完整项目源码、导轨适配及 GPU 扩展源码（权重保留本地，不上传）；`rail_assets/` 保存机器人/夹具模型和离线样本；`camera_calibration/` 保存实机标定工具。它们与下述设备快照一起工作，不需要相邻 FoundationPose 或 bimanual-teleop 仓库。安装与运行见 [预插入说明](../workspace/rail_pose_insert/README.md)，新增文件来源和摘要见 `rail_insert_manifest.json`。

标准 Python 库、CUDA/驱动及设备 SDK 的安装由环境文件和本地构建脚本定义。项目自包含指项目代码、权重及模型资源自包含，不复制整个 Conda 或操作系统。

`third_party` 保存真机评估需要的设备控制、运动学、实时相机读取和SDK资源。本项目采用随仓库保存的快照，真机评估的设备接口不依赖相邻仓库。数据采集、整理和转换由 bimanual-teleop 维护，模型项目直接读取其输出的训练zarr。

```text
third_party/
  bimanual_teleop/          # 真机评估的双臂伺服、无极手与实时相机依赖
    manifest.json          # 来源版本、原文件摘要、快照摘要及本地适配说明
    vendor/tianji/          # 官方 SDK、Linux x86_64 原生库、M6 模型及许可证
```

Python名称为 `diffusion_policy.third_party.bimanual_teleop`，由 `setup.py` 映射到此目录。公开的设备配置维护在 `diffusion_policy/config/`。训练与评估环境配置为 `conda_environment_tianji.yaml`，环境名是 `tianji-rail-diffusion-policy`。`recording/camera.py`、`config.py`和`sink.py`用于实时观测，目录名不表示本项目提供离线数据整理或转换。

## 来源及本地调整

快照来自用户工作区的 `bimanual-teleop`，提交为 `540cb01fd345f1cfb2e562ec90e0fbffc9c252f8`，取入时包含未提交的已验收修改，因此提交号不能独立还原此次快照。[清单](bimanual_teleop/manifest.json)记录取入时每个源文件和本地文件的 SHA-256。只取天机导轨所需的传递依赖，没有迁入完整遥操作程序、Quest 或手套采集入口。

本地调整包括包导入名称、默认配置资源定位及说明链接；伺服约束和设备协议沿用源实现。2026-10-14移除了离线整理、转换、原始缓存存储及其专用辅助代码，保留真机评估依赖。2026-10-16按 bimanual-teleop 提交 `284b1e3` 同步相机约定：`camera_1` 为机器人左臂相机，RGB 读取后旋转 180°；`camera_2` 为右臂相机；相机元数据记录 `image_rotation_deg`。模型及厂家Python／`.so`文件保持原字节，厂家清单继续验证其摘要。

源项目未提供项目级 LICENSE，不能据此声明为某一种开源许可证。天机官方 SDK 的 Apache-2.0 [LICENSE](bimanual_teleop/vendor/tianji/LICENSE)及厂家来源清单原样保留；无极 SDK 和 RealSense 通过环境依赖安装，适用各自的许可证。来源信息用于追溯，运行不访问来源仓库。

## 更新

按需取入真机评估的依赖，保留厂家匹配的封装、库、模型和许可证，更新清单。检查实时观测、控制停止、原生运动学和打包资源；厂家SDK更新仍需现场重新验收。数据整理与转换的更新在 bimanual-teleop 中完成。

使用说明见[导轨任务](../docs/tianji-rail/README.md)和[设备运行](../docs/tianji-rail/device-runtime.md)；数据处理见 bimanual-teleop 的[采集操作](../../bimanual-teleop/docs/recording_guide.md)。
