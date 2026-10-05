# 本地依赖快照

`third_party` 保存本项目天机导轨流程需要的第三方代码与资源。目录名称是一种组织约定，既可以放 Git 子模块，也可以保存代码快照；本项目采用随仓库保存的快照，运行、安装和移动项目均不需要相邻仓库或拉取子模块。

```text
third_party/
  bimanual_teleop/          # 整理转换、双臂伺服、无极手和相机的依赖闭包
    manifest.json          # 来源版本、原文件摘要、快照摘要及本地适配说明
    vendor/tianji/          # 官方 SDK、Linux x86_64 原生库、M6 模型及许可证
```

Python 名称为 `diffusion_policy.third_party.bimanual_teleop`，由根目录 `setup.py` 将包映射到此目录。没有符号链接、外部 editable 安装或运行时下载。公开的设备配置统一维护在 `diffusion_policy/config/`，不在这里再保存第二份配置。安装仅需项目根目录的 `conda_environment_tianji.yaml`；本项目自身的 editable 安装方便开发。

## 来源及本地调整

快照来自用户工作区的 `bimanual-teleop`，提交为 `540cb01fd345f1cfb2e562ec90e0fbffc9c252f8`，取入时包含未提交的已验收修改，因此提交号不能独立还原此次快照。[清单](bimanual_teleop/manifest.json)记录取入时每个源文件和本地文件的 SHA-256。只取天机导轨所需的传递依赖，没有迁入完整遥操作程序、Quest 或手套采集入口。

本地调整包括包导入名称、默认配置资源定位、说明链接及 spool 归档使用相对符号链接；数据转换数学规则、伺服约束和设备协议沿用源实现。模型及厂家 Python／`.so` 文件保持原字节，厂家清单继续验证其摘要。

源项目未提供项目级 LICENSE，不能据此声明为某一种开源许可证。天机官方 SDK 的 Apache-2.0 [LICENSE](bimanual_teleop/vendor/tianji/LICENSE)及厂家来源清单原样保留；无极 SDK 和 RealSense 通过环境依赖安装，适用各自的许可证。来源信息用于追溯，运行不访问来源仓库。

## 更新

按需取入新的依赖闭包，保留厂家匹配的封装、库、模型和许可证，应用上述少量适配后更新清单。先检查严格／修复转换、控制停止和原生运动学回归，再验证 wheel 资源完整及项目迁移后的独立安装。不要用相邻目录或临时 `sys.path` 修改代替快照。厂家 SDK 更新仍需现场重新验收。

使用说明见[导轨任务](../docs/tianji-rail/README.md)、[数据整理](../docs/tianji-rail/data-preparation.md)和[设备运行](../docs/tianji-rail/device-runtime.md)。
