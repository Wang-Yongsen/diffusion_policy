# 官方天机 SDK

此目录包含厂家发布的原始运行文件，来源为 [TJ_FX_ROBOT_CONTRL_SDK](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK/tree/02440e886fb59095711eb9ec6dcbedd8be08922a)，控制库版本 `100343014`，适用 Linux x86_64。

- `SDK_PYTHON/`：原样的控制／运动学 Python 封装与 `libMarvinSDK.so`、`libKine.so`。
- `CommonConfig/ccs_m6_40.MvKDCfg`：M6 4.0 双臂名义几何、关节及耦合限位，不含设备个体标定。
- `LICENSE`：厂家原始许可证。
- `manifest.json`：来源提交、版本、目标平台及各文件 SHA-256。

无需构建或单独安装 SDK。天机接口默认使用本目录；随项目复制后，按本项目环境配置重新创建 Conda 环境即可。导轨策略评估不需要 ADB 或 Quest。`TianjiKinematics` / `TianjiDriver` 的可选 `sdk_root` 参数仅用于显式指定另一套与清单匹配的 SDK，通常无需设置。

SDK 及模型路径通过包位置解析，不依赖当前工作目录或开发者机器路径。更新厂商版本时应一起替换匹配的 Python 文件、库和清单，并重新执行兼容性、离线运动学和实机验收；不要单独替换某个 `.so`。加载错误会明确报告，不自动编译。

项目的 Python 适配层修正官方封装的类型和精度问题，不修改 SDK 文件。官方接口只返回 SDK 接受结果，不能提供底层 UDP 发送回执或原始收包时间，详见[设备运行说明](../../../../docs/tianji-rail/device-runtime.md)。
