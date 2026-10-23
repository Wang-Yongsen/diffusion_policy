# 导轨任务标定工具快照

入口：在 diffusion_policy 根目录运行 `python third_party/camera_calibration/run.py --help`。

保留腕部手眼标定、固定头部相机标定、只读采集、离线求解与通用导出；SDK、URDF 和运动学均引用本项目 `third_party/`。本快照不提供 Isaac Sim 配置写入或独立采样包打包命令。

`wrist_reference.json` 为已有实测腕部标定快照，使用前核对设备序列号、安装结构、URDF 及标定板尺寸。相机安装变化后须重新标定。头部实测外参由实验的 `tools/export_head_calibration.py` 转为 SDK 左臂基座。

来源及修改摘要见 `../rail_insert_manifest.json`。本项目操作步骤见 `../../workspace/rail_pose_insert/README.md`。
