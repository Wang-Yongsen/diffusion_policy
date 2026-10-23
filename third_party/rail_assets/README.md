# 导轨任务资源

`generated/` 保留原 URDF、全部引用网格和几何清单；`source/` 保留清单对应的 CAD 原件。模型字节不变，坐标和单位沿用原定义。设备 SDK 模型继续使用 `../bimanual_teleop/vendor/tianji/`。

`examples/joint_state.json` 是单个录制关节配置；`examples/rgbd/` 是三帧真实录制 RGB-D 和首帧掩码。它们只供离线验证，不包含实测头部到基座外参、正确就位目标或现场障碍物，不可替代真机观测。

来源和文件摘要见 `../rail_insert_manifest.json`；原来源未提供的许可证不作额外授权声明。
