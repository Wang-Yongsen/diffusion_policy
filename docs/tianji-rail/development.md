# 开发与维护

## 模块职责

| 位置 | 职责 |
| --- | --- |
| `dataset/tianji_rail_image_dataset.py` | 数据契约、来源划分、连续窗口 |
| `common/image_preprocessing.py` | 训练与部署共享的确定性预处理 |
| `common/tianji_rail_pose.py` | eef 四元数观测、6D动作与录制／控制 rotvec 表示转换 |
| `policy/` | Transformer、UNet 与观测编码 |
| `workspace/train_image_workspace.py` | 公共训练、验证、EMA 和保存流程 |
| `env_runner/tianji_rail_*_runner.py` | 离线指标或真机试验生命周期 |
| `real_world/tianji/config.py` | 部署配置与覆盖 |
| `real_world/tianji/observation.py` | 相机与实测状态时间对齐 |
| `real_world/tianji/env.py` | 设备生命周期与观测、动作接口 |
| `real_world/tianji/runtime.py` | 动作队列、同步／异步执行与暂停 |
| `real_world/tianji/joint_executor.py` | 关节目标跟随 |
| `real_world/tianji/checkpoint.py`、`inference.py` | 模型加载与独立推理进程 |
| `real_world/tianji/recording.py`、`trace.py` | 事件和有界实时追踪 |
| `real_world/tianji/home.py`、`terminal.py` | 回位与可取消的终端输入 |
| `diagnostics/` | 离线分析、绘图、观测对比 |
| `cli/tianji/` | 参数解析与命令分派 |

业务包不导入根目录脚本。推理进程只加载模型，不持有设备。
控制路径不执行离线绘图、压缩和报告分析。
`third_party/` 不参与日常代码整理，修改须更新摘要并重新验收。

## 旧入口迁移

| 旧命令 | 新命令 |
| --- | --- |
| `python eval_tianji_rail.py` | `python -m diffusion_policy.cli.tianji eval` |
| `python eval_tianji_rail_recorded_obs.py` | `python -m diffusion_policy.cli.tianji replay` |
| `python home_tianji_rail.py` | `python -m diffusion_policy.cli.tianji home` |
| `scripts/analyze_tianji_trace.py` | `python -m diffusion_policy.cli.tianji analyze` |
| `scripts/probe_tianji_obs_gap.py` | `python -m diffusion_policy.cli.tianji probe` |

根目录旧入口是薄转发；`--batch-seed` 仍可作为 `--seed` 的别名。
固定日期的补充诊断和最终批次入口已移除。
训练命令、Hydra 配置名、checkpoint 的模型与工作区类路径保持可用。
末端版已改为14维四元数观测、58维6D旋转动作，旧52维末端版 checkpoint 须重新训练。
旧部署模块导入仅保留兼容转发，新代码使用上表中的包路径。

## 验证

```bash
conda run -n tianji-rail-diffusion-policy python -m unittest discover \
  -s tests -p 'test_tianji_rail*.py' -q
conda run -n tianji-rail-diffusion-policy python -m unittest discover \
  -s tests -p 'test_transformer_workspace_resume.py' -q
conda run -n tianji-rail-diffusion-policy python -m unittest discover \
  -s tests -p 'test_policy_inference_process.py' -q
conda run -n tianji-rail-diffusion-policy python -m unittest discover \
  -s tests -p 'test_decision_trace.py' -q
```

使用测试内的模拟设备。上游 `tests/` 还含相机、机器人和仿真环境脚本，勿直接全量执行。
回归覆盖数据来源划分、预处理、增强、EMA、恢复、调度、暂停和进程退出。

## 代码约定

- 保持上游分层；新增导轨部署功能放入 `real_world/tianji/`。
- 函数围绕一个职责组织，使用完整名称；单位写入参数名或必要注释。
- 默认参数以可执行配置为准，文档只保留常用项。
- 新实验使用独立输出目录，不覆盖旧模型、数据或记录。
