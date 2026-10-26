# 天机导轨：图像与关节观测来源对比

日期：2026-10-19。当前设置：两组均执行完整示教片段，从当前姿态开始，执行前不回初始位置。

## 两组设置

| 实验 | 三路图像 | 机械臂与双手状态 | 参数 |
|---|---|---|---|
| M01 / ER | 数据集 | 真实反馈 | `dataset-images-live-state` |
| M02 / RE | 真实相机 | 数据集 | `live-images-dataset-state` |

“关节观测”按现有模型输入契约处理：`robot_eef_pose` 为双臂实测关节经 FK 得到的12维法兰位姿，`hand_joint` 为双手40维实测关节；两项一起切换。三路 RGB 为主相机、左腕、右腕，一起切换。组合原始数组后，统一使用 checkpoint 内的预处理与 normalizer。

两组使用 epoch50 EMA（SHA256 `575b901e0aa801437046789d3ae626969943043f0d8a2307acb88d4b158286d3`）、CUDA、DDIM8、seed42、两步观测、8步动作、30 Hz，同步执行后再推理。预览预测不占正式采样序列；手动开始后重置 seed，第0块重读真实观测。每组配置1次，执行到所选示教片段末尾，取消10秒和24次决策截断。

数据集沿用 `data/replay_obs/rail_place3_episode_000000.zarr`，episode0的最长连续片段，观测窗口依次为 `[0,1]`、`[8,9]`、`[16,17]`……，执行完整块后才前进。真实部分每轮取最新可对齐的两步窗口；两套时钟分别记录，不能把相同块序号解释为相同任务阶段。

该片段共1549帧，名义时长约51.6秒，共194次决策：前193块各8步，末块仅执行剩余4步，共1548步预测动作（第一帧为观测历史）。实际运行还包括推理与等待观测，通常长于51.6秒；不再用总时长提前截断。正常停止条件是完整片段执行结束，操作员停止、观测故障和动作校验失败仍可提前结束。

两组按 `s` 后从当前姿态开始，不在执行前摆手回位，也不要求通过冻结起点容差核对。沿用 loose 部署限制，跳过首动作接近段，关闭视觉修正和稳定性门控；动作跳变、速度和设备健康检查继续生效。真实窗口暂缺时保持并等待，超时或操作员停止后退出，不回退到完整示教输入。

## 启动

项目根目录执行。先激活环境：

```bash
conda activate tianji-rail-diffusion-policy
```

离线预检不连接设备：

```bash
for slot in M01 M02; do
  python scripts/run_tianji_final_batch_slot.py \
    --manifest diffusion_policy/config/tianji_rail_mixed_obs_batch.json \
    --slot "$slot" --mode offline-preflight \
    --output "data/real_eval/mixed-obs-20261019-full/$slot"
done
```

M01：数据集图像＋真实关节观测。

```bash
python scripts/run_tianji_final_batch_slot.py \
  --manifest diffusion_policy/config/tianji_rail_mixed_obs_batch.json \
  --slot M01 --mode hardware --require-manual-start --no-auto-retry \
  --output data/real_eval/mixed-obs-20261019-full/M01
```

M02：真实图像＋数据集关节观测。

```bash
python scripts/run_tianji_final_batch_slot.py \
  --manifest diffusion_policy/config/tianji_rail_mixed_obs_batch.json \
  --slot M02 --mode hardware --require-manual-start --no-auto-retry \
  --output data/real_eval/mixed-obs-20261019-full/M02
```

观测准备完成后按 `s` 开始，`p` 暂停、`y` 成功、`n` 失败、`q` 退出；这些结束按键会提前停止当前段。无需执行 `--mode home`。已有试验结果不覆盖，不自动补跑；以上命令使用新的 `full` 输出目录，保留之前的试验记录，再次运行须换新目录。

也可直接用回放入口的 `--observation-mode` 选择组合，并加 `--full-segment`；要使用以上 seed、整段设置和完整记录，使用上述 batch 命令。

## 记录与解释

每组保存 `session.json`、`slot.json`、`trial_0001/trial.json`、事件与完整 trace。记录实际混合输入、同期完整真实输入、数据集窗口起点、真实对齐时间戳、来源、预测动作及采样前 RNG；每个会话另存当前源码哈希。起点策略记录为 `initialization_mode=current_pose`、`initialization_check_required=false`；未检查的 `initial_state_comparable` 为 `null`。

`execution_scope=full_segment` 表示整段设置；`planned_windows`、`completed_windows` 和 `full_segment_completed` 区分完整执行与提前结束。自然回放结束记为 `playback_complete`，不自动判定物理任务成功。

整段图像在推理所在进程写入文件映射缓存，控制线程不压缩或写图；低维反馈、命令和状态容量随决策数扩大。运动停止后导出兼容原分析工具的 NPZ，成功后清理缓存；导出失败保留图像缓存及错误记录。当前片段的图像数据约2.0 GiB/组，导出期间缓存和最终文件会暂时同时存在。

与完整示教观测（EE，原 P00）和完整真实观测（RR，原 C0）一起观察图像／状态来源的影响。严格配对需相同代码版本、起点、示教窗口与采样噪声；旧结果若来自不同同步观测实现，只能作历史参照。运动中两组真实场景会随各自轨迹改变，因此跨组差异不能单独证明某一种观测是唯一原因。相对示教标签的动作误差只用于诊断，任务成功由现场标记。

## 已完成的验证

- 两组整段离线预检通过，报告位于 `data/real_eval/mixed-obs-20261019-full/M01/preflight.json` 和 `M02/preflight.json`：1549帧、194次决策、末块4步，无总时长或决策数截断，仍从当前姿态开始。
- 60项相关测试通过。两种输入组合均以模拟设备执行完194块、1548步；模拟时钟超过10秒后继续运行。另覆盖第26块人工提前停止、末块裁切、完整 NPZ 导出、图像缓存清理、导出失败保留缓存，以及既有短段和停止行为。
- 初版已用实际 epoch50 EMA 在 CUDA 上完成两种组合、各两个窗口的离线推理：4份输出均为有限的 `8×52` 动作。报告为 `data/real_eval/mixed-obs-20261019/setup_check/report.json`。现场部分来自已保存的真实观测；原历史 trace 因状态记录溢出标记无效，本项仅检查输入组合和推理链路，不用于行为归因或成功率结论。
- 本次修改仅做离线验证，未启动真机。
