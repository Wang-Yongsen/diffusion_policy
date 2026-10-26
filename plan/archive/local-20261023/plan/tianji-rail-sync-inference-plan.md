# Tianji 同步模式：按序号消费与最新两步观测

更新日期：2026-10-19。

状态：已完成实现，89项离线回归测试全部通过；沿用“检查真机同步推理代码”聊天完成的按序消费与手指不限速改动。本次未连接设备。

## 观测规则

每轮执行完整个动作块后，完成已有稳定性等待，再从持续采集的缓存中选择最新可对齐观测时刻 `t₀`，读取 `[t₀−Δt, t₀]` 两步观测。`Δt=1/policy_hz`，当前30 Hz，对应约33.3 ms，保持与训练采样间隔一致。

`t₀` 是最新可用的传感器时刻，可能因采集延迟略早于动作块完成时刻；历史那一步可以发生在动作块执行期间。查询时刻由当前缓存决定，不再由上一块末两步的计划结束边界推算。

成功读取后固定本次推理输入。等待期间没有合格窗口时，可以随新数据重新选择候选；已经组装好的图像和状态快照不受后续新帧影响。

## 观测实现

- 同步 `get_obs()` 调用 `align_latest_observation()`。在同一份七路缓存快照中，按主相机 `camera_0` 时间戳从新到旧尝试候选；候选不得晚于当前时刻或任一路数据的最新时间戳。
- 候选窗口必须满足 `max_camera_age_s` 的锚点年龄限制。最新候选无法完整对齐时，尝试时效范围内更早的候选；没有合格候选则返回 `ObservationNotReady`，等待新数据或试验超时。
- 三路相机各选两张不同、时间戳递增的图像。在满足单帧偏差和三相机跨度的组合中选择总时间偏差最小者。双臂、双手状态均插值到同一组查询时刻，保留旋转 SLERP、禁止外推和状态插值间隔检查。
- 最新图像和状态数据流的健康检查继续生效。低层显式 `align_observation(..., query_ns=...)` 与 `get_obs(query_ns=...)` 保留供按指定时刻读取；同步 Runner 不再传入动作块边界时刻。
- `sync_observation_window()` 仅原子返回已完成动作块的完成时刻、来源块编号和预测编号；执行中返回 `None`。首轮和后续轮次均使用最新可对齐窗口，不要求锚点严格晚于种子或完成时刻。
- 暂停、重新接合和新试验清除旧完成上下文。推理中停止仍会丢弃迟到预测。已有稳定性等待在观测读取前执行，因而模型使用等待结束后的最新可对齐输入。
- `prediction.alignment` 和完整记录保存本次实际使用的 `query_ns`、`camera_ns`、`state_brackets_ns`、`source_chunk_id`、`source_prediction_id`。调度和完成事件不再预先保存 `next_observation_query_ns`。`chunk_completed_ns` 继续用于 `sync_hold_ms` 统计。
- 异步模式的最新观测选取、过期动作筛选与重规划间隔保持原行为。同步模式仍要求两步观测、动作块至少两步。

## 按序号消费与手指目标

- 保留已有顺序消费：当前步从首次选择时起经过至少一个策略周期，且双臂、左手、右手均有成功下发回执后，才推进一个序号；控制迟到会延长执行，不跳步或压缩后续步骤追赶。
- 末步满足同样条件后才进入保持并允许下一轮推理。成功下发不等于物理到位，不新增停稳门槛。
- 保留手指不限速：不做动作块内手指速度校验或控制周期限速裁剪；回放接近轨迹不因手指速度增加步数。关节范围、目标跳变和设备健康检查继续生效。
- 同步参考目标仍按各路设备成功下发结果更新；`submission` 保存 `generation`、`step_id`、`accepted`，`chunk_complete` 保存 `consumed_steps`。

## 验证与验收

回归覆盖：最新两步窗口及33.3 ms间隔、七路共同覆盖、不同相机帧、状态插值、最新候选不合格时回退、缺失旧窗口后随新数据恢复、锚点时效、稳定性等待后读最新观测、推理输入快照和完整记录、首轮、暂停重启、等待超时及推理中停止。

另有集成测试模拟控制迟到：动作块仍逐步完整消费，但完成后读取的是接近实际完成时刻的新观测，不再读取原计划末两步。顺序消费屏障、在途回执、SDK拒绝、手指不限速、异步模式、推理子进程和示教回放继续回归。

```bash
/home/yuchen/anaconda3/envs/tianji-rail-diffusion-policy/bin/python -m unittest \
  tests.test_tianji_sync_consumption tests.test_tianji_fixed_observation tests.test_tianji_rail_real_eval \
  tests.test_policy_inference_process tests.test_batch_factors \
  tests.test_decision_trace tests.test_recorded_episode \
  tests.test_tianji_rail_standalone -q
```

真机验收时检查 `prediction.alignment.query_ns` 的末项与 `anchor_ns` 相同、两项间隔为一个策略周期；三路图像分别为两张不同帧，四路状态插值区间均覆盖相同查询时刻。控制延迟后观测应跟随当前缓存，而非旧计划边界。

检查同一 `generation` 的三路 `submission.step_id`：允许高频控制重复发送同一步，但各路序号应连续覆盖0到末步。`chunk_complete.consumed_steps` 应等于动作块长度，完成前须存在三路末步成功下发记录。
