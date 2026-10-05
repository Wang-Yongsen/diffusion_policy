# 原始数据整理与训练数据转换

本项目包含完整的整理与转换实现。原始采集程序不在本次迁入范围；将兼容的采集会话放到 `data/recordings/`，或在 `--input` 指定实际数据位置即可，处理过程不导入采集项目。配置和代码见[通用转换配置](../../diffusion_policy/config/recording_conversion.yaml)及[本地实现](../../third_party/bimanual_teleop/recording/)。

## 原始会话与整理

一个会话可以包含多条 episode。采集的 `episode.json` 记录状态，原始分段缓存在 `raw_spool/`。整理将其生成三路视频、`raw.zarr` 和完整 episode 元数据；只有 `complete` 条目进入训练转换。`capturing`／`captured`／`finalizing` 表示不同未完成阶段，失败保留以便排查。

在项目根目录、统一环境中执行，替换 `<session>`：

```bash
python scripts/finalize_recording.py \
  --input data/recordings/<session> --spool-archive data/raw_spools
```

入口递归查找 episode；**标记为 `discarded` 的整条 episode 会被删除**。成功整理后将原始 spool 归档，并在原条目留下相对符号链接；失败条目及其原始输入保留。已有完整条目通常跳过，`--refinalize` 显式重新生成派生产物并可从归档恢复原始 spool。会话与归档保持相对位置一起移动时，新建链接仍有效；历史绝对链接可通过 `--refinalize` 和迁移后的 `--spool-archive` 恢复。归档目录应放在本项目内以方便整体迁移。

## 严格与修复转换

输出路径必须尚不存在；入口通过临时结果和完成发布避免留下可误用的半成品，不原地改写已有训练 zarr。

```bash
python scripts/convert_recording.py \
  --input data/recordings/<session> \
  --output data/<session>/episodes_eef_repaired.zarr --action-space eef \
  --conversion-config diffusion_policy/config/recording_conversion.yaml
```

未传 `--conversion-config` 时保持严格模式。传配置且 `mode: repair` 才启用修复；文件存在不会自动启用。当前通用配置在 30 Hz 网格处理普通短缺口，默认最多补两帧；实测状态在有效两端之间插值，过去成功指令保持，图像按时戳匹配或复用。状态插值跨度上限 100 ms、指令年龄上限 50 ms；具体配置与已验收转换规则为准。超限或标签失效分段，不能用补帧掩盖失效指令。

暂停压缩的接缝须按每条来源复核，未明确批准合并的暂停保留边界。[旧会话配置](../../diffusion_policy/config/recording_conversion_tianji_rail_20260924.yaml)只适用于该会话，不将其复核结论套用于新采集。检查生成报告、来源划分、分段边界和修复追溯数组后，再指定训练的 `task.dataset.dataset_path`。已有训练数据可直接使用，无需再次转换。

## 观测与动作含义

三路 RGB 按固定顺序形成 `camera_0/1/2`；低维观测为双臂实际关节 FK 得到的 12 维末端位姿及双手实际反馈的 40 维关节。动作为成功提交的绝对目标，按左臂 6、右臂 6、左手 20、右手 20 排列，总共 52 维。位置为米，rotvec 和关节为弧度，双臂各自基座系下的法兰为末端，不是两臂共用世界系。

图像匹配、状态插值和动作保持使用共同时间域，但不代表三机同时曝光或已经补偿设备延迟。训练网格、原始来源时戳和暂停压缩后的时间含义见[时间对齐说明](../../wys/2026-10-04-rail-time-alignment/explanation.md)。

本说明提取了源采集项目的整理、数据格式和转换约定；不需要源项目文档。真机评估保存的 `trial.json`／视频／事件日志用于部署诊断，其目录不直接等同于上述训练原始会话格式。
