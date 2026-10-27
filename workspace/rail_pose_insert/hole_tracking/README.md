# 数值 bbox 初始化的双孔轨迹估计

入口是 `../tools/track_rail_holes.py`，单个 Python 文件，可复制使用。依赖 NumPy、SciPy、OpenCV；不调用 GPT、网络 API、FoundationPose 或机器人。默认处理本项目的离线 RGB-D 录制格式，输出两个孔口中心在 **RGB 相机光学坐标系**中的轨迹：X 向右、Y 向下、Z 向前，单位米。

这是上一轮实际采用的“RGB 孔心 + 邻近金属面深度”方法的程序化实现。FoundationPose 原始孔投影偏差较大，因此没有将其作为坐标来源。它是给定目标区域后的局部跟踪程序，不是从整幅图像中识别任意导轨的通用检测器。

## 输入 bbox

提供两个孔在**同一初始帧**的原始 RGB 像素坐标：

```text
[[x_min, y_min, x_max, y_max], [x_min, y_min, x_max, y_max]]
```

采用 `xyxy`，右边界和下边界不包含在框内。不是 `xywh`，也不是归一化坐标。每个框包住一个孔口并留少量周边金属即可，避免同时框住相邻孔。程序不读取图中画出的红框。孔编号 H0/H1 始终对应输入 bbox 的顺序，不随后续画面的左右顺序变化。

本次视频为 640×480。VLC 截图的窗口、黑边和缩放不属于原始图像坐标；1920×983 截图中的坐标不能直接传入。

本次第 960 帧（32.000 秒）可使用：

```json
{
  "seed_frame": 960,
  "bboxes_xyxy": [[258, 328, 272, 342], [407, 320, 421, 333]]
}
```

程序自动在框内拟合孔心；这两个框会得到约 `(264.91,334.38)` 与 `(413.54,326.52)`。初始化失败或框内存在多个相近候选时会报错，需要换较清晰的帧或缩小框，不会把 bbox 中心直接当孔心。

## 运行

现有环境已包含依赖。直接传 bbox：

```bash
/home/yuchen/anaconda3/envs/tianji-rail-diffusion-policy/bin/python \
  /home/yuchen/project/diffusion_policy/workspace/rail_pose_insert/tools/track_rail_holes.py \
  --episode /home/yuchen/project/bimanual-teleop/recordings/record_grasp_pose/episode_000000 \
  --bboxes '258,328,272,342;407,320,421,333' \
  --seed-time 32 --start-time 32 --end-time 34 \
  --output /home/yuchen/project/diffusion_policy/workspace/rail_pose_insert/outputs/my_hole_track
```

`--seed-time` 是 bbox 所在帧的**视频时间**。开始、结束时间均包含端点，以最近的帧为准，且需满足 `start <= seed <= end`。默认从初始帧运行到其后 2 秒。所有输出写入新目录；已经存在的目录会拒绝覆盖。

也可把 `--bboxes` 和 `--seed-time` 两项替换为：

```bash
--init-json /home/yuchen/project/diffusion_policy/workspace/rail_pose_insert/hole_tracking/episode_000000_camera0.json
```

附带 JSON 包含本次录制的元数据摘要，避免误用到其他录制，并将有效深度限制到本场景的 0.60–0.76 米。新录制应重新填写 bbox、seed_frame 和合适的深度范围，不要沿用旧文件的 `episode_sha256`。

`--start-time` 可早于初始帧，程序会先向前/向后分别跟踪，再按时间顺序滤波。向前后处理的目标关联属于离线操作；只有滤波本身是因果滤波，不应将整条流程称为实时算法。`--no-video` 可关闭 MP4 输出，仍保留首末帧 PNG。

## 方法与状态

1. 框内使用暗斑椭圆高斯和逐行背景模型定位孔心，检查对比度、残差、边界和多候选歧义。
2. 后续帧使用金字塔 LK 光流预测局部位置，再重新拟合孔心；检查前后向误差、局部图像相关性、修正幅度和运动速度。
3. 按时间戳寻找最近深度帧。深度设备帧号与 RGB 帧号不同，不能按设备帧号相等匹配。超过 5 ms 的配对拒绝使用。
4. 原始深度反投影后，使用 `depth_to_rgb` 外参变换到 RGB 系。旋转数组按 RealSense 的列主序读取。RGB 射线做畸变校正，不能直接用 RGB 像素索引原始深度图。
5. 在孔两侧的金属窄条中排除孔心区域，稳健拟合局部平面，求孔心射线与平面的交点。检查支持点数、平面退化、残差、射线夹角及 265 mm 名义孔距；孔距只作检查，不强制修改坐标来满足 CAD。
6. 对通过检查的中心坐标作时间相关的 One Euro 滤波，保存原始值和滤波值。丢失或深度失效帧输出 `valid=false` 和空三维坐标，并重置滤波历史。过大的三维跳变也会拒绝。

短暂反光期间允许最多 0.5 秒的内部图像运动预测，供下一帧重新定位使用；预测不作为测量输出。长期失效后输出 `reinitialization_required`，需要给新的 bbox 和初始帧重新运行。不会搜索整幅图像并自动换到另一个孔。

`stationary=true` 表示最近约 0.4 秒的**原始测量**满足位置离散与速度阈值，`stable_xyz_m` 是该窗口的中位数。它是静止性条件，不是精度认证，也不表示轨迹中运动的物体应该被冻结。

## 输出

| 文件/字段 | 含义 |
|---|---|
| `trajectory.jsonl` | 每帧一条，包含时间戳、两孔坐标、图像与深度检查信息、失效原因 |
| `trajectory.csv` | 每帧每孔一行，便于绘制轨迹或导入其他程序 |
| `metadata.json` | bbox、初始化帧、相机系、完整阈值与配置 |
| `summary.json` | 有效帧数量、失效分类、相邻帧变化和滤波偏移 |
| `overlay.mp4` | 原始 RGB 上显示孔心和有效状态；标记是 RGB 测量位置，不是滤波位置投影 |
| `frame_*.png` | 首末帧状态图 |
| `xyz_raw_m` | 两孔未滤波的三维测量；无效时为 `null` |
| `xyz_filtered_m` | 两孔滤波后的三维测量；无效时为 `null` |
| `uv_raw_px` | 原始 RGB 孔心；图像跟踪丢失时为 `null` |
| `xyz_rejected_m` | 仅在三维跳变检查失败时保存的候选值，不能作为有效测量 |

JSON 中 H0/H1 分别为数组第 0/1 项。CSV 的无效坐标留空。程序不会把旧坐标填到失效帧中。

## 中心位置与完整位姿

默认仅输出可靠性经过筛选的 **3D 中心位置**，完整旋转和 `T_camera_holes` 为 `null`。两个点本身不能确定绕两孔连线的旋转；此次窄条带深度的表面法向还有系统性不确定性。

`--surface-orientation` 可显式启用实验性孔面坐标系：X 从 H0 指向 H1，Z 为朝向相机的表面法向，Y=Z×X。只有两孔面法向、几何关系和支撑范围通过检查时才输出。它不是 CAD 坐标系，旋转尚未时序滤波，也未经实物精度验证。`surface_orientation_valid` 只表示内部检查通过，不代表可用于精密插入。

## 配置与适用范围

`--init-json` 的 `settings` 可覆盖 `Settings` 中的参数，包括深度范围、孔距、允许偏差、孔面条带范围、滤波频率和最大速度。默认孔距为 0.265 米、偏差阈值为 0.015 米，属于粗筛选门限，不是允许的定位误差。默认孔面条带针对本次导轨图像尺度；换相机距离、分辨率、物体或孔径后需重新核验。

当前支持本录制的未旋转 RGB 和零畸变原始深度。D436 RGB 的 inverse_brown_conrady 使用 OpenCV 校正；已与本机 librealsense 2.59 反投影比较，目标区域差异小于 0.001 像素。未知畸变模型、非零深度畸变、缺少深度或标定时会拒绝运行。

该程序不保证任意视角、反光、完全遮挡时都能定位。滤波会带来滞后，不能消除标定偏差和深度系统误差，也不能把本次结果升级为经过验证的毫米级精度。

## 验证记录

真实录制以第 960 帧 bbox 初始化，检查 31–36 秒，共 151 帧，85 帧输出有效坐标。32–34 秒的 61 帧中 58 帧有效，3 帧因孔外观不可靠而拒绝，随后恢复。较早的左孔反光与约 34.6 秒后的外观变化触发失效；没有强行填满整条轨迹。

有效连续帧的相邻位置变化 RMS 从 2.67 mm 降到 1.70 mm，滤波与原始测量的中位差为 3.28 mm。这些统计混合了物体运动和测量噪声，只描述平滑程度及滞后，不能作为绝对精度或静止抖动指标。该运动片段未达到静止性判据。

历史验证产物已随插入目录的旧输出清理，统计记录保留于本文。合成几何、数值 bbox、已知移动、遮挡、时间同步和滤波测试位于 `../tests/test_hole_tracker.py`：

```bash
/home/yuchen/anaconda3/envs/tianji-rail-diffusion-policy/bin/python -m unittest discover \
  -s /home/yuchen/project/diffusion_policy/workspace/rail_pose_insert/tests \
  -p test_hole_tracker.py -v
```
