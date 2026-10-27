# FoundationPose 导轨插入

当前只保留右臂流程：**接料抓握 → 初始估姿 → M 搬到上方 → 上方重新估姿并纠偏一轮 → 新按 N 螺旋插入 → 自动松手、抬升、回位**。

## 环境与现场文件

在项目根目录使用 `tianji-rail-diffusion-policy` 环境。首次创建环境：

```bash
conda env create -f conda_environment_tianji.yaml
conda activate tianji-rail-diffusion-policy
python workspace/rail_pose_insert/check.py
```

检查命令验证配置、现场文件、FoundationPose/SAM 权重和 GPU 扩展，不连接机器人。权重准备见 [权重说明](../../third_party/foundationpose/weights/README.md)，原生扩展安装见 [环境记录](../../plan/2026-10-24-FoundationPose本机环境补齐.md)。SAM 权重沿用 `third_party/cnos/pretrained/segment-anything/` 路径，运行流程不使用 CNOS 检测器。

配置统一为 `receive.yaml`，其中路径相对该配置文件解析。需要准备：

| 文件 | 用途 |
|---|---|
| `calibrations/head_to_left_base.json` | 实测头部相机外参，米制 SDK 左臂基座坐标；运行时转换至右臂基座 |
| `targets/rail_goal.json` | 导轨完全正确就位时的 CAD 目标位姿 |
| `sam_prompt.json` | 初始抓持位置的 SAM bbox |
| `sam_prompt_preplace.json` | 导轨台上方的 SAM bbox |
| `hand_profiles/rail_middle_right.json` | 右手张开与抓握手型 |

相机安装或标定变化后需重新确认外参并示教目标。标定工具见 [相机标定说明](../../third_party/camera_calibration/docs/external.md)。导入已求解的实测外参：

```bash
python workspace/rail_pose_insert/tools/export_head_calibration.py /path/to/result.json \
  --camera-serial 405622076394
```

将导轨完全就位后执行目标示教，在图像中标注导轨区域，Enter 确认。此命令只使用相机，不控制机器人，已有目标文件会拒绝覆盖：

```bash
python workspace/rail_pose_insert/teach_target.py
# 需要重新示教时另存，再修改 receive.yaml 中 move_above.target
python workspace/rail_pose_insert/teach_target.py --output /path/to/new_goal.json
```

## 启动与操作

```bash
# 首次运行：在两个实际位置分别绘框
python workspace/rail_pose_insert/insert_spiral.py --draw-bboxes

# 复用已保存的两份 bbox
python workspace/rail_pose_insert/insert_spiral.py

# 右手已抓住导轨：跳过双臂启动回位与张手，重新估姿
python workspace/rail_pose_insert/insert_spiral.py --already-held

# 可选：保留精简运行记录；目录必须尚不存在
python workspace/rail_pose_insert/insert_spiral.py --output /tmp/rail-insert-run
```

普通启动先让左臂、右臂依次回固定初始位姿，右臂到位停稳后切换阻抗保持并张手。准备完成后等待 5 秒闭合，G 可提前闭合；抓握完成后等待 8 秒开始初始估姿。`--already-held` 跳过这段准备动作。

初始采集 3 帧估姿，结合图像时刻的关节反馈保存抓持关系均值。**M 使用已保存均值**搬到装配目标沿右基座 +Y 抬高 15 mm 的位置。上方停稳后，重新运行 SAM 与 FoundationPose，以新抓持关系最多纠偏一轮；完成后不再追加估姿或纠偏，必须**重新按 N**才插入，提前按 N 不排队。

绘框阶段按 **E → 鼠标左键拖框 → Enter**；已有有效框可直接 Enter 确认复用。缺失、无效或相机不匹配的框会在对应阶段等待重新绘制。可用 `set_sam_prompt.py` 单独设置框。`--no-view` 关闭窗口，要求两份框均有效，不能与 `--draw-bboxes` 同用。

| 按键 | 作用 |
|---|---|
| G | 提前闭合右手 |
| M | 使用初始均值开始搬运 |
| N | 单轮纠偏完成后开始插入 |
| E / Enter | 编辑 / 保存或确认当前阶段 bbox |
| Q / 空格 / Ctrl+C；窗口 Esc | 暂停，保持最后手部目标，不自动恢复 |
| X | 退出并停用右手；退出前需承托导轨 |

可点动阶段使用 W/S、R/F、D/A 平移，I/K、J/L、U/O 旋转，+/− 调步长。规划、搬运、纠偏、插入和回位期间禁用点动。偏离上方目标后 N 不会直接下插。

## 参数与完成条件

| 参数 | 当前值 |
|---|---|
| 上方高度 `move_above.height_m` | 15 mm |
| 螺旋半径 / 圈数 | 3 mm / 3 圈 |
| 螺旋相邻目标最大距离 | 0.5 mm |
| 启动与粗搬运速度 / 加速度比例上限 | 3% / 3% |
| 纠偏、插入及松手回位速度 / 加速度比例 | 1% / 1% |
| 张手指令过渡 / 最终指令提交后等待 | 3 秒 / 0.5 秒 |
| 松手后抬升 | 右基座 +Y 方向 15 cm，每段 5 mm |

螺旋在右基座 XZ 平面绕圈、沿 −Y 下降，半径从零扩大再收拢，终点回到示教目标。每条路径先完成 IK 和关节限位检查，再逐段执行；SDK 完成、实测 FK 到位并停稳后才推进。插入到位后先预规划完整抬升和回位路径，规划成功才松手，随后抬升并回到配置的固定初始位姿；回位完成后等待 X，不自动开始下一轮。

本流程没有接触力反馈或完整场景碰撞检测。**位置到位不代表真实落座**，`assembly_verified` 始终为 `false`；松手按指令完成和固定等待推进，`hand_open_confirmed` 为 `false`，不代表测量确认手指已脱离。

默认退出后清理运行文件。使用 `--output` 时只保留：

- `events.jsonl`：阶段变化、操作请求、异常和退出事件。
- `result.json`：当前或最终状态、插入证据、松手回位结果、最后有效反馈及退出原因。`returned_initial` 表示回到固定初始位姿，`insertion` 保留松手前结果。

只读设备状态使用 `python workspace/rail_pose_insert/read_state.py`。代码修改后需重启进程，不要同时运行多个设备控制进程。

## 代码与离线验证

`receive.py` 负责流程编排，`rail_receive/` 包含硬件、感知、纠偏、螺旋和回位逻辑，`rail_common/` 包含共用坐标、规划、相机与示教感知工具。独立孔位跟踪见 [说明](hole_tracking/README.md)。

```bash
python -m unittest discover -s workspace/rail_pose_insert/tests -q
python workspace/rail_pose_insert/check.py
```
