# 导轨搬运至预插入位姿

本轮完成张手接料、按键闭合、停稳估计、固定抓握关系计算、持物碰撞规划，以及到达目标沿插入轴外移 **20 mm** 的位姿。到位后保持夹持和机械臂目标，等待操作员处理。程序的 `end_stage` 固定为 `preinsert`。

完整设计与后续讨论见 [根目录计划](/home/jiyuchen/project/WYS/plan/2026-10-21-真机导轨6D位姿插入实验问题与实施计划.md)。本目录已实现软件流程，真机到位精度和时序仍待现场验收。

## 1. 环境与入口

以下命令在 `/home/jiyuchen/project/WYS` 执行。控制、规划、测试使用已有 Conda 环境 `bimanual-teleop`；感知子进程使用 `foundationpose`，不要求控制环境安装 PyTorch。

```bash
conda run -n bimanual-teleop --no-capture-output python \
  diffusion_policy/workspace/rail_pose_insert/run.py check
```

该命令不连接设备，列出缺少的现场输入。运行位置可在 WYS 或 diffusion_policy 根目录，配置路径统一相对 `config.yaml` 所在目录解析。部署到另一台机器时修改配置中的 `foundationpose_python`。

## 2. 现场输入

1. 将 `calibrations/head_to_left_base.example.json` 作为格式参考，写入 `calibrations/head_to_left_base.json`。矩阵 `T_B_C` 必须把头部彩色光学系坐标转换到 `tianji_left_base`，平移单位米。相机序列号需匹配当前头部 D436。
2. 根据 `calibrations/scene.example.json` 填写 `calibrations/scene.json`：台面、料箱及可能进入搬运路径的其他障碍物，均用左臂基座系下的位姿和米制尺寸。支持 box 或米制网格。带 null 的模板会被拒绝。导轨台由真实目标位姿结合现有 CAD 装配关系自动定位。
3. 记录正确就位的目标，保存为 `targets/rail_goal.json`：

```bash
conda run -n bimanual-teleop --no-capture-output python \
  diffusion_policy/workspace/rail_pose_insert/run.py teach-target
```

导轨已正确就位时，在相机窗口标注可见导轨；`a` 添加轮廓，`h` 排除手或孔内背景，Enter 确认。默认采 3 次位姿，检查一致性后保存原始 CAD 系下的 `T_C_O_goal`。已有目标文件不会被覆盖。

`outward_axis_cad` 默认 `[0,0,1]`，按现有导轨 CAD 定义为离开导轨台的孔轴方向。预插入位置为 `p_goal + 0.02 * R_goal * outward_axis_cad`，朝向保持目标朝向；核对现场符号，不能用基座 Z 加 0.02 替代。

这些现场文件不填示例真值，不使用仿真相机外参充当真机标定。

### 2.1 获取相机外参 `head_to_left_base.json`

现有 [独立标定工具](/home/jiyuchen/project/WYS/bimanual-teleop/camera_calibration/docs/external.md) 可以通过已标定左腕相机求头部固定相机外参。前提是左腕相机支架自手眼标定后没有移动；现有参考文件对应左腕序列号 `408322072104`。若支架移动，先按 [腕部标定说明](/home/jiyuchen/project/WYS/bimanual-teleop/camera_calibration/docs/wrist.md) 重做手眼标定。

本实验模板中的头部 D436 序列号为 **`405622076394`**。下面命令显式指定它，避免采用第三视角标定工具默认的另一台 D435。更换头部设备时，采样和导出两处都填写实际序列号。

在连接相机和机器人反馈的电脑上执行，关闭其他相机和反馈端口占用程序：

```bash
cd /home/jiyuchen/project/WYS/bimanual-teleop
conda activate bimanual-teleop
python -m camera_calibration external collect \
  --wrist-result bimanual_teleop/sim/configs/wrist_camera_calibration_reference.json \
  --side left --external-serial 405622076394 --samples 25 \
  --output camera_calibration/data/head_capture_01
```

让两台相机同时完整看到现有 4×5 非对称圆点板，使用真实板尺寸。头部保持固定，改变板和／或左腕的观察位置与倾角，每组停稳 1～2 秒。覆盖实际接料和预插入区域，按工具提示采满 25 组。采样只读取相机和关节反馈，手臂姿态由操作员通过示教器调整。可以固定左腕、仅移动板，前提是两相机均能看到足够多不同板姿态。

以下求解和实验格式转换均为离线操作；不连接设备。输出目录或文件须尚不存在：

```bash
python -m camera_calibration external solve \
  camera_calibration/data/head_capture_01 \
  --wrist-result bimanual_teleop/sim/configs/wrist_camera_calibration_reference.json \
  --output camera_calibration/data/head_result_01

python /home/jiyuchen/project/WYS/diffusion_policy/workspace/rail_pose_insert/tools/export_head_calibration.py \
  camera_calibration/data/head_result_01/result.json \
  --camera-serial 405622076394
```

检查 `head_result_01/report.md` 和独立验证叠图。导出脚本只接受通过验证的实测结果，验证序列号和 URDF，并用官方离线 FK 转成实验所需的 **SDK 左臂基座**。`camera_calibration` 的通用 `T_base_camera_opencv` 使用 URDF 第一关节系，不能直接改名为 `T_B_C`；当前两种基座有 174.5 mm 固定偏移。转换复用规划中的基座配准，并校验非零关节姿态。

默认保存到 `calibrations/head_to_left_base.json`，包含坐标系、米制单位、相机序列号、4×4 矩阵和来源摘要。若已经有其他方法得到的实测外参，可按现有模板填写，须满足 `p_B = T_B_C @ p_C`，其中 C 为头部原始彩色 OpenCV 光学系、B 为 `tianji_left_base`。

标定报告的通过条件是像素误差和位姿一致性，不能代替工作区绝对精度验收。使用未参与标定的实体点或标定板位置核对坐标链，再进行搬运。

### 2.2 获取目标位姿 `rail_goal.json`

先完成上面的相机外参；本步骤不需要 `scene.json`。相机、机器人基座和导轨台保持固定，将导轨完全正确就位，人员撤手，保留导轨两端的可见区域，然后执行：

```bash
conda run -n bimanual-teleop --no-capture-output python \
  /home/jiyuchen/project/WYS/diffusion_policy/workspace/rail_pose_insert/run.py teach-target
```

窗口中 `a` 添加导轨轮廓，`h` 排除手或孔内背景，Enter 确认。程序采集 3 个估计，检查平移／旋转一致性，取均值并自动生成 `targets/rail_goal.json`，同时保存图像、掩码、位姿叠图、相机序列号和 CAD 摘要。首次 GPU 初始化需要等待。

保存的是 **完全就位的导轨 CAD 位姿 `T_C_O_goal`**，不是抬高后的预插入位姿。20 mm 外移由规划程序计算，无需手工抬高导轨再示教。检查叠图的两端、孔位和 CAD 方向，重复一致性本身不能排除持续的估计偏差。此命令只使用相机，不向机械臂或手发送动作。

### 2.3 获取障碍物 `scene.json`

这一项目前需要现场测量，没有自动扫描场景的入口。复制 [场景模板](/home/jiyuchen/project/WYS/diffusion_policy/workspace/rail_pose_insert/calibrations/scene.example.json) 为同目录的 `scene.json`。对桌面、料箱及其他可能进入路径的物体，先用长方体近似，填写：

- `size_m: [L, W, H]`：长方体三个局部轴上的完整尺寸，单位米；不是半长。
- `T_B_obstacle`：长方体**几何中心**到 SDK 左臂基座的 4×4 位姿。前三列是局部 X/Y/Z 轴在基座系中的方向，最后一列是中心坐标。不能填桌面角点，也不能默认桌面水平就与 SDK 基座三轴对齐。

尺寸用尺测量；位置和朝向可从完成外参标定后的头部 RGB-D 测得，或通过已知偏移的测量指针和实测机器人 FK 测得。使用相机时，必须取已去畸变彩色图上的点以及注册到该图的深度；针孔关系为 `p_C = depth * inverse(K) @ [u,v,1]`，再用 `T_B_C` 转到基座。原始深度像素不能直接配彩色内参。

例如测桌面板：A 为上表面一个角，B、C 分别位于沿长和沿宽的相邻边。将三个点转到基座后，用 `B-A` 定义局部 X，用桌面向上法向定义 Z，再取 `Y=Z×X`，保证 C 位于正 Y 一侧。若长、宽、板厚分别为 L、W、H，则中心为 `A + L/2*X + W/2*Y - H/2*Z`，旋转矩阵为 `[X Y Z]`。桌腿及料箱分别建模，测量误差应通过适当扩大障碍物包络覆盖。

矩阵填写结构如下，字母须替换为实测数字，不能把示意结构直接作为 JSON 保存：

```text
T_B_obstacle = [[Xx, Yx, Zx, center_x],
                [Xy, Yy, Zy, center_y],
                [Xz, Yz, Zz, center_z],
                [ 0,  0,  0,        1]]
```

导轨台和定位销已有 CAD，程序根据示教目标自动定位，不需再次手填同一障碍物。台面、料箱及其他环境障碍仍需保留。头部相机或机器人基座移动后重做外参和目标；导轨台移动后重新示教；其他障碍物移动后更新场景。

三个文件完成后执行第 1 节的 `check`。它验证文件格式、坐标约定及模型一致性，不验证实物测量精度；通过后再进入第 3 节的只读联调。

## 3. 真机运行

退出遥操作、录制及其他占用头部相机或控制设备的进程后启动。机器人应处于已经确定的接料姿态，右臂位于固定待机位置。入口保持启动时的实测机械臂位置，按键触发后逐步张手；不会执行未经规划的回位动作。

先执行只读观测，记录设备实际状态：

```bash
conda run -n bimanual-teleop --no-capture-output python \
  diffusion_policy/workspace/rail_pose_insert/run.py run \
  --output diffusion_policy/workspace/rail_pose_insert/outputs/read_only_01
```

在有桌面和交互终端的本机启动动作入口：

```bash
conda run -n bimanual-teleop --no-capture-output python \
  diffusion_policy/workspace/rail_pose_insert/run.py run --motion \
  --output diffusion_policy/workspace/rail_pose_insert/outputs/live_preinsert_01
```

输出目录必须不存在。动作按键如下：

| 按键 | 作用 |
|---|---|
| O | 在当前接料姿态接合左臂，缓慢张开左手 |
| G | 人递入导轨后，缓慢闭合至候选握持目标 |
| M | 人撤手、导轨稳定后，标注导轨并估计位姿，自动计算抓握关系和路径 |
| E | 路径保存后执行搬运，到预插入位姿停稳，再标注复测导轨 |
| V | 到位后重复视觉复核 |
| Q / 空格 / Ctrl+C | 暂停机械臂推进，手继续保持最后目标；本次试验不自动恢复 |
| X | 退出并停用灵巧手；先用外部支撑接住导轨 |

首次测量及终点复测都会打开掩码标注窗口。机械臂与手的保持线程独立运行。忙碌阶段不接受提前排队的动作键；暂停和退出持续有效。

到位后继续保持，**不会自动松手**。`preinsert_reached` 表示实测法兰到位、视觉复核和名义 CAD 余量检查通过；`preinsert_unverified` 表示终点视觉复核未通过，仍保持夹持，可用 M 重新测量规划。两者都没有执行继续插入。

## 4. 数据、模型与路径检查

- 使用 `hand_profiles/rail_place3_left.json` 的张手和闭合下发角度，来源为 episode 000034。候选演示偏端部握持，中部握持仍需现场调整；实测角度一并保留用于比较。
- 用每张图像采集时刻插值实测左臂关节，再计算每次 `T_F_O`；不拿推理结束时的法兰位姿拼接旧图像。停稳期间允许推理耗时，检查状态历史覆盖及机械臂是否移动。
- 头部 RGB-D 在独立感知进程采集并去畸变、注册深度，支持与已有 D436 录制一致的畸变模型。相机仅由感知进程占用。
- 碰撞计算与 RRT-Connect 在独立规划进程运行。先尝试直达关节路径，碰撞时搜索绕行路径；检查关节及腕部耦合限位，并用分段五次曲线限制速度、加速度。
- 使用现有完整 URDF、手和腕部安装链。通过官方 FK 显式对齐基座，校验非零姿态的一致性；不把 URDF 第一关节系当作 SDK 基座。
- 使用保守凸包检查整个持轨组合、手臂、右臂、导轨台和已填写障碍物。固定抓握组合内的手指/导轨接触、相邻安装链的固有重叠被允许，外部碰撞仍检查。
- 当前是 **离散关节路径碰撞检查**，默认采样间隔最大 0.005 rad，外部间距 2 mm；不提供连续扫掠体证明，也不能发现未建模的障碍物。
- 执行时检查反馈时效、实测跟踪误差、右臂移动和手型变化。外参、目标、网格或环境模型在规划后变化时拒绝执行。
- 暂停保持手指，退出才停用。当前沿用已有末端动力学辨识参数；持轨后的额外负载对实际到位精度的影响需现场验证。

## 5. 日志与结果

每次输出包含：

```text
config.json / effective.yaml       完整实验配置
initial_state.json                 实测关节、覆盖感知时间的状态历史
initial_observation.json           原始时刻的导轨位姿
plan.json                         抓握关系、预插入法兰目标、关节路径及文件摘要
planner.log                       独立规划进程输出
perception/                       RGB-D、掩码、叠加图、感知进程日志
events.jsonl                      阶段、实测状态及原始腕部六维力
final_observation.json             到位后导轨位姿
result.json                       preinsert_reached / preinsert_unverified / paused
```

SDK 接受命令不作为到位依据。终点使用实测法兰误差并持续稳定，再根据当前导轨位姿检查目标误差及全轨相对就位状态的高度余量。此验收仅针对预插入阶段。

## 6. 离线规划、重放与测试

创建明确标记 `test_data=true` 的输入。它只复用录制的关节状态，握持关系、相机和目标是构造的集成测试数据：

```bash
conda run -n bimanual-teleop --no-capture-output python \
  diffusion_policy/workspace/rail_pose_insert/tools/make_offline_case.py \
  --output diffusion_policy/workspace/rail_pose_insert/outputs/offline_demo_01

conda run -n bimanual-teleop --no-capture-output python \
  diffusion_policy/workspace/rail_pose_insert/run.py \
  --config diffusion_policy/workspace/rail_pose_insert/outputs/offline_demo_01/config.yaml plan \
  --state diffusion_policy/workspace/rail_pose_insert/outputs/offline_demo_01/state.json \
  --observation diffusion_policy/workspace/rail_pose_insert/outputs/offline_demo_01/observation.json \
  --output diffusion_policy/workspace/rail_pose_insert/outputs/offline_demo_01/plan.json --allow-test-data

conda run -n bimanual-teleop --no-capture-output python \
  diffusion_policy/workspace/rail_pose_insert/run.py \
  --config diffusion_policy/workspace/rail_pose_insert/outputs/offline_demo_01/config.yaml replay \
  --plan diffusion_policy/workspace/rail_pose_insert/outputs/offline_demo_01/plan.json \
  --output diffusion_policy/workspace/rail_pose_insert/outputs/offline_demo_01/replay

conda run -n bimanual-teleop --no-capture-output python -m unittest discover \
  -s diffusion_policy/workspace/rail_pose_insert/tests -v
```

离线重放输出 `path.png` 和 `result.json`，不访问硬件。带测试标记的标定、目标、环境与观测均被真机入口拒绝；离线感知重放数据也不能代替新的真机观测。

真实录制 RGB-D 的感知接口验证命令：

```bash
conda run -n foundationpose --no-capture-output python \
  diffusion_policy/workspace/rail_pose_insert/perception_worker.py \
  --foundationpose-root /home/jiyuchen/project/WYS/FoundationPose \
  --mesh /home/jiyuchen/project/WYS/FoundationPose/workspace/input/model/rail.obj \
  --scene /home/jiyuchen/project/WYS/FoundationPose/workspace/datasets/rail-episode-000010-camera0 \
  --index 320 --samples 3 \
  --output diffusion_policy/workspace/rail_pose_insert/outputs/perception_demo_01
```

当前已通过 23 项自动检查（含新增外参导出检查）、完整模型离线规划/重放和真实录制的 3 帧 GPU 位姿接口验证。没有进行本轮真机运动；现场待完成外参接入、正确就位目标示教、障碍物测量，以及抓握、时序和最终到位验证。
