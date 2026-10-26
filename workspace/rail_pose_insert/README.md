# 导轨接料与预插入

启动前激活现有环境一次：

```bash
conda activate tianji-rail-diffusion-policy
```

以下命令均在项目根目录执行；也可进入 `workspace/rail_pose_insert` 后直接用 `python receive.py` 等短命令。各动作脚本会控制设备，一次只运行一个。

| 任务 | 直接启动 |
|---|---|
| 提前设置 SAM 范围框和可选点，只读取主视相机 | `python workspace/rail_pose_insert/set_sam_prompt.py` |
| 右手接料、估姿，M 移到上方、N 力控搜索插入 | `python workspace/rail_pose_insert/receive.py` |
| 新版：右手接料、M 搬运与视觉纠偏、N 纯位置螺旋插入，不采集六维力 | `python workspace/rail_pose_insert/insert_spiral.py` |
| 只接料和单次估姿，不自动搬运 | `python workspace/rail_pose_insert/receive.py run --motion --pose-only` |
| 接料后保存位姿均值，按 M 搬运（保留入口，与默认接料相同） | `python workspace/rail_pose_insert/move_above.py` |
| 已握住导轨，保持当前手型估姿，按 M 向目标上方移动 | `python workspace/rail_pose_insert/move_above_held.py` |
| 导轨正确就位后记录目标 | `python workspace/rail_pose_insert/teach_target.py` |
| 左臂接料与预插入 | `python workspace/rail_pose_insert/preinsert.py` |
| 检查两套配置与 GPU，不连接设备 | `python workspace/rail_pose_insert/check.py` |
| 旧固定区域 CNOS 诊断，不控制机器人 | `python workspace/rail_pose_insert/view_detection.py` |
| 检查插入现场参数，不连接设备 | `python workspace/rail_pose_insert/receive.py check --insertion` |
| 只读右臂六维力诊断，不使能动作 | `python workspace/rail_pose_insert/read_insertion_force.py --duration 3` |
| 只读右臂、右手反馈 | `python workspace/rail_pose_insert/read_state.py` |
| 只读左臂预插入设备反馈 | `python workspace/rail_pose_insert/read_preinsert_state.py` |

运行设置修改 `receive.yaml` 或 `config.yaml`；不需要指定输出目录。`run.py` 无参数等同于左臂预插入入口；原有高级子命令仍兼容。右手按 M 搬到目标上方，按 N 启动所选插入流程；左臂流程止于预插入。两者均不自动松手，右臂到达插入深度后等待人工确认。默认力控版的 `insertion` 现场参数留空，未填写时 N 明确拒绝启动；新版 `position_spiral` 不使用这些参数。

## 不使用六维力的简化螺旋插入

```bash
# 接料、估姿，按 M 搬到目标上方并自动视觉纠偏，对准后按 N 插入
python workspace/rail_pose_insert/insert_spiral.py

# 已经抓住导轨：保持当前手型，重新估姿后按 M、N
python workspace/rail_pose_insert/insert_spiral.py --already-held

# 只检查配置和依赖，不连接设备
python workspace/rail_pose_insert/receive.py check --insertion --insertion-mode position_spiral
```

该入口将本次运行设为 `lowering.mode: position_spiral`，原 `receive.py` 默认仍是力控版。也可在 `receive.yaml` 中选择模式，或使用 `receive.py run --motion --insertion-mode position_spiral`。新模式关闭驱动的六维力采集，不读取力、力矩、传感器标定或 `insertion.impedance`，无需填写原力控参数。接料初始化和点动仍沿用原控制方式；M 搬运和 N 螺旋插入使用 SDK 位置控制。

默认流程为：**M 搬到目标上方 15 mm → 停稳后重新 SAM 分割和 FoundationPose 注册 → 根据新测得的位姿自动纠偏 → 对准后等待 N → 位置螺旋下放**。重新分割使用当前 FK、已有抓持关系及 CAD 投影生成临时框，不复用接料区域的旧框、点或 mask；预测位姿只用于确定图像搜索范围。精对准关闭整机方向先验，使用新一轮感知结果，按图像采集时间同步关节反馈，重新计算导轨相对法兰的抓持关系。

纠偏同时处理右基座 XYZ 平移和三轴角度，目标仍为示教装配位姿上方 15 mm。根据新抓持关系换算法兰目标，以 SDK 1% 速度/加速度分段移动；长导轨需要角度修正时，必要时先抬高，再调整姿态并回到 15 mm。每轮停稳后重新估姿，默认最多执行 3 次视觉纠偏，视觉位置/角度误差须进入 **2 mm / 1°**。单轮允许修正的偏差上限为 **40 mm / 10°**；超限、视觉失效或修正未收敛时暂停并保持抓握，不继续下放。此逻辑配置在 `preplace_alignment`，仅用于 `position_spiral`；显式设置 `enabled: false` 可恢复机械到位后直接等待 N 的旧行为。

视觉误差进入容差后，冻结最新抓持关系；若仍有需要消除的小残差，追加一次 `final_pose_lock` 精调，沿相同规划器生成的低速小步轨迹对齐目标，避免把例如 0.9° 的残余角度留到螺旋首段。此次精调不再重新估姿，以 SDK 完成和实测 FK 到位作为结束条件，随后才进入 `above_target` 并接受 N。抬升提供名义轨迹的导轨几何净空界，执行中还监控实测导轨角点高度；这些检查不包含全面的机器人与环境碰撞检测。

N 从机械臂在上方位置的实测点开始，固定目标姿态，在右臂基座 **XZ 平面绕圈、沿 −Y 下降**。中心线从实测起点渐变到已记录的装配目标，半径按 `radius_m × sin(πu)` 先扩大再收拢，角度为 `2π × turns × u`，`u` 从 0 到 1；终点回到目标中心，不在底部追加横向回正。默认覆盖从目标上方 15 mm 到目标的全程，最大半径 1 mm、共 3 圈、相邻轨迹目标距离不超过 0.5 mm。参数在 `position_spiral` 段，属于试运行参数，须按实际孔销间隙和插入深度调整。

M 后的视觉纠偏可修正左右、前后位置以及水平偏航、倾斜角度；N 使用更新后的抓持关系，螺旋下放时固定目标姿态，不再搜索角度。视觉对准表示 CAD 位姿接近示教目标，不能证明孔销已经配合。手动 W/S、D/A 可平移，J/L 可调整水平角度，但点动不会更新插入目标或重新计算抓持关系；N 仍要求当前末端停在已确认的预插入目标，因此不支持保留任意人工偏移后直接插入。

全部离散目标先做 IK 和关节限位检查，再以 SDK 1% 速度/加速度逐段发送；控制器确认完成、实测到位并稳定后才进入下一段。这是分段近似螺旋，不保证匀速或控制器段内严格走笛卡尔直线。保留轨迹偏差、姿态、深度、超时、设备故障和 Q/空格/Ctrl+C 暂停检查，完成后保持抓握，状态为 `awaiting_confirmation`；重复 N 不会重新启动。

**该模式没有接触力限制，也不能识别顶住、卡孔或真实落座，位置到位不代表装配成功。** 出错或暂停时停止，不自动卸载回撤，也不自动松手。可用 `--output /tmp/rail-position-spiral` 保留 `position_spiral_plan.json`、`insertion_result.json` 和 `result.json`；结果明确记录 `force_controlled: false`、`force_sensor_used: false` 和 `assembly_verified: false`。X 退出仍会停用右手。

预放置纠偏另存以下记录：

| 文件 | 内容 |
|---|---|
| `above_arrival.json` | 粗搬运完成时的 SDK 与实测 FK 机械到位结果 |
| `aligned_observation.json` | 最近一轮新采样、更新后的抓持关系及位姿偏差 |
| `preplace_alignment_plan_N.json` | 第 N 次运动的完整 IK 预规划、导轨与法兰路径、抬升量、观测偏差及是否最终精调 |
| `preplace_alignment_result.json` | 各轮观测、分段执行证据、最终精调标记及 `converged` 收敛状态 |
| `above_aligned_plan.json` | 对准完成后的法兰目标和抓持关系，供 N 螺旋下放使用 |

## 右手接料：准备好后等 5 秒抓握，抓完等 8 秒估姿，按 M 搬运

`receive.py` 默认流程：读取抓后 3 帧有效位姿 → 计算并保存抓持关系均值 → **等待 M** → 求解最终目标 IK、切换 SDK 位置控制、直接点到点移动 → 到达记录装配目标上方 **15 mm** → 视觉复核并保持抓握 → **等待 N** → 悬空采集力基线 → 轻触 → 平面螺旋搜索 → 下插 → 保持抓握、等待人工确认。即使位姿已经稳定，也先等待 M。使用 `receive.yaml` 的 `move_above` 配置和 `targets/rail_goal.json`；目标或标定无效时启动失败，初始估姿超时、规划失败或控制异常时暂停并保持最后手目标。估姿、等待 M、规划及搬运阶段禁止点动和编辑 SAM 提示；到位后恢复点动，按 N 才下放，不自动松手。`run --motion` 同样默认启用该流程；`run` 和 `read_state.py` 仍只读。

**M 手动开始搬运（终端或相机窗口）：** 首次收齐 3 帧后，按各帧采集时刻的机械臂反馈计算导轨相对法兰的关系，保存平移均值和 SO(3) 旋转均值，进入 `waiting_move_key`。该均值在等待 M 和粗搬运期间固定，不被连续预览覆盖；`position_spiral` 在预放置视觉纠偏阶段重新计算并更新抓持关系。按 M 直接用已保存均值和当前机械臂反馈规划移动，不重新采样、不要求新画面，也不检查均值是否超过位置/角度稳定门限。均值尚未生成时按 M 会提示稍后重按，不预存动作请求。最初采样仍检查相机/CAD 身份、时间同步及方向约束；规划时保留 IK/限位和方向检查，执行时保留控制器故障检查。已经暂停的机械臂不能用 M 恢复，终端和窗口会显示原因。`--pose-only` 不提供搬运操作。

**N 力控螺旋搜索插入（终端或相机窗口）：** 默认 `lowering.mode: force_spiral`。上方机械到位、视觉复核通过且当前仍停在该位置时，N 提交一次插入请求；现场参数必须完整。机械臂原控制线程先按实测关节切换到专用笛卡尔阻抗配置，确认模式后从实测位姿接管，手部线程继续保持抓握。提前按 N 不会缓存动作，重复 N 不会重启或增加插入深度。

首版固定实际朝向，沿右臂基座 −Y 接近和插入，在 X–Z 平面做有界阿基米德螺旋。轴向用接触力误差产生受限速度，不按圈数强制下降。搜索期间发现实际下移或明显卸载，先冻结横向搜索，再用持续深度进展确认候选。到达记录目标深度并稳定后状态为 `awaiting_confirmation`：保持当前实测位置和抓握，**不自动确认双孔落座、不自动松手**。首版不搜索偏航或倾斜；平移搜索无法修正两孔朝向或孔距不匹配。

每次在无接触、固定姿态和固定抓持下采集六维力基线。扣除基线后，通过 `T_F_S`（传感器测量坐标/原点到法兰，平移单位 m）、`wrench_sign`（统一为外界对工具的力）转换到右基座，再用 `hole_midpoint_cad_m` 和抓持关系将力矩平移到两孔中点。此方法仅消除该固定姿态下的静态负载和零偏，不能替代变姿态重力补偿；姿态或手形漂移超限会停止。禁止把已接触状态重新置零；上方采样必须确实悬空。既有工具动力学参数继续用于底层控制，不假定它们已包含所抓导轨。

`hole_midpoint_cad_m` 已按用户确认的 Ø8.9/Ø8.0、间距 265 mm 两孔，从当前导轨 CAD 填入 `[-0.00003262591430678186, 0.2345, -0.0161090627486111]` m。该点取两孔孔壁中面中心的中点，仅用于力矩参考点换算，不定义接触高度或插入深度，也不是导轨包围盒中心；来源与网格复核见 [装配台几何核对](../../plan/2026-10-25-第八版装配台几何核对.md)。更换导轨 CAD 时需重新核对该点。

`insertion` 中的传感器标定、阻抗参数、速度、搜索圈距和半径、接触力与力矩限制、深度、超时及稳定阈值仍需按现场填写；`null` 不是默认允许值。`raw_force_limit_n/raw_torque_limit_nm` 是可选的**包含重力和零偏**的传感器原始范数硬限，不能直接当作接触力阈值。带默认值的基线时间、滤波和确认时间也应经现场验证。可先运行：

```bash
python workspace/rail_pose_insert/receive.py check --insertion
python workspace/rail_pose_insert/read_insertion_force.py --duration 3 --output /tmp/rail-force-diagnostic.json
```

诊断程序只读机械臂反馈，不开启手、视觉或运动模式；不与已有设备进程同时连接。它可统计原始六维力噪声；有传感器标定时也输出基线扣除后的基座力/力矩。没有抓持关系的诊断力矩参考点是传感器原点，不是两孔中点。诊断结果不能自动推导允许接触载荷，也不会写回标定。

软超限、搜索耗尽、候选未确认或插入停滞时，保持横向位置沿 +Y 卸载，回退到起始净空高度；实际卸载并稳定后标记 `retracted`。硬超限、失效反馈、控制超时或操作员暂停时停止，不自动回退。插入启动后禁点动、禁修改手型；完成或失败均锁存本次尝试，不自动重试。Q/空格/Ctrl+C 可暂停并保持手目标；X 退出会停用手，须先承托导轨。

指定 `--output` 保留运行记录时，`insertion_plan.json` 保存现场参数、目标、抓持关系及输入摘要；`insertion_trace.jsonl` 保存约 20 Hz 及阶段变化时的力/位姿/深度/半径记录，包含原始反馈时间和右臂序列，缓冲丢失会记录 `telemetry_gap`；`insertion_result.json` 和 `result.json` 记录当前或最终结果。每次控制提交前仍独立执行力和反馈检查，日志采样频率不等于控制频率。 原始力与未滤波补偿力对应同一最新反馈；滤波值另带 `filtered_wrench_timestamp_ns`。若暂停恰好发生在 SDK 接受命令与上层提交确认之间，结果会保留 `motion_submission_attempted` / `motion_acceptance_unknown`；无法确定是否接受且尚无已确认动作时，`insertion_executed` 为 `null`，不会误记为未执行。

旧逐段位置下放仅在显式设置 `lowering.mode: position_steps` 时使用，仍按原 5 mm 分段、SDK 1% 速度/加速度及实测到位检查执行，不包含接触力控制。力控配置缺失或运行失败不会自动降级到旧模式。旧状态 `placement_reached` 只表示位置到位，历史说明见 [N 键缓慢下放](../../plan/2026-10-25-N键缓慢下放.md)。新实现与验证记录见 [六维力螺旋搜索插入](../../plan/2026-10-25-六维力螺旋搜索插入.md)。

**原上方视觉复核及 N 重试仅适用于 `force_spiral` 和 `position_steps`，这两个模式的流程保持不变。** 终端显示位置/姿态误差、各自限值和右基座 XYZ 偏差，窗口显示误差与限值，完整结果另存 `above_verification.json`。`above_target_unverified` 表示机械臂末端已到位，但视觉中的导轨未达到要求；此时 **N 只重新采集复核帧，不执行插入**，通过后需再按 N。重试要求机械臂仍停在原上方目标，不使用上次复核的旧帧，也不放宽原来的 5 mm / 约 2° 门槛。复核中再按 N 不会清空正在收集的帧。若新帧持续超限，应先核对抓持关系、目标和视觉定位；重复复核本身不会修正位置偏差。`position_spiral` 使用前述重新注册与主动纠偏流程。

`move_above.start_mode` 默认为 `manual`。显式设为 `auto` 可恢复稳定均值自动启动；若采样不稳定，则保存该均值并等待 M。均值保存后相机失效不阻止 M 发起粗搬运；到位后的视觉流程仍需要相机恢复并提供新的有效观测。`force_spiral` 和 `position_steps` 保留原到位复核；`position_spiral` 默认要求完成预放置视觉纠偏，不因视觉失效自动跳过。

均值生成时立即写入 `held_observation.json`，包含 3 帧记录、平均 `T_F_O` 及保存时间。按 M 后更新 `pose_sampling`，记录均值存放时长、是否跳过稳定性及实际偏差；规划和最终结果保留同一信息，`events.jsonl` 记录手动请求或拒绝原因。默认临时目录退出后清理，需保留时指定 `--output`。代码不会热加载，需要重启程序后生效；重启后重新估姿，不复用上一进程的抓持关系。

仅需检查分割、位姿及键盘点动时，显式使用 `receive.py run --motion --pose-only`。下文涉及单次估姿、T 重估和估姿后编辑提示的操作适用于此模式；默认搬运在估姿启动后禁用提示编辑。程序不会热加载代码，已有进程需要重启；已持料状态使用 `move_above_held.py`，避免重新张手。

右手接料启动后，右臂先到达**固定接料初始位姿**，再张手并进入 5 秒抓握倒计时。该目标以原 `tianji_teleop.yaml` 的右臂初始化关节角 `[-30, -60, 34, -52, -30, 12, -4]` 度为参考，由 FK 得到原法兰位姿，再沿右臂基座 `-Z`（机器人自身左侧）调整 **15 cm**，保持原初始化朝向。参考角固定保存为 `receive.yaml` 的 `initial_reference_joints_deg`，平移保存为 `initial_offset_base_m: [0.0, 0.0, -0.15]`；当前反馈只用于规划起点，每次启动的目标相同，不累积左移。T、E 和点动不重复回位；只读入口和 `move_above_held.py` 不执行启动回位。规划沿用右臂 IK、关节插值和现有速度/加速度限制，无碰撞检测。

启动回位不再因单帧关节跟随误差超过 0.035 rad（约 2°）直接暂停；它等待实测末端到固定目标的位置/朝向容差内并稳定，到位超时或控制器故障仍停止后续张手和抓握。按 M 后的搬运改用 SDK 位置控制及实测到位检查。

启动接料使用独立到位容差 **5 mm / 3°**，适应阻抗控制下的粗定位；搬运阶段原有的 3 mm / 1° 条件不变。每 2 秒打印一次实际位置/姿态误差，已到位并稳定时立即进入张手，不再额外等待名义关节轨迹的全部时长。超时会显示具体剩余误差。

入口为 `receive.py`，默认配置 `receive.yaml` 使用 `grasp_mode: auto`、`grasp_delay_s: 5.0`、`post_grasp_pose_delay_s: 8.0`。设备初始化、接管并张手完成后，进入接料状态，开始完整的 5 秒倒计时；到时按 `hand_profiles/rail_middle_right.json` 的录制手型开始闭合。初始化和张手耗时不计入这 5 秒。闭合且手指稳定后，再独立等待 8 秒，才启动分割和位姿估计。抓前只显示相机，不加载或运行 SAM、FoundationPose；没有检测到导轨也会闭合。右臂点动在接料等待、抓后等待和搬运结束后可使用，估姿及搬运期间禁用。

抓后 8 秒等待结束，用本地 SAM ViT-H 读取已保存的 bbox 和可选正负点，生成 **一个实例 mask**，再交给 FoundationPose 初始化位姿并跟踪，供搬运前采样；`force_spiral` 和 `position_steps` 还用于到位复核。`position_spiral` 在预放置到位后另用当前 CAD 投影生成临时框，重新分割和注册。首次估姿只需设置框，链路为“保存的框/可选点 → SAM 单 mask → FoundationPose”，不运行 CNOS、DINOv2、CAD 模板匹配或 GPT。SAM 图像编码仍读取完整 RGB，分割使用框提示，最终 mask 限定在框内；不再全图生成候选并按 CAD 相似度排名。手两侧不相连的可见导轨区域保留在同一个 mask，不补出被手遮挡的部分。

在 `--pose-only` 模式，完成后停止分割和估姿，释放模型，继续刷新主视相机画面、保持抓握并接受右臂点动。窗口全程保持打开。需要重画范围时，在主窗口按 E，左键拖出新框后按 Enter 保存，程序自动请求“新提示 → SAM 新 mask → FoundationPose 新位姿”，无需另回终端按 T。也可直接在终端按 T，使用现有保存提示重新估姿。两种模式下，抓前或抓后 8 秒等待期间保存提示，只更新提示文件；首次估姿仍等待抓握完成后的完整 8 秒，按 T 也不会跳过等待。

默认搬运窗口显示连续估姿叠图和阶段状态，包括估姿、规划、移动、到位复核及结果；Q/空格/Esc 暂停、X 退出始终可用。在 `--pose-only` 模式，窗口左侧显示实时相机、黄色提示框及正负点。取得 mask 后，右侧先显示 `LAST SAM MASK` 快照；估姿完成后替换为对应 RGB 帧的 CAD 投影、轮廓、包围框及坐标轴。位姿面板标为 `LAST POSE SNAPSHOT (NOT LIVE)`，显示相机系位置、四元数和帧龄。点动时保留原估姿帧，通过 E 或 T 重估后替换，不把旧位姿投到新的相机图像上。快照只存于当前进程的临时感知目录。

**G 可提前直接闭合右手。** 张手过程中也可按 G 打断张手并开始闭合。接料时视觉启动失败不影响定时闭合、G 和键盘点动，但不会继续自动搬运。闭合过程中不重复触发，也不接收新的点动；默认搬运中出现异常则暂停并保持最后手目标。Q/空格暂停时不会继续倒计时闭合。

首次可先单独设置并保存分割范围；此脚本只打开主视相机，不连接机器人、不估姿：

```bash
python workspace/rail_pose_insert/set_sam_prompt.py
```

在窗口按空格冻结当前画面，然后左键拖出大致包围整根可见导轨的框，可以稍宽松。框设置好后可直接 Enter 保存；仅当分割不足时，才在导轨两侧各加正点（左键），在框内可见手部加负点（右键）。不要把被手占据的框中心当成导轨正点。Z 撤销最后一点，C 清空点，B 重画框，空格重新拍摄并清空全部提示，Q/Esc 取消且不覆盖文件。

提示保存到 `receive.yaml` 的 `sam_prompt` 所指向的 `sam_prompt.json`；路径按配置所在目录解析为绝对路径。文件记录 640×480 原始 RGB 像素坐标、相机序列号、框和点，不预填示例框或点。固定框和点不跟随机械臂；导轨移动后要检查并修改，尤其不能让旧正点落在背景上。

随后在项目根目录的交互终端运行，无需指定输出目录：

```bash
python workspace/rail_pose_insert/receive.py
```

1. 右手空载启动，程序从当前实测姿态接管右臂，移动到原初始化位姿向左调整 15 cm 后的固定目标，到位后张手；重复启动不会继续累加偏移。
2. 看到“接料准备完成，开始 5 秒抓握倒计时”后，人支撑导轨并将中部递到右掌。时间到后自动闭合，也可按 G 提前闭合；抓握不依赖提示文件或分割结果。
3. 抓握完成后等待 8 秒，程序自动用保存的框调用 SAM，再用新 mask 初始化 FoundationPose。提示应在估姿启动前设置好；运行 `receive.py` 时不要另开 `set_sam_prompt.py` 抢占同一相机。
4. 看到“位姿均值已保存，等待 M”后，在终端或主窗口按 M，程序使用该均值规划并搬运到装配目标上方 15 mm；Q/空格可暂停。到位后视觉复核并保持抓握，超差显示未验证。默认手型和掌部偏置仍需实物确认，`holding` 只表示手指稳定并保持目标；仅检查抓握及位姿时使用 `--pose-only`。

主窗口按 E 后冻结当前相机画面，并清空本次编辑中的旧框和旧点，直接等待左键拖出新 bbox，避免把拖框误当成加点；尚未保存时不会覆盖提示文件。拖框后可直接 Enter 保存，也可先添加正点（左键）和负点（右键）；B 重画框、C 清点、Z 撤销。Enter 保存后恢复实时预览，并在抓稳且 8 秒等待结束后自动请求新一轮分割、估姿。E 取消编辑，不保存、不触发重估。E 只冻结编辑画面，不暂停抓握倒计时或当前估姿；编辑时 Q/空格仍暂停设备，X 仍退出，Esc 取消编辑并暂停。独立设置脚本只保存提示，不触发估姿；其空格用于重拍，和动作主窗口的空格暂停功能不同。

抓握目标来自 `/home/yuchen/project/bimanual-teleop/recordings/record_grasp_pose/episode_000001` 的第 2 次 B 标记之后（标记位于 27.068 秒）。采用末尾 37.230～38.230 秒的 200 帧右手实测关节角，逐关节取中位数；对应视频中右手已握住并抬起导轨。数据源为 `hands/right`，单位 rad，顺序为 finger1～5、每指 joint1～4。手型文件记录来源摘要和选取区间，运行时直接读取，不依赖原始录制目录。录制时 kp=5、电流上限 1.5 A；本程序仍使用原有 kp=3、电流上限 0.6 A，因此角度复用不代表抓持力相同。手型在启动时读取，已有进程需重启才生效；先支撑导轨再按 X 退出。

针对当前抓握偏松，已在录制基准上将食指、中指、无名指、小指的 joint3、joint4 屈曲目标各增加 0.05 rad（约 2.9°），小幅收紧。手型文件的 `adjustment` 记录偏移，20 个目标角已通过 SDK 关节限位检查，实际贴合仍需现场验证。

拇指在录制对掌方向上，将 joint3 增加 0.05 rad、joint4 增加 0.10 rad，以增加指腹贴合；当前拇指目标为 `[-0.027516, -0.771024, 1.017400, 0.114119]` rad，偏移记录在 `thumb_adjustment`。这是待现场确认的试调姿态，已通过关节限位检查。

| 按键 | 功能 |
|---|---|
| G | 按默认手型直接闭合，不依赖视觉或静止检测 |
| W / S | 右臂基座 +X / -X，前 / 后 |
| R / F | +Y / -Y，上 / 下 |
| D / A | +Z / -Z，向外 / 向内 |
| I / K、J / L、U / O | 分别绕固定基座 X、Y、Z 正向 / 负向转动 |
| + / - | 调整点动步长，默认 1 mm / 0.5° |
| T（终端，仅估姿模式） | 抓取后重新读取提示、分割并估姿一次 |
| E（主窗口有焦点） | 估姿搬运启动前可重画 bbox；仅估姿模式下 Enter 保存后自动重估；搬运及暂停期间禁用 |
| M（终端或主窗口） | 直接使用已保存的位姿均值规划并移动；不重新采样、不检查位姿稳定性；设备已暂停时显示原因 |
| N（终端或主窗口） | 上方到位并复核通过后缓慢下放到装配目标；到位保持抓握，不重复下放 |
| Q / 空格 / Ctrl+C | 暂停并保持最后手目标；本次不恢复 |
| X | 退出并停用右手，先支撑导轨 |

按 M 后的搬运使用 **SDK 位置控制**：只求最终法兰目标的 IK，应用层保留起点和终点，不再生成逐段抬升、平移和旋转的密集轨迹。唯一的右臂线程停止 Cartesian 阻抗伺服，以当前实测关节切入位置模式，再通过 `TianjiDriver.move_joints("right", ...)` 下发最终目标，由控制器完成加减速。速度和加速度比例上限均保持 3%；SDK 模式切换、反馈新鲜度、设备故障及关节限位检查仍有效。

SDK 确认位置模式、目标回读、低速和实测关节到位后，程序还要求法兰误差在 **3 mm / 1°** 内连续 **0.5 秒**，再进行新的视觉复核。SDK 运动超时为 `move_above.position_move_timeout_s: 120.0`；SDK 返回后，末端收敛另有 `reach_timeout_s: 30.0`。不再等待应用层名义轨迹时长，也不将当前位置与远处最终目标的关节差作为旧阻抗跟随误差而暂停。运行时每 2 秒打印距终点的位置和姿态误差，事件日志记录 `right_position_queued`、`right_position_started`、`right_position_complete`。

启动回位及搬运前点动仍沿用原有 Cartesian 阻抗控制。搬运到位后保持位置模式，后续点动也由当前位置求解一个位置目标；上一点动完成前的重复按键不累积。Q/空格/Ctrl+C 会取消正在等待的 SDK 运动并请求停止，手保持最后目标，本次运行不恢复。实现与验证记录见 [M 键位置控制直达目标上方](../../plan/2026-10-25-M键位置控制直达目标上方.md)。

`receive.py` 窗口显示完整主视画面与底部状态面板，实时叠加用户的 SAM 提示。张手阶段提示抓握计时尚未开始，准备好后显示 `Auto close in: 5.0 s` 等倒计时；抓握完成后显示 `Pose recognition starts in: 8.0 s`，再进入分割、估姿就绪或失败状态。`receive_roi_xyxy` 和连续检测计时仅保留给旧 CNOS 诊断入口。配置 YAML 在启动时读取，修改后需重启；每次分割重新读取 SAM 提示 JSON，E 保存后无需重启。

窗口在连接机器人 SDK 前先显示初始化画面，彩色相机同时启动；SDK 初始化在独立线程执行，主线程持续刷新窗口、显示到达的相机帧并处理暂停/退出。需在图形桌面的终端运行；未找到 `DISPLAY` / `WAYLAND_DISPLAY` 时会明确提示窗口未启用。只读反馈入口和显式 `--no-view` 不显示窗口。

抓取前只开启主视相机彩色流；抓取后估姿时切换到 RGB-D。首帧最多等待 10 秒，后续取帧保持 2 秒超时；在此超时内跳过 RGB/深度时间差超过 5 ms 的帧对。默认搬运继续跟踪；`--pose-only` 成功后保留 RGB-D 流，空闲时仅读彩色画面，重估复用相机。仅估姿模式下，缺少提示、空 mask 或估姿失败会恢复彩色预览，可 E 或 T 重试；搬运流程在首次采样或到位复核期间未能取得新鲜位姿时超时暂停，等待 M 期间使用已保存均值，不因新画面缺失而超时。终端与窗口显示打开 RGB-D、加载 SAM、框提示分割及 FoundationPose 初始化等阶段。

仅需检查旧固定区域 CNOS 检测时，可单独运行：

```bash
python workspace/rail_pose_insert/view_detection.py
```

此入口仅供单独检查识别，不连接机器人执行器，达到 1 秒也只显示 `READY - PREVIEW ONLY`；目标移出后重新计时。窗口中 Q、空格、Esc 或 X 退出。正常接料直接运行 `receive.py` 即可，不需要先运行此入口；两者不能同时占用相机。

默认动作入口的窗口区分倒计时、闭合、估姿、规划、移动、视觉复核及暂停。仅估姿模式完成后显示 `POSE READY (T: REFRESH)`，失败显示 `POSE UNAVAILABLE - CAMERA ONLY (T: RETRY)`。窗口 Q/空格/Esc 暂停、X 退出；G、T 和右臂点动在终端操作，并受当前阶段限制。

SAM 使用官方 `SamPredictor` 的 `box`、可选 `point_coords`/`point_labels` 接口，设置 `multimask_output=False`，不再对多个候选进行排名。质量分数是 SAM 预测的 mask IoU，不是导轨识别概率，也不作为自定义置信度门槛。离线抓持样本中，一个稍宽松的框已分出导轨两侧共 5208 个像素，避免此前只选中短端的问题；覆盖整张桌面的大框仍混入背景，加点也未完全消除，因此框应围绕导轨，避免包含整张桌面。这只证明特定样本的分割有所改善，不能保证其他姿态或 FoundationPose 的精度。比较图和实现记录见 [SAM 框提示分割与抓后估姿](../../plan/2026-10-25-SAM框提示分割与抓后估姿.md)。旧 [CNOS 检测诊断](../../plan/2026-10-25-CNOS官方流程与检测诊断.md) 保留为历史记录，不代表当前默认流程。

依赖检查不连接设备：

```bash
python workspace/rail_pose_insert/check.py
```

控制和感知都使用 `tianji-rail-diffusion-policy`，安装记录见 [FoundationPose 共用环境](../../plan/2026-10-24-FoundationPose本机环境补齐.md)。新默认链路复用已有 SAM-H 权重和 FoundationPose，无需下载 DINOv2 或渲染 CAD 模板。`check.py` 检查配置和视觉依赖；接料时视觉异常会提示并保留 G。只读反馈使用 `python workspace/rail_pose_insert/read_state.py`，不使能动作。

默认使用系统临时目录传递感知数据，退出后自动清理，不在项目中创建输出目录。SAM-H 权重复用 `third_party/cnos/pretrained/segment-anything/sam_vit_h_4b8939.pth`；目录名沿用旧缓存位置，不表示默认仍运行 CNOS。用户的 `sam_prompt.json` 会保留，供后续启动读取。如需保留诊断记录，可使用高级命令 `receive.py run --motion --output 新目录`，其中 `grasp_source` 区分 `keyboard` 与 `timer`。测试命令为 `python -m unittest discover -s workspace/rail_pose_insert/tests -v`；本轮验证不执行真机动作。

## 右手抓持后估姿、可视化与移动到目标上方

抓持后位姿现在先按**目标坐标系**消除首尾翻转歧义，再计算抓持关系和规划。只考虑原始姿态，以及绕导轨中部 CAD Z 轴转 180° 的首尾对调；同时修正旋转和端部原点的平移，不能只把坐标轴反向。旧目标文件仍可使用，无需为这次修正重新示教。

`receive.yaml` 的 `pose_alignment` 将 `R_goal.T @ R_observed` 分解为固定 XYZ 的 roll/pitch/yaw，要求**三个分量分别小于 45°**，不是总旋转角小于 45°。三个分量均小于 30° 属于通常范围；30～45° 仍可通过。没有满足条件的候选时拒绝搬运，不截断角度、不强行使用目标朝向。分支对齐在多帧平均之前完成，到位后的新观测也按同一目标方向对齐。M 跳过抓后采样稳定性门限，机械到位容差继续生效；到位视觉复核及 `position_spiral` 的预放置纠偏均检查新观测的采样一致性，并分别使用对应配置的到位容差。规划前还会用当前机械臂反馈复查一次每轴角度。

预览和控制使用同一个修正后的位姿。图像只叠加调整后的绿色 CAD 轮廓、包围框及红 X/绿 Y/蓝 Z 轴，不显示原始位姿或目标/上方参考框；数值面板显示相对目标 XYZ 角度。静态快照保留原始图像区域。默认搬运找不到匹配相机/CAD 的目标时启动失败；对齐后任一轴达到 45° 的观测不会用于搬运，持续无有效观测则超时暂停。仅估姿模式提示对齐失败并保留相机画面，不发布未通过约束的位姿快照。首尾对齐前的成功结果保存在 `T_C_CAD_raw`，`pose_alignment` 记录候选及补偿变换，仅供诊断。首尾对齐不改变跟踪器状态；启用下述整机方向先验时，跟踪器内部使用先验修正后的旋转，既有跳变检查仍生效。模型层判定丢失时需要重新估姿。

该导轨是**近似首尾对称**，不是任意滚转都等价的圆柱；CAD 局部几何差异最大约 4 mm，因此不允许绕 X/Y 的上下翻转。该处理利用“小角度搬运”先验解决方向歧义，不证明视觉位姿已达到插入精度。左臂预插入也使用同一规则，新示教会先对齐样本再平均；旧左臂执行计划需重新生成。实现、CAD 核查和验证边界见 [位姿首尾对齐记录](../../plan/2026-10-25-导轨位姿首尾对齐.md)。

### 整机方向先验与抖动验证

右手抓后估姿（`receive.py`、两个 `move_above` 入口及 `--pose-only`）默认启用 `receive.yaml` 的 `body_orientation_prior`。以整机 URDF `base_link` 为准：+Z 向上、−Y 向前、X 沿左右；CAD +Z 为凹面朝向、Y 为 480 mm 长轴。相机外参先通过经 SDK FK 核验的 URDF 变换转入整机，明确移除 `sim_world` 的 90° 偏航，不把单臂基座当作整机坐标。

初始化在“凹面向上、长轴沿左右”的两种首尾朝向附近生成候选，倾斜搜索范围 15°、水平方向偏差搜索范围 30°。每帧视觉细化后，旋转向最近的整机理想方向修正 25%，单次最多 5°；修正作用于居中网格，保留视觉测得的中心，再正确还原原 CAD 端部坐标。候选在修正后接受 FoundationPose 图像评分，跟踪器保存同一结果。视觉细化结果倾斜超过 30° 或长轴水平方向偏差超过 45° 时拒绝该候选，全部矛盾则重新估姿。上述角度和权重是可调参数；未锁死姿态、未使用目标平移或跨帧平均。

`T_C_CAD_before_body_prior` 记录本次跟踪历史下、施加方向修正前的视觉结果，`orientation_prior` 记录整机变换、角度及修正量；它不是独立无先验基线。运行目录的 `perception/<generation>/pose_records.jsonl` 保存连续估计与失败原因，退出后保留需显式指定 `--output`。设置 `body_orientation_prior.enabled: false` 可恢复原估姿流程。此开关作用于右手初次抓后估姿；`position_spiral` 的预放置重新注册阶段始终关闭此先验，与目标示教使用同样的原始视觉位姿定义。左臂预插入及目标示教也不应用这项方向修正。

离线验证脚本 `tools/validate_body_prior.py` 不连接相机或机器人。对同一静态场景分别运行 `replay --variant baseline`、`replay --variant prior`（可加 `grid_only` 对照），指定相同 `--scene`、`--mask` 及各自 `--output`，随后用 `summarize --runs ... --output comparison.json` 汇总。输入摘要必须一致；同时记录角度、中心、CAD 端部抖动、3 帧一致性和深度残差。实测数据与限制见 [整机先验验证记录](../../plan/2026-10-25-整机方向先验与位姿抖动验证.md)。静态抖动改善不代表绝对位姿或插入精度已达标。

直接运行 `receive.py` 已默认启用以下流程，原抓取后移动脚本仍可使用：

```bash
python workspace/rail_pose_insert/move_above.py
```

接料、G 闭合功能沿用原入口。右手闭合并稳定后，程序用用户设置的 SAM 框/可选点分割当前导轨，将单 mask 交给 FoundationPose 初始化，再跟踪并读取 3 帧有效抓持后位姿；不使用闭合前的观测计算抓持关系。按相机采集时间插值右臂反馈，计算导轨相对右法兰的 `T_F_O`，保存均值并等待 M。初次采样一致性仅记录为诊断信息，不阻止 M 使用已保存均值。FoundationPose 首次在此阶段加载；每次重新分割、注册时按顺序释放旧模型，避免 SAM 与 FoundationPose 两套网络同时占用显存。启动搬运前应确认提示范围覆盖导轨。

抓取后必须重新执行“采集当前 RGB-D → 使用当前框/点调用 SAM → 新 mask 初始化 FoundationPose”，不沿用抓取前的 mask。默认链路不需要 CAD 渲染模板或 DINOv2 特征；CAD 模型仍用于 FoundationPose。`receive.py` 和搬运入口在等待 M 时继续跟踪供预览，但不替换已保存的均值；`force_spiral` 和 `position_steps` 移动后再使用新观测做视觉复核。`position_spiral` 默认在机械到位并稳定后重新分割和注册，更新抓持关系并主动纠偏，完成后等待 N。只有显式 `--pose-only` 在单次估姿后停止推理。

使用现有 `targets/rail_goal.json` 和相机标定，将目标转换到**右臂 SDK 基座**。上方定义为右基座 **+Y**（机器人竖直向上），默认高于目标 **15 mm**，朝向与目标一致；可用 `--above-height 0.08` 改为 80 mm。先对导轨目标增加此高度，再通过已保存的平均抓持关系换算法兰目标，不能直接把法兰移到导轨坐标。M 的粗搬运只求最终目标 IK，通过 SDK 位置控制直接执行关节点到点运动，不增加抬升中间点。`position_spiral` 随后的视觉纠偏使用分段轨迹，并按角度修正需要增加抬升。末端段内路径不保证为直线，也没有场景碰撞检测。上方到位后保持位置模式和抓握。默认 `force_spiral` 通过视觉复核后，按 N 且现场参数完整时才切换至力控插入；`position_spiral` 默认先完成预放置视觉纠偏，按 N 继续用位置控制螺旋插入。均不自动松手。

窗口只绘制约束对齐后的导轨与坐标轴；目标及目标上方位姿仍用于计算，不另外叠加参考框。`--no-view` 关闭窗口，运行期间仍生成叠图供程序读取；仅指定 `--output` 时在退出后保留。终端 Q、空格、Ctrl+C 仍暂停，窗口 Q/空格/Esc 也可暂停；到位后继续保持，X 才退出并停用右手。规划、移动和预放置视觉纠偏期间不接收点动，流程完成并等待 N 时恢复点动。

导轨已经由右手抓持时，运行 `python workspace/rail_pose_insert/move_above_held.py`，从当前实测手型接管，不发送张手命令。**旧接料进程不会热加载本次改动，不能同时启动两个设备控制进程。** 原程序退出时会停用右手，切换前先支撑导轨；新实例按实测手型接管不保证复现旧实例的抓持力。

检查完整配置与 GPU 依赖，不连接机器人：

```bash
python workspace/rail_pose_insert/check.py
```

指定 `--output` 时还会保留 `held_observation.json`、`above_plan.json` 和实时叠图。初次 3 帧采样的稳定性标准为单帧位置相对均值偏差不超过 **10 mm**，由 `move_above.pose_consistency_position_tolerance_m` 配置；角度一致性门限仍为 0.035 rad（约 2°）。抓后均值无论是否稳定，默认都等待 M；M 不受该门限阻止。`force_spiral` 和 `position_steps` 的到位复核仍检查采样一致性，上方阶段的 `result.json` 只有新鲜的到位后视觉误差满足容差才记录 `above_target`；否则记录 `above_target_unverified` 或暂停原因，并保持右手目标。这两个模式的视觉误差容差仍为 **5 mm / 0.035 rad**，M 不跳过复核。`position_spiral` 默认按 `preplace_alignment` 的独立容差重新测量并纠偏，达到 **2 mm / 1°**、完成必要的最终精调并确认 SDK 和实测 FK 到位后，才记录 `above_target`；关闭该功能时仅表示机械到位。所有模式下，目标不可达或反馈失效都不会被标成已到位。

软件验证覆盖完整模拟流程、真实 SDK IK/FK、原 Cartesian 伺服数值回放、GUI 显示及合成 RGB-D GPU 初始化/跟踪。**尚未进行真机到位验证**；无纹理 CAD 合成回放存在位置偏差与旋转漂移，不能将“输出有效位姿”视为精度合格。详细结果见 [本轮实现与验证记录](../../plan/2026-10-24-右手抓持估姿与目标上方移动.md)。

## 左臂搬运至预插入位姿（已有入口）

本轮完成张手接料、按键闭合、停稳估计、固定抓握关系计算、目标 IK 与限速关节轨迹，以及到达目标沿插入轴外移 **20 mm** 的位姿。到位后保持夹持和机械臂目标，等待操作员处理。程序的 `end_stage` 固定为 `preinsert`。

本目录已实现软件流程，真机到位精度和时序仍待现场验收。FoundationPose 源码、运动学、SDK、机器人模型及标定工具已内置到本项目 `third_party/`；无需相邻项目。迁移记录见 [依赖内置化计划](../../plan/2026-10-23-预插入依赖内置化.md)。

当前运动方式为 **目标 IK + 关节插值**。已移除 FCL 碰撞检测和 RRT 绕行，不再检查机器人自碰撞、持轨组合与环境障碍物，也不再要求 `scene.json`。仍检查关节/腕部耦合限位、速度和加速度、反馈时效、跟踪误差，以及终点视觉位姿误差。

## 1. 环境与入口

以下命令均在 **diffusion_policy 项目根目录** 执行，项目可以复制到任意位置。控制、规划、标定和 FoundationPose GPU 感知共用 `tianji-rail-diffusion-policy` 环境。标准 Python 包、CUDA/驱动和设备连接仍需准备。

首次创建环境：

```bash
PYTHONNOUSERSITE=1 PIP_USER=false conda env create -f conda_environment_tianji.yaml
```

权重保留在本地，不随 GitHub 仓库发布；运行前按 [权重放置说明](../../third_party/foundationpose/weights/README.md) 准备两组模型。FoundationPose 原生扩展的官方源码构建步骤和本机兼容版本见 [环境安装记录](../../plan/2026-10-24-FoundationPose本机环境补齐.md)。旧的 `third_party/foundationpose/workspace/scripts/install.sh` 面向独立环境，不用于本项目的共用环境安装。先核对全部内置资源：

```bash
python workspace/rail_pose_insert/tools/verify_bundle.py --source-only
```

```bash
python workspace/rail_pose_insert/check.py
```

该命令不连接设备，列出缺少的现场输入。运行时不依赖 shell 工作目录，配置路径统一相对 `config.yaml` 所在目录解析。感知进程默认使用当前 Python 解释器，与控制共用已激活的 DP 环境；也可设置 `FOUNDATIONPOSE_PYTHON` 或配置中的 `foundationpose_python`，无需填写开发机的固定路径。

## 2. 现场输入

1. 本机默认使用已导入的实测外参 `calibrations/head_to_left_base.json`（来源及复用条件见第 2.1 节）。矩阵 `T_B_C` 把头部彩色光学系坐标转换到 `tianji_left_base`，平移单位米。相机序列号需匹配当前头部 D436；重新标定时可参考 `calibrations/head_to_left_base.example.json`。
2. 记录正确就位的目标，保存为 `targets/rail_goal.json`：

```bash
python workspace/rail_pose_insert/teach_target.py
```

导轨已正确就位时，在相机窗口标注可见导轨；`a` 添加轮廓，`h` 排除手或孔内背景，Enter 确认。默认采 3 次位姿，检查一致性后保存原始 CAD 系下的 `T_C_O_goal`。已有目标文件不会被覆盖。

`outward_axis_cad` 默认 `[0,0,1]`，按现有导轨 CAD 定义为离开导轨台的孔轴方向。预插入位置为 `p_goal + 0.02 * R_goal * outward_axis_cad`，朝向保持目标朝向；核对现场符号，不能用基座 Z 加 0.02 替代。

这些现场文件不填示例真值，不使用仿真相机外参充当真机标定。

### 2.1 获取相机外参 `head_to_left_base.json`

本机已导入 2026-10-23 的主相机实测结果 `external_20261023_175718_solve`，相机为 D436 **`405622076394`**。原始 `result.json`、`report.md` 和 5 张独立验证叠图保存在 `calibration_runs/external_20261023_175718_solve/`；运行时只读取 `calibrations/head_to_left_base.json`，无需相邻项目。上述原始记录及运行外参均按现有 `.gitignore` 规则保存在本地，新检出项目需要单独复制或重新导入。

该结果为 `external_eye_to_hand_result_v2`，直接使用 SDK 的 `tianji_left_base`，机器人模型摘要与本项目一致。导入时原样使用其米制 `T_base_camera_opencv` 作为 `T_B_C`，不施加旧格式的 URDF 基座偏移。独立验证的像素 RMS 中位数为 0.831 px、最大值为 1.291 px，平移一致性中位数为 1.358 mm、最大值为 2.697 mm，旋转一致性最大值为 0.730°；这些指标不等于绝对定位精度。仅在主相机与机器人基座的相对安装未改变时复用。

导入新版求解结果的命令如下；已有输出文件不会被覆盖，重新导出时用 `--output` 指定新文件：

```bash
python workspace/rail_pose_insert/tools/export_head_calibration.py \
  workspace/rail_pose_insert/calibration_runs/external_20261023_175718_solve/result.json \
  --camera-serial 405622076394
```

以下为本项目内置旧版工具的重新标定流程：

现有 [独立标定工具](../../third_party/camera_calibration/docs/external.md) 可以通过已标定左腕相机求头部固定相机外参。前提是左腕相机支架自手眼标定后没有移动；现有参考文件对应左腕序列号 `408322072104`。若支架移动，先按 [腕部标定说明](../../third_party/camera_calibration/docs/wrist.md) 重做手眼标定。

本实验模板中的头部 D436 序列号为 **`405622076394`**。下面命令显式指定它，避免采用第三视角标定工具默认的另一台 D435。更换头部设备时，采样和导出两处都填写实际序列号。

在连接相机和机器人反馈的电脑上执行，关闭其他相机和反馈端口占用程序：

```bash
conda activate tianji-rail-diffusion-policy
python third_party/camera_calibration/run.py external collect \
  --wrist-result third_party/camera_calibration/wrist_reference.json \
  --side left --external-serial 405622076394 --samples 25 \
  --output workspace/rail_pose_insert/calibration_runs/head_capture_01
```

让两台相机同时完整看到现有 4×5 非对称圆点板，使用真实板尺寸。头部保持固定，改变板和／或左腕的观察位置与倾角，每组停稳 1～2 秒。覆盖实际接料和预插入区域，按工具提示采满 25 组。采样只读取相机和关节反馈，手臂姿态由操作员通过示教器调整。可以固定左腕、仅移动板，前提是两相机均能看到足够多不同板姿态。

以下求解和实验格式转换均为离线操作；不连接设备。输出目录或文件须尚不存在：

```bash
python third_party/camera_calibration/run.py external solve \
  workspace/rail_pose_insert/calibration_runs/head_capture_01 \
  --wrist-result third_party/camera_calibration/wrist_reference.json \
  --output workspace/rail_pose_insert/calibration_runs/head_result_01

python workspace/rail_pose_insert/tools/export_head_calibration.py \
  workspace/rail_pose_insert/calibration_runs/head_result_01/result.json \
  --camera-serial 405622076394
```

检查 `head_result_01/report.md` 和独立验证叠图。导出脚本只接受通过验证的实测结果。对于旧格式 `external_camera_result_v1`，脚本验证序列号和 URDF，并用官方离线 FK 转成实验所需的 **SDK 左臂基座**。本项目内置旧版 `camera_calibration` 的通用 `T_base_camera_opencv` 使用 URDF 第一关节系，不能直接改名为 `T_B_C`；当前两种基座有 174.5 mm 固定偏移。转换复用规划中的基座配准，并校验非零关节姿态。新版眼在手外结果已经使用 SDK 原生基座，导出时检查模型摘要、左臂坐标系和原始相机方向，无需上述转换。

默认保存到 `calibrations/head_to_left_base.json`，包含坐标系、米制单位、相机序列号、4×4 矩阵和来源摘要。若已经有其他方法得到的实测外参，可按现有模板填写，须满足 `p_B = T_B_C @ p_C`，其中 C 为头部原始彩色 OpenCV 光学系、B 为 `tianji_left_base`。

标定报告的通过条件是像素误差和位姿一致性，不能代替工作区绝对精度验收。使用未参与标定的实体点或标定板位置核对坐标链，再进行搬运。

### 2.2 获取目标位姿 `rail_goal.json`

先完成上面的相机外参。相机、机器人基座和导轨台保持固定，将导轨完全正确就位，人员撤手，保留导轨两端的可见区域，然后执行：

```bash
python workspace/rail_pose_insert/teach_target.py
```

窗口中 `a` 添加导轨轮廓，`h` 排除手或孔内背景，Enter 确认。程序采集 3 个估计，检查平移／旋转一致性，取均值并自动生成 `targets/rail_goal.json`，保存观测、相机序列号和 CAD 摘要；图像、掩码和叠图仅在临时目录中生成，完成后清理。首次 GPU 初始化需要等待。

保存的是 **完全就位的导轨 CAD 位姿 `T_C_O_goal`**，不是抬高后的预插入位姿。20 mm 外移由规划程序计算，无需手工抬高导轨再示教。检查叠图的两端、孔位和 CAD 方向，重复一致性本身不能排除持续的估计偏差。此命令只使用相机，不向机械臂或手发送动作。

两个现场文件完成后执行第 1 节的 `check`。它验证文件格式、坐标约定及模型一致性，不验证实物测量精度。无需准备障碍物配置。

## 3. 真机运行

退出遥操作、录制及其他占用头部相机或控制设备的进程后启动。机器人应处于已经确定的接料姿态，右臂位于固定待机位置。入口保持启动时的实测机械臂位置，按键触发后逐步张手；不会执行未经规划的回位动作。

先执行只读观测，记录设备实际状态：

```bash
python workspace/rail_pose_insert/read_preinsert_state.py
```

在有桌面和交互终端的本机启动动作入口：

```bash
python workspace/rail_pose_insert/preinsert.py
```

默认不保留运行目录，无需输出参数。动作按键如下：

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

到位后继续保持，**不会自动松手**。`preinsert_reached` 表示实测法兰到位、视觉位置和朝向误差检查通过；`preinsert_unverified` 表示终点视觉复核未通过，仍保持夹持，可用 M 重新测量规划。两者都没有执行继续插入。

## 4. 数据、模型与路径检查

- 使用 `hand_profiles/rail_place3_left.json` 的张手和闭合下发角度，来源为 episode 000034。候选演示偏端部握持，中部握持仍需现场调整；实测角度一并保留用于比较。
- 用每张图像采集时刻插值实测左臂关节，再计算每次 `T_F_O`；不拿推理结束时的法兰位姿拼接旧图像。停稳期间允许推理耗时，检查状态历史覆盖及机械臂是否移动。
- 头部 RGB-D 在独立感知进程采集并去畸变、注册深度，支持与已有 D436 录制一致的畸变模型。相机仅由感知进程占用。
- 独立规划进程求解预插入目标的关节角，从当前关节角直接生成五次插值轨迹，限制速度和加速度。直达指关节空间插值，末端在空间中的轨迹不保证为直线。
- 使用现有完整 URDF、手和腕部安装链。通过官方 FK 显式对齐基座，校验非零姿态的一致性；不把 URDF 第一关节系当作 SDK 基座。
- 执行时检查反馈时效、实测跟踪误差、右臂移动和手型变化。外参、目标、网格或机器人模型在规划后变化时拒绝执行。
- 暂停保持手指，退出才停用。当前沿用已有末端动力学辨识参数；持轨后的额外负载对实际到位精度的影响需现场验证。

## 5. 日志与结果

接料与预插入默认临时运行并在退出后清理。仅显式指定 `--output` 保留诊断记录时，预插入记录包含：

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

SDK 接受命令不作为到位依据。终点使用实测法兰误差并持续稳定，再根据当前导轨位姿检查位置和朝向误差。此验收仅针对预插入阶段。

## 6. 离线规划、重放与测试

创建明确标记 `test_data=true` 的输入。它只复用录制的关节状态，握持关系、相机和目标是构造的集成测试数据：

```bash
python workspace/rail_pose_insert/tools/make_offline_case.py \
  --output workspace/rail_pose_insert/outputs/offline_demo_01

python workspace/rail_pose_insert/run.py \
  --config workspace/rail_pose_insert/outputs/offline_demo_01/config.yaml plan \
  --state workspace/rail_pose_insert/outputs/offline_demo_01/state.json \
  --observation workspace/rail_pose_insert/outputs/offline_demo_01/observation.json \
  --output workspace/rail_pose_insert/outputs/offline_demo_01/plan.json --allow-test-data

python workspace/rail_pose_insert/run.py \
  --config workspace/rail_pose_insert/outputs/offline_demo_01/config.yaml replay \
  --plan workspace/rail_pose_insert/outputs/offline_demo_01/plan.json \
  --output workspace/rail_pose_insert/outputs/offline_demo_01/replay

python -m unittest discover \
  -s workspace/rail_pose_insert/tests -v
```

离线重放只检查关节限位并输出 `path.png` 和 `result.json`，不访问硬件、不进行碰撞检测。规划和重放结果标记 `collision_checked: false`。旧版 `plan.json` 需要重新生成。带测试标记的标定、目标与观测均被真机入口拒绝；离线感知重放数据也不能代替新的真机观测。

真实录制 RGB-D 的感知接口验证命令（以下样本仅保留在本地，不进入 Git；其他机器需将 `--scene` 改为自行准备的 RGB-D 场景）：

```bash
python workspace/rail_pose_insert/perception_worker.py \
  --foundationpose-root third_party/foundationpose \
  --mesh third_party/foundationpose/workspace/input/model/rail.obj \
  --scene third_party/rail_assets/examples/rgbd \
  --index 0 --samples 3 \
  --output workspace/rail_pose_insert/outputs/perception_demo_01
```

已有核心自动检查、完整模型离线规划/重放和真实录制的 3 帧 GPU 位姿接口验证。本次移除碰撞检测后的验证结果见 `outputs/no_collision_validation.json`；迁移时的历史验证见 `outputs/standalone_validation.json`。尚无本轮真机运动验收；现场仍需外参接入、正确就位目标示教，以及抓握、时序和最终到位验证。左臂仍止于预插入；右臂已提供独立的六维力螺旋搜索插入，真机参数与接触验收仍待现场完成。
