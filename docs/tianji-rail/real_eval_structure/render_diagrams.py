"""核对当前实现后的文件职责与真机评估流程；只生成图片，不连接硬件。"""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'rail_diagram_renderer', ROOT.parent / 'task_flow_explanation/render_diagrams.py')
draw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(draw)
draw.ROOT = ROOT
node, edge, group = draw.node, draw.edge, draw.group

FIGURES = [
    dict(name='01-files', size=(1800, 1390),
         title='真机评估：文件与职责',
         subtitle='入口加载模型；Runner组织试验；Env管理观测与设备；Runtime负责定时控制。',
         groups=[
             group('cfg', (40, 155, 550, 290), '配置'),
             group('entry', (625, 155, 550, 290), '独立启动入口'),
             group('model', (1210, 155, 550, 290), '模型与预处理契约'),
             group('env', (40, 515, 550, 335), '实时环境 Env'),
             group('runner', (625, 515, 550, 335), '试验组织 Runner'),
             group('runtime', (1210, 515, 550, 335), '动作执行 Runtime'),
             group('vendor', (40, 965, 840, 295), '本项目 third_party 〔Env / Runtime使用〕'),
             group('logs', (920, 965, 840, 295), '评估记录 〔Runner异步写入〕'),
         ], nodes=[
             node('config', (65, 225, 500, 190), 'diffusion_policy/config/',
                  'tianji_rail_real_eval.yaml', 'tianji_teleop.yaml / wuji_teleop.yaml', 'recording.yaml：相机顺序与序列号', group='cfg'),
             node('main', (650, 225, 500, 190), 'eval_tianji_rail.py',
                  '参数与模型契约检查', '加载 EMA / 模型及归一化器', '模拟输入预检查；选择运行模式', group='entry'),
             node('checkpoint', (1235, 225, 500, 190), 'checkpoint + Policy',
                  '权重、归一化器、输入尺寸及时间窗口', 'policy/diffusion_transformer_', 'hybrid_image_policy.py', group='model'),
             node('environment', (65, 590, 500, 230), 'real_world/tianji_rail_env.py',
                  '三路RGB + 双臂FK + 双手实际关节', '历史缓存、对齐及图像预处理', 'start / get_obs / engage', 'exec_actions / pause / close', group='env'),
             node('trials', (650, 590, 500, 230), 'env_runner/tianji_rail_real_runner.py',
                  '试验开始 / 结束；独立GPU推理线程', '操作员命令、故障与超时', 'TrialRecorder：异步日志与视频', 'summarize_trials：结果及成功率', group='runner'),
             node('targets', (1235, 590, 500, 230), 'real_world/tianji_policy_runtime.py',
                  '52维拆分、名义时刻及过期筛选', '工作空间 / 跳变 / 速度限制', '双臂200 Hz；左右手各120 Hz', '覆盖耗尽或设备异常时请求暂停', group='runtime'),
             node('devices', (65, 1040, 790, 190), 'third_party/bimanual_teleop/',
                  'control/arm/：约束笛卡尔执行器与伺服', 'devices/tianji/、devices/wuji/：驱动与运动学',
                  'recording/camera.py：实时相机；vendor/tianji/：SDK与模型', group='vendor'),
             node('records', (945, 1040, 790, 190), 'data/real_eval/<session>/',
                  'session.json / summary.json', 'trial_0001/：trial.json、events.jsonl、timing.json',
                  'camera_0.mp4 / camera_1.mp4 / camera_2.mp4', group='logs'),
         ], edges=[
             edge('config', 'main', ('right', 'left')),
             edge('checkpoint', 'main', ('left', 'right')),
             edge('main', 'trials'),
             edge('trials', 'environment', ('left', 'right'), label='观测'),
             edge('trials', 'targets', ('right', 'left'), label='动作'),
         ], notes=[
             'Runner通过Env发布动作；下层快照只保留真机评估依赖，recording目录用于实时相机观测。',
             '文件位置以当前工作区为准；图中控制频率为配置值，不表示已达到硬件实测指标。']),
    dict(name='02-flow', size=(1800, 1770),
         title='真机评估：启动、闭环与停止',
         subtitle='三种运行模式共用模型加载；只有运动模式接合设备并提交控制目标。',
         groups=[
             group('startup', (40, 140, 1720, 525), 'A · 启动与模式选择'),
             group('loop', (40, 705, 1720, 925), 'B · 在线试验：观测 → 推理 → 定时目标 → 实际反馈'),
         ], nodes=[
             node('load', (650, 215, 500, 170), '入口校验与模型预检查',
                  '读取部署配置及 checkpoint', '检查输入契约、EMA与归一化器', '运动模式先检查工作空间已填写', group='startup'),
             node('only', (65, 465, 500, 165), '--check-only',
                  '模拟观测做 batch=1 推理', '保存 preflight.json 后退出', '不连接任何设备', group='startup'),
             node('connect', (650, 465, 500, 165), '连接相机和设备反馈',
                  '启动观测、控制及记录所需线程', '默认影子模式：只读、不使能', '当前模型预处理以 checkpoint 为准', group='startup'),
             node('motion', (1235, 465, 500, 165), '--motion',
                  '获得运动模式的试验入口', '等待操作员输入 s 并按Enter', '每次试验都需重新开始', group='startup'),
             node('begin', (650, 785, 500, 150), 'Runner开始一条试验',
                  '影子自动开始；运动由s触发', '运动接合设备；每条试验重置策略', '启动异步记录及本次推理', group='loop'),
             node('observe', (650, 1010, 500, 145), 'Env.get_obs：七路历史对齐',
                  '三路RGB + 双臂实际FK + 双手关节', '取有效主相机锚点t0；检查覆盖与时效', '通常查询 t0−33.3 ms 与 t0', group='loop'),
             node('predict', (650, 1225, 500, 145), 'Runner：预处理 + GPU推理',
                  'Policy.predict_action', '当前配置返回8步 × 52维动作', '实际观测与动作长度从checkpoint读取', group='loop'),
             node('schedule', (650, 1440, 500, 155), 'Env.exec_actions',
                  '保留 t0+i/30；丢弃过期动作', '影子：只记录 shadow_schedule', '运动：校验目标并更新Runtime队列', group='loop'),
             node('control', (1235, 1440, 500, 155), 'Runtime独立推进目标',
                  '双臂200 Hz；左右手各120 Hz', '按时刻选目标，再伺服 / 限速提交', '相机与实际反馈持续更新历史', group='loop'),
             node('stop', (65, 1010, 500, 240), '结束本次试验 / 请求暂停',
                  '人工p / y / n；超时；故障', '清空未来队列；迟到预测作废', '双臂请求保持', '健康手维持最后已接受的目标', group='loop'),
             node('save', (65, 1360, 500, 235), '记录结果与汇总',
                  'trial.json + 三路视频 + 事件与时序', 'summary.json：所有已开始运动试验', '成功率 = 成功次数 / 运动试验次数',
                  'q / Ctrl+C：退出清理并关闭驱动', group='loop'),
         ], edges=[
             edge('load', 'only', ('left', 'top'), [(315, 300)], label='预检查模式'),
             edge('load', 'connect'),
             edge('connect', 'motion', ('right', 'left'), label='运动'),
             edge('connect', 'begin', label='影子自动开始'),
             edge('motion', 'begin', ('bottom', 'right'), [(1485, 860)], label='手动开始'),
             edge('begin', 'observe'), edge('observe', 'predict'), edge('predict', 'schedule'),
             edge('schedule', 'control', ('right', 'left'), label='运动'),
             edge('control', 'observe', ('right', 'right'), [(1780, 1517), (1780, 1082)], label='实际反馈', dashed=True),
             edge('schedule', 'observe', ('left', 'left'), [(610, 1517), (610, 1082)], dashed=True),
             edge('observe', 'stop', ('left', 'right'), [(595, 1082), (595, 1130)], label='失效'),
             edge('stop', 'save'),
         ], notes=[
             '“30 Hz”是训练和目标时间网格；重规划锚点至少前进66.7 ms，实际推理频率受耗时与观测可用性影响。',
             '人工结束、超时和故障可终止试验；SDK接受不等于物理到位；q退出不会自动回位。']),
]

if __name__ == '__main__':
    for figure in FIGURES:
        draw.render(figure)
    print(f'生成 {len(FIGURES)} 张PNG及对应Mermaid源文件：{ROOT}')
