# FoundationPose 与 DP 共用环境

按用户要求，仅使用现有 `tianji-rail-diffusion-policy` Conda 环境。此前自行建立的独立 `foundationpose` 路线已停止，自写安装脚本已删除。

## 官方依据与来源

- https://github.com/NVlabs/FoundationPose#env-setup-option-2-conda-local
- https://github.com/NVlabs/FoundationPose/blob/main/environment.yml
- https://github.com/NVlabs/FoundationPose/blob/main/requirements.txt
- https://github.com/NVlabs/FoundationPose/blob/main/build_all_conda.sh
- 官方 HEAD：`a1b694b83e633c2cb6115b9063d940a687759392`。
- 原始官方仓库与安装前记录：`workspace/rail_pose_insert/outputs/foundationpose_dp_setup_20261024/`，不覆盖项目中已经接入的导轨代码。

## 安装原则

1. 保留 DP 已安装的 Python 3.10.22、PyTorch 2.7.1 / CUDA 12.9、torchvision 0.22.1、NumPy 1.26.4、SciPy 1.15.3、Pandas 1.5.3 等核心版本。
2. 从 conda-forge 补齐官方列出的 CMake、Ninja、Eigen、Boost、pybind11 和 C++ 编译器，并准备与当前 PyTorch 匹配的 CUDA 12.9 Toolkit。
3. 按官方步骤从 PyTorch3D 与 NVlabs/NVDiffRast 的 GitHub 源码，在 DP 环境内使用 `--no-build-isolation` 构建；按官方依赖清单安装缺少的 Python 包，并运行官方 `build_all_conda.sh` 构建 `mycpp`。
4. 官方当前模板默认 Python 3.11，要求 NumPy ≥ 2；这与现有 DP 固定版本不同。本次保留现有 Python 和 NumPy，明确记录这两项兼容偏离，不宣称原样复现官方环境。实际已安装的 OmegaConf 2.4.0 满足官方 ≥ 2.3 的要求，不另行替换。
5. 接料和预插入配置统一指向 DP 环境。完成官方导入、真实录制 RGB-D 推理和 DP 回归检查后，清理本轮误建的独立环境。所有验证均不执行真机动作。

## 本机执行记录

已下载官方 FoundationPose、PyTorch3D 和 nvdiffrast 仓库，未覆盖本项目已有的导轨适配代码。源码提交：

- FoundationPose：`a1b694b83e633c2cb6115b9063d940a687759392`
- PyTorch3D：`204fe5831b3a5bbe079f09d72afb4515178f8892`
- nvdiffrast：`253ac4fcea7de5f396371124af597e6cc957bfae`

安装前保存了 `conda-before.json`、`pip-before.txt` 和全部现有包的约束文件；`pip check` 安装前无依赖冲突。Conda 和 pip 都先检查安装计划，确认不替换 DP 核心版本，再执行安装。

Conda 补齐命令：

```bash
conda install -y -n tianji-rail-diffusion-policy --override-channels \
  -c conda-forge --freeze-installed cmake ninja eigen boost-cpp \
  pybind11 cxx-compiler cuda-nvcc=12.9 cuda-libraries-dev=12.9 \
  cuda-cccl=12.9 tbb=2021.13.0
```

Python 包来自官方 `requirements.txt`，保留 DP 核心包约束；兼容后的直接依赖记录为 `workspace/rail_pose_insert/requirements-foundationpose.txt`，新增依赖也已写入 `conda_environment_tianji.yaml`。

GPU 扩展使用上述两个官方源码检出目录，在 DP 环境中执行 `python -m pip install --no-build-isolation`；未复用此前 Python 3.11 环境的二进制。重建时可按官方 GitHub 安装形式执行：

```bash
conda activate tianji-rail-diffusion-policy
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export CUB_HOME="$CONDA_PREFIX/targets/x86_64-linux/include"
export MAX_JOBS=4
# 本机 RTX 3070 Ti 为 8.6；更换 GPU 时重新获取架构。
export TORCH_CUDA_ARCH_LIST=8.6
python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/pytorch3d.git@204fe5831b3a5bbe079f09d72afb4515178f8892'
python -m pip install --no-build-isolation \
  'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'
```

`mycpp` 实际运行下载的官方 `build_all_conda.sh`，生成 CPython 3.10 扩展后放入本项目适配器读取的 `third_party/foundationpose/mycpp/native/`。已核对两份 `mycpp` 的 C++ 源码一致。重新构建项目内相同源码时，可执行 `bash third_party/foundationpose/build_all_conda.sh`，其差异仅为保留构建目录、限制并行数并把输出写到 `native/`。

安装日志、解析计划、原始官方源码和后续验证产物位于 `workspace/rail_pose_insert/outputs/foundationpose_dp_setup_20261024/`；该目录被 Git 忽略。不要执行旧的 `third_party/foundationpose/workspace/scripts/install.sh`，它会按旧快照建立独立环境。

## 验证记录

- 补齐 Python 依赖后 `pip check`：无冲突。Python 3.10.22、PyTorch 2.7.1 / CUDA 12.9、torchvision 0.22.1、NumPy 1.26.4、SciPy 1.15.3、Pandas 1.5.3、Hydra 1.2.0、OmegaConf 2.4.0 均保持安装前版本。
- DP 回归：`python -m unittest tests.test_tianji_rail tests.test_tianji_rail_pose tests.test_tianji_rail_training -v`，17/17 通过，包含真实 CPU 训练、EMA、保存与重载检查点、策略推理；无设备连接。
- 导轨接料及预插入测试：46 项中 44 项通过，包含新增接料测试 15 项。两项失败位于 `test_standalone.py` 的搬迁隔离检查，原因是现有 editable 安装向测试暴露原项目源码路径；没有为通过环境检查而修改这些测试或设备代码。详见 `rail-tests.log`。
- 官方来源扩展全部构建并加载成功：PyTorch3D 0.7.9、nvdiffrast 0.4.0、CPython 3.10 的 `mycpp`。CUDA KNN、CUDA 光栅渲染以及 Warp 1.18.0 的实际 GPU 内核编译和运行通过。
- 已用 `outputs/teach_target_20261024_152106/scene` 中的真实录制 RGB-D 验证一次初始化及四次跟踪，5/5 有效；初始化 26.23 s，跟踪 47.92–74.82 ms。叠图已查看。该记录明确标为 `replay_only: true`，不作为现场自动抓握输入，也不代替绝对位姿精度验收。
- 同一 DP 环境下 `receive.py check` 返回 `ready: true`、`problems: []`；`run.py check` 返回 `ready_for_live_configuration: true`、`problems: []`。最终 `pip check` 无冲突。完整结果为输出目录中的 `verification.json`。
- 全部验证没有连接机器人或灵巧手，没有执行真机动作；在线相机时序、现场抓握和点动仍待实机验收。
- DP 环境推理与预检通过后，已通过 `conda env remove -n foundationpose -y` 删除本轮误建的独立环境；接料运行无需该环境。
