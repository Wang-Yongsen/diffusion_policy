#!/usr/bin/env bash
set -euo pipefail
task_workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_repo="$(dirname -- "$task_workspace")"
task_env="${FOUNDATIONPOSE_ENV:-foundationpose}"
task_conda="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$task_conda" ]]; then
  echo "Conda is required. Install Miniforge/Miniconda first." >&2
  exit 1
fi
task_conda_root="$(dirname -- "$(dirname -- "$task_conda")")"
source "$task_conda_root/etc/profile.d/conda.sh"
if ! conda run -n "$task_env" python -c 'import sys; assert sys.version_info[:2] == (3,11)' >/dev/null 2>&1; then
  conda create -y -n "$task_env" --override-channels -c conda-forge -c nvidia \
    python=3.11 pip 'cmake<4' ninja eigen boost-cpp pybind11 \
    gcc_linux-64=13 gxx_linux-64=13 cuda-toolkit=12.8
else
  conda install -y -n "$task_env" --override-channels -c conda-forge -c nvidia \
    python=3.11 pip 'cmake<4' ninja eigen boost-cpp pybind11 \
    gcc_linux-64=13 gxx_linux-64=13 cuda-toolkit=12.8
fi
conda activate "$task_env"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if [[ -f "$task_workspace/requirements-runtime.txt" ]]; then
  python -m pip install -r "$task_workspace/requirements-runtime.txt"
else
  python -m pip install -r "$task_repo/requirements.txt" pytest cadquery pyrealsense2 gdown nvidia-ml-py
fi
export TORCH_CUDA_ARCH_LIST="$(python -c 'import torch; a,b=torch.cuda.get_device_capability(); print(f"{a}.{b}")')"
python -m pip install --no-build-isolation "$task_workspace/vendor/nvdiffrast" "$task_workspace/vendor/pytorch3d"
# Build beside the upstream sources without deleting an existing build.
cmake -S "$task_repo/mycpp" -B "$task_repo/mycpp/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$task_repo/mycpp/native" \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
cmake --build "$task_repo/mycpp/build" -j "$MAX_JOBS"
python "$task_workspace/run.py" check-env
python "$task_workspace/scripts/write_lock.py"
conda env export --no-builds | sed '/^prefix:/d' > "$task_workspace/environment-lock.yml"
echo "Environment ready. Run workspace/run.py from the local desktop for camera/GUI validation."
