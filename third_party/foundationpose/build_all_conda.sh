#!/usr/bin/env bash
# Build native extensions for a local conda environment.
# BundleSDF CUDA ops (bundlesdf/mycuda) are omitted; model-free NeRF path needs them separately.
set -euo pipefail

PROJ_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

: "${CONDA_PREFIX:?Set CONDA_PREFIX by activating your conda env first.}"

export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH:-}"

cmake -S "${PROJ_ROOT}/mycpp" -B "${PROJ_ROOT}/mycpp/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${PROJ_ROOT}/mycpp/native" \
  -DPython3_ROOT_DIR="${CONDA_PREFIX}" \
  -DPYBIND11_PYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python"
cmake --build "${PROJ_ROOT}/mycpp/build" -j"${MAX_JOBS:-4}"

cd "${PROJ_ROOT}"
