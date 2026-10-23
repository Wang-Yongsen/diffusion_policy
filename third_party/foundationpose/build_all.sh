#!/usr/bin/env bash
# Build this local snapshot without requiring a separate /kaolin checkout.
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$task_repo/build_all_conda.sh"
