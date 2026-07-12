#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FTGSPP_ROOT="$(cd "${ROOT}/../FreeTimeGSPlusPlus" && pwd)"
CONFIG="${ROOT}/configs/scene1_opera_c_soft_av_gaussians_stereo_reg.yaml"

PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
uv run --no-sync --with soundfile --with pyyaml --with scipy python -m avfusion.train.train_soft_av_gaussians \
  --config "${CONFIG}"
