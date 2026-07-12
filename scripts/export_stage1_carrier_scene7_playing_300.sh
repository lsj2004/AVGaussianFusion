#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/gaussians.pt" >&2
  exit 2
fi

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
FTGSPP_ROOT="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
RUN_DIR="${ROOT}/runs/scene7_playing_300_a"

cd "${FTGSPP_ROOT}"
PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync python -m avfusion.visual.export_ftgspp_carrier \
  --checkpoint "$1" \
  --output "${RUN_DIR}/carrier.pt"
