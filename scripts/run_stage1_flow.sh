#!/usr/bin/env bash
set -euo pipefail

FTGSPP="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
CONFIG="${ROOT}/configs/ftgspp_scene1_opera/scene1_opera.toml"

cd "${FTGSPP}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync python -m ftgspp.data.flow "${CONFIG}" \
  --device cuda \
  --cameras all
