#!/usr/bin/env bash
set -euo pipefail

FTGSPP="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
OUT="${ROOT}/runs/scene1_opera_a/ftgspp"

cd "${FTGSPP}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync ./run dynerf "${ROOT}/configs/ftgspp_scene1_opera" "${OUT}" \
  --scenes scene1_opera \
  --from init \
  --to eval
