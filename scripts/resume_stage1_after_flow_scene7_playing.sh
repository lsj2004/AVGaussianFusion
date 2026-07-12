#!/usr/bin/env bash
set -euo pipefail

FTGSPP="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
OUT="${ROOT}/runs/scene7_playing_a/ftgspp"

cd "${FTGSPP}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync ./run dynerf "${ROOT}/configs/ftgspp_scene7_playing" "${OUT}" \
  --scenes Scene7playing \
  --from init \
  --to eval
