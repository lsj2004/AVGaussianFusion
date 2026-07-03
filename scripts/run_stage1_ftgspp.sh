#!/usr/bin/env bash
set -euo pipefail

FTGSPP="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
OUT="/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/ftgspp"

cd "${FTGSPP}"
uv run --no-sync ./run dynerf "${ROOT}/configs/ftgspp_scene1_opera" "${OUT}" \
  --scenes scene1_opera \
  --from extract \
  --to eval
