#!/usr/bin/env bash
set -euo pipefail

FTGSPP="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
OUT="/mnt/sda/lisujing/Dataset/AVGaussianFusion/runs/scene1_opera_a/ftgspp"

cd "${FTGSPP}"
uv run --no-sync ./run dynerf configs/dynerf/ftgspp "${OUT}" \
  --scenes scene1_opera \
  --from extract \
  --to render
