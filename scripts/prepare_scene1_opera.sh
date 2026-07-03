#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene1_opera_a"
mkdir -p "${RUN_DIR}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run python -m avfusion.data.build_scene_manifest \
  --scene-id scene1_opera \
  --visual-root /mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/scene1_opera \
  --audio-root /mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/scene1_opera \
  --heldout-camera cam10 \
  --output "${RUN_DIR}/scene_manifest.json"
