#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene7_playing_300_a"
mkdir -p "${RUN_DIR}"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run python -m avfusion.data.build_scene_manifest \
  --scene-id Scene7playing \
  --visual-root /mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_dynerf/Scene7playing \
  --audio-root /mnt/sda/lisujing/Dataset/Sampled_data/v5_0630_audiogs_audio/Scene7playing \
  --heldout-camera cam10 \
  --num-frames 301 \
  --output "${RUN_DIR}/scene_manifest.json"
