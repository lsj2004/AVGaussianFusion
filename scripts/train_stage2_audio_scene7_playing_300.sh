#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene7_playing_300_a"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run python -m avfusion.train.train_frozen_carrier_audio \
  --manifest "${RUN_DIR}/scene_manifest.json" \
  --carrier "${RUN_DIR}/carrier.pt" \
  --output "${RUN_DIR}/stage2_audio.pt" \
  --steps 1000 \
  --lr 0.0005 \
  --top-k 8192
