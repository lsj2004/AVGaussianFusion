#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene1_opera_a"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run python -m avfusion.eval.eval_audio \
  --manifest "${RUN_DIR}/scene_manifest.json" \
  --checkpoint "${RUN_DIR}/stage2_audio.pt" \
  --output-dir "${RUN_DIR}/eval"
