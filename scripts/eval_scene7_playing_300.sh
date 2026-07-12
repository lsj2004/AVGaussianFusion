#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene7_playing_300_a"

PYTHONPATH="${ROOT}:${PYTHONPATH:-}" \
conda run -n avcloud python -m avfusion.eval.eval_audio \
  --manifest "${RUN_DIR}/scene_manifest.json" \
  --checkpoint "${RUN_DIR}/stage2_audio.pt" \
  --output-dir "${RUN_DIR}/eval"
