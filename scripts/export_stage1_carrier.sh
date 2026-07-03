#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CHECKPOINT" >&2
  exit 2
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" uv run python -m avfusion.visual.export_ftgspp_carrier \
  --checkpoint "$1" \
  --output runs/scene1_opera_a/carrier.pt
