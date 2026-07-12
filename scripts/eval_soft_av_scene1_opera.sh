#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FTGSPP_ROOT="$(cd "${ROOT}/../FreeTimeGSPlusPlus" && pwd)"
RUN_DIR="${ROOT}/runs/scene1_opera_c_soft_av_gaussians"
MANIFEST="${ROOT}/runs/scene1_opera_a/scene_manifest.json"
CHECKPOINT="${RUN_DIR}/soft_av.pt"
EVAL_DIR="${RUN_DIR}/eval"

mkdir -p "${EVAL_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing Route C soft checkpoint: ${CHECKPOINT}" >&2
  echo "run scripts/train_soft_av_scene1_opera.sh first" >&2
  exit 1
fi

PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with pyyaml --with soundfile python -m avfusion.eval.eval_soft_audio \
  --manifest "${MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_DIR}" \
  --skip-padding-windows
