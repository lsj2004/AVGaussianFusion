#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FTGSPP_ROOT="$(cd "${ROOT}/../FreeTimeGSPlusPlus" && pwd)"
RUN_DIR="${ROOT}/runs/scene1_opera_c_soft_av_gaussians"
MANIFEST="${ROOT}/runs/scene1_opera_a/scene_manifest.json"
CHECKPOINT="${RUN_DIR}/soft_av.pt"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing Route C soft checkpoint: ${CHECKPOINT}" >&2
  echo "run scripts/train_soft_av_scene1_opera.sh first" >&2
  exit 1
fi

PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with pyyaml --with soundfile python -m avfusion.eval.eval_soft_audio_fulltrack \
  --manifest "${MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_DIR}/eval_fulltrack_audiogs_3s_nonoverlap" \
  --protocol audiogs_3s_nonoverlap_fulltrack \
  --audio-window-seconds 3.0

PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with pyyaml --with soundfile python -m avfusion.eval.eval_soft_audio_fulltrack \
  --manifest "${MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${RUN_DIR}/eval_fulltrack_visual_center_0p5s_ola" \
  --protocol visual_center_overlap_add_fulltrack \
  --audio-window-seconds 0.5
