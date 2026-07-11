#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
FTGSPP_ROOT="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
RUN_DIR="${ROOT}/runs/scene1_opera_b_joint_av_audiogs_head_start"
MANIFEST="${ROOT}/runs/scene1_opera_a/scene_manifest.json"
CHECKPOINT="${RUN_DIR}/joint_finetune.pt"
EVAL_DIR="${RUN_DIR}/eval"

cd "${FTGSPP_ROOT}"
mkdir -p "${EVAL_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing start-anchored AudioGS-head checkpoint: ${CHECKPOINT}" >&2
  echo "run scripts/train_joint_audiogs_head_start_scene1_opera.sh first" >&2
  exit 1
fi

PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with pyyaml --with soundfile --with scipy python -m avfusion.eval.eval_joint_audio \
  --manifest "${MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_DIR}" \
  --audio-eval-protocol audiogs_start_nonoverlap \
  --include-dpam

PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with pyyaml --with soundfile --with scipy python -m avfusion.eval.eval_joint_visual \
  --manifest "${MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_DIR}"
