#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
FTGSPP_ROOT="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
RUN_DIR="${ROOT}/runs/scene1_opera_b_joint_av_spectral_no_warmup_audiogs_strict"
MANIFEST="${ROOT}/runs/scene1_opera_a/scene_manifest.json"
CHECKPOINT="${RUN_DIR}/joint_finetune.pt"
EVAL_DIR="${RUN_DIR}/eval"

cd "${FTGSPP_ROOT}"
mkdir -p "${EVAL_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing strict AudioGS checkpoint: ${CHECKPOINT}" >&2
  echo "run scripts/train_joint_spectral_no_warmup_scene1_opera_audiogs_strict.sh first" >&2
  exit 1
fi

PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with pyyaml --with soundfile --with scipy python -m avfusion.eval.eval_joint_audio \
  --manifest "${MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_DIR}" \
  --include-dpam

PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with pyyaml --with soundfile --with scipy python -m avfusion.eval.eval_joint_visual \
  --manifest "${MANIFEST}" \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${EVAL_DIR}"
