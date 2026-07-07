#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
FTGSPP_ROOT="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
CONFIG="${ROOT}/configs/scene7_playing_300_b_joint_av_audiogs_head_strict.yaml"
RUN_DIR="${ROOT}/runs/scene7_playing_300_b_joint_av_audiogs_head_strict"
MANIFEST="${ROOT}/runs/scene7_playing_300_a/scene_manifest.json"
FTGSPP_CHECKPOINT="${ROOT}/runs/scene7_playing_300_a/ftgspp/Scene7playing/00/gaussians.pt"

mkdir -p "${RUN_DIR}"

valid_joint_steps() {
  cd "${FTGSPP_ROOT}"
  PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
  uv run --no-sync --with soundfile --with scipy python -m avfusion.eval.fair_baseline_comparison \
    --print-valid-joint-steps \
    --manifest "${MANIFEST}" \
    --audio-window-seconds 3.0
}

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing AudioGS-head config: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "missing Route A manifest: ${MANIFEST}" >&2
  exit 1
fi

if [[ ! -f "${FTGSPP_CHECKPOINT}" ]]; then
  echo "missing Route A FTGS++ checkpoint: ${FTGSPP_CHECKPOINT}" >&2
  exit 1
fi

JOINT_STEPS="$(valid_joint_steps)"
echo "AudioGS-like head Route B scene7_playing_300 joint_steps=${JOINT_STEPS}"

cd "${FTGSPP_ROOT}"
PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with soundfile --with pyyaml --with scipy python -m avfusion.train.train_joint_av_gaussians \
  --config "${CONFIG}" \
  --joint-steps "${JOINT_STEPS}" \
  --audio-window-seconds 3.0
