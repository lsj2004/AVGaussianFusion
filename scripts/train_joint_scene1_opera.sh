#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
CONFIG="${ROOT}/configs/scene1_opera_b_joint_av.yaml"
RUN_DIR="${ROOT}/runs/scene1_opera_b_joint_av"
MANIFEST="${ROOT}/runs/scene1_opera_a/scene_manifest.json"
FTGSPP_CHECKPOINT="${ROOT}/runs/scene1_opera_a/ftgspp/scene1_opera/00/gaussians.pt"
OUTPUT="${RUN_DIR}/joint_initialized.pt"

cd "${ROOT}"
mkdir -p "${RUN_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing Route B config: ${CONFIG}" >&2
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

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run python -m avfusion.train.train_joint_av_gaussians \
  --manifest "${MANIFEST}" \
  --ftgspp-checkpoint "${FTGSPP_CHECKPOINT}" \
  --output "${OUTPUT}" \
  --warmup-steps 1000 \
  --joint-steps 1000 \
  --top-k 8192 \
  --audio-lr 0.0005 \
  --shared-lr 0.00001
