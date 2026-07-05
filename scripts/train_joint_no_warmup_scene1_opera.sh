#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
FTGSPP_ROOT="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
CONFIG="${ROOT}/configs/scene1_opera_b_joint_av_no_warmup.yaml"
RUN_DIR="${ROOT}/runs/scene1_opera_b_joint_av_no_warmup"
MANIFEST="${ROOT}/runs/scene1_opera_a/scene_manifest.json"
FTGSPP_CHECKPOINT="${ROOT}/runs/scene1_opera_a/ftgspp/scene1_opera/00/gaussians.pt"

mkdir -p "${RUN_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing Route B no-warmup config: ${CONFIG}" >&2
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

cd "${FTGSPP_ROOT}"
PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with soundfile --with pyyaml python -m avfusion.train.train_joint_av_gaussians \
  --config "${CONFIG}"
