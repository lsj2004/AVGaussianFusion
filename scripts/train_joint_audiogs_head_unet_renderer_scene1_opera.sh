#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
FTGSPP_ROOT="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
CONFIG="${ROOT}/configs/scene1_opera_b_joint_av_audiogs_head_unet_renderer.yaml"
RUN_DIR="${ROOT}/runs/scene1_opera_b_joint_av_audiogs_head_unet_renderer"

mkdir -p "${RUN_DIR}"

EXTRA_ARGS=()
if [[ -n "${JOINT_STEPS_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--joint-steps "${JOINT_STEPS_OVERRIDE}")
  echo "AudioGS U-Net renderer Route B scene1_opera joint_steps=${JOINT_STEPS_OVERRIDE} (override)"
else
  echo "AudioGS U-Net renderer Route B scene1_opera using config joint_steps"
fi

cd "${FTGSPP_ROOT}"
PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with soundfile --with pyyaml --with scipy python -m avfusion.train.train_joint_av_gaussians \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"
