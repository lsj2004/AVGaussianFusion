#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
FTGSPP_ROOT="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
CONFIG="${ROOT}/configs/scene7_playing_300_b_joint_av_audiogs_head_camera_frame.yaml"
RUN_DIR="${ROOT}/runs/scene7_playing_300_b_joint_av_audiogs_head_camera_frame"
MANIFEST="${ROOT}/runs/scene7_playing_300_a/scene_manifest.json"
FTGSPP_CHECKPOINT="${ROOT}/runs/scene7_playing_300_a/ftgspp/Scene7playing/00/gaussians.pt"

mkdir -p "${RUN_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "missing camera-frame AudioGS-head config: ${CONFIG}" >&2
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

EXTRA_ARGS=()
if [[ -n "${JOINT_STEPS_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--joint-steps "${JOINT_STEPS_OVERRIDE}")
  echo "AudioGS camera-frame Route B scene7_playing_300 joint_steps=${JOINT_STEPS_OVERRIDE} (override)"
else
  echo "AudioGS camera-frame Route B scene7_playing_300 using config joint_steps"
fi

cd "${FTGSPP_ROOT}"
PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}" \
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}" \
uv run --no-sync --with soundfile --with pyyaml --with scipy python -m avfusion.train.train_joint_av_gaussians \
  --config "${CONFIG}" \
  --audio-window-seconds 3.0 \
  "${EXTRA_ARGS[@]}"
