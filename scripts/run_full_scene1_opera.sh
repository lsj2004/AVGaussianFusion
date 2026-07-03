#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/sda/lisujing/Dataset/AVGaussianFusion"
RUN_DIR="${ROOT}/runs/scene1_opera_a"
LOG_DIR="${RUN_DIR}/logs"
FTGS_CKPT="${RUN_DIR}/ftgspp/scene1_opera/00/gaussians.pt"

mkdir -p "${LOG_DIR}"

cd "${ROOT}"

echo "[1/5] prepare manifest"
scripts/prepare_scene1_opera.sh 2>&1 | tee "${LOG_DIR}/01_prepare.log"

echo "[2/5] train and evaluate FreeTimeGS++ visual carrier"
scripts/run_stage1_ftgspp.sh 2>&1 | tee "${LOG_DIR}/02_stage1_ftgspp.log"

if [[ ! -f "${FTGS_CKPT}" ]]; then
  echo "missing FTGS++ checkpoint: ${FTGS_CKPT}" >&2
  exit 1
fi

echo "[3/5] export frozen visual carrier"
scripts/export_stage1_carrier.sh "${FTGS_CKPT}" 2>&1 | tee "${LOG_DIR}/03_export_carrier.log"

echo "[4/5] train AudioGS-style acoustic parameters"
scripts/train_stage2_audio.sh 2>&1 | tee "${LOG_DIR}/04_stage2_audio.log"

echo "[5/5] evaluate held-out audio reconstruction"
scripts/eval_scene1_opera.sh 2>&1 | tee "${LOG_DIR}/05_eval_audio.log"

echo "visual metrics: ${RUN_DIR}/ftgspp/scene1_opera/00/metrics.json"
echo "audio metrics: ${RUN_DIR}/eval/audio_summary.json"
