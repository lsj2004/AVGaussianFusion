#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
RUN_DIR="${ROOT}/runs/scene1_opera_b_joint_av"
CHECKPOINT="${RUN_DIR}/joint_initialized.pt"
EVAL_DIR="${ROOT}/runs/scene1_opera_b_joint_av/eval"
STATUS_FILE="${EVAL_DIR}/route_b_eval_status.txt"

cd "${ROOT}"
mkdir -p "${EVAL_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing Route B joint checkpoint: ${CHECKPOINT}" >&2
  echo "run scripts/train_joint_scene1_opera.sh first" >&2
  exit 1
fi

MESSAGE="Route B checkpoint schema is not yet compatible with avfusion.eval.eval_audio; this smoke/status placeholder writes no metrics. Add a Route B joint AV evaluator before reporting metrics."
{
  printf '%s\n' "${MESSAGE}"
  printf 'metrics=none\n'
  printf 'checkpoint=%s\n' "${CHECKPOINT}"
} > "${STATUS_FILE}"

echo "${MESSAGE}"
echo "wrote smoke eval status: ${STATUS_FILE}"
