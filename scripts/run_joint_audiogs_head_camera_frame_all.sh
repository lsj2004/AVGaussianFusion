#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
MODE="${MODE:-train_eval}"

case "${MODE}" in
  train)
    "${SCRIPT_DIR}/train_joint_audiogs_head_camera_frame_all.sh"
    ;;
  eval)
    "${SCRIPT_DIR}/eval_joint_audiogs_head_camera_frame_all.sh"
    ;;
  train_eval)
    "${SCRIPT_DIR}/train_joint_audiogs_head_camera_frame_all.sh"
    "${SCRIPT_DIR}/eval_joint_audiogs_head_camera_frame_all.sh"
    ;;
  *)
    echo "unknown MODE: ${MODE}" >&2
    echo "supported MODE values: train eval train_eval" >&2
    exit 1
    ;;
esac
