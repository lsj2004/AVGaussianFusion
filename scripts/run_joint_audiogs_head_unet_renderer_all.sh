#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SCENES="${SCENES:-scene1_opera scene7_playing_300}"
MODE="${MODE:-train_eval}"

run_scene() {
  local scene="$1"
  local action="$2"
  case "${scene}:${action}" in
    scene1_opera:train)
      "${SCRIPT_DIR}/train_joint_audiogs_head_unet_renderer_scene1_opera.sh"
      ;;
    scene1_opera:eval)
      "${SCRIPT_DIR}/eval_joint_audiogs_head_unet_renderer_scene1_opera.sh"
      ;;
    scene7_playing_300:train)
      "${SCRIPT_DIR}/train_joint_audiogs_head_unet_renderer_scene7_playing_300.sh"
      ;;
    scene7_playing_300:eval)
      "${SCRIPT_DIR}/eval_joint_audiogs_head_unet_renderer_scene7_playing_300.sh"
      ;;
    *)
      echo "unknown scene/action: ${scene}:${action}" >&2
      exit 1
      ;;
  esac
}

for scene in ${SCENES}; do
  case "${MODE}" in
    train)
      run_scene "${scene}" train
      ;;
    eval)
      run_scene "${scene}" eval
      ;;
    train_eval)
      run_scene "${scene}" train
      run_scene "${scene}" eval
      ;;
    *)
      echo "MODE must be train, eval, or train_eval; got ${MODE}" >&2
      exit 1
      ;;
  esac
done
