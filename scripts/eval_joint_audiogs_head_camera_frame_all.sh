#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

SCENES="${SCENES:-scene1_opera scene7_playing_300}"

run_scene() {
  local scene="$1"
  case "${scene}" in
    scene1_opera)
      "${SCRIPT_DIR}/eval_joint_audiogs_head_camera_frame_scene1_opera.sh"
      ;;
    scene7_playing_300)
      "${SCRIPT_DIR}/eval_joint_audiogs_head_camera_frame_scene7_playing_300.sh"
      ;;
    *)
      echo "unknown scene: ${scene}" >&2
      echo "supported scenes: scene1_opera scene7_playing_300" >&2
      exit 1
      ;;
  esac
}

for scene in ${SCENES}; do
  echo "==> eval camera-frame ${scene}"
  run_scene "${scene}"
done
