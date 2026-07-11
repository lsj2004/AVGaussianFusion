#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

"${SCRIPT_DIR}/train_joint_audiogs_head_start_all.sh"
"${SCRIPT_DIR}/eval_joint_audiogs_head_start_all.sh"
