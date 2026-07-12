#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
FTGSPP_ROOT="/mnt/sda/lisujing/Dataset/FreeTimeGSPlusPlus"
AUDIOGS_ROOT="/mnt/sda/lisujing/Dataset/audioGS-replay"
OUT_DIR="${ROOT}/runs/fair_baseline_comparison"
STRICT_AUDIOGS_SUMMARY="${AUDIOGS_ROOT}/runs/strict_audiogs_cam10_allcams/strict_audiogs_cam10_allcams_summary.json"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/avgaussianfusion-uv-cache}"
export PYTHONPATH="${ROOT}:${FTGSPP_ROOT}:${PYTHONPATH:-}"

valid_joint_steps() {
  local manifest="$1"
  local window_seconds="$2"
  cd "${FTGSPP_ROOT}"
  uv run --no-sync --with soundfile python -m avfusion.eval.fair_baseline_comparison \
    --print-valid-joint-steps \
    --manifest "${manifest}" \
    --audio-window-seconds "${window_seconds}"
}

summary_matches_steps() {
  local summary="$1"
  local expected="$2"
  python3 -c 'import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
if not p.exists():
    raise SystemExit(1)
data = json.loads(p.read_text())
raise SystemExit(0 if int(data.get("joint_steps", -1)) == expected else 1)
' "${summary}" "${expected}"
}

ensure_strict_audiogs() {
  if [[ -f "${STRICT_AUDIOGS_SUMMARY}" ]]; then
    echo "[fair] reuse strict AudioGS summary: ${STRICT_AUDIOGS_SUMMARY}"
    return
  fi
  echo "[fair] missing strict AudioGS all-cams summary; running AudioGS source-code baseline"
  cd "${AUDIOGS_ROOT}"
  bash scripts/prepare_strict_audiogs_cam10_avgaussianfusion.sh
  bash scripts/run_strict_audiogs_cam10_avgaussianfusion.sh
  bash scripts/eval_strict_audiogs_cam10_avgaussianfusion.sh
}

ensure_route_a_eval() {
  local dataset="$1"
  local audio_summary="$2"
  local eval_script="$3"
  if [[ -f "${audio_summary}" ]]; then
    echo "[fair] reuse Route A eval for ${dataset}: ${audio_summary}"
    return
  fi
  echo "[fair] missing Route A eval for ${dataset}; running ${eval_script}"
  cd "${ROOT}"
  bash "${eval_script}"
}

ensure_ftgspp_eval() {
  local dataset="$1"
  local config_dir="$2"
  local scene="$3"
  local out_dir="$4"
  if [[ -f "${out_dir}/summary.json" ]]; then
    echo "[fair] reuse FreeTimeGS++ visual eval for ${dataset}: ${out_dir}/summary.json"
    return
  fi
  echo "[fair] missing FreeTimeGS++ visual eval for ${dataset}; running FTGS++ eval"
  cd "${FTGSPP_ROOT}"
  uv run --no-sync ./run dynerf "${config_dir}" "${out_dir}" \
    --scenes "${scene}" \
    --from eval \
    --to eval
}

train_joint_if_needed() {
  local label="$1"
  local config="$2"
  local manifest="$3"
  local checkpoint="$4"
  local run_dir="$5"
  local window_seconds="$6"
  local steps
  steps="$(valid_joint_steps "${manifest}" "${window_seconds}")"
  echo "[fair] ${label}: valid_aligned_samples=${steps}, checkpoint=${checkpoint}"
  mkdir -p "${run_dir}"
  if [[ "${FORCE_FAIR:-0}" != "1" && -f "${checkpoint}" ]] \
    && summary_matches_steps "${run_dir}/train_summary.json" "${steps}"; then
    echo "[fair] reuse ${label}; train_summary joint_steps already ${steps}"
    return
  fi
  cd "${FTGSPP_ROOT}"
  uv run --no-sync --with soundfile --with pyyaml python -m avfusion.train.train_joint_av_gaussians \
    --config "${config}" \
    --joint-steps "${steps}" \
    --audio-window-seconds "${window_seconds}"
}

eval_joint_if_needed() {
  local label="$1"
  local manifest="$2"
  local checkpoint="$3"
  local eval_dir="$4"
  if [[ "${FORCE_EVAL:-0}" != "1" \
    && -f "${eval_dir}/audio_summary.json" \
    && -f "${eval_dir}/visual_summary.json" ]]; then
    echo "[fair] reuse eval for ${label}: ${eval_dir}"
    return
  fi
  mkdir -p "${eval_dir}"
  cd "${FTGSPP_ROOT}"
  uv run --no-sync --with pyyaml --with soundfile python -m avfusion.eval.eval_joint_audio \
    --manifest "${manifest}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${eval_dir}" \
    --include-dpam
  uv run --no-sync --with pyyaml --with soundfile python -m avfusion.eval.eval_joint_visual \
    --manifest "${manifest}" \
    --checkpoint "${checkpoint}" \
    --output-dir "${eval_dir}"
}

run_dataset() {
  local dataset="$1"
  local route_a_run="$2"
  local warm_config="$3"
  local nowarm_config="$4"
  local warm_run="$5"
  local nowarm_run="$6"
  local spectral_nowarm_config="$7"
  local spectral_nowarm_run="$8"
  local route_a_eval_script="$9"
  local ftgspp_config_dir="${10}"
  local ftgspp_scene="${11}"
  local manifest="${ROOT}/runs/${route_a_run}/scene_manifest.json"
  local window_seconds="0.5"

  ensure_ftgspp_eval \
    "${dataset}" \
    "${ftgspp_config_dir}" \
    "${ftgspp_scene}" \
    "${ROOT}/runs/${route_a_run}/ftgspp"

  ensure_route_a_eval \
    "${dataset}" \
    "${ROOT}/runs/${route_a_run}/eval_avcloud/audio_summary.json" \
    "${route_a_eval_script}"

  train_joint_if_needed \
    "${dataset} AVFusion joint warmup" \
    "${ROOT}/configs/${warm_config}" \
    "${manifest}" \
    "${ROOT}/runs/${warm_run}/joint_finetune.pt" \
    "${ROOT}/runs/${warm_run}" \
    "${window_seconds}"
  eval_joint_if_needed \
    "${dataset} AVFusion joint warmup" \
    "${manifest}" \
    "${ROOT}/runs/${warm_run}/joint_finetune.pt" \
    "${ROOT}/runs/${warm_run}/eval"

  train_joint_if_needed \
    "${dataset} AVFusion joint no warmup" \
    "${ROOT}/configs/${nowarm_config}" \
    "${manifest}" \
    "${ROOT}/runs/${nowarm_run}/joint_finetune.pt" \
    "${ROOT}/runs/${nowarm_run}" \
    "${window_seconds}"
  eval_joint_if_needed \
    "${dataset} AVFusion joint no warmup" \
    "${manifest}" \
    "${ROOT}/runs/${nowarm_run}/joint_finetune.pt" \
    "${ROOT}/runs/${nowarm_run}/eval"

  train_joint_if_needed \
    "${dataset} AVFusion spectral audio head no warmup" \
    "${ROOT}/configs/${spectral_nowarm_config}" \
    "${manifest}" \
    "${ROOT}/runs/${spectral_nowarm_run}/joint_finetune.pt" \
    "${ROOT}/runs/${spectral_nowarm_run}" \
    "${window_seconds}"
  eval_joint_if_needed \
    "${dataset} AVFusion spectral audio head no warmup" \
    "${manifest}" \
    "${ROOT}/runs/${spectral_nowarm_run}/joint_finetune.pt" \
    "${ROOT}/runs/${spectral_nowarm_run}/eval"
}

ensure_strict_audiogs

run_dataset \
  scene1_opera \
  scene1_opera_a \
  scene1_opera_b_joint_av_fair.yaml \
  scene1_opera_b_joint_av_no_warmup_fair.yaml \
  scene1_opera_b_joint_av_fair \
  scene1_opera_b_joint_av_no_warmup_fair \
  scene1_opera_b_joint_av_spectral_no_warmup.yaml \
  scene1_opera_b_joint_av_spectral_no_warmup \
  "${ROOT}/scripts/eval_scene1_opera.sh" \
  "${ROOT}/configs/ftgspp_scene1_opera" \
  scene1_opera

run_dataset \
  scene7_playing_300 \
  scene7_playing_300_a \
  scene7_playing_300_b_joint_av_fair.yaml \
  scene7_playing_300_b_joint_av_no_warmup_fair.yaml \
  scene7_playing_300_b_joint_av_fair \
  scene7_playing_300_b_joint_av_no_warmup_fair \
  scene7_playing_300_b_joint_av_spectral_no_warmup.yaml \
  scene7_playing_300_b_joint_av_spectral_no_warmup \
  "${ROOT}/scripts/eval_scene7_playing_300.sh" \
  "${ROOT}/configs/ftgspp_scene7_playing_300" \
  Scene7playing

cd "${FTGSPP_ROOT}"
uv run --no-sync --with soundfile python -m avfusion.eval.fair_baseline_comparison \
  --root "${ROOT}" \
  --output-dir "${OUT_DIR}" \
  --protocol-report \
  --check-artifacts \
  --link-media

echo "[fair] metrics: ${OUT_DIR}/metrics/comparison.md"
echo "[fair] media: ${OUT_DIR}/media"
