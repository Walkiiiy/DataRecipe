#!/usr/bin/env bash
set -euo pipefail

# Batch run Recipe Evolution decoupled training + EM eval on:
# - dolly:     epoch1-target-size 130, 650, 1300
# - dialogsum: epoch1-target-size 140, 700, 1000
# - lawyer:    epoch1-target-size 200, 1000, 2000
#
# Usage:
#   bash src/4.3/run_recipe_target_sweep.sh
#
# Optional env vars:
#   PYTHON_BIN=python
#   BASE_MODEL=Qwen/Qwen2.5-0.5B
#   MODEL_SOURCE=modelscope
#   RUN_TAG=target_sweep_YYYYmmdd_HHMMSS
#   CONTINUE_ON_ERROR=0|1
#   NUM_CHUNKS=10
#   TOTAL_EPOCHS=4
#   MAX_SEQ_LENGTH=512
#   TRAIN_BATCH_SIZE=1
#   GRAD_ACC_STEPS=4
#   ANCHOR_SIZE_PER_CAP=2
#   ANCHOR_BATCH_SIZE=2
#   KEEP_THRESHOLD=0.5
#   ALPHA_EMA=0.1
#   BETA_EMA=0.2
#   SCORE_EMA_MOMENTUM=0.02
#   MIN_KEEP_PROB=0.05
#   KEEP_EVERY_N=32
#   MIN_RESPONSE_TOKENS=4
#   MAX_GRAD_DIM=65536
#   SEED=42

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

NUM_CHUNKS="${NUM_CHUNKS:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-4}"
ANCHOR_SIZE_PER_CAP="${ANCHOR_SIZE_PER_CAP:-2}"
ANCHOR_BATCH_SIZE="${ANCHOR_BATCH_SIZE:-2}"
KEEP_THRESHOLD="${KEEP_THRESHOLD:-0.5}"
ALPHA_EMA="${ALPHA_EMA:-0.1}"
BETA_EMA="${BETA_EMA:-0.2}"
SCORE_EMA_MOMENTUM="${SCORE_EMA_MOMENTUM:-0.02}"
MIN_KEEP_PROB="${MIN_KEEP_PROB:-0.05}"
KEEP_EVERY_N="${KEEP_EVERY_N:-32}"
MIN_RESPONSE_TOKENS="${MIN_RESPONSE_TOKENS:-4}"
MAX_GRAD_DIM="${MAX_GRAD_DIM:-65536}"
SEED="${SEED:-42}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${RUN_TAG:-target_sweep_${TIMESTAMP}}"
LOG_ROOT="logs/recipe_target_sweep/${RUN_TAG}"
SUMMARY_CSV="${LOG_ROOT}/summary.csv"

mkdir -p "${LOG_ROOT}"

declare -A TARGETS_MAP=(
  [dolly]="130 650 1300"
  [dialogsum]="140 700 1000"
  [lawyer]="200 1000 2000"
)

declare -A EVAL_SIZE_MAP=(
  [dolly]="1446"
  [dialogsum]="1446"
  [lawyer]="1446"
)

print_cmd() {
  local out=""
  local x
  for x in "$@"; do
    printf -v xq '%q' "${x}"
    out+="${xq} "
  done
  echo "${out}"
}

run_and_log() {
  local logfile="$1"
  shift
  mkdir -p "$(dirname "${logfile}")"
  local cmd_str
  cmd_str="$(print_cmd "$@")"
  {
    echo "======================================================================"
    echo "[$(date '+%F %T')] CMD: ${cmd_str}"
    echo "======================================================================"
  } | tee -a "${logfile}"

  "$@" 2>&1 | tee -a "${logfile}"
}

append_summary() {
  local dataset="$1"
  local target="$2"
  local status="$3"
  local output_dir="$4"
  local final_ckpt="$5"
  local train_log="$6"
  local eval_log="$7"
  echo "${dataset},${target},${status},${output_dir},${final_ckpt},${train_log},${eval_log}" >> "${SUMMARY_CSV}"
}

check_required_files() {
  local dataset="$1"
  local train_jsonl="data/${dataset}/train.jsonl"
  local test_jsonl="data/${dataset}/test.jsonl"
  local topk_jsonl="data/${dataset}/train_coarse_topk5.jsonl"
  local score1="data/${dataset}/score/alpagasus_improved_mapped.jsonl"
  local score2="data/${dataset}/score/delta_improved_mapped.jsonl"
  local score3="data/${dataset}/score/srm_from_topk5_only.jsonl"
  local score4="data/${dataset}/score/pdm_scored.jsonl"

  local f
  for f in "${train_jsonl}" "${test_jsonl}" "${topk_jsonl}" "${score1}" "${score2}" "${score3}" "${score4}"; do
    if [[ ! -f "${f}" ]]; then
      echo "Missing required file: ${f}" >&2
      return 1
    fi
  done
}

echo "dataset,epoch1_target_size,status,output_dir,final_checkpoint,train_log,eval_log" > "${SUMMARY_CSV}"

for dataset in dolly dialogsum lawyer; do
  check_required_files "${dataset}"

  train_jsonl="data/${dataset}/train.jsonl"
  test_jsonl="data/${dataset}/test.jsonl"
  topk_jsonl="data/${dataset}/train_coarse_topk5.jsonl"
  score_alpa="alpagasus_improved::data/${dataset}/score/alpagasus_improved_mapped.jsonl"
  score_delta="delta_improved::data/${dataset}/score/delta_improved_mapped.jsonl"
  score_srm="srm::data/${dataset}/score/srm_from_topk5_only.jsonl"
  score_pdm="pdm::data/${dataset}/score/pdm_scored.jsonl"
  eval_size="${EVAL_SIZE_MAP[${dataset}]}"

  for target_size in ${TARGETS_MAP[${dataset}]}; do
    run_name="re_ep1_${target_size}"
    output_dir="data/${dataset}/exp4.3/${RUN_TAG}/${run_name}"
    train_log="${LOG_ROOT}/${dataset}_${target_size}_train.log"
    eval_log="${LOG_ROOT}/${dataset}_${target_size}_eval.log"
    final_ckpt="${output_dir}/static_stage/run_recipe_selected/final_checkpoint"
    eval_out_dir="${output_dir}/em_eval"

    train_cmd=(
      "${PYTHON_BIN}" "src/4.3/run_orchestrator.py"
      "--train-raw-jsonl" "${train_jsonl}"
      "--anchor-jsonl" "${train_jsonl}"
      "--topk-jsonl" "${topk_jsonl}"
      "--score-path" "${score_alpa}"
      "--score-path" "${score_delta}"
      "--score-path" "${score_srm}"
      "--score-path" "${score_pdm}"
      "--output-dir" "${output_dir}"
      "--base-model" "${BASE_MODEL}"
      "--model-source" "${MODEL_SOURCE}"
      "--num-chunks" "${NUM_CHUNKS}"
      "--total-epochs" "${TOTAL_EPOCHS}"
      "--anchor-size-per-cap" "${ANCHOR_SIZE_PER_CAP}"
      "--anchor-batch-size" "${ANCHOR_BATCH_SIZE}"
      "--max-seq-length" "${MAX_SEQ_LENGTH}"
      "--train-batch-size" "${TRAIN_BATCH_SIZE}"
      "--gradient-accumulation-steps" "${GRAD_ACC_STEPS}"
      "--keep-policy" "threshold"
      "--keep-threshold" "${KEEP_THRESHOLD}"
      "--alpha-ema" "${ALPHA_EMA}"
      "--beta-ema" "${BETA_EMA}"
      "--score-ema-momentum" "${SCORE_EMA_MOMENTUM}"
      "--min-keep-prob" "${MIN_KEEP_PROB}"
      "--keep-every-n" "${KEEP_EVERY_N}"
      "--min-response-tokens" "${MIN_RESPONSE_TOKENS}"
      "--max-grad-dim" "${MAX_GRAD_DIM}"
      "--focus-anchor-by-chunk" "1"
      "--shared-eval-jsonl" "${test_jsonl}"
      "--run-static-stage" "1"
      "--seed" "${SEED}"
      "--epoch1-target-size" "${target_size}"
    )

    if ! run_and_log "${train_log}" "${train_cmd[@]}"; then
      echo "Train failed: dataset=${dataset}, target=${target_size}" | tee -a "${train_log}"
      append_summary "${dataset}" "${target_size}" "train_failed" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        continue
      fi
      exit 1
    fi

    if [[ ! -d "${final_ckpt}" ]]; then
      echo "Final checkpoint missing: ${final_ckpt}" | tee -a "${train_log}"
      append_summary "${dataset}" "${target_size}" "missing_checkpoint" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        continue
      fi
      exit 1
    fi

    eval_cmd=(
      "${PYTHON_BIN}" "src/4.1/EXP/eval_checkpoints_em.py"
      "--eval-source-jsonl" "${test_jsonl}"
      "--eval-size" "${eval_size}"
      "--base_model" "${BASE_MODEL}"
      "--model_source" "${MODEL_SOURCE}"
      "--run" "re::${final_ckpt}"
      "--output-dir" "${eval_out_dir}"
      "--seed" "${SEED}"
    )

    if ! run_and_log "${eval_log}" "${eval_cmd[@]}"; then
      echo "Eval failed: dataset=${dataset}, target=${target_size}" | tee -a "${eval_log}"
      append_summary "${dataset}" "${target_size}" "eval_failed" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        continue
      fi
      exit 1
    fi

    append_summary "${dataset}" "${target_size}" "ok" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
  done
done

echo "All runs done. Summary: ${SUMMARY_CSV}"
