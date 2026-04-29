#!/usr/bin/env bash
set -euo pipefail

# Score-file ablation for Recipe Evolution decoupled training.
#
# Test all combinations of 4 score files except:
# - single-score combinations (size=1): excluded
# - all-4 combination (size=4): excluded
# Included combinations:
# - all size=2 combinations: C(4,2)=6
# - all size=3 combinations: C(4,3)=4
# Total = 10 combinations.
#
# Datasets and fixed epoch1 target sizes:
# - dolly:     1300
# - dialogsum: 1000
# - lawyer:    2000
#
# Usage:
#   bash run_recipe_score_ablation.sh
#
# Optional env vars:
#   PYTHON_BIN=python
#   BASE_MODEL=Qwen/Qwen2.5-0.5B
#   MODEL_SOURCE=modelscope
#   RUN_TAG=score_ablation_YYYYmmdd_HHMMSS
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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
RUN_TAG="${RUN_TAG:-score_ablation_${TIMESTAMP}}"
LOG_ROOT="logs/recipe_score_ablation/${RUN_TAG}"
SUMMARY_CSV="${LOG_ROOT}/summary.csv"

mkdir -p "${LOG_ROOT}"

declare -A TARGET_SIZE_MAP=(
  [dolly]="1300"
  [dialogsum]="1000"
  [lawyer]="2000"
)

declare -A EVAL_SIZE_MAP=(
  [dolly]="1446"
  [dialogsum]="1446"
  [lawyer]="1446"
)

SCORE_KEYS=("alpagasus_improved" "delta_improved" "srm" "pdm")

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
  local combo_size="$3"
  local combo_key="$4"
  local status="$5"
  local output_dir="$6"
  local final_ckpt="$7"
  local train_log="$8"
  local eval_log="$9"
  echo "${dataset},${target},${combo_size},${combo_key},${status},${output_dir},${final_ckpt},${train_log},${eval_log}" >> "${SUMMARY_CSV}"
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

build_score_arg() {
  local dataset="$1"
  local key="$2"
  case "${key}" in
    alpagasus_improved)
      echo "alpagasus_improved::data/${dataset}/score/alpagasus_improved_mapped.jsonl"
      ;;
    delta_improved)
      echo "delta_improved::data/${dataset}/score/delta_improved_mapped.jsonl"
      ;;
    srm)
      echo "srm::data/${dataset}/score/srm_from_topk5_only.jsonl"
      ;;
    pdm)
      echo "pdm::data/${dataset}/score/pdm_scored.jsonl"
      ;;
    *)
      echo "Unknown score key: ${key}" >&2
      return 1
      ;;
  esac
}

echo "dataset,epoch1_target_size,combo_size,combo_key,status,output_dir,final_checkpoint,train_log,eval_log" > "${SUMMARY_CSV}"

for dataset in dolly dialogsum lawyer; do
  check_required_files "${dataset}"

  train_jsonl="data/${dataset}/train.jsonl"
  test_jsonl="data/${dataset}/test.jsonl"
  topk_jsonl="data/${dataset}/train_coarse_topk5.jsonl"
  target_size="${TARGET_SIZE_MAP[${dataset}]}"
  eval_size="${EVAL_SIZE_MAP[${dataset}]}"

  # size=2 combos
  for ((i = 0; i < ${#SCORE_KEYS[@]}; i++)); do
    for ((j = i + 1; j < ${#SCORE_KEYS[@]}; j++)); do
      combo_keys=("${SCORE_KEYS[i]}" "${SCORE_KEYS[j]}")
      combo_size="2"
      combo_tag="${combo_keys[0]}__${combo_keys[1]}"

      score_args=()
      for key in "${combo_keys[@]}"; do
        score_args+=("--score-path" "$(build_score_arg "${dataset}" "${key}")")
      done

      run_name="re_ablation_${combo_tag}"
      output_dir="data/${dataset}/exp4.3/${RUN_TAG}/${run_name}"
      train_log="${LOG_ROOT}/${dataset}_${combo_tag}_train.log"
      eval_log="${LOG_ROOT}/${dataset}_${combo_tag}_eval.log"
      final_ckpt="${output_dir}/static_stage/run_recipe_selected/final_checkpoint"
      eval_out_dir="${output_dir}/em_eval"

      train_cmd=(
        "${PYTHON_BIN}" "src/4.3/run_orchestrator.py"
        "--train-raw-jsonl" "${train_jsonl}"
        "--anchor-jsonl" "${train_jsonl}"
        "--topk-jsonl" "${topk_jsonl}"
        "${score_args[@]}"
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
        echo "Train failed: dataset=${dataset}, combo=${combo_tag}" | tee -a "${train_log}"
        append_summary "${dataset}" "${target_size}" "${combo_size}" "${combo_tag}" "train_failed" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
          continue
        fi
        exit 1
      fi

      if [[ ! -d "${final_ckpt}" ]]; then
        echo "Final checkpoint missing: ${final_ckpt}" | tee -a "${train_log}"
        append_summary "${dataset}" "${target_size}" "${combo_size}" "${combo_tag}" "missing_checkpoint" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
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
        echo "Eval failed: dataset=${dataset}, combo=${combo_tag}" | tee -a "${eval_log}"
        append_summary "${dataset}" "${target_size}" "${combo_size}" "${combo_tag}" "eval_failed" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
          continue
        fi
        exit 1
      fi

      append_summary "${dataset}" "${target_size}" "${combo_size}" "${combo_tag}" "ok" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
    done
  done

  # size=3 combos
  for ((i = 0; i < ${#SCORE_KEYS[@]}; i++)); do
    for ((j = i + 1; j < ${#SCORE_KEYS[@]}; j++)); do
      for ((k = j + 1; k < ${#SCORE_KEYS[@]}; k++)); do
        combo_keys=("${SCORE_KEYS[i]}" "${SCORE_KEYS[j]}" "${SCORE_KEYS[k]}")
        combo_size="3"
        combo_tag="${combo_keys[0]}__${combo_keys[1]}__${combo_keys[2]}"

        score_args=()
        for key in "${combo_keys[@]}"; do
          score_args+=("--score-path" "$(build_score_arg "${dataset}" "${key}")")
        done

        run_name="re_ablation_${combo_tag}"
        output_dir="data/${dataset}/exp4.3/${RUN_TAG}/${run_name}"
        train_log="${LOG_ROOT}/${dataset}_${combo_tag}_train.log"
        eval_log="${LOG_ROOT}/${dataset}_${combo_tag}_eval.log"
        final_ckpt="${output_dir}/static_stage/run_recipe_selected/final_checkpoint"
        eval_out_dir="${output_dir}/em_eval"

        train_cmd=(
          "${PYTHON_BIN}" "src/4.3/run_orchestrator.py"
          "--train-raw-jsonl" "${train_jsonl}"
          "--anchor-jsonl" "${train_jsonl}"
          "--topk-jsonl" "${topk_jsonl}"
          "${score_args[@]}"
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
          echo "Train failed: dataset=${dataset}, combo=${combo_tag}" | tee -a "${train_log}"
          append_summary "${dataset}" "${target_size}" "${combo_size}" "${combo_tag}" "train_failed" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
          if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
            continue
          fi
          exit 1
        fi

        if [[ ! -d "${final_ckpt}" ]]; then
          echo "Final checkpoint missing: ${final_ckpt}" | tee -a "${train_log}"
          append_summary "${dataset}" "${target_size}" "${combo_size}" "${combo_tag}" "missing_checkpoint" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
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
          echo "Eval failed: dataset=${dataset}, combo=${combo_tag}" | tee -a "${eval_log}"
          append_summary "${dataset}" "${target_size}" "${combo_size}" "${combo_tag}" "eval_failed" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
          if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
            continue
          fi
          exit 1
        fi

        append_summary "${dataset}" "${target_size}" "${combo_size}" "${combo_tag}" "ok" "${output_dir}" "${final_ckpt}" "${train_log}" "${eval_log}"
      done
    done
  done
done

echo "All runs done. Summary: ${SUMMARY_CSV}"
