#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash exprun3.sh
# Optional env vars:
#   PYTHON_BIN=python3
#   BASE_MODEL=Qwen/Qwen2.5-0.5B
#   MODEL_SOURCE=modelscope
#   EVAL_RATIO=1
#   EPOCHS=4
#   SEED=42
#   DRY_RUN=1
#   LAWYER_EVAL_SOURCE=data/lawyer/test.jsonl
#   LAWYER_EVAL_SIZE=2147
#   LAWYER_OUTPUT_ROOT=data/lawyer/exp/shared_eval
#   DIALOGSUM_EVAL_SOURCE=data/dialogsum/test.jsonl
#   DIALOGSUM_EVAL_SIZE=1446
#   DIALOGSUM_OUTPUT_ROOT=data/dialogsum/exp/shared_eval
#   DOLLY_EVAL_SOURCE=data/dolly/test.jsonl
#   DOLLY_EVAL_SIZE=1501
#   DOLLY_OUTPUT_ROOT=data/dolly/exp/shared_eval

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
EVAL_RATIO="${EVAL_RATIO:-1}"
EPOCHS="${EPOCHS:-4}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

LAWYER_EVAL_SOURCE="${LAWYER_EVAL_SOURCE:-data/lawyer/test.jsonl}"
LAWYER_EVAL_SIZE="${LAWYER_EVAL_SIZE:-2147}"
LAWYER_OUTPUT_ROOT="${LAWYER_OUTPUT_ROOT:-data/lawyer/exp/shared_eval}"

DIALOGSUM_EVAL_SOURCE="${DIALOGSUM_EVAL_SOURCE:-data/dialogsum/test.jsonl}"
DIALOGSUM_EVAL_SIZE="${DIALOGSUM_EVAL_SIZE:-1446}"
DIALOGSUM_OUTPUT_ROOT="${DIALOGSUM_OUTPUT_ROOT:-data/dialogsum/exp/shared_eval}"

DOLLY_EVAL_SOURCE="${DOLLY_EVAL_SOURCE:-data/dolly/test.jsonl}"
DOLLY_EVAL_SIZE="${DOLLY_EVAL_SIZE:-1501}"
DOLLY_OUTPUT_ROOT="${DOLLY_OUTPUT_ROOT:-data/dolly/exp/shared_eval}"

TRAIN_SCRIPT="src/4.1/EXP/sft_lora_train_shared_eval.py"
EVAL_SCRIPT="src/4.1/EXP/eval_checkpoints_em.py"

METHODS=(mig datawhisperer)
LAWYER_SIZES=(200 1000 2000)
DIALOGSUM_SIZES=(140 700 1000)
DOLLY_SIZES=(130 650 1300)

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${ROOT_DIR}/logs/exprun3/${TIMESTAMP}"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/main.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

STEP_NO=0

slugify() {
  local s="$1"
  s="${s,,}"
  s="${s// /_}"
  s="$(echo "$s" | tr -cd 'a-z0-9_-')"
  if [[ -z "$s" ]]; then
    s="step"
  fi
  echo "$s"
}

line() {
  printf '%*s\n' 90 '' | tr ' ' '='
}

info() {
  echo "[INFO] $*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing required file: $path"
    exit 1
  fi
}

run_step() {
  local title="$1"
  shift
  STEP_NO=$((STEP_NO + 1))

  local idx
  idx="$(printf '%02d' "$STEP_NO")"
  local slug
  slug="$(slugify "$title")"
  local step_log="${LOG_DIR}/${idx}_${slug}.log"

  line
  echo "[STEP ${idx}] ${title}"
  echo "Start: $(date '+%F %T')"
  echo "Log:   ${step_log}"
  echo "Cmd:"
  printf '  %q' "$@"
  echo
  line

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] Skip execution."
    return 0
  fi

  set +e
  "$@" 2>&1 | tee "$step_log"
  local rc=${PIPESTATUS[0]}
  set -e

  if ((rc != 0)); then
    echo "[FAILED] Step ${idx} (${title}) exit_code=${rc}"
    echo "[FAILED] See log: ${step_log}"
    exit "$rc"
  fi

  echo "[DONE] Step ${idx} (${title}) at $(date '+%F %T')"
}

dataset_file() {
  local dataset="$1"
  local method="$2"
  local size="$3"
  echo "data/${dataset}/exp4.2/dataset_${method}_${size}.jsonl"
}

run_dir() {
  local dataset="$1"
  local method="$2"
  local size="$3"
  echo "data/${dataset}/exp4.2/run_${method}_${size}_shared_eval"
}

train_run() {
  local dataset="$1"
  local method="$2"
  local size="$3"
  local eval_source="$4"
  local output_root="$5"

  local dataset_jsonl
  dataset_jsonl="$(dataset_file "$dataset" "$method" "$size")"
  local output_dir
  output_dir="$(run_dir "$dataset" "$method" "$size")"

  run_step "${dataset} train ${method} ${size}" \
    "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --run "${method}::${dataset_jsonl}::${output_dir}" \
    --eval-source-jsonl "$eval_source" \
    --eval-ratio "$EVAL_RATIO" \
    --seed "$SEED" \
    --num_train_epochs "$EPOCHS" \
    --output-root "$output_root" \
    --base_model "$BASE_MODEL"
}

eval_run() {
  local dataset="$1"
  local method="$2"
  local size="$3"
  local eval_source="$4"
  local eval_size="$5"

  local checkpoint_dir
  checkpoint_dir="$(run_dir "$dataset" "$method" "$size")/final_checkpoint"

  run_step "${dataset} eval ${method} ${size}" \
    "$PYTHON_BIN" "$EVAL_SCRIPT" \
    --eval-source-jsonl "$eval_source" \
    --eval-size "$eval_size" \
    --base_model "$BASE_MODEL" \
    --model_source "$MODEL_SOURCE" \
    --run "${method}::${checkpoint_dir}"
}

run_method_sizes() {
  local dataset="$1"
  local method="$2"
  local eval_source="$3"
  local eval_size="$4"
  local output_root="$5"
  local -n sizes_ref="$6"

  local size
  local ds_path
  for size in "${sizes_ref[@]}"; do
    ds_path="$(dataset_file "$dataset" "$method" "$size")"
    require_file "$ds_path"
    train_run "$dataset" "$method" "$size" "$eval_source" "$output_root"
    eval_run "$dataset" "$method" "$size" "$eval_source" "$eval_size"
  done
}

check_method_sizes() {
  local dataset="$1"
  local -n methods_ref="$2"
  local -n sizes_ref="$3"

  local method
  local size
  local ds_path
  for method in "${methods_ref[@]}"; do
    for size in "${sizes_ref[@]}"; do
      ds_path="$(dataset_file "$dataset" "$method" "$size")"
      require_file "$ds_path"
    done
  done
}

info "Workspace: $ROOT_DIR"
info "Main log:  $MAIN_LOG"
info "DRY_RUN:   $DRY_RUN"

require_file "$TRAIN_SCRIPT"
require_file "$EVAL_SCRIPT"
require_file "$LAWYER_EVAL_SOURCE"
require_file "$DIALOGSUM_EVAL_SOURCE"
require_file "$DOLLY_EVAL_SOURCE"

check_method_sizes "lawyer" METHODS LAWYER_SIZES
check_method_sizes "dialogsum" METHODS DIALOGSUM_SIZES
check_method_sizes "dolly" METHODS DOLLY_SIZES

for method in "${METHODS[@]}"; do
  run_method_sizes "lawyer" "$method" \
    "$LAWYER_EVAL_SOURCE" "$LAWYER_EVAL_SIZE" "$LAWYER_OUTPUT_ROOT" LAWYER_SIZES
done

for method in "${METHODS[@]}"; do
  run_method_sizes "dialogsum" "$method" \
    "$DIALOGSUM_EVAL_SOURCE" "$DIALOGSUM_EVAL_SIZE" "$DIALOGSUM_OUTPUT_ROOT" DIALOGSUM_SIZES
done

for method in "${METHODS[@]}"; do
  run_method_sizes "dolly" "$method" \
    "$DOLLY_EVAL_SOURCE" "$DOLLY_EVAL_SIZE" "$DOLLY_OUTPUT_ROOT" DOLLY_SIZES
done

line
echo "[SUCCESS] All requested steps finished."
echo "[SUCCESS] Main log:  $MAIN_LOG"
echo "[SUCCESS] Step logs: $LOG_DIR"
line
