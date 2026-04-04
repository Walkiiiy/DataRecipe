#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash exprun.sh
# Optional env vars:
#   PYTHON_BIN=python3
#   BASE_MODEL=Qwen/Qwen2.5-0.5B
#   MODEL_SOURCE=modelscope
#   EVAL_SOURCE=data/dolly/test.jsonl
#   EVAL_SIZE=1446
#   EVAL_RATIO=1
#   EPOCHS=4
#   SEED=42
#   OUTPUT_ROOT=data/dolly/exp/shared_eval
#   DRY_RUN=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
EVAL_SOURCE="${EVAL_SOURCE:-data/dolly/test.jsonl}"
EVAL_SIZE="${EVAL_SIZE:-1446}"
EVAL_RATIO="${EVAL_RATIO:-1}"
EPOCHS="${EPOCHS:-4}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/dolly/exp/shared_eval}"
DRY_RUN="${DRY_RUN:-0}"

TRAIN_SCRIPT="src/4.1/EXP/sft_lora_train_shared_eval.py"
EVAL_SCRIPT="src/4.1/EXP/eval_checkpoints_em.py"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${ROOT_DIR}/logs/exprun/${TIMESTAMP}"
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

train_run() {
  local method="$1"
  local dataset_jsonl="$2"
  local run_dir="$3"
  run_step "Train ${method}" \
    "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    --run "${method}::${dataset_jsonl}::${run_dir}" \
    --eval-source-jsonl "$EVAL_SOURCE" \
    --eval-ratio "$EVAL_RATIO" \
    --seed "$SEED" \
    --num_train_epochs "$EPOCHS" \
    --output-root "$OUTPUT_ROOT" \
    --base_model "$BASE_MODEL"
}

eval_run() {
  local method="$1"
  local ckpt_dir="$2"
  run_step "Eval ${method}" \
    "$PYTHON_BIN" "$EVAL_SCRIPT" \
    --eval-source-jsonl "$EVAL_SOURCE" \
    --eval-size "$EVAL_SIZE" \
    --base_model "$BASE_MODEL" \
    --model_source "$MODEL_SOURCE" \
    --run "${method}::${ckpt_dir}"
}

info "Workspace: $ROOT_DIR"
info "Main log:  $MAIN_LOG"
info "DRY_RUN:   $DRY_RUN"

require_file "$TRAIN_SCRIPT"
require_file "$EVAL_SCRIPT"
require_file "$EVAL_SOURCE"

for f in \
  data/dolly/exp4.2/dataset_delta_origin_130.jsonl \
  data/dolly/exp4.2/dataset_delta_origin_650.jsonl \
  data/dolly/exp4.2/dataset_delta_origin_1300.jsonl \
  data/dolly/exp4.2/dataset_delta_improved_130.jsonl \
  data/dolly/exp4.2/dataset_delta_improved_650.jsonl \
  data/dolly/exp4.2/dataset_delta_improved_1300.jsonl \
  data/dolly/exp4.2/dataset_alpagasus_origin_130.jsonl \
  data/dolly/exp4.2/dataset_alpagasus_origin_650.jsonl \
  data/dolly/exp4.2/dataset_alpagasus_origin_1300.jsonl \
  data/dolly/exp4.2/dataset_alpagasus_improved_130.jsonl \
  data/dolly/exp4.2/dataset_alpagasus_improved_650.jsonl \
  data/dolly/exp4.2/dataset_alpagasus_improved_1300.jsonl
do
  require_file "$f"
done

# ------------------------------------------------------------------------------
# Delta Origin
# ------------------------------------------------------------------------------
train_run "delta_origin" \
  "data/dolly/exp4.2/dataset_delta_origin_650.jsonl" \
  "data/dolly/exp4.2/run_delta_origin_650_shared_eval"
eval_run "delta_origin" \
  "data/dolly/exp4.2/run_delta_origin_650_shared_eval/final_checkpoint"

train_run "delta_origin" \
  "data/dolly/exp4.2/dataset_delta_origin_130.jsonl" \
  "data/dolly/exp4.2/run_delta_origin_130_shared_eval"
eval_run "delta_origin" \
  "data/dolly/exp4.2/run_delta_origin_130_shared_eval/final_checkpoint"

train_run "delta_origin" \
  "data/dolly/exp4.2/dataset_delta_origin_1300.jsonl" \
  "data/dolly/exp4.2/run_delta_origin_1300_shared_eval"
eval_run "delta_origin" \
  "data/dolly/exp4.2/run_delta_origin_1300_shared_eval/final_checkpoint"

# ------------------------------------------------------------------------------
# Delta Improved
# ------------------------------------------------------------------------------
train_run "delta_improved" \
  "data/dolly/exp4.2/dataset_delta_improved_650.jsonl" \
  "data/dolly/exp4.2/run_delta_improved_650_shared_eval"
eval_run "delta_improved" \
  "data/dolly/exp4.2/run_delta_improved_650_shared_eval/final_checkpoint"

train_run "delta_improved" \
  "data/dolly/exp4.2/dataset_delta_improved_130.jsonl" \
  "data/dolly/exp4.2/run_delta_improved_130_shared_eval"
eval_run "delta_improved" \
  "data/dolly/exp4.2/run_delta_improved_130_shared_eval/final_checkpoint"

train_run "delta_improved" \
  "data/dolly/exp4.2/dataset_delta_improved_1300.jsonl" \
  "data/dolly/exp4.2/run_delta_improved_1300_shared_eval"
eval_run "delta_improved" \
  "data/dolly/exp4.2/run_delta_improved_1300_shared_eval/final_checkpoint"

# ------------------------------------------------------------------------------
# Alpagasus Origin
# ------------------------------------------------------------------------------
train_run "alpagasus_origin" \
  "data/dolly/exp4.2/dataset_alpagasus_origin_130.jsonl" \
  "data/dolly/exp4.2/run_alpagasus_origin_130_shared_eval"
eval_run "alpagasus_origin" \
  "data/dolly/exp4.2/run_alpagasus_origin_130_shared_eval/final_checkpoint"

train_run "alpagasus_origin" \
  "data/dolly/exp4.2/dataset_alpagasus_origin_650.jsonl" \
  "data/dolly/exp4.2/run_alpagasus_origin_650_shared_eval"
eval_run "alpagasus_origin" \
  "data/dolly/exp4.2/run_alpagasus_origin_650_shared_eval/final_checkpoint"

train_run "alpagasus_origin" \
  "data/dolly/exp4.2/dataset_alpagasus_origin_1300.jsonl" \
  "data/dolly/exp4.2/run_alpagasus_origin_1300_shared_eval"
eval_run "alpagasus_origin" \
  "data/dolly/exp4.2/run_alpagasus_origin_1300_shared_eval/final_checkpoint"

# ------------------------------------------------------------------------------
# Alpagasus Improved
# ------------------------------------------------------------------------------
train_run "alpagasus_improved" \
  "data/dolly/exp4.2/dataset_alpagasus_improved_130.jsonl" \
  "data/dolly/exp4.2/run_alpagasus_improved_130_shared_eval"
eval_run "alpagasus_improved" \
  "data/dolly/exp4.2/run_alpagasus_improved_130_shared_eval/final_checkpoint"

train_run "alpagasus_improved" \
  "data/dolly/exp4.2/dataset_alpagasus_improved_650.jsonl" \
  "data/dolly/exp4.2/run_alpagasus_improved_650_shared_eval"
eval_run "alpagasus_improved" \
  "data/dolly/exp4.2/run_alpagasus_improved_650_shared_eval/final_checkpoint"

train_run "alpagasus_improved" \
  "data/dolly/exp4.2/dataset_alpagasus_improved_1300.jsonl" \
  "data/dolly/exp4.2/run_alpagasus_improved_1300_shared_eval"
eval_run "alpagasus_improved" \
  "data/dolly/exp4.2/run_alpagasus_improved_1300_shared_eval/final_checkpoint"

line
echo "[SUCCESS] All steps finished."
echo "[SUCCESS] Main log:  $MAIN_LOG"
echo "[SUCCESS] Step logs: $LOG_DIR"
line
