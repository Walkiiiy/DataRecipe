#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash exprun2.sh
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
ALPAGASUS_IMPROVED_SAMPLE_SCRIPT="src/4.2/alpagasus/alpagasus_improved_sample.py"
INSTAG_SAMPLE_SCRIPT="src/4.2/instag/instag_sample.py"

LAWYER_SIZES=(200 1000 2000)
DIALOGSUM_SIZES=(140 700 1000)
DOLLY_SIZES=(130 650 1300)

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${ROOT_DIR}/logs/exprun2/${TIMESTAMP}"
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

ensure_alpagasus_improved_dataset() {
  local dataset="$1"
  local size="$2"
  local out_path
  out_path="$(dataset_file "$dataset" "alpagasus_improved" "$size")"
  if [[ -f "$out_path" ]]; then
    return 0
  fi

  local score_path="data/${dataset}/score/alpagasus_improved_mapped.jsonl"
  local data_path="data/${dataset}/train.jsonl"
  require_file "$ALPAGASUS_IMPROVED_SAMPLE_SCRIPT"
  require_file "$score_path"
  require_file "$data_path"

  run_step "${dataset} prep alpagasus_improved ${size}" \
    "$PYTHON_BIN" "$ALPAGASUS_IMPROVED_SAMPLE_SCRIPT" \
    --score_path "$score_path" \
    --data_path "$data_path" \
    --output_path "$out_path" \
    --num_samples "$size"
}

ensure_instag_dataset() {
  local dataset="$1"
  local size="$2"
  local out_path
  out_path="$(dataset_file "$dataset" "instag" "$size")"
  if [[ -f "$out_path" ]]; then
    return 0
  fi

  local data_path="data/${dataset}/train.jsonl"
  local tag_path="data/${dataset}/score/instag_tags.jsonl"
  require_file "$INSTAG_SAMPLE_SCRIPT"
  require_file "$data_path"
  require_file "$tag_path"

  run_step "${dataset} prep instag ${size}" \
    "$PYTHON_BIN" "$INSTAG_SAMPLE_SCRIPT" \
    --data_path "$data_path" \
    --tag_path "$tag_path" \
    --output_path "$out_path" \
    --num_samples "$size"
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
    if [[ "$DRY_RUN" != "1" ]]; then
      require_file "$ds_path"
    fi
    train_run "$dataset" "$method" "$size" "$eval_source" "$output_root"
    eval_run "$dataset" "$method" "$size" "$eval_source" "$eval_size"
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

# Requested scope only:
# 1) lawyer alpagasus_origin
# 2) lawyer + dialogsum alpagasus_improved
# 3) lawyer + dialogsum + dolly instag

# 1) lawyer alpagasus_origin on 200/1000/2000
run_method_sizes "lawyer" "alpagasus_origin" \
  "$LAWYER_EVAL_SOURCE" "$LAWYER_EVAL_SIZE" "$LAWYER_OUTPUT_ROOT" LAWYER_SIZES

# 2) lawyer/dialogsum alpagasus_improved on their three sizes
for size in "${LAWYER_SIZES[@]}"; do
  ensure_alpagasus_improved_dataset "lawyer" "$size"
done
run_method_sizes "lawyer" "alpagasus_improved" \
  "$LAWYER_EVAL_SOURCE" "$LAWYER_EVAL_SIZE" "$LAWYER_OUTPUT_ROOT" LAWYER_SIZES

for size in "${DIALOGSUM_SIZES[@]}"; do
  ensure_alpagasus_improved_dataset "dialogsum" "$size"
done
run_method_sizes "dialogsum" "alpagasus_improved" \
  "$DIALOGSUM_EVAL_SOURCE" "$DIALOGSUM_EVAL_SIZE" "$DIALOGSUM_OUTPUT_ROOT" DIALOGSUM_SIZES

# 3) instag on lawyer/dialogsum/dolly with each dataset's three sizes
for size in "${LAWYER_SIZES[@]}"; do
  ensure_instag_dataset "lawyer" "$size"
done
run_method_sizes "lawyer" "instag" \
  "$LAWYER_EVAL_SOURCE" "$LAWYER_EVAL_SIZE" "$LAWYER_OUTPUT_ROOT" LAWYER_SIZES

for size in "${DIALOGSUM_SIZES[@]}"; do
  ensure_instag_dataset "dialogsum" "$size"
done
run_method_sizes "dialogsum" "instag" \
  "$DIALOGSUM_EVAL_SOURCE" "$DIALOGSUM_EVAL_SIZE" "$DIALOGSUM_OUTPUT_ROOT" DIALOGSUM_SIZES

for size in "${DOLLY_SIZES[@]}"; do
  ensure_instag_dataset "dolly" "$size"
done
run_method_sizes "dolly" "instag" \
  "$DOLLY_EVAL_SOURCE" "$DOLLY_EVAL_SIZE" "$DOLLY_OUTPUT_ROOT" DOLLY_SIZES

line
echo "[SUCCESS] All requested steps finished."
echo "[SUCCESS] Main log:  $MAIN_LOG"
echo "[SUCCESS] Step logs: $LOG_DIR"
line
