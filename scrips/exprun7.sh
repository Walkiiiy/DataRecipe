#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash exprun7.sh
# Optional env vars:
#   PYTHON_BIN=python3
#   BASE_MODEL=Qwen/Qwen2.5-0.5B
#   MODEL_SOURCE=modelscope
#   DRY_RUN=1
#   DATASETS="dialogsum lawyer"        # optional: limit datasets to evaluate
#   EXTRA_EVAL_ARGS="--batch_size 8"   # optional: extra args passed to eval script
#   DIALOGSUM_EVAL_SOURCE=data/dialogsum/test.jsonl
#   DIALOGSUM_EVAL_SIZE=1446

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
DRY_RUN="${DRY_RUN:-0}"
DATASETS="${DATASETS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

EVAL_SCRIPT="src/4.1/EXP/eval_checkpoints_em.py"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${ROOT_DIR}/logs/exprun7/${TIMESTAMP}"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/main.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

STEP_NO=0
TOTAL_EVALS=0
TOTAL_SKIPPED=0

line() {
  printf '%*s\n' 90 '' | tr ' ' '='
}

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*"
}

error_exit() {
  echo "[ERROR] $*"
  exit 1
}

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

dataset_env_prefix() {
  local s="$1"
  s="$(echo "$s" | tr '[:lower:]-.' '[:upper:]__' | tr -cd 'A-Z0-9_')"
  if [[ -z "$s" ]]; then
    s="DATASET"
  fi
  echo "$s"
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

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    error_exit "Missing required file: $path"
  fi
}

DATASET_EXP_ROOTS=()

discover_dataset_exp_roots() {
  DATASET_EXP_ROOTS=()

  if [[ -n "$DATASETS" ]]; then
    local raw
    raw="${DATASETS//,/ }"
    local dataset
    for dataset in $raw; do
      local exp_root="data/${dataset}/exp4.3"
      if [[ -d "$exp_root" ]]; then
        DATASET_EXP_ROOTS+=("$exp_root")
      else
        warn "Skip dataset=${dataset}: missing ${exp_root}"
      fi
    done
  else
    while IFS= read -r exp_root; do
      [[ -n "$exp_root" ]] && DATASET_EXP_ROOTS+=("$exp_root")
    done < <(find data -mindepth 2 -maxdepth 2 -type d -name 'exp4.3' | sort)
  fi

  if (( ${#DATASET_EXP_ROOTS[@]} == 0 )); then
    error_exit "No exp4.3 directory found under data/*/exp4.3"
  fi
}

resolve_eval_source() {
  local dataset="$1"
  local prefix
  prefix="$(dataset_env_prefix "$dataset")"
  local var_name="${prefix}_EVAL_SOURCE"
  local default_source="data/${dataset}/test.jsonl"
  local eval_source="${!var_name:-$default_source}"
  echo "$eval_source"
}

resolve_eval_size() {
  local dataset="$1"
  local eval_source="$2"

  local prefix
  prefix="$(dataset_env_prefix "$dataset")"
  local var_name="${prefix}_EVAL_SIZE"

  if [[ -n "${!var_name:-}" ]]; then
    echo "${!var_name}"
    return 0
  fi

  if [[ ! -f "$eval_source" ]]; then
    return 1
  fi

  local n
  n="$(awk 'NF{c++} END{print c+0}' "$eval_source")"
  if [[ "$n" -le 0 ]]; then
    return 1
  fi

  echo "$n"
}

CHECKPOINTS=()

discover_checkpoints() {
  local exp_root="$1"
  CHECKPOINTS=()

  while IFS= read -r ckpt; do
    [[ -n "$ckpt" ]] && CHECKPOINTS+=("$ckpt")
  done < <(find "$exp_root" -mindepth 1 -maxdepth 4 -type d \( -name 'checkpoint-*' -o -name 'final' -o -name 'final_checkpoint' \) | sort -V)
}

make_run_tag() {
  local dataset="$1"
  local exp_root="$2"
  local checkpoint_dir="$3"

  local rel="${checkpoint_dir#${exp_root}/}"
  local tag="${dataset}__${rel//\//__}"
  tag="$(echo "$tag" | tr ' ' '_' | tr -cd 'A-Za-z0-9_.-')"
  if [[ -z "$tag" ]]; then
    tag="${dataset}__checkpoint"
  fi
  echo "$tag"
}

EXTRA_EVAL_ARGS_ARR=()
if [[ -n "$EXTRA_EVAL_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_EVAL_ARGS_ARR=($EXTRA_EVAL_ARGS)
fi

info "Workspace: $ROOT_DIR"
info "Main log:  $MAIN_LOG"
info "DRY_RUN:   $DRY_RUN"
info "BASE_MODEL:$BASE_MODEL"
info "MODEL_SRC: $MODEL_SOURCE"

require_file "$EVAL_SCRIPT"
discover_dataset_exp_roots

for exp_root in "${DATASET_EXP_ROOTS[@]}"; do
  dataset="$(basename "$(dirname "$exp_root")")"

  eval_source="$(resolve_eval_source "$dataset")"
  if [[ ! -f "$eval_source" ]]; then
    warn "Skip dataset=${dataset}: missing eval source ${eval_source}"
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
    continue
  fi

  if ! eval_size="$(resolve_eval_size "$dataset" "$eval_source")"; then
    warn "Skip dataset=${dataset}: failed to resolve eval size for ${eval_source}"
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
    continue
  fi

  discover_checkpoints "$exp_root"
  if (( ${#CHECKPOINTS[@]} == 0 )); then
    warn "Skip dataset=${dataset}: no checkpoints found under ${exp_root}"
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
    continue
  fi

  info "Dataset=${dataset} eval_source=${eval_source} eval_size=${eval_size} checkpoints=${#CHECKPOINTS[@]}"

  for checkpoint_dir in "${CHECKPOINTS[@]}"; do
    rel="${checkpoint_dir#${exp_root}/}"
    run_tag="$(make_run_tag "$dataset" "$exp_root" "$checkpoint_dir")"

    run_step "${dataset} eval ${rel}" \
      "$PYTHON_BIN" "$EVAL_SCRIPT" \
      --eval-source-jsonl "$eval_source" \
      --eval-size "$eval_size" \
      --base_model "$BASE_MODEL" \
      --model_source "$MODEL_SOURCE" \
      --run "${run_tag}::${checkpoint_dir}" \
      "${EXTRA_EVAL_ARGS_ARR[@]}"

    TOTAL_EVALS=$((TOTAL_EVALS + 1))
  done
done

line
echo "[SUCCESS] Finished exp4.3 checkpoint evaluation sweep."
echo "[SUCCESS] Total eval runs: ${TOTAL_EVALS}"
echo "[SUCCESS] Total skipped:   ${TOTAL_SKIPPED}"
echo "[SUCCESS] Main log:  ${MAIN_LOG}"
echo "[SUCCESS] Step logs: ${LOG_DIR}"
line
