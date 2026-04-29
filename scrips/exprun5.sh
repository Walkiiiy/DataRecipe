#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   bash exprun5.sh
# Optional env vars:
#   PYTHON_BIN=python3
#   BASE_MODEL=Qwen/Qwen2.5-0.5B
#   MODEL_SOURCE=modelscope
#   DRY_RUN=1
#   LAWYER_EVAL_SOURCE=data/lawyer/test.jsonl
#   LAWYER_EVAL_SIZE=2147
#   DIALOGSUM_EVAL_SOURCE=data/dialogsum/test.jsonl
#   DIALOGSUM_EVAL_SIZE=1446
#   DOLLY_EVAL_SOURCE=data/dolly/test.jsonl
#   DOLLY_EVAL_SIZE=1501

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
DRY_RUN="${DRY_RUN:-0}"

LAWYER_EVAL_SOURCE="${LAWYER_EVAL_SOURCE:-data/lawyer/test.jsonl}"
LAWYER_EVAL_SIZE="${LAWYER_EVAL_SIZE:-2147}"

DIALOGSUM_EVAL_SOURCE="${DIALOGSUM_EVAL_SOURCE:-data/dialogsum/test.jsonl}"
DIALOGSUM_EVAL_SIZE="${DIALOGSUM_EVAL_SIZE:-1446}"

DOLLY_EVAL_SOURCE="${DOLLY_EVAL_SOURCE:-data/dolly/test.jsonl}"
DOLLY_EVAL_SIZE="${DOLLY_EVAL_SIZE:-1501}"

EVAL_SCRIPT="src/4.1/EXP/eval_checkpoints_em.py"
BASE_EVAL_METHOD="base_model"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${ROOT_DIR}/logs/exprun5/${TIMESTAMP}"
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

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "[ERROR] Missing required path: $path"
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

BASE_MODEL_EVAL_PATH=""
resolve_base_model_eval_path() {
  if [[ -e "$BASE_MODEL" ]]; then
    BASE_MODEL_EVAL_PATH="$BASE_MODEL"
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    BASE_MODEL_EVAL_PATH="$BASE_MODEL"
    return 0
  fi

  if [[ "$MODEL_SOURCE" == "modelscope" ]]; then
    BASE_MODEL_EVAL_PATH="$(
      BASE_MODEL_ID="$BASE_MODEL" MODELSCOPE_LOG_LEVEL=40 "$PYTHON_BIN" - <<'PY' | awk 'NF{last=$0} END{print last}'
import os
from modelscope import snapshot_download

print(snapshot_download(model_id=os.environ["BASE_MODEL_ID"]))
PY
    )"
    return 0
  fi

  echo "[ERROR] For base-model eval, BASE_MODEL must be a local path when MODEL_SOURCE=hf."
  echo "[ERROR] Current BASE_MODEL=$BASE_MODEL"
  exit 1
}

eval_base_model() {
  local dataset="$1"
  local eval_source="$2"
  local eval_size="$3"

  run_step "${dataset} eval ${BASE_EVAL_METHOD}" \
    "$PYTHON_BIN" "$EVAL_SCRIPT" \
    --eval-source-jsonl "$eval_source" \
    --eval-size "$eval_size" \
    --base_model "$BASE_MODEL_EVAL_PATH" \
    --model_source "$MODEL_SOURCE" \
    --run "${BASE_EVAL_METHOD}::${BASE_MODEL_EVAL_PATH}"
}

info "Workspace: $ROOT_DIR"
info "Main log:  $MAIN_LOG"
info "DRY_RUN:   $DRY_RUN"

require_file "$EVAL_SCRIPT"
require_file "$LAWYER_EVAL_SOURCE"
require_file "$DIALOGSUM_EVAL_SOURCE"
require_file "$DOLLY_EVAL_SOURCE"

resolve_base_model_eval_path
if [[ "$DRY_RUN" != "1" ]]; then
  require_path "$BASE_MODEL_EVAL_PATH"
fi

eval_base_model "lawyer" "$LAWYER_EVAL_SOURCE" "$LAWYER_EVAL_SIZE"
eval_base_model "dialogsum" "$DIALOGSUM_EVAL_SOURCE" "$DIALOGSUM_EVAL_SIZE"
eval_base_model "dolly" "$DOLLY_EVAL_SOURCE" "$DOLLY_EVAL_SIZE"

line
echo "[SUCCESS] All requested steps finished."
echo "[SUCCESS] Main log:  $MAIN_LOG"
echo "[SUCCESS] Step logs: $LOG_DIR"
line
