#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${DATASET:-dolly}"
TRAIN_JSONL="${TRAIN_JSONL:-data/${DATASET}/train.jsonl}"
TOP_K_PATH="${TOP_K_PATH:-data/${DATASET}/train_coarse_topk5.jsonl}"
OUT_DIR="${OUT_DIR:-data/${DATASET}/exp4.3/run_${DATASET}_recipe}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
MODELSCOPE_CACHE_DIR="${MODELSCOPE_CACHE_DIR:-}"

BATCH="${BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
MAX_LEN="${MAX_LEN:-256}"
SEED="${SEED:-42}"

# 用户重点参数：第一轮采样 data_size，后续 epoch 复训该子集
DATA_SIZE="${DATA_SIZE:-${TRAIN_SIZE:-${STEPS:-130}}}"
EPOCH="${EPOCH:-${EPOCHS:-2}}"

# Recipe Evolution 超参
GAMMA_T="${GAMMA_T:-0.8}"
GAMMA_ALPHA="${GAMMA_ALPHA:-1.0}"
ALPHA_EPS="${ALPHA_EPS:-0.05}"
ETA_BETA="${ETA_BETA:-0.2}"

ANCHOR_SIZE="${ANCHOR_SIZE:-4}"
ANCHOR_REFRESH="${ANCHOR_REFRESH:-20}"
ANCHOR_CHUNK="${ANCHOR_CHUNK:-2}"
MAX_CAP_DIM="${MAX_CAP_DIM:-0}"
VMAP_CHUNK="${VMAP_CHUNK:-4}"

LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

TORCH_DTYPE="${TORCH_DTYPE:-auto}"
GRAD_CKPT="${GRAD_CKPT:-1}"
TARGET_LAYER="${TARGET_LAYER:-lm_head}"
TOP_K_MAX="${TOP_K_MAX:-5}"

DRY_RUN="${DRY_RUN:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[ERR] train jsonl not found: ${TRAIN_JSONL}" >&2
  exit 1
fi

# score 文件：
# 1) 若传入 SCORE_FILES（逗号分隔），优先使用
# 2) 否则自动从常见 vector 评分文件中择优存在者
SCORE_ARGS=()
if [[ -n "${SCORE_FILES:-}" ]]; then
  IFS=',' read -r -a _arr <<< "${SCORE_FILES}"
  for raw in "${_arr[@]}"; do
    p="$(echo "${raw}" | xargs)"
    [[ -z "${p}" ]] && continue
    SCORE_ARGS+=("--score-path" "${p}")
  done
else
  CANDIDATES=(
    "data/${DATASET}/score/pdm_scored.jsonl"
    "data/${DATASET}/score/srm_from_topk5_only.jsonl"
    "data/${DATASET}/score/delta_improved_mapped.jsonl"
    "data/${DATASET}/score/alpagasus_improved_mapped.jsonl"
    "data/${DATASET}/score/datawhisperer_icl_mapped.jsonl"
  )
  for p in "${CANDIDATES[@]}"; do
    if [[ -f "${p}" ]]; then
      SCORE_ARGS+=("--score-path" "${p}")
    fi
  done
fi

if [[ ${#SCORE_ARGS[@]} -eq 0 ]]; then
  echo "[ERR] no score files resolved. Set SCORE_FILES or prepare data/${DATASET}/score/*.jsonl" >&2
  exit 1
fi

CMD=(
  python src/4.3/run.py
  --train-jsonl "${TRAIN_JSONL}"
  --output-dir "${OUT_DIR}"
  --base_model "${BASE_MODEL}"
  --model_source "${MODEL_SOURCE}"
  --max_seq_length "${MAX_LEN}"
  --learning_rate "${LR}"
  --train_batch_size "${BATCH}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --warmup_ratio "${WARMUP_RATIO}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --logging_steps "${LOG_EVERY}"
  --save_steps "${SAVE_EVERY}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --data_size "${DATA_SIZE}"
  --epoch "${EPOCH}"
  --gamma_t "${GAMMA_T}"
  --gamma_alpha "${GAMMA_ALPHA}"
  --alpha_epsilon "${ALPHA_EPS}"
  --eta_beta "${ETA_BETA}"
  --anchor_size "${ANCHOR_SIZE}"
  --anchor_refresh_steps "${ANCHOR_REFRESH}"
  --anchor_batch_size "${ANCHOR_CHUNK}"
  --max_anchor_capabilities "${MAX_CAP_DIM}"
  --target_layer "${TARGET_LAYER}"
  --vmap_chunk_size "${VMAP_CHUNK}"
  --torch_dtype "${TORCH_DTYPE}"
  --grad_ckpt "${GRAD_CKPT}"
  --seed "${SEED}"
  --top_k_max "${TOP_K_MAX}"
  --dry_run "${DRY_RUN}"
  --log_level "${LOG_LEVEL}"
)

if [[ -n "${MODELSCOPE_CACHE_DIR}" ]]; then
  CMD+=(--modelscope_cache_dir "${MODELSCOPE_CACHE_DIR}")
fi

if [[ -f "${TOP_K_PATH}" ]]; then
  CMD+=(--top-k-path "${TOP_K_PATH}")
fi

CMD+=("${SCORE_ARGS[@]}")

echo "[INFO] running command:"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
