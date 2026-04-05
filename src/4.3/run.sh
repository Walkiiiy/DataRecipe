#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/4.3/run.sh
# Optional overrides:
#   STEPS=300 BATCH=8 ANCHOR_REFRESH=20 MAX_CAP_DIM=128 MAX_SAMPLES=0 bash src/4.3/run.sh

ROOT_DIR="/home/walkiiiy/DataRecipe"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-$HOME/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B}"
OUT_DIR="${OUT_DIR:-data/dialogsum/exp4.3/run_dialogsum_recipe}"

STEPS="${STEPS:-200}"
BATCH="${BATCH:-4}"
ANCHOR_SIZE="${ANCHOR_SIZE:-16}"
ANCHOR_REFRESH="${ANCHOR_REFRESH:-20}"
ANCHOR_CHUNK="${ANCHOR_CHUNK:-8}"
MAX_LEN="${MAX_LEN:-384}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_CAP_DIM="${MAX_CAP_DIM:-128}"

LR="${LR:-2e-4}"
ETA_BETA="${ETA_BETA:-0.1}"
GAMMA_ALPHA="${GAMMA_ALPHA:-5.0}"
EPSILON="${EPSILON:-0.05}"
GAMMA_T="${GAMMA_T:-1.0}"
ANCHOR_EMA_MOMENTUM="${ANCHOR_EMA_MOMENTUM:-0.8}"
PRUNE_PATIENCE="${PRUNE_PATIENCE:-3}"
PRUNE_REWARD_THRESHOLD="${PRUNE_REWARD_THRESHOLD:--0.05}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
DRY_RUN="${DRY_RUN:-0}"

CMD=(
  python src/4.3/run.py
  --train-jsonl data/dialogsum/train.jsonl
  --score-path data/dialogsum/score/pdm_scored.jsonl
  --score-path data/dialogsum/score/delta_improved_mapped.jsonl
  --score-path data/dialogsum/score/srm_from_topk5_only.jsonl
  --score-path data/dialogsum/score/alpagasus_improved_mapped.jsonl
  --top-k-path data/dialogsum/train_coarse_topk5.jsonl
  --output-dir "$OUT_DIR"
  --base-model "$BASE_MODEL"
  --num-steps "$STEPS"
  --batch-size "$BATCH"
  --anchor-size-per-capability "$ANCHOR_SIZE"
  --anchor-refresh-interval "$ANCHOR_REFRESH"
  --anchor-chunk-size "$ANCHOR_CHUNK"
  --max-length "$MAX_LEN"
  --lr "$LR"
  --eta-beta "$ETA_BETA"
  --gamma-alpha "$GAMMA_ALPHA"
  --epsilon "$EPSILON"
  --gamma-T "$GAMMA_T"
  --anchor-ema-momentum "$ANCHOR_EMA_MOMENTUM"
  --prune-patience "$PRUNE_PATIENCE"
  --prune-reward-threshold "$PRUNE_REWARD_THRESHOLD"
  --mapper-utility-mode beta_weighted
  --layer-name auto
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lora-dropout "$LORA_DROPOUT"
  --log-every 5
  --save-every 50
  --seed 42
)

if [[ "$MAX_SAMPLES" -gt 0 ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$MAX_CAP_DIM" -gt 0 ]]; then
  CMD+=(--max-capability-dim "$MAX_CAP_DIM")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(--dry-run)
fi

"${CMD[@]}"
