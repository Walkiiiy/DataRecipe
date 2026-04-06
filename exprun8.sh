STEPS=130 DATASET=dolly BATCH=4 OUT_DIR=data/dolly/exp4.3.1/run_dolly_recipe_130 bash  src/4.3/run.sh


# 环境变量设置
BASE_MODEL="Qwen/Qwen2.5-0.5B" \
DATASET="dolly" \
OUT_DIR="data/dolly/exp4.3.1/run_dolly_recipe" \
BATCH=4 \
GRAD_ACCUM=4 \
LR=1e-4 \
ANCHOR_SIZE=32 \
ANCHOR_CHUNK=4 \
ANCHOR_REFRESH=30 \
MAX_LEN=384 \
GAMMA_T=0.8 \
STEPS=1300 \
bash src/4.3/run.sh


########################################################
DATASET="dolly" OUT_DIR="data/dolly/exp4.4.0/run_dolly_recipe_final" BATCH=16 GRAD_ACCUM=4 VMAP_CHUNK=4 LR=1e-4 ANCHOR_SIZE=32 ANCHOR_CHUNK=4  ANCHOR_REFRESH=20  MAX_LEN=384 GAMMA_T=0.8 STEPS=1300 EMA_MOMENTUM=0.9 FREQ_PENALTY=0.1 PRUNE_PATIENCE=3 bash src/4.3/run.sh