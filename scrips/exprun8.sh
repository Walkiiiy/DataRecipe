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



DATASET="dolly" \
OUT_DIR="data/dolly/exp4.4.0/run_dolly_recipe" \
BATCH=4 \
GRAD_ACCUM=2 \
MAX_LEN=256 \
ANCHOR_SIZE=8 \
ANCHOR_REFRESH=20 \
ANCHOR_CHUNK=8 \
MAX_CAP_DIM=64 \
bash src/4.3/run.sh




DATASET="dolly" OUT_DIR="data/dolly/exp4.4.0/run_dolly_recipe" BATCH=4 STEPS=1300 GRAD_ACCUM=2 MAX_LEN=256 ANCHOR_SIZE=8 ANCHOR_REFRESH=20 ANCHOR_CHUNK=8 MAX_CAP_DIM=64 bash src/4.3/run.sh


DATASET="dialogsum" OUT_DIR="data/dialogsum/exp4.4.0/run_dialogsum_recipe" BATCH=4 STEPS=1000 GRAD_ACCUM=2 MAX_LEN=256 ANCHOR_SIZE=8 ANCHOR_REFRESH=20 ANCHOR_CHUNK=8 MAX_CAP_DIM=64 bash src/4.3/run.sh


DATASET="lawyer" OUT_DIR="data/lawyer/exp4.4.0/run_lawyer_recipe" BATCH=4 STEPS=2000 GRAD_ACCUM=2 MAX_LEN=256 ANCHOR_SIZE=8 ANCHOR_REFRESH=20 ANCHOR_CHUNK=8 MAX_CAP_DIM=64 bash src/4.3/run.sh


###########################################################################
BATCH=4  GRAD_ACCUM=8  GAMMA_ALPHA=100.0 GAMMA_T=0.5  


DATASET="dolly" OUT_DIR="data/dolly/exp4.4.0/run_dolly_recipe" BATCH=4 STEPS=1300 GRAD_ACCUM=8  GAMMA_ALPHA=100.0 GAMMA_T=0.5   MAX_LEN=256 ANCHOR_SIZE=8 ANCHOR_REFRESH=20 ANCHOR_CHUNK=8 MAX_CAP_DIM=64 bash src/4.3/run.sh


DATASET="dialogsum" OUT_DIR="data/dialogsum/exp4.4.0/run_dialogsum_recipe" BATCH=4 STEPS=1000 GRAD_ACCUM=8  GAMMA_ALPHA=100.0 GAMMA_T=0.5   MAX_LEN=256 ANCHOR_SIZE=8 ANCHOR_REFRESH=20 ANCHOR_CHUNK=8 MAX_CAP_DIM=64 bash src/4.3/run.sh


DATASET="lawyer" OUT_DIR="data/lawyer/exp4.4.0/run_lawyer_recipe" BATCH=4 STEPS=2000 GRAD_ACCUM=8  GAMMA_ALPHA=100.0 GAMMA_T=0.5   MAX_LEN=256 ANCHOR_SIZE=8 ANCHOR_REFRESH=20 ANCHOR_CHUNK=8 MAX_CAP_DIM=64 bash src/4.3/run.sh




# 帮我在主目录写一个消融实验的脚本，测试除了四个评分文件全用以外的所有评分文件的组合（只用一个文件的就不用了，用两个和用三个的所有组合都要测试，一共是10种组合）。训练和评估用和之前一样的设置，但是不用每个数据集取三种不同的数据量了，dolly的epoch1-target-size为1300，dialogsum的epoch1-target-size为1000，lawyer为2000.



# 