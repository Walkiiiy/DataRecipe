python src/4.1/EXP/eval_checkpoints_em.py   --eval-source-jsonl data/lawyer/test.jsonl   --eval-size 1446   --base_model Qwen/Qwen2.5-0.5B   --model_source modelscope   --run re::



STEPS=70 BATCH=8 bash src/4.3/run.sh



python src/4.1/EXP/eval_checkpoints_em.py \
  --eval-source-jsonl data/dialogsum/test.jsonl \
  --eval-size 1446 \
  --base_model Qwen/Qwen2.5-0.5B \
  --model_source modelscope \
  --run delta_origin::data/dialogsum/exp4.3/run_dialogsum_recipe_700/final