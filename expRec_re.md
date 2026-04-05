python src/4.1/EXP/eval_checkpoints_em.py   --eval-source-jsonl data/dialogsum/test.jsonl   --eval-size 1446   --base_model Qwen/Qwen2.5-0.5B   --model_source modelscope   --run re::data/dialogsum/exp4.3/run_dialogsum_recipe/checkpoint-200 



STEPS=70 BATCH=8 bash src/4.3/run.sh