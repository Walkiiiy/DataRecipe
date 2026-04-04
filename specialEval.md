python src/4.1/EXP/eval_checkpoints_em.py   --eval-source-jsonl data/lawyer/test.jsonl   --eval-size 1446   --base_model Qwen/Qwen2.5-0.5B   --model_source modelscope   --run delta_origin::data/lawyer/exp4.2/run_delta_origin_2000_shared_eval/checkpoint-440

python src/4.1/EXP/eval_checkpoints_em.py   --eval-source-jsonl data/lawyer/test.jsonl   --eval-size 1446   --base_model Qwen/Qwen2.5-0.5B   --model_source modelscope   --run random::data/lawyer/exp4.2/run_random_2000_shared_eval/checkpoint-480

python src/4.1/EXP/eval_checkpoints_em.py   --eval-source-jsonl data/dolly/test.jsonl   --eval-size 1446   --base_model Qwen/Qwen2.5-0.5B   --model_source modelscope   --run alpagasus_origin::data/dolly/exp4.2/run_alpagasus_origin_650_shared_eval/checkpoint-39

python src/4.1/EXP/eval_checkpoints_em.py   --eval-source-jsonl data/dolly/test.jsonl   --eval-size 1446   --base_model Qwen/Qwen2.5-0.5B   --model_source modelscope   --run alpagasus_origin::data/dolly/exp4.2/run_alpagasus_origin_1300_shared_eval/checkpoint-27