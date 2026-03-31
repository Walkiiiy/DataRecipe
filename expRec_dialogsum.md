# 能力树
- 重构4.1为模型聚类
- python3 /home/walkiiiy/DataRecipe/src/4.1/llm_capability_tree_builder.py \
  --input-jsonl /home/walkiiiy/DataRecipe/data/dialogsum/train.jsonl \
  --api-key "sk-ab412f420cd540888da4732a35600c4a" \
  --base-url https://api.deepseek.com \
  --model deepseek-chat \
  --max-samples 300 \
  --print-tree-every  1 \
  --patience-structure-stable 40 \
  --max-leaf-nodes 100 \
  --min-node-size-for-split  4 \
  --stale-singleton-rounds 30

--print-tree-every（默认 20，0 表示不周期打印树）
--tree-print-max-lines（默认 120，0 表示不截断）
--patience-structure-stable（默认 0，关闭早停）
--min-node-size-for-split
--max-leaf-nodes 


- 预处理：用簇心向量对每一条数据寻找topk相似
python3 src/4.2/stage_2_top_k_routing.py \
  --dataset_path data/dialogsum/train.jsonl \
  --centroids_path data/dialogsum/capability_tree_final.json \
  --centroid_source auto \
  --model_name auto \
  --top_k 5 \
  --output_path data/dialogsum/train_coarse_topk5.jsonl \
  --output_centroid_manifest data/dialogsum/centroid_manifest.json




export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
# 评分
- SRM
- python3 src/4.2/SRM/SRM.py \
  --input_path data/dialogsum/train_coarse_topk5.jsonl \
  --data_path data/dialogsum/train.jsonl \
  --api_key "sk-ab412f420cd540888da4732a35600c4a" \
  --output_path data/dialogsum/score/srm_from_topk5_only.jsonl \
  --model_name deepseek-chat \
  --include_debug_fields


python3 /home/walkiiiy/DataRecipe/src/4.2/SRM/srm_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/dialogsum/score/srm_from_topk5_only.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/dialogsum/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/dialogsum/exp4.2/dataset_srm.jsonl \
  --num_samples 1000

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run srm::data/dialogsum/exp4.2/dataset_srm.jsonl::data/dialogsum/exp4.2/run_srm_shared_eval \
  --eval-source-jsonl data/dialogsum/train.jsonl \
  --eval-ratio 0.01 \
  --seed 42 \
  --num_train_epochs 6 \
  --output-root data/dialogsum/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct 

------------------------------------------------------------------------------------------------
- delta_improved（用 train_coarse_topk5.jsonl 的 top5 names 构 prompt，输出统一 mapped_vector）
python3 src/4.2/delta/delta_improved.py \
  --data_path data/banking77/train.jsonl \
  --routing_path data/banking77/train_coarse_topk5.jsonl \
  --output_path data/banking77/score/delta_improved_mapped.jsonl \
  --concurrency 16 \
  --quality_mode batch \
  --cluster_batch_size 5

python3 /home/walkiiiy/DataRecipe/src/4.2/delta/delta_improved_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/banking77/score/delta_improved_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/banking77/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/banking77/selected/dataset_delta_improved.jsonl \
  --meta_output_path /home/walkiiiy/DataRecipe/data/banking77/selected/delta_improved_selected_3000_meta.json \
  --num_samples 1000

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run srm::data/banking77/exp4.2/dataset_delta_improved.jsonl::data/banking77/exp4.2/run_delta_improved_shared_eval \
  --eval-source-jsonl data/banking77/train.jsonl \
  --eval-ratio 0.01 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/banking77/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct 
============================================================================

- delta_origin（输出统一 mapped_vector）
python3 src/4.2/delta/delta_origin.py \
  --data_path data/banking77/train.jsonl \
  --routing_path data/banking77/train_coarse_topk5.jsonl \
  --output_path data/banking77/score/delta_origin_mapped.jsonl \
  --concurrency 16 \
  --turn_aggregation sum \
  --routing_weight_mode coarse

python3 /home/walkiiiy/DataRecipe/src/4.2/delta/delta_origin_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/banking77/score/delta_origin_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/banking77/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/banking77/exp4.2/dataset_delta_origin.jsonl \
  --meta_output_path /home/walkiiiy/DataRecipe/data/banking77/selected/delta_origin_selected_3000_meta.json \
  --num_samples 1000

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run srm::data/banking77/exp4.2/dataset_delta_origin.jsonl::data/banking77/exp4.2/run_delta_origin_shared_eval \
  --eval-source-jsonl data/banking77/train.jsonl \
  --eval-ratio 0.01 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/banking77/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct 
=========================================================================================


-  alpagasus_origin：原始标量评分（不做能力簇评分）
python3 src/4.2/alpagasus/alpagasus_origin.py \
  --data_path data/banking77/train.jsonl \
  --output_path data/banking77/score/alpagasus_origin_scored.jsonl \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --temperature 0.01 \
  --max_tokens 200 \
  --concurrency 16



python3 /home/walkiiiy/DataRecipe/src/4.2/alpagasus/alpagasus_origin_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/banking77/score/alpagasus_origin_scored.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/banking77/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/banking77/exp4.2/dataset_alpagasus_origin.jsonl \
  --meta_output_path /home/walkiiiy/DataRecipe/data/banking77/selected/alpagasus_origin_selected_3000_meta.json \
  --num_samples 1000

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run srm::data/banking77/exp4.2/dataset_alpagasus_origin.jsonl::data/banking77/exp4.2/run_alpagasus_origin_shared_eval \
  --eval-source-jsonl data/banking77/train.jsonl \
  --eval-ratio 0.01 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/banking77/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct 
==========================================================================================



- alpagasus_improved：保持原评分逻辑 + 映射到能力维向量
python3 src/4.2/alpagasus/alpagasus_improved.py \
  --data_path data/banking77/train.jsonl \
  --routing_path data/banking77/train_coarse_topk5.jsonl \
  --output_path data/banking77/score/alpagasus_improved_mapped.jsonl \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --temperature 0.01 \
  --max_tokens 200 \
  --routing_weight_mode coarse \
  --concurrency 16


python3 /home/walkiiiy/DataRecipe/src/4.2/alpagasus/alpagasus_improved_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/banking77/score/alpagasus_improved_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/banking77/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/banking77/exp4.2/dataset_alpagasus_improved.jsonl \
  --meta_output_path /home/walkiiiy/DataRecipe/data/banking77/selected/alpagasus_improved_selected_3000_meta.json \
  --num_samples 1000


python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run srm::data/banking77/exp4.2/dataset_alpagasus_improved.jsonl::data/banking77/exp4.2/run_alpagasus_improved_shared_eval \
  --eval-source-jsonl data/banking77/train.jsonl \
  --eval-ratio 0.01 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/banking77/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct 

================================================================================
- instag
python3 src/4.2/instag/instag.py \
  --data_path data/banking77/train.jsonl \
  --output_path data/banking77/score/instag_tags.jsonl \
  --model deepseek-chat \
  --concurrency 16


采样
python3 src/4.2/instag/instag_sample.py \
  --data_path data/banking77/train.jsonl \
  --tag_path data/banking77/score/instag_tags.jsonl \
  --output_path data/banking77/exp4.2/dataset_instag.jsonl \
  --num_samples 1000

- MIG
- python3 src/4.2/mig/mig.py \
  --data_path data/banking77/train.jsonl \
  --instag_path data/banking77/score/instag_tags.jsonl \
  --delta_path data/banking77/score/delta_origin_mapped.jsonl \
  --output_path data/banking77/score/mig_scored.jsonl \
  --cluster_output_path data/banking77/score/mig_tag_clusters.json \
  --valid_tag_output_path data/banking77/score/mig_valid_tags.json \
  --tag_merge_eps 0.05 \
  --delta_scalar_mode auto \
  --embedding_backend auto

采样
- python3 src/4.2/mig/mig_sample.py \
  --scored_path data/banking77/score/mig_scored.jsonl \
  --output_path data/banking77/exp4.2/dataset_mig.jsonl \
  --num_samples 1000 \
  --meta_output_path data/banking77/score/mig_sample_meta.json \
  --embedding_backend auto



- 先对这两个跑实验
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run instag::data/banking77/exp4.2/dataset_instag.jsonl::data/banking77/exp4.2/run_instag_shared_eval \
  --run mig::data/banking77/exp4.2/dataset_mig.jsonl::data/banking77/exp4.2/run_mig_shared_eval \
  --eval-source-jsonl data/banking77/train.jsonl \
  --eval-ratio 0.01 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/banking77/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \

============================================================================

- datawhisperer
python3 src/4.2/datawhisperer/datawhisperer.py \
  --data_path data/banking77/train.jsonl \
  --output_path data/banking77/score/datawhisperer_icl_mapped.jsonl \
  --model_name_or_path ~/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
  --n_demonstrations 10 \
  --n_queries 5 \
  --n_iterations 10 \
  --attention_layer 13 \
  --icl_metric accuracy \
  --concurrency 2 \
  --max_new_tokens 16 \
  --max_input_tokens 2048 \
  --select_top_k 1000

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run instag::data/banking77/exp4.2/dataset_datawhisperer.jsonl::data/banking77/exp4.2/run_datawhisperer_shared_eval \
  --eval-source-jsonl data/banking77/train.jsonl \
  --eval-ratio 0.01 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/banking77/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct

===============================================================================================
- 随机采样
python src/scripts/sample_train_subset.py \
  --input-jsonl data/dialogsum/train.jsonl \
  --output-jsonl data/dialogsum/exp4.2/dataset_random.jsonl \
  --sample-size 1000 \
  --seed 42

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run instag::data/banking77/exp4.2/dataset_random.jsonl::data/banking77/exp4.2/run_random_shared_eval \
  --eval-source-jsonl data/banking77/train.jsonl \
  --eval-ratio 0.01 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/banking77/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct


  --sample-ratio 0.1 \


# EM评估
python src/4.1/EXP/eval_checkpoints_em.py \
  --eval-source-jsonl data/banking77/train.jsonl \
  --eval-size 500 \
  --base_model Qwen/Qwen2.5-0.5B \
  --model_source modelscope \
  --run ours::data/banking77/exp4.2/run_ours_shared_eval/final_checkpoint \
  --run random::data/banking77/exp4.2/run_random_shared_eval/final_checkpoint \
  --run instag::data/banking77/exp4.2/run_instag_shared_eval/final_checkpoint \
  --run datawhisperer::data/banking77/exp4.2/run_datawhisperer_shared_eval/final_checkpoint \
  --run delta_origin::data/banking77/exp4.2/run_delta_origin_shared_eval/final_checkpoint \
  --run delta_improved::data/banking77/exp4.2/run_delta_improved_shared_eval/final_checkpoint \
  --run delta_origin::data/banking77/exp4.2/run_delta_origin_shared_eval/final_checkpoint \
  --run alpagasus_improved::data/banking77/exp4.2/run_alpagasus_improved_shared_eval/final_checkpoint \
  --run alpagasus_origin::data/banking77/exp4.2/run_alpagasus_origin_shared_eval/final_checkpoint \
  --output-dir data/banking77/exp4.2/em_eval
