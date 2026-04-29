# train 重新划分
python3 src/scripts/split_train_test_random.py \
  --train-jsonl data/lawyer/train.jsonl


# 能力树
- 重构4.1为模型聚类
- python3 /home/walkiiiy/DataRecipe/src/4.1/llm_capability_tree_builder.py \
  --input-jsonl /home/walkiiiy/DataRecipe/data/lawyer/train.jsonl \
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
  --dataset_path data/lawyer/train.jsonl \
  --centroids_path data/lawyer/capability_tree_final.json \
  --centroid_source auto \
  --model_name auto \
  --top_k 5 \
  --output_path data/lawyer/train_coarse_topk5.jsonl \
  --output_centroid_manifest data/lawyer/centroid_manifest.json




export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
# 评分
# SRM
- python3 src/4.2/SRM/SRM.py \
  --input_path data/lawyer/train_coarse_topk5.jsonl \
  --data_path data/lawyer/train.jsonl \
  --api_key "sk-ab412f420cd540888da4732a35600c4a" \
  --output_path data/lawyer/score/srm_from_topk5_only.jsonl \
  --model_name deepseek-chat 
  
  --include_debug_fields


python3 /home/walkiiiy/DataRecipe/src/4.2/SRM/srm_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/lawyer/score/srm_from_topk5_only.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/lawyer/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/lawyer/exp4.2/dataset_srm_1000.jsonl \
  --num_samples 1000



------------------------------------------------------------------------------------------------
# delta_improved（用 train_coarse_topk5.jsonl 的 top5 names 构 prompt，输出统一 mapped_vector）
python3 src/4.2/delta/delta_improved.py \
  --data_path data/lawyer/train.jsonl \
  --routing_path data/lawyer/train_coarse_topk5.jsonl \
  --output_path data/lawyer/score/delta_improved_mapped.jsonl \
  --concurrency 16 \
  --quality_mode batch \
  --cluster_batch_size 5

python3 /home/walkiiiy/DataRecipe/src/4.2/delta/delta_improved_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/lawyer/score/delta_improved_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/lawyer/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/lawyer/exp4.2/dataset_delta_improved_1000.jsonl \
  --num_samples 1000

============================================================================

# delta_origin（输出统一 mapped_vector）
python3 src/4.2/delta/delta_origin.py \
  --data_path data/lawyer/train.jsonl \
  --routing_path data/lawyer/train_coarse_topk5.jsonl \
  --output_path data/lawyer/score/delta_origin_mapped.jsonl \
  --concurrency 16 \
  --turn_aggregation sum \
  --routing_weight_mode coarse

python3 /home/walkiiiy/DataRecipe/src/4.2/delta/delta_origin_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/lawyer/score/delta_origin_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/lawyer/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/lawyer/exp4.2/dataset_delta_origin_1000.jsonl \
  --num_samples 1000

=========================================================================================


#  alpagasus_origin：原始标量评分（不做能力簇评分）
python3 src/4.2/alpagasus/alpagasus_origin.py \
  --data_path data/lawyer/train.jsonl \
  --output_path data/lawyer/score/alpagasus_origin_scored.jsonl \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --temperature 0.01 \
  --max_tokens 1000 \
  --concurrency 16



python3 /home/walkiiiy/DataRecipe/src/4.2/alpagasus/alpagasus_origin_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/lawyer/score/alpagasus_origin_scored.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/lawyer/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/lawyer/exp4.2/dataset_alpagasus_origin_1000.jsonl \
  --num_samples 1000

==========================================================================================



#  alpagasus_improved：保持原评分逻辑 + 映射到能力维向量
python3 src/4.2/alpagasus/alpagasus_improved.py \
  --data_path data/lawyer/train.jsonl \
  --routing_path data/lawyer/train_coarse_topk5.jsonl \
  --output_path data/lawyer/score/alpagasus_improved_mapped.jsonl \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --temperature 0.01 \
  --max_tokens 1000 \
  --routing_weight_mode coarse \
  --concurrency 16

python3 /home/walkiiiy/DataRecipe/src/4.2/alpagasus/alpagasus_improved_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/lawyer/score/alpagasus_improved_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/lawyer/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/lawyer/exp4.2/dataset_alpagasus_improved_1000.jsonl \
  --num_samples 1000
================================================================================
# instag
python3 src/4.2/instag/instag.py \
  --data_path data/lawyer/train.jsonl \
  --output_path data/lawyer/score/instag_tags.jsonl \
  --model deepseek-chat \
  --concurrency 16


采样
python3 src/4.2/instag/instag_sample.py \
  --data_path data/lawyer/train.jsonl \
  --tag_path data/lawyer/score/instag_tags.jsonl \
  --output_path data/lawyer/exp4.2/dataset_instag_1000.jsonl \
  --num_samples 1000
=======================================================
# MIG
- python3 src/4.2/mig/mig.py \
  --data_path data/lawyer/train.jsonl \
  --instag_path data/lawyer/score/instag_tags.jsonl \
  --delta_path data/lawyer/score/delta_origin_mapped.jsonl \
  --output_path data/lawyer/score/mig_scored.jsonl \
  --cluster_output_path data/lawyer/score/mig_tag_clusters.json \
  --valid_tag_output_path data/lawyer/score/mig_valid_tags.json \
  --tag_merge_eps 0.05 \
  --delta_scalar_mode auto \
  --embedding_backend auto

采样
- python3 src/4.2/mig/mig_sample.py \
  --scored_path data/lawyer/score/mig_scored.jsonl \
  --output_path data/lawyer/exp4.2/dataset_mig_2000.jsonl \
  --num_samples 2000 




============================================================================

# datawhisperer
python3 src/4.2/datawhisperer/datawhisperer.py \
  --data_path data/lawyer/train.jsonl \
  --output_path data/lawyer/score/datawhisperer_icl_mapped.jsonl \
  --model_name_or_path ~/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
  --n_demonstrations 10 \
  --n_queries 5 \
  --n_iterations 10 \
  --attention_layer 13 \
  --icl_metric accuracy \
  --concurrency 2 \
  --max_new_tokens 16 \
  --max_input_tokens 2048 \
  --select_top_k 2000

==============================================================
# PDM
python src/4.2/PDM/pdm.py \
  --data_path data/lawyer/train.jsonl \
  --srm_path data/lawyer/score/srm_from_topk5_only.jsonl \
  --output_path data/lawyer/score/pdm_scored.jsonl \
  --model_name_or_path ~/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B \
  --context_size 4 \
  --global_trials 3 \
  --max_seq_len 1024 \
  --nll_batch_size 8 \
  --no-cache_token_ids \
  --device_map auto


python src/4.2/PDM/pdm_sample.py \
  --score_path data/lawyer/score/pdm_scored.jsonl \
  --data_path data/lawyer/train.jsonl \
  --output_path data/lawyer/exp4.2/dataset_pdm_1000.jsonl \
  --num_samples 1000 \
  --annotate_selection
===============================================================================================
# 随机采样
python src/scripts/sample_train_subset.py \
  --input-jsonl data/lawyer/train.jsonl \
  --output-jsonl data/lawyer/exp4.2/dataset_random_1000.jsonl \
  --sample-size 1000 \
  --seed 42



# 训练
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --eval-source-jsonl data/lawyer/test.jsonl \
  --eval-ratio 1 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/lawyer/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B \
  --run alpagasus_origin::data/lawyer/exp4.2/dataset_alpagasus_origin_1000.jsonl::data/lawyer/exp4.2/run_alpagasus_origin_1000_shared_eval 

  --run delta_improved::data/lawyer/exp4.2/dataset_delta_improved_1000.jsonl::data/lawyer/exp4.2/run_delta_improved_1000_shared_eval 
  

  <!-- --base_model Qwen/Qwen3-4B-Instruct-2507   JunHowie/Qwen3-8B-Instruct -->

--run delta_origin::data/lawyer/exp4.2/dataset_delta_origin_1000.jsonl::data/lawyer/exp4.2/run_delta_origin_1000_shared_eval \
  
  --run srm::data/lawyer/exp4.2/dataset_srm_1000.jsonl::data/lawyer/exp4.2/run_srm_1000_shared_eval \
  --run random::data/lawyer/exp4.2/dataset_random_1000.jsonl::data/lawyer/exp4.2/run_random_1000_shared_eval \




# EM评估
python src/4.1/EXP/eval_checkpoints_em.py \
  --eval-source-jsonl data/lawyer/test.jsonl \
  --eval-size 1446 \
  --base_model Qwen/Qwen2.5-0.5B \
  --model_source modelscope \
  --run alpagasus_origin::data/lawyer/exp4.2/run_alpagasus_origin_1000_shared_eval/final_checkpoint 
 
 
  --run delta_improved::data/lawyer/exp4.2/run_delta_improved_1000_shared_eval/final_checkpoint 

  --run delta_origin::data/lawyer/exp4.2/run_delta_origin_1000_shared_eval/final_checkpoint 


  --run random::data/lawyer/exp4.2/run_random_1000_shared_eval/final_checkpoint \
  --run srm::data/lawyer/exp4.2/run_srm_1000_shared_eval/final_checkpoint



 


  --run delta_improved::data/lawyer/exp4.2/run_delta_improved_1000_shared_eval/final_checkpoint \

  run_delta_origin_1000_shared_eval/final_checkpoint \
  --run alpagasus_origin::data/lawyer/exp4.2/
  run_alpagasus_origin_1000_shared_eval/final_checkpoint \
  --output-dir data/lawyer/exp4.2/em_eval


# bert-score
python src/4.1/EXP/eval_checkpoints_bertscore.py \
  --eval-source-jsonl data/lawyer/train.jsonl \
  --eval-size 500 \
  --bertscore-lang-mode auto \
  --run srm::data/lawyer/exp4.2/run_srm_1000_shared_eval/final_checkpoint \
  --run random::data/lawyer/exp4.2/run_random_1000_shared_eval/final_checkpoint \
  --output-dir data/lawyer/exp4.2/bertscore_eval \
  --save-per-run-jsonl
