# train 重新划分
python3 src/scripts/split_train_test_random.py \
  --train-jsonl data/magicoder/train.jsonl


# 能力树
- 重构4.1为模型聚类
- python3 /home/walkiiiy/DataRecipe/src/4.1/llm_capability_tree_builder.py \
  --input-jsonl /home/walkiiiy/DataRecipe/data/magicoder/train.jsonl \
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
  --dataset_path data/magicoder/train.jsonl \
  --centroids_path data/magicoder/capability_tree_final.json \
  --centroid_source auto \
  --model_name auto \
  --top_k 5 \
  --output_path data/magicoder/train_coarse_topk5.jsonl \
  --output_centroid_manifest data/magicoder/centroid_manifest.json




export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
# 评分
# SRM
- python3 src/4.2/SRM/SRM.py \
  --input_path data/magicoder/train_coarse_topk5.jsonl \
  --data_path data/magicoder/train.jsonl \
  --api_key "sk-ab412f420cd540888da4732a35600c4a" \
  --output_path data/magicoder/score/srm_from_topk5_only.jsonl \
  --model_name deepseek-chat \
  --include_debug_fields


python3 /home/walkiiiy/DataRecipe/src/4.2/SRM/srm_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/magicoder/score/srm_from_topk5_only.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/magicoder/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/magicoder/exp4.2/dataset_srm_700.jsonl \
  --num_samples 700



------------------------------------------------------------------------------------------------
# delta_improved（用 train_coarse_topk5.jsonl 的 top5 names 构 prompt，输出统一 mapped_vector）
python3 src/4.2/delta/delta_improved.py \
  --data_path data/magicoder/train.jsonl \
  --routing_path data/magicoder/train_coarse_topk5.jsonl \
  --output_path data/magicoder/score/delta_improved_mapped.jsonl \
  --concurrency 16 \
  --quality_mode batch \
  --cluster_batch_size 5

python3 /home/walkiiiy/DataRecipe/src/4.2/delta/delta_improved_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/magicoder/score/delta_improved_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/magicoder/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/magicoder/exp4.2/dataset_delta_improved_700.jsonl \
  --num_samples 700

============================================================================

# delta_origin（输出统一 mapped_vector）
python3 src/4.2/delta/delta_origin.py \
  --data_path data/magicoder/train.jsonl \
  --routing_path data/magicoder/train_coarse_topk5.jsonl \
  --output_path data/magicoder/score/delta_origin_mapped.jsonl \
  --concurrency 16 \
  --turn_aggregation sum \
  --routing_weight_mode coarse

python3 /home/walkiiiy/DataRecipe/src/4.2/delta/delta_origin_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/magicoder/score/delta_origin_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/magicoder/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/magicoder/exp4.2/dataset_delta_origin_700.jsonl \
  --num_samples 700

=========================================================================================


#  alpagasus_origin：原始标量评分（不做能力簇评分）
python3 src/4.2/alpagasus/alpagasus_origin.py \
  --data_path data/magicoder/train.jsonl \
  --output_path data/magicoder/score/alpagasus_origin_scored.jsonl \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --temperature 0.01 \
  --max_tokens 200 \
  --concurrency 16



python3 /home/walkiiiy/DataRecipe/src/4.2/alpagasus/alpagasus_origin_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/magicoder/score/alpagasus_origin_scored.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/magicoder/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/magicoder/exp4.2/dataset_alpagasus_origin_700.jsonl \
  --num_samples 700

==========================================================================================



#  alpagasus_improved：保持原评分逻辑 + 映射到能力维向量
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


================================================================================
# instag
python3 src/4.2/instag/instag.py \
  --data_path data/magicoder/train.jsonl \
  --output_path data/magicoder/score/instag_tags.jsonl \
  --model deepseek-chat \
  --concurrency 16


采样
python3 src/4.2/instag/instag_sample.py \
  --data_path data/banking77/train.jsonl \
  --tag_path data/banking77/score/instag_tags.jsonl \
  --output_path data/banking77/exp4.2/dataset_instag.jsonl \
  --num_samples 700
=======================================================
# MIG
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
  --num_samples 700 \
  --meta_output_path data/banking77/score/mig_sample_meta.json \
  --embedding_backend auto




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
  --select_top_k 700


===============================================================================================
# 随机采样
python src/scripts/sample_train_subset.py \
  --input-jsonl data/magicoder/train.jsonl \
  --output-jsonl data/magicoder/exp4.2/dataset_random_700.jsonl \
  --sample-size 700 \
  --seed 42



# 训练
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run srm::data/magicoder/exp4.2/dataset_srm_700.jsonl::data/magicoder/exp4.2/run_srm_700_shared_eval \
  --run random::data/magicoder/exp4.2/dataset_random_700.jsonl::data/magicoder/exp4.2/run_random_700_shared_eval \
  --eval-source-jsonl data/magicoder/test.jsonl \
  --eval-ratio 1 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/magicoder/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B

  <!-- --base_model Qwen/Qwen3-4B-Instruct-2507   JunHowie/Qwen3-8B-Instruct -->

  --run delta_improved::data/magicoder/exp4.2/dataset_delta_improved_700.jsonl::data/magicoder/exp4.2/run_delta_improved_700_shared_eval \
  --run delta_origin::data/magicoder/exp4.2/dataset_delta_origin_700.jsonl::data/magicoder/exp4.2/run_delta_origin_700_shared_eval \
  --run alpagasus_origin::data/magicoder/exp4.2/dataset_alpagasus_origin_700.jsonl::data/magicoder/exp4.2/run_alpagasus_origin_700_shared_eval \


# EM评估
python src/4.1/EXP/eval_checkpoints_em.py \
  --eval-source-jsonl data/magicoder/test.jsonl \
  --eval-size 1500 \
  --base_model Qwen/Qwen2.5-0.5B \
  --model_source modelscope \
  --run srm::data/magicoder/exp4.2/run_srm_700_shared_eval/final_checkpoint \
  --run random::data/magicoder/exp4.2/run_random_700_shared_eval/final_checkpoint 


  --run delta_origin::data/magicoder/exp4.2/run_delta_origin_700_shared_eval/final_checkpoint \
  --run alpagasus_origin::data/magicoder/exp4.2/run_alpagasus_origin_700_shared_eval/final_checkpoint 



  --run delta_improved::data/magicoder/exp4.2/run_delta_improved_700_shared_eval/final_checkpoint \







  run_delta_origin_700_shared_eval/final_checkpoint \
  --run alpagasus_origin::data/magicoder/exp4.2/
  run_alpagasus_origin_700_shared_eval/final_checkpoint \
  --output-dir data/magicoder/exp4.2/em_eval



