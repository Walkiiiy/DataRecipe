# train 重新划分
python3 src/scripts/split_train_test_random.py \
  --train-jsonl data/dolly/train.jsonl


# 能力树
- 重构4.1为模型聚类
- python3 /home/walkiiiy/DataRecipe/src/4.1/llm_capability_tree_builder.py \
  --input-jsonl /home/walkiiiy/DataRecipe/data/dolly/train.jsonl \
  --api-key "sk-ab412f420cd540888da4732a35600c4a" \
  --base-url https://api.deepseek.com \
  --model deepseek-chat \
  --max-samples 500 \
  --print-tree-every  1 \
  --patience-structure-stable 40 \
  --min-node-size-for-split  4 \
  --stale-singleton-rounds 50

--print-tree-every（默认 20，0 表示不周期打印树）
--tree-print-max-lines（默认 120，0 表示不截断）
--patience-structure-stable（默认 0，关闭早停）
--min-node-size-for-split
--max-leaf-nodes 


- 预处理：用簇心向量对每一条数据寻找topk相似
python3 src/4.2/stage_2_top_k_routing.py \
  --dataset_path data/dolly/train.jsonl \
  --centroids_path data/dolly/capability_tree_final.json \
  --centroid_source auto \
  --model_name auto \
  --top_k 5 \
  --output_path data/dolly/train_coarse_topk5.jsonl \
  --output_centroid_manifest data/dolly/centroid_manifest.json




export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
# 评分
# SRM
- python3 src/4.2/SRM/SRM.py \
  --input_path data/dolly/train_coarse_topk5.jsonl \
  --data_path data/dolly/train.jsonl \
  --api_key "sk-ab412f420cd540888da4732a35600c4a" \
  --output_path data/dolly/score/srm_from_topk5_only.jsonl \
  --model_name deepseek-chat 
  
  --include_debug_fields


python3 /home/walkiiiy/DataRecipe/src/4.2/SRM/srm_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/dolly/score/srm_from_topk5_only.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/dolly/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/dolly/exp4.2/dataset_srm_130.jsonl \
  --num_samples 130



------------------------------------------------------------------------------------------------
# delta_improved（用 train_coarse_topk5.jsonl 的 top5 names 构 prompt，输出统一 mapped_vector）
python3 src/4.2/delta/delta_improved.py \
  --data_path data/dolly/train.jsonl \
  --routing_path data/dolly/train_coarse_topk5.jsonl \
  --output_path data/dolly/score/delta_improved_mapped.jsonl \
  --concurrency 32 \
  --quality_mode batch \
  --cluster_batch_size 5

python3 /home/walkiiiy/DataRecipe/src/4.2/delta/delta_improved_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/dolly/score/delta_improved_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/dolly/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/dolly/exp4.2/dataset_delta_improved_130.jsonl \
  --num_samples 130

============================================================================

# delta_origin（输出统一 mapped_vector）
python3 src/4.2/delta/delta_origin.py \
  --data_path data/dolly/train.jsonl \
  --routing_path data/dolly/train_coarse_topk5.jsonl \
  --output_path data/dolly/score/delta_origin_mapped.jsonl \
  --concurrency 16 \
  --turn_aggregation sum \
  --routing_weight_mode coarse

python3 /home/walkiiiy/DataRecipe/src/4.2/delta/delta_origin_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/dolly/score/delta_origin_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/dolly/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/dolly/exp4.2/dataset_delta_origin_130.jsonl \
  --num_samples 130

=========================================================================================


#  alpagasus_origin：原始标量评分（不做能力簇评分）
python3 src/4.2/alpagasus/alpagasus_origin.py \
  --data_path data/dolly/train.jsonl \
  --output_path data/dolly/score/alpagasus_origin_scored.jsonl \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --temperature 0.01 \
  --max_tokens 130 \
  --concurrency 16



python3 /home/walkiiiy/DataRecipe/src/4.2/alpagasus/alpagasus_origin_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/dolly/score/alpagasus_origin_scored.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/dolly/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/dolly/exp4.2/dataset_alpagasus_origin_130.jsonl \
  --num_samples 130

==========================================================================================



#  alpagasus_improved：保持原评分逻辑 + 映射到能力维向量
python3 src/4.2/alpagasus/alpagasus_improved.py \
  --data_path data/dolly/train.jsonl \
  --routing_path data/dolly/train_coarse_topk5.jsonl \
  --output_path data/dolly/score/alpagasus_improved_mapped.jsonl \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --temperature 0.01 \
  --max_tokens 130 \
  --routing_weight_mode coarse \
  --concurrency 16

python3 /home/walkiiiy/DataRecipe/src/4.2/alpagasus/alpagasus_improved_sample.py \
  --score_path /home/walkiiiy/DataRecipe/data/dolly/score/alpagasus_improved_mapped.jsonl \
  --data_path /home/walkiiiy/DataRecipe/data/dolly/train.jsonl \
  --output_path /home/walkiiiy/DataRecipe/data/dolly/exp4.2/dataset_alpagasus_improved_650.jsonl \
  --num_samples 650

================================================================================
# instag
python3 src/4.2/instag/instag.py \
  --data_path data/dolly/train.jsonl \
  --output_path data/dolly/score/instag_tags.jsonl \
  --model deepseek-chat \
  --concurrency 16


采样
python3 src/4.2/instag/instag_sample.py \
  --data_path data/dolly/train.jsonl \
  --tag_path data/dolly/score/instag_tags.jsonl \
  --output_path data/dolly/exp4.2/dataset_instag_130.jsonl \
  --num_samples 130
=======================================================
# MIG
- python3 src/4.2/mig/mig.py \
  --data_path data/dolly/train.jsonl \
  --instag_path data/dolly/score/instag_tags.jsonl \
  --delta_path data/dolly/score/delta_origin_mapped.jsonl \
  --output_path data/dolly/score/mig_scored.jsonl \
  --cluster_output_path data/dolly/score/mig_tag_clusters.json \
  --valid_tag_output_path data/dolly/score/mig_valid_tags.json \
  --tag_merge_eps 0.05 \
  --delta_scalar_mode auto \
  --embedding_backend auto

采样
- python3 src/4.2/mig/mig_sample.py \
  --scored_path data/dolly/score/mig_scored.jsonl \
  --output_path data/dolly/exp4.2/dataset_mig_130.jsonl \
  --num_samples 130 




============================================================================

# datawhisperer
python3 src/4.2/datawhisperer/datawhisperer.py \
  --data_path data/dolly/train.jsonl \
  --output_path data/dolly/score/datawhisperer_icl_mapped.jsonl \
  --model_name_or_path ~/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
  --n_demonstrations 10 \
  --n_queries 5 \
  --n_iterations 10 \
  --attention_layer 13 \
  --icl_metric accuracy \
  --concurrency 2 \
  --max_new_tokens 16 \
  --max_input_tokens 2048 \
  --select_top_k 130

==============================================================
# PDM
python src/4.2/PDM/pdm.py \
  --data_path data/dolly/train.jsonl \
  --srm_path data/dolly/score/srm_from_topk5_only.jsonl \
  --output_path data/dolly/score/pdm_scored.jsonl \
  --model_name_or_path ~/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B \
  --context_size 4 \
  --global_trials 3 \
  --max_seq_len 1024 \
  --nll_batch_size 8 \
  --no-cache_token_ids \
  --device_map auto


python src/4.2/PDM/pdm_sample.py \
  --score_path data/dolly/score/pdm_scored.jsonl \
  --data_path data/dolly/train.jsonl \
  --output_path data/dolly/exp4.2/dataset_pdm_650.jsonl \
  --num_samples 650 \
  --annotate_selection
===============================================================================================
# 随机采样
python src/scripts/sample_train_subset.py \
  --input-jsonl data/dolly/train.jsonl \
  --output-jsonl data/dolly/exp4.2/dataset_random_130.jsonl \
  --sample-size 130 \
  --seed 42



# 训练
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --eval-source-jsonl data/dolly/test.jsonl \
  --eval-ratio 1 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/dolly/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B
  --run alpagasus_origin::data/dolly/exp4.2/dataset_improved_130.jsonl::data/dolly/exp4.2/run_alpagasus_improved_130_shared_eval 
  
  dataset_alpagasus_origin_130.jsonl::data/dolly/exp4.2/run_alpagasus_origin_130_shared_eval \




  --run delta_improved::data/dolly/exp4.2/dataset_delta_improved_130.jsonl::data/dolly/exp4.2/run_delta_improved_130_shared_eval \

  --run delta_origin::data/dolly/exp4.2/dataset_delta_origin_130.jsonl::data/dolly/exp4.2/run_delta_origin_130_shared_eval \

  --run srm::data/dolly/exp4.2/dataset_srm_130.jsonl::data/dolly/exp4.2/run_srm_130_shared_eval \
  --run random::data/dolly/exp4.2/dataset_random_130.jsonl::data/dolly/exp4.2/run_random_130_shared_eval \



  --run delta_origin::data/dolly/exp4.2/dataset_delta_origin_130.jsonl::data/dolly/exp4.2/run_delta_origin_130_shared_eval \


  
# re
python src/4.3/run_orchestrator.py \
  --train-raw-jsonl data/dolly/train.jsonl \
  --anchor-jsonl data/dolly/train.jsonl \
  --topk-jsonl data/dolly/train_coarse_topk5.jsonl \
  --score-path alpagasus_improved::data/dolly/score/alpagasus_improved_mapped.jsonl \
  --score-path delta_improved::data/dolly/score/delta_improved_mapped.jsonl \
  --score-path srm::data/dolly/score/srm_from_topk5_only.jsonl \
  --score-path pdm::data/dolly/score/pdm_scored.jsonl \
  --output-dir data/dolly/exp4.3/decoupled_run1 \
  --base-model Qwen/Qwen2.5-0.5B \
  --model-source modelscope \
  --num-chunks 10 \
  --total-epochs 4 \
  --anchor-size-per-cap 2 \
  --anchor-batch-size 2 \
  --max-seq-length 256 \
  --train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --keep-policy threshold \
  --keep-threshold 0.5 \
  --alpha-ema 0.1 \
  --beta-ema 0.2 \
  --shared-eval-jsonl data/dolly/test.jsonl \
  --run-static-stage 1 \
  --seed 42 \
  --epoch1-target-size 130



# EM评估
python src/4.1/EXP/eval_checkpoints_em.py \
  --eval-source-jsonl data/dolly/test.jsonl \
  --eval-size 1446 \
  --base_model Qwen/Qwen2.5-0.5B \
  --model_source modelscope \
  --run re::data/dolly/exp4.3/decoupled_run1/static_stage/run_recipe_selected/final_checkpoint




  --run alpagasus_origin::data/dolly/exp4.2/run_alpagasus_improved_130_shared_eval/final_checkpoint 

  --run alpagasus_origin::data/dolly/exp4.2/run_alpagasus_origin_130_shared_eval/final_checkpoint 


  --run delta_origin::data/dolly/exp4.2/run_delta_origin_130_shared_eval/final_checkpoint 


  --run random::data/dolly/exp4.2/run_random_130_shared_eval/final_checkpoint \
  --run srm::data/dolly/exp4.2/run_srm_130_shared_eval/final_checkpoint

  --run delta_improved::data/dolly/exp4.2/run_delta_improved_130_shared_eval/final_checkpoint

  
  run_delta_origin_130_shared_eval/final_checkpoint \
  --run alpagasus_origin::data/dolly/exp4.2/
  run_alpagasus_origin_130_shared_eval/final_checkpoint \
  --output-dir data/dolly/exp4.2/em_eval



# bert-score
python src/4.1/EXP/eval_checkpoints_bertscore.py \
  --eval-source-jsonl data/dolly/train.jsonl \
  --eval-size 500 \
  --bertscore-lang-mode auto \
  --run srm::data/dolly/exp4.2/run_srm_130_shared_eval/final_checkpoint \
  --run random::data/dolly/exp4.2/run_random_130_shared_eval/final_checkpoint \
  --output-dir data/dolly/exp4.2/bertscore_eval \
  --save-per-run-jsonl
