#  数据映射
- delta方法在capability tree的投影
#  delta origin
export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
python src/4.2/delta_origin.py   --data_path data/banking77/train.jsonl   --output_path data/banking77/train_delta_origin_scored.jsonl --max_samples 500 --concurrancy 32



# 先构建能力簇总结脚本
python3 src/4.2/stage_1_capability_extraction.py \
  --tree-json data/banking77/capability_tree_final_pruned.json \
  --profile-jsonl data/banking77/banking77_cdt_profile.jsonl \
  --api-key "sk-ab412f420cd540888da4732a35600c4a" \
  --model deepseek-chat \
  --base-url https://api.deepseek.com \
  --cluster-scope leaf \
  --samples-per-cluster 12 \
  --concurrency 8 \
  --resume



python3 src/4.2/stage_1_capability_extraction.py \
  --tree-json data/dialogsum/capability_tree_final_pruned.json \
  --profile-jsonl data/dialogsum/dialogsum_cdt_profile.jsonl \
  --api-key "sk-ab412f420cd540888da4732a35600c4a" \
  --model deepseek-chat \
  --base-url https://api.deepseek.com \
  --cluster-scope leaf \
  --samples-per-cluster 12 \
  --concurrency 8 \
  --resume



# 改进 delta 评分脚本
<!-- python3 src/4.2/delta/delta_improved.py \
  --data_path data/banking77/train.jsonl \
  --cluster_path data/banking77/capability_cluster_names_leaf.jsonl \
  --output_path data/banking77/train_delta_array_scored.jsonl \
  --api_key "sk-ab412f420cd540888da4732a35600c4a" \
  --concurrency 32 \
  --cluster_batch_size 12 \
  --max_samples 500 -->


python3 src/4.2/delta/delta_improved.py \
  --data_path data/banking77/train.jsonl \
  --cluster_path data/banking77/capability_cluster_names_leaf.jsonl \
  --output_path data/banking77/train_delta_improved_scored.jsonl \
  --api_key "sk-ab412f420cd540888da4732a35600c4a" \
  --quality_mode batch \
  --cluster_batch_size 12 \
  --cluster_definition_max_chars 100 \
  --quality_max_tokens 384 \
  --concurrency 32

