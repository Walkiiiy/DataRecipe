<!-- - 数据集tag
```
OPENAI_API_KEY='sk-ab412f420cd540888da4732a35600c4a' OPENAI_BASE_URL='https://api.deepseek.com/v1' python src/4.1/stage1_atomic_profile.py   --model deepseek-chat   --max-samples 1000   --batch-size 8   --concurrency 10   --max-tokens 700   --output data/alpaca_with_tags.jsonl
```
- 初始层次聚类
python src/4.1/stage2_vectorize_cluster.py   --input-jsonl data/alpaca-gpt4-data-en/alpaca_with_tags_first1000.jsonl   --output-jsonl data/alpaca-gpt4-data-en/alpaca_with_initial_clusters.jsonl   --output-mapping-json data/alpaca-gpt4-data-en/sample_to_initial_clusters.json   --output-tag-cluster-json data/alpaca-gpt4-data-en/tag_to_cluster.json   --distance-threshold 0.7   --embedding-batch-size 256 -->
- 数据集tag+description
```
OPENAI_API_KEY='sk-ab412f420cd540888da4732a35600c4a' \
OPENAI_BASE_URL='https://api.deepseek.com/v1' \
python src/4.1/stage1_atomic_profile.py \
  --model deepseek-chat \
  --max-samples 1000 \
  --batch-size 8 \
  --concurrency 10 \
  --max-tokens 700 \
  --output data/alpaca-gpt4-data-en/alpaca_capability_profile.jsonl
```


- 目前的版本没有动态添加数据到能力锚点集直到锚点集稳定的功能
- - 初始层次聚类
python src/4.1/stage2_vectorize_cluster.py   --input-jsonl data/alpaca-gpt4-data-en/alpaca_capability_profile.jsonl   --output-jsonl data/alpaca-gpt4-data-en/alpaca_with_initial_clusters.jsonl   --output-mapping-json data/alpaca-gpt4-data-en/sample_to_initial_clusters.json   --output-tag-cluster-json data/alpaca-gpt4-data-en/tag_to_cluster.json   --distance-threshold 0.7   --embedding-batch-size 256 