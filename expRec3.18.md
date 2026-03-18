- 能力树采样方法有问题，第二节实验需要重跑
- 这么对能力树方法进行采样有问题。
把现在的总采样脚本src/4.1/EXP/data_sampling.py中的能力树采样（ours）去掉，然后改成data_sampling_by_random_and_kmeans.py。一会我告诉你怎么单独重写一个能力树采样脚本
- python src/4.1/EXP/data_sampling_by_capability_tree.py \
  --tree-json data/dolly-15k/capability_tree_final_pruned.json \
  --profile-jsonl data/dolly-15k/dolly-15k_cdt_profile.jsonl \
  --out-dir data/dolly-15k/exp \
  --min-valid-cluster-size 10 \
  --budget-n 2443 \
  --radius-quantile 0.9 \
  --radius-scale 1.0 \
  --random-seed 42


- 统一验证集训练
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --ours-dataset-path data/dolly-15k/exp/dataset_ours.jsonl \
  --kmeans-dataset-path data/dolly-15k/exp/dataset_kmeans.jsonl \
  --random-dataset-path data/dolly-15k/exp/dataset_random.jsonl \
  --category-dataset-path data/dolly-15k/exp/dataset_category.jsonl \
  --ours-output-dir data/dolly-15k/exp/run_ours_shared_eval \
  --kmeans-output-dir data/dolly-15k/exp/run_kmeans_shared_eval \
  --random-output-dir data/dolly-15k/exp/run_random_shared_eval \
  --category-output-dir data/dolly-15k/exp/run_category_shared_eval \
  --eval-source-jsonl data/dolly-15k/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --output-root data/dolly-15k/exp/shared_eval

