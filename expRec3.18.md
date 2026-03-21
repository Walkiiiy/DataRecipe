- 能力树采样方法有问题，第二节实验需要重跑
- 这么对能力树方法进行采样有问题。
把现在的总采样脚本src/4.1/EXP/data_sampling.py中的能力树采样（ours）去掉，然后改成data_sampling_by_random_and_kmeans.py。一会我告诉你怎么单独重写一个能力树采样脚本

- 能力树单独采样脚本
- python src/4.1/EXP/data_sampling_by_capability_tree.py \
  --tree-json data/dolly-15k/capability_tree_final_pruned.json \
  --profile-jsonl data/dolly-15k/dolly-15k_cdt_profile.jsonl \
  --out-dir data/dolly-15k/exp \
  --min-valid-cluster-size 10 \
  --budget-n 2443 \
  --radius-quantile 0.9 \
  --radius-scale 1.0 \
  --random-seed 42

- 分验证集单独训练
python src/4.1/EXP/sft_lora_train.py \
  --dataset_path data/dolly-15k/exp/dataset_ours.jsonl \
  --output_dir data/dolly-15k/exp/run_ours


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
- 效果还是不好。估计时因为dolly是指令微调数据集。换别的数据试试
-----------------------------------------------------------------------------------------
- 动态可扩展的统一验证集对比训练脚本：
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/dolly-15k/exp/dataset_ours.jsonl::data/dolly-15k/exp/run_ours_shared_eval \
  --run kmeans::data/dolly-15k/exp/dataset_kmeans.jsonl::data/dolly-15k/exp/run_kmeans_shared_eval \
  --run random::data/dolly-15k/exp/dataset_random.jsonl::data/dolly-15k/exp/run_random_shared_eval \
  --run category::data/dolly-15k/exp/dataset_category.jsonl::data/dolly-15k/exp/run_category_shared_eval \
  --eval-source-jsonl data/dolly-15k/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --output-root data/dolly-15k/exp/shared_eval

格式：--run <name>::<dataset_jsonl>::<output_dir>
--------------------------------------------------------------------------------------------

- 单独给data_ablation在source_type上采样：
- python src/4.1/EXP/data_sampling_by_source_type.py \
  --profile-jsonl data/data_ablation_full59K/train.jsonl \
  --out-dir data/data_ablation_full59K/exp \
  --source-field source_type \
  --min-valid-source-size 0 \
  --random-seed 42 \
  --budget-n 317


- 能力树重采样：
- python src/4.1/EXP/data_sampling_by_capability_tree.py \
  --tree-json data/data_ablation_full59K/capability_tree_final_pruned.json \
  --profile-jsonl data/data_ablation_full59K/data_ablation_full59K_cdt_profile.jsonl \
  --out-dir data/data_ablation_full59K/exp \
  --min-valid-cluster-size 10 \
  --radius-quantile 0.9 \
  --radius-scale 1.0 \
  --random-seed 42


- 统一测试集训练
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/data_ablation_full59K/exp/dataset_ours.jsonl::data/data_ablation_full59K/exp/run_ours_shared_eval \
  --run kmeans::data/data_ablation_full59K/exp/dataset_kmeans.jsonl::data/data_ablation_full59K/exp/run_kmeans_shared_eval \
  --run random::data/data_ablation_full59K/exp/dataset_random.jsonl::data/data_ablation_full59K/exp/run_random_shared_eval \
  --run category::data/data_ablation_full59K/exp/dataset_source_type.jsonl::data/data_ablation_full59K/exp/run_source_type_shared_eval \
  --eval-source-jsonl data/data_ablation_full59K/train.jsonl \
  --eval-ratio 0.1 \
  --seed 42 \
  --output-root data/data_ablation_full59K/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct


- 画图
python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/data_ablation_full59K/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/data_ablation_full59K/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/data_ablation_full59K/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/data_ablation_full59K/exp/run_source_type_shared_eval/train_eval_log.csv \
  --out-dir data/data_ablation_full59K/exp/figures


- 效果不错
-----------------------------------
重跑alpaca
- 能力树重采样
- python src/4.1/EXP/data_sampling_by_capability_tree.py \
  --tree-json data/alpaca-gpt4-data-en/capability_tree_final_pruned.json \
  --profile-jsonl data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl \
  --out-dir data/alpaca-gpt4-data-en/exp \
  --min-valid-cluster-size 10 \
  --radius-quantile 0.9 \
  --radius-scale 1.0 \
  --random-seed 42

- 统一测试集训练
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/alpaca-gpt4-data-en/exp/dataset_ours.jsonl::data/alpaca-gpt4-data-en/exp/run_ours_shared_eval \
  --run kmeans::data/alpaca-gpt4-data-en/exp/dataset_kmeans.jsonl::data/alpaca-gpt4-data-en/exp/run_kmeans_shared_eval \
  --run random::data/alpaca-gpt4-data-en/exp/dataset_random.jsonl::data/alpaca-gpt4-data-en/exp/run_random_shared_eval \
  --eval-source-jsonl data/alpaca-gpt4-data-en/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --output-root data/alpaca-gpt4-data-en/exp/shared_eval \
  --base_model EleutherAI/gpt-neo-125m

  - 画图
python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/alpaca-gpt4-data-en/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/alpaca-gpt4-data-en/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/alpaca-gpt4-data-en/exp/run_random_shared_eval/train_eval_log.csv \
  --out-dir data/alpaca-gpt4-data-en/exp/figures


---------------------------------------------------------

## MMLU数据集
- 格式化
- python src/scripts/prepare_mmlu.py \
  --input-parquet data/mmlu/all/auxiliary_train-00000-of-00001.parquet \
  --output-jsonl data/mmlu/train.jsonl \
  --category-field subject \
  --fallback-category mmlu_auxiliary_train

- 预处理
-export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/mmlu/train.jsonl \
  --output data/mmul/mmlu_cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat \
  --base-url https://api.deepseek.com



- 生成能力树：
python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/mmlu/mmlu_cdt_profile.jsonl \
  --max-samples 1500 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 200 \
  --max-layers 15 \
  --log-level INFO

- 减枝
python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/mmlu/capability_tree_final.json \
  --output-tree-json data/mmlu/capability_tree_final_pruned.json \
  --output-summary-json data/mmlu/capability_tree_summary_pruned.json 


- 采样
python src/4.1/EXP/data_sampling_by_capability_tree.py \
  --tree-json data/mmlu/capability_tree_final_pruned.json \
  --profile-jsonl data/mmlu/mmlu_cdt_profile.jsonl \
  --out-dir data/mmlu/exp \
  --min-valid-cluster-size 10 \
  --random-seed 42 




- python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/mmlu/train.jsonl \
  --out-dir data/mmlu/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1495 \
  --random-seed 42

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/mmlu/mmlu_cdt_profile.jsonl \
  --out-dir data/mmlu/exp \
  --budget-n 1495 \
  --kmeans-k 18


- 统一测试集训练
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/mmlu/exp/dataset_ours.jsonl::data/mmlu/exp/run_ours_shared_eval \
  --run kmeans::data/mmlu/exp/dataset_kmeans.jsonl::data/mmlu/exp/run_kmeans_shared_eval \
  --run random::data/mmlu/exp/dataset_random.jsonl::data/mmlu/exp/run_random_shared_eval \
  --eval-source-jsonl data/mmlu/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --output-root data/mmlu/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct


- 画图
python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/mmlu/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/mmlu/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/mmlu/exp/run_random_shared_eval/train_eval_log.csv \
  --out-dir data/mmlu/exp/figures



-----------------------------------------
## dialogsum数据集
- export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/gsm/train.jsonl \
  --output data/gsm/gsm_cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat \
  --base-url https://api.deepseek.com


- 生成能力树：
python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/gsm/gsm_cdt_profile.jsonl \
  --max-samples 1000000 \
  --d-max 0.7 \
  --log-every 100 \
  --patience-no-1to2-growth 200 \
  --log-level INFO


-python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/gsm/capability_tree_final.json \
  --output-tree-json data/gsm/capability_tree_final_pruned.json \
  --output-summary-json data/gsm/capability_tree_summary_pruned.json 

- 采样

-python src/4.1/EXP/data_sampling_by_capability_tree.py \
  --tree-json data/gsm/capability_tree_final_pruned.json \
  --profile-jsonl data/gsm/gsm_cdt_profile.jsonl \
  --out-dir data/gsm/exp \
  --min-valid-cluster-size 10 \
  --random-seed 42 


-python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/gsm/train.jsonl \
  --out-dir data/gsm/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 2903 \
  --random-seed 42


  Total categories=8





python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/gsm/train.jsonl

- python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/gsm/gsm_cdt_profile.jsonl \
  --out-dir data/gsm/exp \
  --budget-n 2903 \
  --kmeans-k 3

- 统一测试集训练
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/gsm/exp/dataset_ours.jsonl::data/gsm/exp/run_ours_shared_eval \
  --run kmeans::data/gsm/exp/dataset_kmeans.jsonl::data/gsm/exp/run_kmeans_shared_eval \
  --run random::data/gsm/exp/dataset_random.jsonl::data/gsm/exp/run_random_shared_eval \
  --run category::data/gsm/exp/dataset_category.jsonl::data/data_ablation_full59K/exp/run_category_shared_eval \
  --eval-source-jsonl data/gsm/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/gsm/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct


- 画图
python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/gsm/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/gsm/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/gsm/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/dialigsum/exp/run_source_type_shared_eval/train_eval_log.csv \
  --out-dir data/gsm/exp/figures



-----------------------------------------
# gsm数据集
- 缩减到8k数据
- 预处理
- export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/gsm/train.jsonl \
  --output data/gsm/gsm_cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat \
  --base-url https://api.deepseek.com

 
 - 生成能力树：
python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/gsm/gsm_cdt_profile.jsonl \
  --max-samples 1000000 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 300 \
  --log-level INFO


python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/gsm/capability_tree_final.json \
  --output-tree-json data/gsm/capability_tree_final_pruned.json \
  --output-summary-json data/gsm/capability_tree_summary_pruned.json 


采样
python src/4.1/EXP/data_sampling_by_capability_tree.py \
  --tree-json data/gsm/capability_tree_final_pruned.json \
  --profile-jsonl data/gsm/gsm_cdt_profile.jsonl \
  --out-dir data/gsm/exp \
  --min-valid-cluster-size 10 \
  --random-seed 42
  




-python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/gsm/train.jsonl \
  --out-dir data/gsm/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1485 \
  --random-seed 42


python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/gsm/train.jsonl

- python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/gsm/gsm_cdt_profile.jsonl \
  --out-dir data/gsm/exp \
  --budget-n 1485 \
  --kmeans-k 4

- 统一测试集训练
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/gsm/exp/dataset_ours.jsonl::data/gsm/exp/run_ours_shared_eval \
  --run kmeans::data/gsm/exp/dataset_kmeans.jsonl::data/gsm/exp/run_kmeans_shared_eval \
  --run random::data/gsm/exp/dataset_random.jsonl::data/gsm/exp/run_random_shared_eval \
  --run category::data/gsm/exp/dataset_category.jsonl::data/gsm/exp/run_category_shared_eval \
  --eval-source-jsonl data/gsm/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 2 \
  --output-root data/gsm/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct


python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/gsm/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/gsm/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/gsm/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/gsm/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/gsm/exp/figures


----------------------------------------------
3.20总结
目前4.1效果比较好的数据集：
alpaca 指令微调 无自带分类 在150M预训练模型上 略微超过
data_ablation 数学推理 有自带分类 在Qwen0.5b 大幅超过
dialogsum 对话总结 有自带分类 在Qwen0.5b 略微超过
