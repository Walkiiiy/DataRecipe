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
  --budget-n 1382 \
  --random-seed 42

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/mmlu/mmlu_cdt_profile.jsonl \
  --out-dir data/mmlu/exp \
  --budget-n 1382 \
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

改进采样脚本
python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/mmlu/capability_tree_final_pruned.json   --profile-jsonl data/mmlu/mmlu_cdt_profile.jsonl   --out-dir data/mmlu/exp   --min-valid-cluster-size 20   --random-seed 42 --budget-n 1000
  




-python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/gsm/train.jsonl \
  --out-dir data/gsm/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42


python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/gsm/train.jsonl

- python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/gsm/gsm_cdt_profile.jsonl \
  --out-dir data/gsm/exp \
  --budget-n 1000 \
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



---------------------------------------------------
# bioinstruct数据集
- 
export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/bioinstruct/train.jsonl \
  --output data/bioinstruct/bioinstruct_cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat \
  --base-url https://api.deepseek.com
- 
python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/bioinstruct/bioinstruct_cdt_profile.jsonl \
  --max-samples 1500 \
  --d-max 0.9 \
  --log-every 100 \
  --patience-no-1to2-growth 200 \
  --max-layers 15 \
  --log-level INFO



python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/bioinstruct/capability_tree_final.json \
  --output-tree-json data/bioinstruct/capability_tree_final_pruned.json \
  --output-summary-json data/bioinstruct/capability_tree_summary_pruned.json 


采样
python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/bioinstruct/capability_tree_final_pruned.json   --profile-jsonl data/bioinstruct/bioinstruct_cdt_profile.jsonl   --out-dir data/bioinstruct/exp   --min-valid-cluster-size 9   --random-seed 42 --budget-n 1000
  

python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/bioinstruct/train.jsonl \
  --out-dir data/bioinstruct/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42


python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/bioinstruct/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/bioinstruct/bioinstruct_cdt_profile.jsonl \
  --out-dir data/bioinstruct/exp \
  --budget-n 1000 \
  --kmeans-k 3



- 统一测试集训练
python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/bioinstruct/exp/dataset_ours.jsonl::data/bioinstruct/exp/run_ours_shared_eval \
  --run kmeans::data/bioinstruct/exp/dataset_kmeans.jsonl::data/bioinstruct/exp/run_kmeans_shared_eval \
  --run random::data/bioinstruct/exp/dataset_random.jsonl::data/bioinstruct/exp/run_random_shared_eval \
  --run category::data/bioinstruct/exp/dataset_category.jsonl::data/bioinstruct/exp/run_category_shared_eval \
  --eval-source-jsonl data/bioinstruct/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 2 \
  --output-root data/bioinstruct/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct


- 画图
python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/bioinstruct/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/bioinstruct/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/bioinstruct/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/bioinstruct/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/bioinstruct/exp/figures


效果不佳

-----------------------------------------
接下来对六个数据集逐一实验：
# sharegpt 
export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/sharegpt/train.jsonl \
  --output data/sharegpt/sharegpt_cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat \
  --base-url https://api.deepseek.com

python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/sharegpt/sharegpt_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 1500 \
  --max-layers 15 \
  --log-level INFO




python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/sharegpt/capability_tree_final.json \
  --output-tree-json data/sharegpt/capability_tree_final_pruned.json \
  --output-summary-json data/sharegpt/capability_tree_summary_pruned.json 


采样
python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/sharegpt/capability_tree_final_pruned.json   --profile-jsonl data/sharegpt/sharegpt_cdt_profile.jsonl   --out-dir data/sharegpt/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000
  

python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/sharegpt/train.jsonl \
  --out-dir data/sharegpt/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42


python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/sharegpt/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/sharegpt/sharegpt_cdt_profile.jsonl \
  --out-dir data/sharegpt/exp \
  --budget-n 1000 \
  --kmeans-k 3





python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/sharegpt/exp/dataset_ours.jsonl::data/sharegpt/exp/run_ours_shared_eval \
  --eval-source-jsonl data/sharegpt/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 2 \
  --output-root data/sharegpt/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct

  --run kmeans::data/sharegpt/exp/dataset_kmeans.jsonl::data/sharegpt/exp/run_kmeans_shared_eval \
  --run random::data/sharegpt/exp/dataset_random.jsonl::data/sharegpt/exp/run_random_shared_eval \
  --run category::data/sharegpt/exp/dataset_category.jsonl::data/sharegpt/exp/run_category_shared_eval \

- 画图
python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/sharegpt/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/sharegpt/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/sharegpt/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/sharegpt/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/sharegpt/exp/figures

# lawyer 
export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/lawyer/train.jsonl \
  --output data/lawyer/lawyer_cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat \
  --base-url https://api.deepseek.com

python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/lawyer/lawyer_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 1000 \
  --max-layers 15 \
  --log-level INFO

python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/lawyer/capability_tree_final.json \
  --output-tree-json data/lawyer/capability_tree_final_pruned.json \
  --output-summary-json data/lawyer/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/lawyer/capability_tree_final_pruned.json   --profile-jsonl data/lawyer/lawyer_cdt_profile.jsonl   --out-dir data/lawyer/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000


python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/lawyer/train.jsonl \
  --out-dir data/lawyer/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/lawyer/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/lawyer/lawyer_cdt_profile.jsonl \
  --out-dir data/lawyer/exp \
  --budget-n 1000 \
  --kmeans-k 3

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/lawyer/exp/dataset_ours.jsonl::data/lawyer/exp/run_ours_shared_eval \
  --eval-source-jsonl data/lawyer/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 2 \
  --output-root data/lawyer/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct

  --run kmeans::data/lawyer/exp/dataset_kmeans.jsonl::data/lawyer/exp/run_kmeans_shared_eval \
  --run random::data/lawyer/exp/dataset_random.jsonl::data/lawyer/exp/run_random_shared_eval \
  --run category::data/lawyer/exp/dataset_category.jsonl::data/lawyer/exp/run_category_shared_eval \

python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/lawyer/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/lawyer/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/lawyer/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/lawyer/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/lawyer/exp/figures

# evolinstruct

export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/evolinstruct/train.jsonl \
  --output data/evolinstruct/evolinstruct_cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat \
  --base-url https://api.deepseek.com

python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/evolinstruct/evolinstruct_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO


python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/evolinstruct/capability_tree_final.json \
  --output-tree-json data/evolinstruct/capability_tree_final_pruned.json \
  --output-summary-json data/evolinstruct/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/evolinstruct/capability_tree_final_pruned.json   --profile-jsonl data/evolinstruct/evolinstruct_cdt_profile.jsonl   --out-dir data/evolinstruct/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000


python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/evolinstruct/train.jsonl \
  --out-dir data/evolinstruct/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/evolinstruct/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/evolinstruct/evolinstruct_cdt_profile.jsonl \
  --out-dir data/evolinstruct/exp \
  --budget-n 1000 \
  --kmeans-k 3

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/evolinstruct/exp/dataset_ours.jsonl::data/evolinstruct/exp/run_ours_shared_eval \
  --eval-source-jsonl data/evolinstruct/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 2 \
  --output-root data/evolinstruct/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct


  --run kmeans::data/evolinstruct/exp/dataset_kmeans.jsonl::data/evolinstruct/exp/run_kmeans_shared_eval \
  --run random::data/evolinstruct/exp/dataset_random.jsonl::data/evolinstruct/exp/run_random_shared_eval \
  --run category::data/evolinstruct/exp/dataset_category.jsonl::data/evolinstruct/exp/run_category_shared_eval \


python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/evolinstruct/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/evolinstruct/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/evolinstruct/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/evolinstruct/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/evolinstruct/exp/figures

# magicoder


export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/magicoder/train.jsonl \
  --output data/magicoder/magicoder_cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat \



python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/magicoder/magicoder_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO


python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/magicoder/capability_tree_final.json \
  --output-tree-json data/magicoder/capability_tree_final_pruned.json \
  --output-summary-json data/magicoder/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/magicoder/capability_tree_final_pruned.json   --profile-jsonl data/magicoder/magicoder_cdt_profile.jsonl   --out-dir data/magicoder/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000


python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/magicoder/train.jsonl \
  --out-dir data/magicoder/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/magicoder/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/magicoder/magicoder_cdt_profile.jsonl \
  --out-dir data/magicoder/exp \
  --budget-n 1000 \
  --kmeans-k 12

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/magicoder/exp/dataset_ours.jsonl::data/magicoder/exp/run_ours_shared_eval \
  --run kmeans::data/magicoder/exp/dataset_kmeans.jsonl::data/magicoder/exp/run_kmeans_shared_eval \
  --run random::data/magicoder/exp/dataset_random.jsonl::data/magicoder/exp/run_random_shared_eval \
  --run category::data/magicoder/exp/dataset_category.jsonl::data/magicoder/exp/run_category_shared_eval \
  --eval-source-jsonl data/magicoder/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/magicoder/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct



python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/magicoder/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/magicoder/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/magicoder/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/magicoder/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/magicoder/exp/figures













<!-- # spider


# bird


--------------------------------------------------------------------------------
# HDBSCAN聚类

python src/4.1HDBSCN/stage2_hdbscan_clustering.py \
  --input-jsonl data/magicoder/magicoder_cdt_profile.jsonl \
  --output-json data/magicoder/exp/hdbscan_cluster_result.json \
  --max-samples -1 \
  --min-cluster-size 15 \
  --random-seed 42 \
  --log-level INFO

python src/4.1HDBSCN/EXP/data_sampling_by_hdbscan.py \
  --profile-jsonl data/magicoder/magicoder_cdt_profile.jsonl \
  --cluster-json data/magicoder/exp/hdbscan_cluster_result.json \
  --out-dir data/magicoder/exp \
  --budget-n 1000 \
  --random-seed 42 \
  --log-level INFO





python src/4.1HDBSCN/stage2_hdbscan_clustering.py \
  --input-jsonl data/sharegpt/sharegpt_cdt_profile.jsonl \
  --output-json data/sharegpt/exp/hdbscan_cluster_result.json \
  --max-samples -1 \
  --min-cluster-size 2 \
  --random-seed 42 \
  --log-level INFO

python src/4.1HDBSCN/EXP/data_sampling_by_hdbscan.py \
  --profile-jsonl data/sharegpt/sharegpt_cdt_profile.jsonl \
  --cluster-json data/sharegpt/exp/hdbscan_cluster_result.json \
  --out-dir data/sharegpt/exp \
  --budget-n 1000 \
  --random-seed 42 \
  --log-level INFO



python src/4.1HDBSCN/stage2_hdbscan_clustering.py \
  --input-jsonl data/evolinstruct/evolinstruct_cdt_profile.jsonl \
  --output-json data/evolinstruct/exp/evolinstruct_cluster_result.json \
  --max-samples -1 \
  --min-cluster-size 2 \
  --random-seed 42 \
  --log-level INFO

python src/4.1HDBSCN/EXP/data_sampling_by_hdbscan.py \
  --profile-jsonl data/evolinstruct/evolinstruct_cdt_profile.jsonl \
  --cluster-json data/evolinstruct/exp/evolinstruct_cluster_result.json \
  --out-dir data/evolinstruct/exp \
  --budget-n 1000 \
  --random-seed 42 \
  --log-level INFO -->





  -----------------------------------------------
  # 重跑
  # magicoder


export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/magicoder/train.jsonl \
  --output data/magicoder/magicoder_cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat \



python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/magicoder/magicoder_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO


python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/magicoder/capability_tree_final.json \
  --output-tree-json data/magicoder/capability_tree_final_pruned.json \
  --output-summary-json data/magicoder/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/magicoder/capability_tree_final_pruned.json   --profile-jsonl data/magicoder/magicoder_cdt_profile.jsonl   --out-dir data/magicoder/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000


python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/magicoder/train.jsonl \
  --out-dir data/magicoder/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/magicoder/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/magicoder/magicoder_cdt_profile.jsonl \
  --out-dir data/magicoder/exp \
  --budget-n 1000 \
  --kmeans-k 12

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run kmeans::data/magicoder/exp/dataset_kmeans.jsonl::data/magicoder/exp/run_kmeans_shared_eval \
  --eval-source-jsonl data/magicoder/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/magicoder/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct


  --run ours::data/magicoder/exp/dataset_ours.jsonl::data/magicoder/exp/run_ours_shared_eval \
  --run random::data/magicoder/exp/dataset_random.jsonl::data/magicoder/exp/run_random_shared_eval \
  --run category::data/magicoder/exp/dataset_category.jsonl::data/magicoder/exp/run_category_shared_eval \

python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/magicoder/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/magicoder/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/magicoder/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/magicoder/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/magicoder/exp/figures




# alpaca




python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/alpaca/alpaca_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 0.75 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO



python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/alpaca/capability_tree_final.json \
  --output-tree-json data/alpaca/capability_tree_final_pruned.json \
  --output-summary-json data/alpaca/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/alpaca/capability_tree_final_pruned.json   --profile-jsonl data/alpaca/alpaca_cdt_profile.jsonl   --out-dir data/alpaca/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000


python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/alpaca/train.jsonl \
  --out-dir data/alpaca/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/alpaca/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/alpaca/alpaca_cdt_profile.jsonl \
  --out-dir data/alpaca/exp \
  --budget-n 1000 \
  --kmeans-k 8

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/alpaca/exp/dataset_ours.jsonl::data/alpaca/exp/run_ours_shared_eval \
  --run kmeans::data/alpaca/exp/dataset_kmeans.jsonl::data/alpaca/exp/run_kmeans_shared_eval \
  --run random::data/alpaca/exp/dataset_random.jsonl::data/alpaca/exp/run_random_shared_eval \
  --run category::data/alpaca/exp/dataset_category.jsonl::data/alpaca/exp/run_category_shared_eval \
  --eval-source-jsonl data/alpaca/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/alpaca/exp/shared_eval \
  --base_model EleutherAI/gpt-neo-125m




python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/alpaca/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/alpaca/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/alpaca/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/alpaca/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/alpaca/exp/figures



# dialogsum
python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/dialogsum/dialogsum_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 0.85 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO



python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/dialogsum/capability_tree_final.json \
  --output-tree-json data/dialogsum/capability_tree_final_pruned.json \
  --output-summary-json data/dialogsum/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/dialogsum/capability_tree_final_pruned.json   --profile-jsonl data/dialogsum/dialogsum_cdt_profile.jsonl   --out-dir data/dialogsum/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000


python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/dialogsum/train.jsonl \
  --out-dir data/dialogsum/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/dialogsum/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/dialogsum/dialogsum_cdt_profile.jsonl \
  --out-dir data/dialogsum/exp \
  --budget-n 1000 \
  --kmeans-k 3

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/dialogsum/exp/dataset_ours.jsonl::data/dialogsum/exp/run_ours_shared_eval \
  --run kmeans::data/dialogsum/exp/dataset_kmeans.jsonl::data/dialogsum/exp/run_kmeans_shared_eval \
  --run random::data/dialogsum/exp/dataset_random.jsonl::data/dialogsum/exp/run_random_shared_eval \
  --run category::data/dialogsum/exp/dataset_category.jsonl::data/dialogsum/exp/run_category_shared_eval \
  --eval-source-jsonl data/dialogsum/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/dialogsum/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct



python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/dialogsum/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/dialogsum/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/dialogsum/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/dialogsum/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/dialogsum/exp/figures




#  data_ablation

export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/data_ablation/train_full.jsonl \
  --output data/data_ablation/data_ablation_cdt_profile.jsonl \
  --max-samples 2000 \
  --concurrency 32 \
  --model deepseek-chat \
  --base-url https://api.deepseek.com

python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/data_ablation/data_ablation_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 0.8 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO \
  --decay-rate 0.85

python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/data_ablation/data_ablation_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO 

python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/data_ablation/capability_tree_final.json \
  --output-tree-json data/data_ablation/capability_tree_final_pruned.json \
  --output-summary-json data/data_ablation/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/data_ablation/capability_tree_final_pruned.json   --profile-jsonl data/data_ablation/data_ablation_cdt_profile.jsonl   --out-dir data/data_ablation/exp   --min-valid-cluster-size 2   --random-seed 42 --budget-n 1000


python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/data_ablation/train_full.jsonl \
  --out-dir data/data_ablation/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/data_ablation/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/data_ablation/data_ablation_cdt_profile.jsonl \
  --out-dir data/data_ablation/exp \
  --budget-n 1000 \
  --kmeans-k 10

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/data_ablation/exp/dataset_ours.jsonl::data/data_ablation/exp/run_ours_shared_eval \
  --eval-source-jsonl data/data_ablation/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 5 \
  --output-root data/data_ablation/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --eval-steps-mode manual \
  --eval_steps 5

  --run kmeans::data/data_ablation/exp/dataset_kmeans.jsonl::data/data_ablation/exp/run_kmeans_shared_eval \
  --run random::data/data_ablation/exp/dataset_random.jsonl::data/data_ablation/exp/run_random_shared_eval \
  --run category::data/data_ablation/exp/dataset_category.jsonl::data/data_ablation/exp/run_category_shared_eval \





python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/data_ablation/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/data_ablation/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/data_ablation/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/data_ablation/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/data_ablation/exp/figures






# alpaca效果不好 再试一次 mllu

python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/mmlu/mmlu_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 0.9 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO




python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/mmlu/capability_tree_final.json \
  --output-tree-json data/mmlu/capability_tree_final_pruned.json \
  --output-summary-json data/mmlu/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/mmlu/capability_tree_final_pruned.json   --profile-jsonl data/mmlu/mmlu_cdt_profile.jsonl   --out-dir data/mmlu/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000


python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/mmlu/train.jsonl \
  --out-dir data/mmlu/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/mmlu/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/mmlu/mmlu_cdt_profile.jsonl \
  --out-dir data/mmlu/exp \
  --budget-n 1000 \
  --kmeans-k 8

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/mmlu/exp/dataset_ours.jsonl::data/mmlu/exp/run_ours_shared_eval \
  --run kmeans::data/mmlu/exp/dataset_kmeans.jsonl::data/mmlu/exp/run_kmeans_shared_eval \
  --run random::data/mmlu/exp/dataset_random.jsonl::data/mmlu/exp/run_random_shared_eval \
  --run category::data/mmlu/exp/dataset_category.jsonl::data/mmlu/exp/run_category_shared_eval \
  --eval-source-jsonl data/mmlu/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/mmlu/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct



python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/mmlu/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/mmlu/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/mmlu/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/mmlu/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/mmlu/exp/figures

效果不错
# data_ablation效果不佳，再试一次 gsm

python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/gsm/gsm_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO

python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/gsm/capability_tree_final.json \
  --output-tree-json data/gsm/capability_tree_final_pruned.json \
  --output-summary-json data/gsm/capability_tree_summary_pruned.json 



python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/gsm/capability_tree_final_pruned.json   --profile-jsonl data/gsm/gsm_cdt_profile.jsonl   --out-dir data/gsm/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000


python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/gsm/train.jsonl \
  --out-dir data/gsm/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/gsm/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/gsm/gsm_cdt_profile.jsonl \
  --out-dir data/gsm/exp \
  --budget-n 1000 \
  --kmeans-k 4

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/gsm/exp/dataset_ours.jsonl::data/gsm/exp/run_ours_shared_eval \
  --eval-source-jsonl data/gsm/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/gsm/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct


  --run ours::data/gsm/exp/dataset_ours.jsonl::data/gsm/exp/run_ours_shared_eval \
  --run kmeans::data/gsm/exp/dataset_kmeans.jsonl::data/gsm/exp/run_kmeans_shared_eval \
  --run random::data/gsm/exp/dataset_random.jsonl::data/gsm/exp/run_random_shared_eval \
  --run category::data/gsm/exp/dataset_category.jsonl::data/gsm/exp/run_category_shared_eval \



python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/gsm/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/gsm/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/gsm/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/gsm/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/gsm/exp/figures

  也不好
# lawyer
python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/lawyer/lawyer_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO

python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/lawyer/capability_tree_final.json \
  --output-tree-json data/lawyer/capability_tree_final_pruned.json \
  --output-summary-json data/lawyer/capability_tree_summary_pruned.json 



python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/lawyer/capability_tree_final_pruned.json   --profile-jsonl data/lawyer/lawyer_cdt_profile.jsonl   --out-dir data/lawyer/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000


python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/lawyer/train.jsonl \
  --out-dir data/lawyer/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/gsm/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/gsm/gsm_cdt_profile.jsonl \
  --out-dir data/gsm/exp \
  --budget-n 1000 \
  --kmeans-k 4

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/lawyer/exp/dataset_ours.jsonl::data/lawyer/exp/run_ours_shared_eval \
  --eval-source-jsonl data/lawyer/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/lawyer/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct



  --run kmeans::data/lawyer/exp/dataset_kmeans.jsonl::data/lawyer/exp/run_kmeans_shared_eval \
  --run random::data/lawyer/exp/dataset_random.jsonl::data/lawyer/exp/run_random_shared_eval \
  --run category::data/lawyer/exp/dataset_category.jsonl::data/lawyer/exp/run_category_shared_eval \

python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/lawyer/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/lawyer/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/lawyer/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/lawyer/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/lawyer/exp/figures


# 新数据集 banking77
banking77

export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/banking77/train.jsonl \
  --output data/banking77/banking77_cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat



python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/banking77/banking77_cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 0.55 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO

python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/banking77/capability_tree_final.json \
  --output-tree-json data/banking77/capability_tree_final_pruned.json \
  --output-summary-json data/banking77/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/banking77/capability_tree_final_pruned.json   --profile-jsonl data/banking77/banking77_cdt_profile.jsonl   --out-dir data/banking77/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000



python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/banking77/train.jsonl \
  --out-dir data/banking77/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/banking77/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/banking77/banking77_cdt_profile.jsonl \
  --out-dir data/banking77/exp \
  --budget-n 1000 \
  --kmeans-k 18

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run kmeans::data/banking77/exp/dataset_kmeans.jsonl::data/banking77/exp/run_kmeans_shared_eval \
  --eval-source-jsonl data/banking77/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/banking77/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct
  --eval-steps-mode manual \
  --eval_steps 5

  --run ours::data/banking77/exp/dataset_ours.jsonl::data/banking77/exp/run_ours_shared_eval \
  --run random::data/banking77/exp/dataset_random.jsonl::data/banking77/exp/run_random_shared_eval \
  --run category::data/banking77/exp/dataset_category.jsonl::data/banking77/exp/run_category_shared_eval \

python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/banking77/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/banking77/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/banking77/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/banking77/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/banking77/exp/figures




# 3.27 stage3的embedding换了Qwen embedding

# clinc150
export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/clinc150/train.jsonl \
  --output data/clinc150/cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat



python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/clinc150/cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 0.5 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO

0.55   2026-03-27 15:07:56,433 [INFO] Processed=2000 | depth=3 | level_counts={0: 1, 1: 102, 2: 40, 3: 4} | global_J=2060.633912 | no_1to2_streak=81
0.5    2026-03-27 15:15:13,741 [INFO] Processed=2000 | depth=2 | level_counts={0: 1, 1: 222, 2: 4} | global_J=2157.432189 | no_1to2_streak=9

python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/clinc150/capability_tree_final.json \
  --output-tree-json data/clinc150/capability_tree_final_pruned.json \
  --output-summary-json data/clinc150/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/clinc150/capability_tree_final_pruned.json   --profile-jsonl data/clinc150/cdt_profile.jsonl   --out-dir data/clinc150/exp   --min-valid-cluster-size 5   --random-seed 42 --budget-n 1000



python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/clinc150/train.jsonl \
  --category-jsonl data/clinc150/train.jsonl \
  --out-dir data/clinc150/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/clinc150/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/clinc150/cdt_profile.jsonl \
  --out-dir data/clinc150/exp \
  --budget-n 1000 \
  --kmeans-k 125


python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/clinc150/exp/dataset_ours.jsonl::data/clinc150/exp/run_ours_shared_eval \
  --run kmeans::data/clinc150/exp/dataset_kmeans.jsonl::data/clinc150/exp/run_kmeans_shared_eval \
  --run random::data/clinc150/exp/dataset_random.jsonl::data/clinc150/exp/run_random_shared_eval \
  --run category::data/clinc150/exp/dataset_category.jsonl::data/clinc150/exp/run_category_shared_eval \
  --eval-source-jsonl data/clinc150/train.jsonl \
  --eval-ratio 0.05 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/clinc150/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct
  --eval-steps-mode manual \
  --eval_steps 5



python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/clinc150/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/clinc150/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/clinc150/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/clinc150/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/clinc150/exp/figures


不行


# sciq 

export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
  python src/4.1/stage1_atomic_profile.py \
  --input data/sciq/train.jsonl \
  --output data/sciq/cdt_profile.jsonl \
  --max-samples 10000000 \
  --concurrency 32 \
  --model deepseek-chat


python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/sciq/cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 1 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO

python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/sciq/capability_tree_final.json \
  --output-tree-json data/sciq/capability_tree_final_pruned.json \
  --output-summary-json data/sciq/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/sciq/capability_tree_final_pruned.json   --profile-jsonl data/sciq/cdt_profile.jsonl   --out-dir data/sciq/exp   --min-valid-cluster-size 2   --random-seed 42 --budget-n 1000



python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/sciq/train.jsonl \
  --category-jsonl data/sciq/train.jsonl \
  --out-dir data/sciq/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/sciq/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/sciq/train.jsonl \
  --out-dir data/sciq/exp \
  --budget-n 1000 \
  --kmeans-k 9

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/sciq/exp/dataset_ours.jsonl::data/sciq/exp/run_ours_shared_eval \
  --run kmeans::data/sciq/exp/dataset_kmeans.jsonl::data/sciq/exp/run_kmeans_shared_eval \
  --eval-source-jsonl data/sciq/train.jsonl \
  --eval-ratio 0.01 \
  --seed 42 \
  --num_train_epochs 8 \
  --output-root data/sciq/exp/shared_eval \
  --base_model EleutherAI/gpt-neo-125m \
  --eval-steps-mode manual \
  --eval_steps 10


  --run random::data/sciq/exp/dataset_random.jsonl::data/sciq/exp/run_random_shared_eval \
  --run category::data/sciq/exp/dataset_category.jsonl::data/sciq/exp/run_category_shared_eval \




python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/sciq/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/sciq/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/sciq/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/sciq/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/sciq/exp/figures

# gsm


python src/4.1/stage3_overlapping_incremental_hierarchy.py \
  --input-jsonl data/gsm/cdt_profile.jsonl \
  --max-samples 2000 \
  --d-max 0.9 \
  --log-every 100 \
  --patience-no-1to2-growth 2000 \
  --max-layers 15 \
  --log-level INFO

python src/4.1/stage4_prune_singleton_tree.py \
  --input-tree-json data/gsm/capability_tree_final.json \
  --output-tree-json data/gsm/capability_tree_final_pruned.json \
  --output-summary-json data/gsm/capability_tree_summary_pruned.json 


python src/4.1/EXP/data_sampling_by_capability_tree.py   --tree-json data/gsm/capability_tree_final_pruned.json   --profile-jsonl data/gsm/cdt_profile.jsonl   --out-dir data/gsm/exp   --min-valid-cluster-size 2   --random-seed 42 --budget-n 1000



python src/4.1/EXP/data_sampling_by_category.py \
  --profile-jsonl data/gsm/train.jsonl \
  --category-jsonl data/gsm/train.jsonl \
  --out-dir data/gsm/exp \
  --min-valid-category-size 10 \
  --category-mode proportional \
  --budget-n 1000 \
  --random-seed 42



python src/4.1/baseline_Kmeans_clustering.py \
  --input-jsonl data/gsm/train.jsonl

python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
  --profile-jsonl data/gsm/train.jsonl \
  --out-dir data/gsm/exp \
  --budget-n 1000 \
  --kmeans-k 4

python src/4.1/EXP/sft_lora_train_shared_eval.py \
  --run ours::data/gsm/exp/dataset_ours.jsonl::data/gsm/exp/run_ours_shared_eval \
  --eval-source-jsonl data/gsm/train.jsonl \
  --eval-ratio 0.01 \
  --seed 42 \
  --num_train_epochs 4 \
  --output-root data/gsm/exp/shared_eval \
  --base_model Qwen/Qwen2.5-0.5B-Instruct \
  --eval-steps-mode manual \
  --eval_steps 10

  --run kmeans::data/gsm/exp/dataset_kmeans.jsonl::data/gsm/exp/run_kmeans_shared_eval \
  --run random::data/gsm/exp/dataset_random.jsonl::data/gsm/exp/run_random_shared_eval \
  --run category::data/gsm/exp/dataset_category.jsonl::data/gsm/exp/run_category_shared_eval \




python src/4.1/EXP/visualize_results.py \
  --ours-log-csv data/gsm/exp/run_ours_shared_eval/train_eval_log.csv \
  --kmeans-log-csv data/gsm/exp/run_kmeans_shared_eval/train_eval_log.csv \
  --random-log-csv data/gsm/exp/run_random_shared_eval/train_eval_log.csv \
  --category-log-csv data/gsm/exp/run_category_shared_eval/train_eval_log.csv \
  --out-dir data/gsm/exp/figures