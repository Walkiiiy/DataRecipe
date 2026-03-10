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
- 用tag进行初始层次聚类
```
python src/4.1/stage2_vectorize_cluster.py   --input-jsonl data/alpaca-gpt4-data-en/alpaca_capability_profile.jsonl   --output-jsonl data/alpaca-gpt4-data-en/alpaca_with_initial_clusters.jsonl   --output-mapping-json data/alpaca-gpt4-data-en/sample_to_initial_clusters.json   --output-tag-cluster-json data/alpaca-gpt4-data-en/tag_to_cluster.json   --distance-threshold 0.7   --embedding-batch-size 256 
```
- 用文本描述进行簇间分裂合并
```
python src/4.1/stage3_dynamic_capability_space.py \
  --profile-jsonl data/alpaca-gpt4-data-en/alpaca_capability_profile.jsonl \
  --cluster-dict-json data/alpaca-gpt4-data-en/sample_to_initial_clusters.json \
  --output-cluster-dict-json data/alpaca-gpt4-data-en/cluster_dict_evolved.json \
  --output-metrics-json data/alpaca-gpt4-data-en/cluster_metrics_evolved.json \
  --tau 1.0 \
  --epsilon 0.05 \
  --merge-cos-threshold 0.85 \
  --max-iter 20
```
- 需要检查：现在的循环中分裂合并的决定信号是否是单向的？如果是单向的那不能完全算一个循环
- 实验：用这种聚类方法得到的数据，采样后训练的效果优于均匀采样或者R&B的k-means聚类   先就用这1000条数据对比采样试试

```
你好，我正在为一篇大模型数据配方论文编写 Baseline（基线）实验代码。请帮我用 Python 实现一个“基于原始文本 Embedding 与轮廓系数峰值搜索 (Silhouette Peak Search) 的最优 K-Means 聚类采样”脚本。核心任务：对 alpaca 数据集（本地alpaca_1000.JSONL 文件）的原始文本进行向量化。然后，实现一个带早停机制（Early Stopping）的动态搜索算法，寻找轮廓系数的“峰值（拐点）”以确定最佳的 $k$ 值。最后，使用该最佳 $k$ 值进行聚类，并从各簇中均匀采样。具体流程要求：数据准备与拼接：加载数据集，将 instruction, input, 和 output 拼接成完整的纯文本字符串。向量化 (Embedding)：使用 sentence-transformers 将纯文本转化为稠密向量（支持 GPU 加速）。动态寻找最优 $k$ 值 (Dynamic Optimal K Search via Silhouette Peak)：设定一个初始 $k$ 值（如 start_k=10）和搜索步长（如 step=10）。在一个 while 循环中递增 $k$，执行 K-Means 聚类，并使用 sklearn.metrics.silhouette_score（开启 sample_size=10000 参数以加速计算）计算当前的轮廓系数。峰值检测与早停逻辑：维护一个历史最高轮廓系数 best_score 和对应的 best_k。设定一个容忍度参数（如 patience=3）。如果在连续 patience 个步长内，新计算的轮廓系数都低于当前的 best_score，则判定已经越过“峰值”。此时立即终止搜索（break），并输出找到的 best_k。均匀采样 (Uniform Sampling)：接收采样比例参数（如 sample_ratio=0.1）。重新使用 best_k 对全量数据跑一次 K-Means 拿到最终标签，按比例从每个簇中随机无放回抽取样本。数据持久化：将采样得到的数据保存为标准 JSONL 文件（如 alpaca_kmeans_peak_baseline.jsonl）。工程与环境要求：运行环境：Ubuntu，Python 3。监控与日志：使用 logging 模块详细打印每一次搜索的 $k$ 值及其轮廓系数，并在触发早停时打印提示信息（例如 "Peak detected at k=..., stopping search"）。代码结构：请提供高内聚、低耦合的规范代码，包含详细的注释。
```