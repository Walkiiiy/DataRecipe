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

# 改进state1 capability extraction使其能在获取簇名的同时获取簇的中心向量

python3 src/4.2/stage_1_capability_extraction.py \
  --tree-json data/banking77/capability_tree_final_pruned.json \
  --profile-jsonl data/banking77/banking77_cdt_profile.jsonl \
  --api-key "sk-ab412f420cd540888da4732a35600c4a" \
  --model deepseek-chat \
  --base-url https://api.deepseek.com \
  --cluster-scope leaf \
  --samples-per-cluster 12 \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --concurrency 8 




# 修改修改代码，只对数据本身的 instruction + input + output/response 组合文本作为目标进行计算簇心（原始数据簇簇心）


```
请在src/4.2/stage_2_top_k_routing.py中实现：
我正在实现大模型微调数据筛选流程中的“基于 Embedding 的粗粒度稠密路由 (Coarse Dense Routing via Embeddings)”模块。目标是编写一个解耦的、独立的 Python 脚本，用于对海量文本数据集进行预处理。该脚本需要计算单条数据与预先定义好的多维“能力空间 (Capability Space)”质心之间的语义相关性，并仅保留 Top-K 的候选簇，从而为下游的 LLM 评估器缩小搜索空间。数学逻辑（请严格遵循以下公式）
对于每个数据点 $x$ 以及 $m$ 个能力质心 $\{\mu_1, ..., \mu_m\}$ ：
1. 计算余弦相似度：$S_k = \cos(E(x), \mu_k)$ ，其中 $k \in \{1...m\}$。
2. 选取 Top-K 的索引：$\mathcal{K}^* = \arg\max_{top-K} S_k$ 。
3. 对 Top-K 的值进行带温度缩放的 Softmax 计算：
$Z = \sum_{j \in \mathcal{K}^*} \exp(S_j / \tau)$ $S_{emb}^{(k)} = \exp(S_k / \tau) / Z$ 
4. 未进入 Top-K 的维度默认视作 0（在输出文件中无需保存，以实现稀疏化）。技术与工程要求技术栈：使用 PyTorch，transformers（或 sentence-transformers）以及 HuggingFace 的 datasets 库。极致效率：脚本必须支持 GPU 运算。计算余弦相似度时，必须使用批处理和矩阵乘法（例如 torch.matmul 或张量广播），绝对不能使用 for 循环逐条计算数据点与质心的相似度。CLI 输入参数 (使用 argparse 封装)：--dataset_path：本地 JSONL 文件或 HuggingFace 数据集的路径。--centroids_path：包含 $m$ 个预计算质心向量的 .npy 或 .pt 文件路径。--model_name：Embedding 模型名称（默认 "BAAI/bge-base-en-v1.5"）。--top_k：保留的候选簇数量（整数，默认 3）。--temperature：温度参数 $\tau$（浮点数，默认 0.1）。输出格式：将结果保存为 .jsonl 或 .parquet 文件。每一条记录应包含：原始的文本标识符（如 id）、top_k_indices（整数列表）以及 top_k_scores（经过温度 Softmax 后的 $S_{emb}$ 浮点数列表）。
其中能力簇的簇心向量被保存在data/数据集name/capability_cluster_names_leaf.jsonl,数据点本身就是data/数据集name/train.jsonl中的 instruction + input + output/response 组合文本
```



- python3 src/4.2/stage_2_top_k_routing.py \
  --dataset_path data/banking77/train.jsonl \
  --centroids_path data/banking77/capability_cluster_names_leaf.jsonl \
  --output_path data/banking77/top_k_routing.jsonl \
  --model_name BAAI/bge-base-en-v1.5 \
  --top_k 3 \
  --temperature 0.1 \
  --batch_size 256 \
  --device auto




# 改用Qwen/Qwen3-Embedding-0.6B做embedding

python3 src/4.2/stage_1_capability_extraction.py \
  --tree-json data/banking77/capability_tree_final_pruned.json \
  --profile-jsonl data/banking77/banking77_cdt_profile.jsonl \
  --api-key "sk-ab412f420cd540888da4732a35600c4a" \
  --model deepseek-chat \
  --base-url https://api.deepseek.com \
  --cluster-scope leaf \
  --samples-per-cluster 12 \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --concurrency 8 


python3 src/4.2/stage_2_top_k_routing.py \
  --dataset_path data/banking77/train.jsonl \
  --centroids_path data/banking77/capability_cluster_names_leaf.jsonl \
  --output_path data/banking77/top_k_routing.jsonl \
  --model_name Qwen/Qwen3-Embedding-0.6B \
  --top_k 3 \
  --temperature 0.1 \
  --batch_size 256 \
  --device auto


- 可以考虑把之前的能力树构建的嵌入模型也换成这个

# SRM
```
接下来在src/4.2/SRM/SRM.py中实现：
我正在实现大模型微调数据筛选流程中的“细粒度深层验证与后期融合 (Fine Deep Verification via LLM & Late Fusion)”模块。目标是编写一个基于 asyncio 的高并发 Python 脚本。该脚本需要读取上一阶段（粗排）生成的包含 Top-K 候选簇的数据，调用 LLM 进行零样本逻辑验证，并将 LLM 的离散评分与粗排的空间缩放分数进行乘法融合，最终输出稀疏的 $m$ 维能力特征向量。数学逻辑与处理流程（请严格遵循）输入解析：读取数据点 $x$ 及其对应的 top_k_indices 和经过温度缩放的 top_k_scores（即论文中的 $S_{emb}^{(k)}$）。深层验证 (Fine Deep Verification)：针对每一个保留的候选簇 $C_k \in Top\text{-}K$，构建包含文本 $x$ 和能力 $C_k$ 描述的 Prompt，请求 LLM。LLM 的输出必须被解析为离散逻辑门控置信度 $W_{llm}(x,C_k) \in \{0, w, 1\}$，分别对应 "unrelated"（不相关，0）、"weakly related"（弱相关，$w$）或 "strongly related"（强相关，1）。阻尼参数（Damping parameter）$w$ 默认为 0.1。后期融合 (Late Fusion)：计算融合后的最终特征向量的分量 $v_{x,k} = S_{emb}^{(k)} \cdot W_{llm}^{(k)}$。未进入 Top-K 的维度保持为 0。技术与工程要求技术栈：使用 Python, asyncio, aiohttp (或 openai 的 AsyncClient)。使用 tqdm.asyncio 显示并发进度。高并发与鲁棒性：必须使用异步并发机制请求 LLM，并通过 asyncio.Semaphore 控制最大并发数（如 max_concurrent_requests=50）。必须包含重试机制（Retrying）和异常捕获，防止因 API 速率限制（Rate Limit）或超时导致整个批次崩溃。Prompt 模板设计：在代码中预留一个清晰的 Prompt 模板函数，要求 LLM 强制输出 JSON 格式，例如 {"relation": "strongly related"}，以便代码稳定解析为 $1, w, 0$。CLI 输入参数 (argparse)：--input_path：第一阶段粗排输出的 .jsonl 文件路径。--capabilities_path：包含能力簇 ID 和文本描述的 .json 文件路径。--m_dimensions：能力空间的总维度数 $m$（整数，用于构建最终的稀疏/稠密向量）。--api_key / --base_url / --model_name：LLM API 的配置项。--damping_w：阻尼参数 $w$（浮点数，默认 0.1）。--output_path：最终输出路径。输出格式：将结果保存为 .jsonl 文件。每条记录包含原始 id 和最终的 mapped_vector（可以是长度为 $m$ 的数组，也可以是 {index: score} 的稀疏字典形式以节省空间）。
以banking77数据集为例，本地data目录中，train.jsonl记录了原始数据，top_k_routing.jsonl记录了topk相似的能力簇，而能力簇的文字描述可以在capability_cluster_names_leaf.jsonl中找到，对每条数据应该在top_k_routing.jsonl中找簇号，然后在capability_cluster_names_leaf.jsonl中找对应簇的文字描述来构建prompt
```



- python3 src/4.2/SRM/SRM.py \
  --input_path data/banking77/top_k_routing.jsonl \
  --capabilities_path data/banking77/capability_cluster_names_leaf.jsonl \
  --data_path data/banking77/train.jsonl \
  --api_key "sk-ab412f420cd540888da4732a35600c4a" \
  --base_url https://api.deepseek.com \
  --model_name deepseek-chat \
  --damping_w 0.1 \
  --max_concurrent_requests 20 \
  --output_path data/banking77/srm_mapped_vectors.jsonl
