# 4.1.2 LLM 驱动的能力空间构建算法

## 概览

本模块在 4.1.1 的"LLM 路由 + 增量能力树"框架上，将分裂准则升级为 `docs/Capability Space Construction.md` 中的能力空间构建思想：

1. **Warm Start**：用前 N 条样本做 Sentence Embedding，以 Silhouette 最优的 KMeans 自动选 k，初始化根层能力节点；
2. **LLM 路由**：对后续每条样本，由 LLM 决定 assign（归入现有节点）或 create（新建节点）；
3. **Silhouette 选 k**：若目标节点样本数满足阈值，枚举 `k=2..max_split_k` 的 KMeans 候选，选轮廓系数最高的 `k_best`；
4. **Lemma-1 门禁**：对 `k_best` 计算收益-代价不等式，严格通过才执行分裂。

Warm Start 的 k 选择与增量分裂的 k 选择使用完全相同的 Silhouette 逻辑（`_best_silhouette_split`），无需用户手动指定初始簇数。

相较 4.1.1，本版本不再使用 `J_split > J_add` 的启发式父层目标，而是把分裂决策拆成两个职责清晰的步骤：

```text
Silhouette: 选最佳局部 k
Lemma Gate: 判断这个 k 是否值得并入全局能力空间
```

---

## 目录结构

```text
src/4.1.2/
├── clustering_objective.py          # 结构风险密度与 Lemma-1 分裂门禁
├── llm_capability_tree_builder.py   # 主程序：增量能力树构建
└── EXP/                             # 实验采样与评估脚本，沿用 4.1.1 兼容流程
```

---

## 核心算法

### 1. Warm Start

程序启动后，首先对前 `warm_start_samples` 条样本做 Sentence Embedding，然后用与增量分裂完全相同的 Silhouette 最优选 k 逻辑（枚举 `k=2..max_split_k`，选轮廓系数最高的 `k_best`）做 KMeans 聚类，将结果作为根层初始能力节点。命名通过 LLM 批量完成。

Warm Start 始终执行，`--warm-start-samples` 控制消耗的样本数（默认 50）。

### 2. 结构风险密度

对一个簇 `C`，代码显式记录三层量：

```text
total_sse = sum ||x - mu||^2
mean_sq_radius = total_sse / N
structural_risk_density = mean_sq_radius / N
```

其中 `mean_sq_radius` 对应文档中的 `I(C)`，最终进入 Lemma 的是：

```text
R(C) = I(C) / N
```

这样能同时惩罚"簇很散"和"子簇样本太少"两种风险。

### 3. 候选 k 选择

当新样本被路由到目标叶节点 `T` 后，若 `T` 的当前样本数大于 `min_node_size_for_split`，算法会把：

```text
P = T 的现有样本 + 新样本
```

作为待评估父簇，并枚举：

```text
k = 2..max_split_k
```

每个候选必须满足：

```text
每个子簇样本数 >= min_child_size_for_split
```

然后只用 Silhouette 对候选排序，选择 Silhouette 最高的候选 `k_best` 进入 Lemma gate。Silhouette 不作为额外分裂门槛；若没有满足子簇大小约束的有效候选，直接执行 add。

### 4. Lemma-1 分裂门禁

对 `k_best` 产生的子簇 `{C_1, ..., C_k}`，计算：

```text
risk_delta = max_j R(C_j) - R(P)
bias_gain = beta * (m^(-2/d) - (m+k-1)^(-2/d))
```

其中：

```text
beta = split_gain_scale = K_d / L^2 的工程缩放项
d = intrinsic_dim
m = 当前 active capability basis 数量，工程上使用分裂前叶节点数
```

判定规则：

```text
split iff risk_delta < bias_gain
```

通过则把旧叶节点替换为一个同名中间节点，并挂载 `k_best` 个新叶节点；否则新样本直接加入原目标叶节点。

---

## 完整处理流程

```text
[启动阶段] Warm Start
  Step 1  取前 warm_start_samples 条样本做 Sentence Embedding
  Step 2  枚举 k=2..max_split_k，按 Silhouette 选最优 k_best
  Step 3  KMeans 聚成 k_best 个初始根层节点，LLM 批量命名

[逐样本阶段] 对 samples[warm_start_samples:] 中的每条样本：
  Step 1  对样本文本做 Sentence Embedding
  Step 2  写入 vector_store / text_store / row_store
  Step 3  LLM 路由：assign（归入已有叶节点）或 create（新建叶节点）
  Step 4a create -> LLM 命名新节点，挂入指定父节点
  Step 4b assign -> 若目标节点样本数 <= min_node_size_for_split，直接 add
  Step 5  对 T + 新样本枚举 k=2..max_split_k
  Step 6  过滤无效候选，按 Silhouette 选 k_best
  Step 7  对 k_best 计算 Lemma-1 收益-代价
  Step 8  通过门禁 -> split，LLM 命名各子节点；否则 add
```

---

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--warm-start-samples` | 50 | Warm Start 消耗的初始样本数；k 由 Silhouette 自动选择 |
| `--min-node-size-for-split` | 3 | 目标节点现有样本数小于等于该值时跳过分裂评估 |
| `--max-split-k` | 6 | Warm Start 和增量分裂枚举的最大 KMeans k |
| `--min-child-size-for-split` | 2 | 候选分裂中每个子簇的最小样本数 |
| `--intrinsic-dim` | 16.0 | Lemma 中的内在维度 d |
| `--split-gain-scale` | 5.0 | `K_d / L^2` 的工程缩放项 beta；数值越大，Lemma gate 越容易接受分裂 |
| `--split-random-seed` | 42 | KMeans 候选分裂随机种子 |
| `--capability-text-mode` | raw | embedding/routing 使用的文本字段；可选 `raw` / `instruction` / `instruction_input` |
| `--patience-structure-stable` | 0 | 连续多少条样本无结构变化后早停，0 表示禁用 |
| `--min-processed-before-early-stop` | 0 | 至少处理多少条样本后才启用结构稳定早停 |
| `--max-leaf-nodes` | 0 | 叶节点数上限，0 表示禁用 |
| `--stale-singleton-rounds` | 0 | 单例节点长期不增长时剪枝，0 表示禁用 |

---

## 输出文件

默认写到输入 JSONL 所在目录：

| 文件 | 内容 |
|------|------|
| `capability_tree_final.json` | 完整能力树 |
| `capability_tree_summary.json` | 运行统计、配置、分裂准则说明 |
| `capability_tree_decisions.jsonl` | 每条样本的路由与分裂判定记录 |

`capability_tree_decisions.jsonl` 中与 4.1.2 相关的关键字段包括：

```text
split_strategy
split_candidates
best_split_k
best_split_silhouette
best_split_child_sizes
lemma_split_decision
split_reject_reason
```

`lemma_split_decision` 会记录：

```text
basis_count_before
basis_count_after
intrinsic_dim
split_gain_scale
parent_risk_density
child_risk_densities
max_child_risk_density
risk_delta
bias_gain
accepted
```

运行中还会对 embedding 和 sklearn 候选分裂做数值防护：

```text
非有限 embedding 会被 nan_to_num 后重新归一化
非有限聚类矩阵会在进入 KMeans/Silhouette 前清理
KMeans/Silhouette 的数值 RuntimeWarning 会被记录为候选跳过
```

---

## 示例命令

```bash
export DEEPSEEK_API_KEY="your_api_key"

python src/4.1.2/llm_capability_tree_builder.py \
    --input-jsonl data/dolly/train.jsonl \
    --max-samples 1000 \
    --shuffle \
    --shuffle-seed 42 \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --device auto \
    --model deepseek-chat \
    --base-url https://api.deepseek.com \
    --capability-text-mode raw \
    --warm-start-samples 200 \
    --min-node-size-for-split 4 \
    --max-split-k 6 \
    --min-child-size-for-split 2 \
    --intrinsic-dim 16 \
    --split-gain-scale 5.0 \
    --split-random-seed 42 \
    --patience-structure-stable 0 \
    --stale-singleton-rounds 0 \
    --log-every 50 \
    --print-tree-every 50
```

---

## EXP 流程

采样、训练、评估脚本沿用 4.1.1 的接口。能力树采样脚本会从树 JSON 中按 DFS 收集 `data_ids`，因此 4.1.2 的多叉分裂输出与现有 EXP 流程兼容。

```bash
python src/4.1.2/EXP/data_sampling_by_capability_tree.py \
    --tree-json data/dolly/capability_tree_final.json \
    --profile-jsonl data/dolly/train.jsonl \
    --budget-n 500 \
    --out-dir data/dolly/exp \
    --random-seed 42
```

---

## 依赖

```text
sentence-transformers
torch
scikit-learn
numpy
```

LLM 接口默认使用 DeepSeek API（兼容 OpenAI 格式），通过环境变量 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 传入密钥，`OPENAI_BASE_URL` 可自定义 API 地址。
