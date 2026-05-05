# 4.1 LLM 驱动的增量能力树聚类算法

## 概览

本模块实现了一套以大语言模型（LLM）为路由引擎、向量 embedding 为度量基础的**增量能力树构建算法**。其核心思想是：逐条处理原始指令数据，通过 LLM 判断每条样本应归入已有簇还是新建节点，同时用量化的聚类目标函数 $J$ 决定是否对目标节点执行二分裂，从而自动生长出一棵带语义标签的层次化能力分类树。

---

## 目录结构

```
src/4.1/
├── clustering_objective.py          # 聚类目标函数 J 的实现
├── llm_capability_tree_builder.py   # 主程序：增量能力树构建
└── EXP/                             # 实验采样与评估脚本
    ├── data_sampling_by_capability_tree.py   # 基于能力树的比例采样（我方策略）
    ├── data_sampling_by_random_and_kmeans.py # 随机采样 + KMeans 均匀采样（基线对比）
    ├── data_sampling_by_category.py          # 按类别采样
    ├── data_sampling_by_source_type.py       # 按数据源类型采样
    ├── sft_lora_train_shared_eval.py         # SFT LoRA 训练与评估
    ├── eval_checkpoints_bertscore.py         # BERTScore 评测
    ├── eval_checkpoints_em.py                # Exact Match 评测
    ├── test_eval_and_plot.py                 # 测试集评估与绘图
    └── visualize_results.py                  # 结果可视化
```

---

## 核心算法

### 1. 聚类目标函数 $J$（`clustering_objective.py`）

目标函数 $J$ 同时衡量簇内紧密度（Density）与簇间分离度（Separation），用于指导 **是否将目标节点一分为二**。

#### 1.1 密度（Density）

$$
\text{Dens}(C_k) = n_k \cdot \left(2 - \bar{d}_k\right)
$$

其中 $n_k$ 是簇 $C_k$ 的样本数，$\bar{d}_k$ 是簇内各样本到质心的平均距离。

- 样本越集中（$\bar{d}_k$ 越小），密度越高；
- 单样本簇特殊处理：$\bar{d}_k = 0$，故 $\text{Dens} = 2$；
- 空簇返回 0。

#### 1.2 分离度（Separation）

$$
\text{Sep}(C_k) = \frac{1}{K-1} \sum_{j \neq k} \| c_k - c_j \|_2
$$

即簇 $k$ 的质心到所有其他簇质心的平均欧氏距离。只有一个簇时返回 0。

#### 1.3 全局目标

$$
J(C) = \sum_{k=1}^{K} \text{Dens}(C_k) \cdot \text{Sep}(C_k)
$$

$J$ 越大，表示整体聚类质量越高（簇内紧、簇间散）。

---

### 2. 增量能力树构建算法（`llm_capability_tree_builder.py`）

#### 2.1 数据结构

- **TreeNode**：树节点，存储节点 ID、语义名称 `cluster_name`、直接持有的样本 ID 列表 `data_ids`、子节点列表 `children`，以及该节点覆盖所有样本的质心向量 `center`。
- **向量存储**：`vector_store`（sample_id → 归一化 embedding）、`text_store`（原始文本）、`row_store`（原始行数据）。
- **树索引**：`nodes`（node_id → TreeNode）、`parent`（node_id → parent_id）。

#### 2.2 单样本处理流程（`process_one`）

每进来一条新样本，执行以下步骤：

```
Step 1  对样本文本做 Sentence Embedding（归一化到单位球）
Step 2  向量写入 vector_store / text_store / row_store
Step 3  树为空 → 直接新建第一个叶节点（LLM 命名），结束
Step 4  树非空 → 调用 LLM 路由（routing）
Step 5  解析 LLM 决策：assign（归入已有叶节点）or create（新建节点）
Step 6a [create] → 在指定父节点下追加新叶，LLM 命名，结束
Step 6b [assign] → 进入 add-vs-split 判断
Step 7  [assign] 目标节点样本数 ≤ min_node_size_for_split → 直接 add，跳过分裂评估
Step 8  [assign] 计算 add 后的父级目标值 J_add
Step 9  [assign] 对目标节点现有样本 + 新样本执行 KMeans(k=2)，得到候选分裂 split_groups
Step 10 [assign] 若 split_groups 有效，计算分裂后的父级目标值 J_split
Step 11 [assign] J_split > J_add → 执行 split；否则执行 add
```

#### 2.3 LLM 路由（Routing）

路由 Prompt 包含：
- 当前整棵树的文本快照（节点 ID、名称、计数）；
- 新样本原始文本；
- 按 embedding 距离最近的 Top-5 候选叶节点；
- 可选父节点列表（用于 create）。

LLM 返回严格 JSON，二选一：

```json
{"decision": "assign", "target_node_id": "Nxx", "reason": "..."}
{"decision": "create", "parent_node_id": "Nxx", "new_cluster_name": "...", "reason": "..."}
```

若 LLM 输出非法，fallback 为按 embedding 距离最近叶节点执行 assign。

**路由约束**：
- assign 的 `target_node_id` 必须在 Top-5 候选集内，且必须是叶节点；
- create 的 `parent_node_id` 必须在预生成的可选父节点集内；
- 违规时自动 fallback 并记录计数器 `router_invalid_fallback`。

#### 2.4 Add vs Split 决策

在父节点 P 的视角下，局部评估：

| 方案 | 含义 | 目标值 |
|------|------|--------|
| **Add** | 将新样本加入目标叶节点 T，T 扩大 1 | $J_{\text{add}}$：以 T 扩大后重算父级 J |
| **Split** | 用 KMeans(k=2) 将 T + 新样本二分为 T_a、T_b | $J_{\text{split}}$：以 T_a、T_b 替换 T 后重算父级 J |

选择规则：$J_{\text{split}} > J_{\text{add}}$ 则分裂，否则直接添加。

**分裂时的树结构变更**：
1. 移除旧叶节点 T；
2. 创建一个与 T 同名的中间父节点 `split_parent`；
3. 在 `split_parent` 下挂两个新叶节点 T_a、T_b；
4. 用 LLM 为 T_a、T_b 生成互相对比的新名称（参考旧名 + 各分组的代表样本文本）；
5. 逐层向上更新质心。

#### 2.5 节点命名策略

所有节点命名均调用 LLM，并遵循**最小必要差异原则**：

- **新建节点命名**：提供父节点名、当前深度、最近已有簇对比信息，让 LLM 仅写出与最近已有簇的关键区分点，不得过细；
- **分裂命名**：提供旧节点名 + 两个子簇的代表样本（距各自质心最近的样本），生成保留公共核心、突显差异的两个子名；
- **名称压缩**：`_compact_child_name` 自动裁剪子节点名中与父节点重复的前缀词，保持树路径上的冗余信息最小化。

深度对应粒度策略：

| 深度 | 粒度 |
|------|------|
| 0（ROOT 子） | COARSE：宽泛可复用的能力家族 |
| 1–2 | MID：中等粒度，适用于本分支下多条样本 |
| ≥3 | FINE：可较具体，但仍需泛化一类样本 |

#### 2.6 稳定性与早停

- **结构稳定早停**（`patience_structure_stable`）：连续 K 条样本均未触发 create/split，判定树已收敛，提前结束。
- **叶节点数上限**（`max_leaf_nodes`）：叶节点数超过阈值时停止。
- **过时单例节点剪枝**（`stale_singleton_rounds`）：若某非根节点子树长期（N 轮）仅有 1 条样本且不再增长，将其数据合并到父节点并删除该节点，防止碎片化。

---

## 脚本执行流程

### 主程序：构建能力树

```bash
export DEEPSEEK_API_KEY="your_api_key"

python src/4.1.1/llm_capability_tree_builder.py \
    --input-jsonl data/dolly/train.jsonl \
    --max-samples 100 \
    --shuffle \
    --shuffle-seed 42 \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --api-key "sk-e02008970d074a01ae71f371e217f9aa" \
    --device auto \
    --model deepseek-chat \
    --base-url https://api.deepseek.com \
    --min-node-size-for-split 3 \
    --patience-structure-stable 50 \
    --stale-singleton-rounds 30 \
    --log-every 20 \
    --print-tree-every 20
```

**关键参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-jsonl` | `data/alpaca-gpt4-data-en/train.jsonl` | 输入数据，每行 JSON，需含 `instruction`/`input`/`output` 字段 |
| `--max-samples` | 1000 | 处理样本数上限，`-1` 表示全量 |
| `--shuffle` / `--no-shuffle` | shuffle | 是否随机打乱输入顺序 |
| `--shuffle-seed` | None | 打乱随机种子（留空为非确定性） |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence Transformer 模型名 |
| `--device` | `auto` | 推理设备（auto/cpu/cuda） |
| `--model` | `deepseek-chat` | LLM 模型名 |
| `--min-node-size-for-split` | 3 | 节点样本数 ≤ 此值时跳过分裂评估 |
| `--patience-structure-stable` | 0 | 结构稳定早停轮数，0 表示禁用 |
| `--max-leaf-nodes` | 0 | 叶节点数上限，0 表示禁用 |
| `--stale-singleton-rounds` | 0 | 单例节点剪枝轮数，0 表示禁用 |
| `--temperature` | 0.0 | LLM 温度（路由建议设 0 保证确定性） |

**输出文件**（默认同 input-jsonl 所在目录）：

| 文件 | 内容 |
|------|------|
| `capability_tree_final.json` | 完整能力树（含所有节点、data_ids、质心向量） |
| `capability_tree_summary.json` | 运行统计（节点数、深度、各类决策计数、LLM 调用次数等） |
| `capability_tree_decisions.jsonl` | 每条样本的处理决策记录（路由结果、J 值、是否分裂等） |

---

### EXP：基于能力树的数据采样（我方策略）

```bash
python src/4.1/EXP/data_sampling_by_capability_tree.py \
    --tree-json data/alpaca-gpt4-data-en/capability_tree_final.json \
    --profile-jsonl data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl \
    --budget-n 500 \
    --out-dir data/alpaca-gpt4-data-en/exp \
    --random-seed 42
```

**算法流程**：
1. DFS 遍历能力树，收集所有节点的 `data_ids`（全局去重，同一样本只归属首次遇到的节点）；
2. 按节点大小比例用 Hamilton 最大余数法分配预算配额；
3. 从每个节点组内随机无放回采样指定配额；
4. 对总量不足时做全局补齐（日志警告）；
5. 输出 `dataset_ours.jsonl` 与采样统计 `sampling_meta_ours_tree_v3.json`。

---

### EXP：基线对比采样（Random + KMeans）

```bash
python src/4.1/EXP/data_sampling_by_random_and_kmeans.py \
    --profile-jsonl data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl \
    --budget-n 500 \
    --kmeans-k 8 \
    --out-dir data/alpaca-gpt4-data-en/exp \
    --random-seed 42
```

**算法流程**：
1. 对所有样本的 `instruction` 字段做 Sentence Embedding；
2. **KMeans 均匀采样**：KMeans(k=`kmeans_k`) 分成 k 个簇，按均匀配额（预算/k，最后一个簇补余数）从每簇随机采样，容量不足时从剩余池补齐；
3. **随机采样**：全量数据中直接随机无放回抽取 N 条；
4. 两个子集均严格等于预算 N，输出 `dataset_kmeans.jsonl`、`dataset_random.jsonl` 及 `sampling_meta.json`。

---

## 完整实验流程（端到端）

```
原始数据 (train.jsonl)
        │
        ▼
[Step 1] llm_capability_tree_builder.py
        → capability_tree_final.json
        → capability_tree_decisions.jsonl
        │
        ▼
[Step 2] data_sampling_by_capability_tree.py   (我方)
         data_sampling_by_random_and_kmeans.py  (基线)
        → dataset_ours.jsonl
        → dataset_kmeans.jsonl
        → dataset_random.jsonl
        │
        ▼
[Step 3] sft_lora_train_shared_eval.py
        → 各方案 checkpoint
        │
        ▼
[Step 4] eval_checkpoints_bertscore.py / eval_checkpoints_em.py
        → 评测分数
        │
        ▼
[Step 5] visualize_results.py / test_eval_and_plot.py
        → 结果图表
```

---

## 依赖

```
sentence-transformers
torch
scikit-learn
numpy
```

LLM 接口默认使用 DeepSeek API（兼容 OpenAI 格式），通过环境变量 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 传入密钥，`OPENAI_BASE_URL` 可自定义 API 地址。





DEEPSEEK_API_KEY="$(tr -d '\n\r' < .apikey)" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n DataRecipe python src/4.1.2/llm_capability_tree_builder.py \
  --input-jsonl data/dolly/train.jsonl \
  --output-tree-json data/dolly/capability_tree_final_4_1_2_fixed_anchor.json \
  --output-summary-json data/dolly/capability_tree_summary_4_1_2_fixed_anchor.json \
  --output-decisions-jsonl data/dolly/capability_tree_decisions_4_1_2_fixed_anchor.jsonl \
  --max-samples 1000 --shuffle --shuffle-seed 42 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 --device auto \
  --capability-text-mode category_aware --routing-mode hybrid \
  --min-assign-cosine 0.08 \
  --warm-start-mode category --warm-start-samples 50 \
  --min-node-size-for-split 10 --max-split-k 6 \
  --min-child-size-for-split 3 --min-split-silhouette 0.10 \
  --intrinsic-dim 16 --split-gain-scale 1.0 --split-random-seed 42 \
  --patience-structure-stable 0 --stale-singleton-rounds 0 \
  --log-every 50 --print-tree-every 50
