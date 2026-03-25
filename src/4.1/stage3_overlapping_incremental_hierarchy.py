"""阶段三：基于 J 目标函数的动态增量层次聚类树（无 K-Means 分裂）。

核心机制：
1) Create New：与当前层所有分支距离都过大，则在当前层直接新增叶子分支。
2) Merge：与最近分支相关，且并入后 J 不下降，则并入并继续向更深层递归路由。
3) Split（层级分化）：与最近分支相关但并入会拉低 J，则创建新的中间父节点，将
   原最近分支与新样本叶子作为该父节点的两个子节点。

说明：
- 本实现复用 stage2 的 ObjectiveEvaluator，J = sum(Dens * Sep)。
- data_ids 在叶子节点持有；内部节点通过子树聚合计算 center。
- 每个节点维护自己的 d_max；根节点由命令行参数初始化。Split 时会将更小的 d_max 下放到子层。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stage2_objective_evaluator import ObjectiveEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    torch = None
    SentenceTransformer = None


@dataclass
class CapabilityNode:
    """能力树节点。

    字段含义：
    - node_id: 节点唯一标识
    - center: 簇中心（子树向量均值）
    - data_ids: 叶子节点保存的数据 ID；内部节点一般为空
    - children: 子节点列表
    - d_max: 当前节点的动态距离阈值
    """

    node_id: str
    center: np.ndarray | None = None
    data_ids: list[str] = field(default_factory=list)
    children: list["CapabilityNode"] = field(default_factory=list)
    d_max: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class Config:
    input_jsonl: Path
    output_tree_json: Path
    output_summary_json: Path
    embedding_model: str
    device: str
    max_samples: int | None
    max_layers: int
    root_d_max: float
    decay_rate: float
    epsilon: float
    log_every: int
    patience_no_1to2_growth: int
    shuffle: bool
    shuffle_seed: int
    log_level: str


def resolve_repo_path(path: Path) -> Path:
    """将相对路径稳定解析到仓库根目录，避免受启动目录影响。"""
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def default_output_paths(input_jsonl: Path) -> tuple[Path, Path]:
    """根据输入数据集目录生成 stage3 默认输出路径。"""
    base_dir = input_jsonl.parent
    return (
        base_dir / "capability_tree_final.json",
        base_dir / "capability_tree_summary.json",
    )


class DenseVectorizer:
    """将文本转为稠密向量。"""

    def __init__(self, model_name: str, device: str):
        if SentenceTransformer is None:
            raise ImportError("Missing sentence-transformers/torch. Please install them first.")
        self.model = SentenceTransformer(model_name, device=self._resolve_device(device))

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def encode_one(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vec, dtype=np.float64)


class IncrementalHierarchicalTree:
    """动态增量层次聚类树引擎。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.objective = ObjectiveEvaluator(eps=cfg.epsilon)
        self.vectorizer = DenseVectorizer(cfg.embedding_model, cfg.device)
        self.vector_store: dict[str, np.ndarray] = {}
        self.node_seq = 0
        self.root = self._new_internal_node(cfg.root_d_max)
        self._grew_1_to_2_in_last_insert = False

    # -----------------------------
    # 节点与中心更新辅助函数
    # -----------------------------
    def _new_node_id(self) -> str:
        node_id = f"N{self.node_seq}"
        self.node_seq += 1
        return node_id

    def _new_internal_node(self, d_max: float) -> CapabilityNode:
        return CapabilityNode(node_id=self._new_node_id(), center=None, data_ids=[], children=[], d_max=d_max)

    def _new_leaf_node(self, data_id: str, d_max: float) -> CapabilityNode:
        return CapabilityNode(
            node_id=self._new_node_id(),
            center=self.vector_store[data_id].copy(),
            data_ids=[data_id],
            children=[],
            d_max=d_max,
        )

    def _new_leaf_from_ids(self, ids: list[str], d_max: float) -> CapabilityNode:
        leaf = CapabilityNode(
            node_id=self._new_node_id(),
            center=None,
            data_ids=list(dict.fromkeys(ids)),
            children=[],
            d_max=d_max,
        )
        self._refresh_center(leaf)
        return leaf

    def _get_subtree_ids(self, node: CapabilityNode) -> list[str]:
        # 无论是否为叶子，都先采集当前节点自身携带的数据，
        # 以兼容“同时拥有 data_ids 与 children”的混合节点形态。
        merged = list(node.data_ids)
        for child in node.children:
            merged.extend(self._get_subtree_ids(child))
        # 子树中去重，避免重复样本 ID 影响中心计算。
        return list(dict.fromkeys(merged))

    def _refresh_center(self, node: CapabilityNode) -> None:
        ids = self._get_subtree_ids(node)
        if not ids:
            node.center = None
            return
        mat = np.stack([self.vector_store[i] for i in ids], axis=0)
        node.center = mat.mean(axis=0)

    def _refresh_center_upward(self, path: list[CapabilityNode]) -> None:
        """沿路径自底向上刷新中心。path[-1] 是当前节点。"""
        for node in reversed(path):
            self._refresh_center(node)

    def _subtree_depth_edges(self, node: CapabilityNode) -> int:
        """返回子树深度（边数）：叶子为 0。"""
        if not node.children:
            return 0
        return 1 + max(self._subtree_depth_edges(c) for c in node.children)

    def _cap_subtree_d_max(self, node: CapabilityNode, upper_bound: float) -> None:
        """将子树中各节点 d_max 限制在给定上界内，维持由上到下的阈值递减。"""
        node.d_max = min(node.d_max, upper_bound)
        for child in node.children:
            self._cap_subtree_d_max(child, node.d_max)

    def _rebalance_layer(self, node: CapabilityNode) -> None:
        """同层 K+1 均值重平衡。

        使用场景：
        - 在“状态 1: Create New”新增一个分支后，
          对 node.children 这一层进行局部重分配，降低顺序依赖。

        算法要点：
        1) 每轮先冻结当前层各簇中心 centers。
        2) 对每个 child 内部元素逐个做最近中心路由：
           - data_ids 元素按样本向量路由；
           - children 元素按子节点 center 路由。
        3) 若元素更靠近其他兄弟簇，则迁移过去。
        4) 最多 5 轮；若一轮无迁移则提前停止。
        """
        max_iter = 10

        for _ in range(max_iter):
            # 清理空簇，避免无效中心参与距离计算。
            node.children = [c for c in node.children if c.data_ids or c.children]
            if len(node.children) <= 1:
                break

            # 固定本轮中心（K+1 个簇中心），在本轮迁移过程中不动态更新。
            centers: list[np.ndarray] = []
            center_ready = True
            for c in node.children:
                if c.center is None:
                    self._refresh_center(c)
                if c.center is None:
                    center_ready = False
                    break
                centers.append(c.center.copy())

            if (not center_ready) or len(centers) <= 1:
                break

            changed = False
            current_children = list(node.children)  # 快照，保证遍历稳定

            for src_idx, child in enumerate(current_children):
                # ------- 1) 迁移 data_ids（原子级） -------
                for sample_id in list(child.data_ids):
                    vec = self.vector_store.get(sample_id)
                    if vec is None:
                        continue
                    dists = [float(np.linalg.norm(vec - ctr)) for ctr in centers]
                    best_idx = int(np.argmin(np.asarray(dists)))
                    if best_idx != src_idx and sample_id in child.data_ids:
                        child.data_ids.remove(sample_id)
                        dst = node.children[best_idx]
                        if sample_id not in dst.data_ids:
                            dst.data_ids.append(sample_id)
                        changed = True

                # ------- 2) 迁移 child.children（子树级） -------
                for sub in list(child.children):
                    if sub.center is None:
                        self._refresh_center(sub)
                    if sub.center is None:
                        continue
                    dists = [float(np.linalg.norm(sub.center - ctr)) for ctr in centers]
                    best_idx = int(np.argmin(np.asarray(dists)))
                    if best_idx != src_idx and sub in child.children:
                        child.children.remove(sub)
                        dst = node.children[best_idx]
                        if sub not in dst.children:
                            dst.children.append(sub)
                        changed = True

            # 清理空簇，并重算存活簇中心，为下一轮做准备。
            node.children = [c for c in node.children if c.data_ids or c.children]
            for c in node.children:
                self._refresh_center(c)

            if not changed:
                break

    def _rebalance_split(
        self,
        target_node: CapabilityNode,
        new_vec: np.ndarray,
        new_id: str,
        split_d_max: float,
    ) -> CapabilityNode:
        """局部 2-Means 重平衡分裂。

        思路：
        1) 以 target_node.center 和 new_vec 初始化两个中心 C1/C2。
        2) 在局部候选项上迭代 2-Means（最多 10 轮）做重分配。
        3) 用最终两组构造 child1/child2，并统一收紧到 split_d_max。
        4) 返回新的父节点 P_new（children=[child1, child2]）。
        """

        def run_local_2means(item_vecs: list[np.ndarray], c1_init: np.ndarray, c2_init: np.ndarray) -> tuple[list[int], list[int]]:
            c1 = c1_init.copy()
            c2 = c2_init.copy()
            prev_assign: tuple[tuple[int, ...], tuple[int, ...]] | None = None

            for _ in range(10):
                group1: list[int] = []
                group2: list[int] = []
                for i, vec in enumerate(item_vecs):
                    d1 = float(np.linalg.norm(vec - c1))
                    d2 = float(np.linalg.norm(vec - c2))
                    if d1 <= d2:
                        group1.append(i)
                    else:
                        group2.append(i)

                # 避免空簇：若某组为空，从另一组挪一个“离本组中心最远”的样本。
                if not group1 and group2:
                    move_i = max(group2, key=lambda idx: float(np.linalg.norm(item_vecs[idx] - c2)))
                    group2.remove(move_i)
                    group1.append(move_i)
                elif not group2 and group1:
                    move_i = max(group1, key=lambda idx: float(np.linalg.norm(item_vecs[idx] - c1)))
                    group1.remove(move_i)
                    group2.append(move_i)

                assign = (tuple(sorted(group1)), tuple(sorted(group2)))
                c1 = np.stack([item_vecs[i] for i in group1], axis=0).mean(axis=0)
                c2 = np.stack([item_vecs[i] for i in group2], axis=0).mean(axis=0)
                if assign == prev_assign:
                    return list(assign[0]), list(assign[1])
                prev_assign = assign

            # 达到最大迭代次数，返回最后一次分配
            assert prev_assign is not None
            return list(prev_assign[0]), list(prev_assign[1])

        c1_init = target_node.center.copy() if target_node.center is not None else new_vec.copy()
        c2_init = new_vec.copy()

        if target_node.is_leaf:
            # 叶子分裂：在 data_ids + new_id 上做局部重平衡。
            item_ids = list(dict.fromkeys(list(target_node.data_ids) + [new_id]))
            if len(item_ids) <= 1:
                # 兜底：极端情况下没有可分样本，退化成“原叶子 + 新叶子”。
                child1 = self._new_leaf_from_ids(item_ids, d_max=split_d_max)
                child2 = self._new_leaf_node(new_id, d_max=split_d_max)
            else:
                item_vecs = [new_vec if sid == new_id else self.vector_store[sid] for sid in item_ids]
                g1, g2 = run_local_2means(item_vecs, c1_init, c2_init)
                child1_ids = [item_ids[i] for i in g1]
                child2_ids = [item_ids[i] for i in g2]
                child1 = self._new_leaf_from_ids(child1_ids, d_max=split_d_max)
                child2 = self._new_leaf_from_ids(child2_ids, d_max=split_d_max)
        else:
            # 内部节点分裂：在 children + new_leaf 上做局部重平衡。
            new_leaf = self._new_leaf_node(new_id, d_max=split_d_max)
            items: list[CapabilityNode] = list(target_node.children) + [new_leaf]
            item_vecs: list[np.ndarray] = []
            for item in items:
                if item.center is None:
                    self._refresh_center(item)
                item_vecs.append(item.center.copy())

            g1, g2 = run_local_2means(item_vecs, c1_init, c2_init)
            group1_nodes = [items[i] for i in g1]
            group2_nodes = [items[i] for i in g2]

            def build_child(nodes: list[CapabilityNode]) -> CapabilityNode:
                if len(nodes) == 1:
                    return nodes[0]
                internal = self._new_internal_node(split_d_max)
                internal.children = list(nodes)
                self._refresh_center(internal)
                return internal

            child1 = build_child(group1_nodes)
            child2 = build_child(group2_nodes)

        # 分裂后统一收紧两侧子树阈值，保证层级阈值单调递减。
        self._cap_subtree_d_max(child1, split_d_max)
        self._cap_subtree_d_max(child2, split_d_max)

        p_new = self._new_internal_node(split_d_max)
        p_new.children = [child1, child2]
        self._refresh_center(p_new)
        return p_new

    # -----------------------------
    # J 计算辅助函数
    # -----------------------------
    def _child_clusters(self, node: CapabilityNode) -> list[np.ndarray]:
        clusters: list[np.ndarray] = []
        for child in node.children:
            ids = self._get_subtree_ids(child)
            if not ids:
                continue
            mat = np.stack([self.vector_store[i] for i in ids], axis=0)
            clusters.append(mat)
        return clusters

    def _compute_j_children(self, node: CapabilityNode) -> float:
        clusters = self._child_clusters(node)
        if len(clusters) <= 1:
            return 0.0
        return self.objective.evaluate(clusters)

    def _compute_j_with_virtual_merge(self, node: CapabilityNode, nearest_idx: int, vector: np.ndarray) -> float:
        """假想把当前样本并入最近子节点后，当前层级的 J。"""
        clusters = []
        for i, child in enumerate(node.children):
            ids = self._get_subtree_ids(child)
            if not ids:
                continue
            mat = np.stack([self.vector_store[j] for j in ids], axis=0)
            if i == nearest_idx:
                mat = np.concatenate([mat, vector.reshape(1, -1)], axis=0)
            clusters.append(mat)
        if len(clusters) <= 1:
            return 0.0
        return self.objective.evaluate(clusters)

    # -----------------------------
    # 插入逻辑（核心）
    # -----------------------------
    def insert(self, node: CapabilityNode, vector: np.ndarray, data_id: str, path: list[CapabilityNode]) -> None:
        """递归插入引擎。

        path 用于在局部结构变更后向上刷新 center。
        """
        # Step 1: 边界判定
        # 1) root 且无子节点：创建第一个叶子分支
        # 2) 非 root 叶子：直接吸收 data_id（叶子簇内合并），不再无限下挂新叶
        if node.is_leaf:
            if node is self.root:
                node.children.append(self._new_leaf_node(data_id, d_max=node.d_max))
            else:
                before = len(node.data_ids)
                if data_id not in node.data_ids:
                    node.data_ids.append(data_id)
                after = len(node.data_ids)
                # 记录本次插入是否出现“节点样本量从 1 增长到 2”的事件
                if before == 1 and after == 2:
                    self._grew_1_to_2_in_last_insert = True
                self._refresh_center(node)
            self._refresh_center_upward(path)
            return

        # Step 2: 距离评估，找最近 child。
        dists = []
        for child in node.children:
            if child.center is None:
                self._refresh_center(child)
            d = float(np.linalg.norm(vector - child.center))
            dists.append(d)
        nearest_idx = int(np.argmin(np.asarray(dists)))
        d_min = dists[nearest_idx]
        nearest = node.children[nearest_idx]

        # 到达最底层后，只允许并入最近分支，不再 Create New / Split。
        current_layer = len(path)  # root = 1
        if current_layer >= self.cfg.max_layers:
            self.insert(nearest, vector, data_id, path + [nearest])
            self._refresh_center_upward(path)
            return

        # 状态 1: Create New（距离过远）
        if d_min >= node.d_max:
            node.children.append(self._new_leaf_node(data_id, d_max=node.d_max))
            # 核心修复：新增分支后执行同层重平衡，
            # 让旧兄弟簇中“更靠近新分支”的元素迁移过来，降低顺序依赖。
            self._rebalance_layer(node)
            self._refresh_center_upward(path)
            return

        # 具备相关性，比较 J_old 与 J_new。
        j_old = self._compute_j_children(node)
        j_new = self._compute_j_with_virtual_merge(node, nearest_idx, vector)

        # 状态 2: Merge（J 不下降）-> 并入最近分支并继续递归向下。
        if j_new >= j_old:
            self.insert(nearest, vector, data_id, path + [nearest])
            self._refresh_center_upward(path)
            return

        # 在 max_layers-1 层禁止 Split（否则会把原子树整体下压到 max_layers+1）。
        if current_layer >= self.cfg.max_layers - 1:
            self.insert(nearest, vector, data_id, path + [nearest])
            self._refresh_center_upward(path)
            return

        # 即使当前层不深，Split 也会把 nearest 整棵子树下压一层；若会突破全局层数上限则禁用 Split。
        nearest_subtree_depth = self._subtree_depth_edges(nearest)
        if current_layer + 2 + nearest_subtree_depth > self.cfg.max_layers:
            self.insert(nearest, vector, data_id, path + [nearest])
            self._refresh_center_upward(path)
            return

        # 状态 3: Split（J 下降且未触底）-> 触发局部重平衡分裂
        # 用统一比例衰减收紧阈值，再在 nearest 子树 + 新样本上执行局部 2-Means 重分配。
        split_d_max = node.d_max * self.cfg.decay_rate
        p_new = self._rebalance_split(nearest, vector, data_id, split_d_max)
        node.children[nearest_idx] = p_new
        self._refresh_center_upward(path)

    def insert_one(self, data_id: str, text: str) -> tuple[int, bool]:
        before = self.node_seq
        self._grew_1_to_2_in_last_insert = False
        vec = self.vectorizer.encode_one(text)
        self.vector_store[data_id] = vec
        self.insert(self.root, vec, data_id, path=[self.root])
        # 返回本次插入引入的新节点数量，供收敛判定使用
        return self.node_seq - before, self._grew_1_to_2_in_last_insert

    # -----------------------------
    # 打印与统计
    # -----------------------------
    def _subtree_size(self, node: CapabilityNode) -> int:
        return len(self._get_subtree_ids(node))

    def _iter_nodes(self, node: CapabilityNode | None = None) -> list[CapabilityNode]:
        if node is None:
            node = self.root
        out: list[CapabilityNode] = [node]
        for child in node.children:
            out.extend(self._iter_nodes(child))
        return out

    def depth(self) -> int:
        def _depth(n: CapabilityNode) -> int:
            if not n.children:
                return 0
            return 1 + max(_depth(c) for c in n.children)

        return _depth(self.root)

    def level_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}

        def _walk(n: CapabilityNode, lv: int) -> None:
            counts[lv] = counts.get(lv, 0) + 1
            for c in n.children:
                _walk(c, lv + 1)

        _walk(self.root, 0)
        return dict(sorted(counts.items(), key=lambda kv: kv[0]))

    def global_j(self) -> float:
        # 全局 J：对整棵树每个“有至少2个子节点”的内部节点计算局部 J，再求和。
        # 这样不会因叶子普遍单元素而恒为 0，更能反映层级结构质量。
        total = 0.0
        for n in self._iter_nodes(self.root):
            if len(n.children) >= 2:
                total += self._compute_j_children(n)
        return float(total)

    def print_tree(self, node: CapabilityNode | None = None, prefix: str = "", is_last: bool = True) -> None:
        """ASCII 树打印。"""
        if node is None:
            node = self.root

        connector = "└── " if is_last else "├── "
        center_norm = float(np.linalg.norm(node.center)) if node.center is not None else 0.0
        logging.info(
            "%s%s%s [subtree_size=%d, leaf_payload=%d, children=%d, d_max=%.6f, |center|=%.4f]",
            prefix,
            connector,
            node.node_id,
            self._subtree_size(node),
            len(node.data_ids),
            len(node.children),
            node.d_max,
            center_norm,
        )

        if not node.children:
            return
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            self.print_tree(child, child_prefix, i == len(node.children) - 1)

    def export_tree_dict(self, node: CapabilityNode | None = None) -> dict[str, Any]:
        if node is None:
            node = self.root
        return {
            "node_id": node.node_id,
            "subtree_size": self._subtree_size(node),
            "leaf_payload_size": len(node.data_ids),
            "children_count": len(node.children),
            "d_max": node.d_max,
            "center_norm": float(np.linalg.norm(node.center)) if node.center is not None else 0.0,
            "data_ids": list(node.data_ids),
            "children": [self.export_tree_dict(c) for c in node.children],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Incremental Hierarchical Clustering Tree")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=REPO_ROOT / "data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl",
    )
    parser.add_argument(
        "--output-tree-json",
        type=Path,
        default=None,
        help="最终能力树结构输出路径",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=None,
        help="最终统计信息输出路径",
    )
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=1000, help="-1 for full dataset")
    parser.add_argument("--max-layers", type=int, default=30, help="能力树最大层数（root=1）；最底层仅允许 Merge。")
    parser.add_argument(
        "--root-d-max",
        "--d-max",
        dest="root_d_max",
        type=float,
        default=0.35,
        help="根节点初始相关性阈值 D_max；后续节点在 split 过程中动态收缩。",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.95,
        help="Split 时 d_max 的统一比例衰减系数，split_d_max = node.d_max * decay_rate。",
    )
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=100, help="每处理多少条打印一次阶段快照")
    parser.add_argument(
        "--patience-no-1to2-growth",
        type=int,
        default=0,
        help="连续多少步没有发生“节点数据量 1->2 增长”则提前收敛；0 表示关闭早停",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在插入前随机打乱样本顺序（默认开启）。",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42, help="随机打乱样本顺序的种子。")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def stream_rows(path: Path, max_samples: int | None, shuffle: bool, shuffle_seed: int):
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            data_id = str(row.get("id", idx))
            text = str(row.get("CDT_description", "")).strip()
            if not text:
                text = str(row.get("T_description", "")).strip()
            if not text:
                continue
            rows.append((data_id, text))

    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(rows)

    if max_samples is not None:
        rows = rows[:max_samples]

    for item in rows:
        yield item


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    max_samples = None if args.max_samples == -1 else max(0, args.max_samples)
    input_jsonl = resolve_repo_path(args.input_jsonl)
    default_tree_json, default_summary_json = default_output_paths(input_jsonl)
    cfg = Config(
        input_jsonl=input_jsonl,
        output_tree_json=resolve_repo_path(args.output_tree_json) if args.output_tree_json else default_tree_json,
        output_summary_json=(
            resolve_repo_path(args.output_summary_json) if args.output_summary_json else default_summary_json
        ),
        embedding_model=args.embedding_model,
        device=args.device,
        max_samples=max_samples,
        max_layers=max(1, args.max_layers),
        root_d_max=max(1e-8, args.root_d_max),
        decay_rate=max(1e-8, args.decay_rate),
        epsilon=max(1e-12, args.epsilon),
        log_every=max(1, args.log_every),
        patience_no_1to2_growth=max(0, args.patience_no_1to2_growth),
        shuffle=bool(args.shuffle),
        shuffle_seed=args.shuffle_seed,
        log_level=args.log_level,
    )
    if not cfg.input_jsonl.exists():
        raise FileNotFoundError(f"Input not found: {cfg.input_jsonl}")

    tree = IncrementalHierarchicalTree(cfg)
    processed = 0
    no_1to2_streak = 0
    for processed, (data_id, text) in enumerate(
        stream_rows(cfg.input_jsonl, cfg.max_samples, cfg.shuffle, cfg.shuffle_seed), start=1
    ):
        _created, grew_1_to_2 = tree.insert_one(data_id, text)
        if grew_1_to_2:
            no_1to2_streak = 0
        else:
            no_1to2_streak += 1

        if processed % cfg.log_every == 0:
            logging.info(
                "Processed=%d | depth=%d | level_counts=%s | global_J=%.6f | no_1to2_streak=%d",
                processed,
                tree.depth(),
                tree.level_counts(),
                tree.global_j(),
                no_1to2_streak,
            )

        if cfg.patience_no_1to2_growth > 0 and no_1to2_streak >= cfg.patience_no_1to2_growth:
            logging.info(
                "Early convergence: no 1->2 node growth for %d consecutive steps at processed=%d.",
                no_1to2_streak,
                processed,
            )
            break

    if processed == 0:
        raise ValueError("No valid samples found.")

    logging.info("All samples inserted: %d", processed)
    tree.print_tree()

    cfg.output_tree_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    tree_payload = tree.export_tree_dict()
    summary_payload = {
        "processed": processed,
        "total_nodes": len(tree._iter_nodes(tree.root)),
        "depth": tree.depth(),
        "level_counts": tree.level_counts(),
        "global_J": tree.global_j(),
        "max_layers": cfg.max_layers,
        "root_d_max": cfg.root_d_max,
        "decay_rate": cfg.decay_rate,
        "patience_no_1to2_growth": cfg.patience_no_1to2_growth,
        "final_no_1to2_streak": no_1to2_streak,
        "converged_early": bool(
            cfg.patience_no_1to2_growth > 0 and no_1to2_streak >= cfg.patience_no_1to2_growth
        ),
    }
    with cfg.output_tree_json.open("w", encoding="utf-8") as f:
        json.dump(tree_payload, f, ensure_ascii=False, indent=2)
    with cfg.output_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    logging.info("Saved tree to: %s", cfg.output_tree_json)
    logging.info("Saved summary to: %s", cfg.output_summary_json)


if __name__ == "__main__":
    main()
