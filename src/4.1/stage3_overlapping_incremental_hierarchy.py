"""阶段三：基于 J 目标函数的动态增量层次聚类树（无 K-Means 分裂）。

核心机制：
1) Create New：与当前层所有分支距离都过大，则在当前层直接新增叶子分支。
2) Merge：与最近分支相关，且并入后 J 不下降，则并入并继续向更深层递归路由。
3) Split（层级分化）：与最近分支相关但并入会拉低 J，则创建新的中间父节点，将
   原最近分支与新样本叶子作为该父节点的两个子节点。

说明：
- 本实现复用 stage2 的 ObjectiveEvaluator，J = sum(Dens * Sep)。
- data_ids 在叶子节点持有；内部节点通过子树聚合计算 center。
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stage2_objective_evaluator import ObjectiveEvaluator

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
    """

    node_id: str
    center: np.ndarray | None = None
    data_ids: list[str] = field(default_factory=list)
    children: list["CapabilityNode"] = field(default_factory=list)

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
    d_max: float
    epsilon: float
    log_every: int
    patience_no_new_node: int
    log_level: str


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
        self.root = self._new_internal_node()

    # -----------------------------
    # 节点与中心更新辅助函数
    # -----------------------------
    def _new_node_id(self) -> str:
        node_id = f"N{self.node_seq}"
        self.node_seq += 1
        return node_id

    def _new_internal_node(self) -> CapabilityNode:
        return CapabilityNode(node_id=self._new_node_id(), center=None, data_ids=[], children=[])

    def _new_leaf_node(self, data_id: str) -> CapabilityNode:
        return CapabilityNode(
            node_id=self._new_node_id(),
            center=self.vector_store[data_id].copy(),
            data_ids=[data_id],
            children=[],
        )

    def _new_leaf_from_ids(self, ids: list[str]) -> CapabilityNode:
        leaf = CapabilityNode(
            node_id=self._new_node_id(),
            center=None,
            data_ids=list(dict.fromkeys(ids)),
            children=[],
        )
        self._refresh_center(leaf)
        return leaf

    def _get_subtree_ids(self, node: CapabilityNode) -> list[str]:
        if node.is_leaf:
            return list(node.data_ids)
        merged: list[str] = []
        for child in node.children:
            merged.extend(self._get_subtree_ids(child))
        # 子树中去重，避免潜在重复 id
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
        # Step 1: 边界判定。当前节点没有 children -> 直接挂一个新叶子。
        if node.is_leaf:
            # 若该节点本身已有 data_ids，说明它从“叶子”演化为“内部节点”。
            # 需要先把已有 payload 下沉为一个旧叶子，再追加新样本叶子。
            if node.data_ids:
                node.children.append(self._new_leaf_from_ids(node.data_ids))
                node.data_ids = []

            node.children.append(self._new_leaf_node(data_id))
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

        # 状态 1: Create New（距离过远）
        if d_min >= self.cfg.d_max:
            node.children.append(self._new_leaf_node(data_id))
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

        # 状态 3: Split（J 下降）-> 创建中间父节点 P_new，替换最近 child。
        p_new = self._new_internal_node()
        new_leaf = self._new_leaf_node(data_id)
        p_new.children = [nearest, new_leaf]
        self._refresh_center(p_new)
        node.children[nearest_idx] = p_new
        self._refresh_center_upward(path)

    def insert_one(self, data_id: str, text: str) -> int:
        before = self.node_seq
        vec = self.vectorizer.encode_one(text)
        self.vector_store[data_id] = vec
        self.insert(self.root, vec, data_id, path=[self.root])
        # 返回本次插入引入的新节点数量，供收敛判定使用
        return self.node_seq - before

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
        # 全局用“叶子集合”作为最终簇集合
        leaves = [n for n in self._iter_nodes(self.root) if n.is_leaf and n.data_ids]
        if len(leaves) <= 1:
            return 0.0
        clusters = []
        for leaf in leaves:
            mat = np.stack([self.vector_store[i] for i in leaf.data_ids], axis=0)
            clusters.append(mat)
        return self.objective.evaluate(clusters)

    def print_tree(self, node: CapabilityNode | None = None, prefix: str = "", is_last: bool = True) -> None:
        """ASCII 树打印。"""
        if node is None:
            node = self.root

        connector = "└── " if is_last else "├── "
        center_norm = float(np.linalg.norm(node.center)) if node.center is not None else 0.0
        logging.info(
            "%s%s%s [subtree_size=%d, leaf_payload=%d, children=%d, |center|=%.4f]",
            prefix,
            connector,
            node.node_id,
            self._subtree_size(node),
            len(node.data_ids),
            len(node.children),
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
            "center_norm": float(np.linalg.norm(node.center)) if node.center is not None else 0.0,
            "data_ids": list(node.data_ids),
            "children": [self.export_tree_dict(c) for c in node.children],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Incremental Hierarchical Clustering Tree")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl"),
    )
    parser.add_argument(
        "--output-tree-json",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/capability_tree_final.json"),
        help="最终能力树结构输出路径",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/capability_tree_summary.json"),
        help="最终统计信息输出路径",
    )
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-samples", type=int, default=1000, help="-1 for full dataset")
    parser.add_argument("--d-max", type=float, default=0.35, help="基础相关性阈值 D_max")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=100, help="每处理多少条打印一次阶段快照")
    parser.add_argument(
        "--patience-no-new-node",
        type=int,
        default=0,
        help="连续多少步未新增节点则提前收敛停止；0 表示关闭早停",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def stream_rows(path: Path, max_samples: int | None):
    with path.open("r", encoding="utf-8") as f:
        count = 0
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
            yield data_id, text
            count += 1
            if max_samples is not None and count >= max_samples:
                break


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    max_samples = None if args.max_samples == -1 else max(0, args.max_samples)
    cfg = Config(
        input_jsonl=args.input_jsonl,
        output_tree_json=args.output_tree_json,
        output_summary_json=args.output_summary_json,
        embedding_model=args.embedding_model,
        device=args.device,
        max_samples=max_samples,
        d_max=max(1e-8, args.d_max),
        epsilon=max(1e-12, args.epsilon),
        log_every=max(1, args.log_every),
        patience_no_new_node=max(0, args.patience_no_new_node),
        log_level=args.log_level,
    )
    if not cfg.input_jsonl.exists():
        raise FileNotFoundError(f"Input not found: {cfg.input_jsonl}")

    tree = IncrementalHierarchicalTree(cfg)
    processed = 0
    no_new_streak = 0
    for processed, (data_id, text) in enumerate(stream_rows(cfg.input_jsonl, cfg.max_samples), start=1):
        created = tree.insert_one(data_id, text)
        if created > 0:
            no_new_streak = 0
        else:
            no_new_streak += 1

        if processed % cfg.log_every == 0:
            logging.info(
                "Processed=%d | depth=%d | level_counts=%s | global_J=%.6f | no_new_streak=%d",
                processed,
                tree.depth(),
                tree.level_counts(),
                tree.global_j(),
                no_new_streak,
            )

        if cfg.patience_no_new_node > 0 and no_new_streak >= cfg.patience_no_new_node:
            logging.info(
                "Early convergence: no new node for %d consecutive steps at processed=%d.",
                no_new_streak,
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
        "patience_no_new_node": cfg.patience_no_new_node,
        "final_no_new_streak": no_new_streak,
        "converged_early": bool(cfg.patience_no_new_node > 0 and no_new_streak >= cfg.patience_no_new_node),
    }
    with cfg.output_tree_json.open("w", encoding="utf-8") as f:
        json.dump(tree_payload, f, ensure_ascii=False, indent=2)
    with cfg.output_summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    logging.info("Saved tree to: %s", cfg.output_tree_json)
    logging.info("Saved summary to: %s", cfg.output_summary_json)


if __name__ == "__main__":
    main()
