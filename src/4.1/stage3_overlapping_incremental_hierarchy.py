"""阶段三：基于 J 目标函数的多路径增量层次聚类树（Overlapping Incremental Hierarchical Clustering）。

实现要点：
1) 数据结构：CapabilityNode（树节点）
2) 递归插入：insert(node, v_i, data_id)
3) 软分配路由：一个样本可路由到多个子节点（重叠簇）
4) 拓扑演化：在同层 children 上比较 absorb / split / merge 三种状态的 J 值并贪心选择
5) 流式处理：按行读取 alpaca_cdt_profile.jsonl，逐条向根节点插入
6) 可观测性：每处理 N 条打印树深、层级节点数、全局 J 和 ASCII 树
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, deque
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

    属性说明：
    - node_id: 节点唯一 ID
    - center: 当前节点中心向量（由 data_ids 对应向量的均值计算）
    - data_ids: 属于该节点的数据 ID 列表
    - children: 子节点列表
    - level: 层级（Root=0）
    """

    node_id: str
    center: np.ndarray | None
    data_ids: list[str] = field(default_factory=list)
    children: list["CapabilityNode"] = field(default_factory=list)
    level: int = 0

    def add_data_id(self, data_id: str) -> None:
        """将样本 ID 加入节点（去重），用于维护节点样本集合。"""
        if data_id not in self.data_ids:
            self.data_ids.append(data_id)

    def recompute_center(self, vector_store: dict[str, np.ndarray]) -> None:
        """基于当前 data_ids 重新计算中心向量。"""
        if not self.data_ids:
            self.center = None
            return
        mat = np.stack([vector_store[did] for did in self.data_ids], axis=0)
        self.center = mat.mean(axis=0)

    def radius(self, vector_store: dict[str, np.ndarray]) -> float:
        """节点内部最大半径 D，用于软路由动态阈值。"""
        if self.center is None or len(self.data_ids) <= 1:
            return 0.0
        mat = np.stack([vector_store[did] for did in self.data_ids], axis=0)
        dists = np.linalg.norm(mat - self.center.reshape(1, -1), axis=1)
        return float(np.max(dists))


@dataclass
class TreeConfig:
    input_jsonl: Path
    max_samples: int | None
    embedding_model: str
    device: str
    log_every: int
    route_top_k: int
    route_radius_scale: float
    split_min_size: int
    split_density_threshold: float
    split_trials: int
    random_state: int
    epsilon: float
    max_tree_print_lines: int


class DenseVectorizer:
    """将 CDT_description 转为稠密向量。

    默认使用 sentence-transformers。若环境缺依赖，会给出明确报错。
    """

    def __init__(self, model_name: str, device: str):
        if SentenceTransformer is None:
            raise ImportError(
                "Missing sentence-transformers/torch. "
                "Please install: pip install sentence-transformers torch"
            )
        resolved_device = self._resolve_device(device)
        self.model = SentenceTransformer(model_name, device=resolved_device)

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
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        return np.asarray(vec, dtype=np.float64)


class OverlappingIncrementalHierarchy:
    """多路径增量层次聚类树引擎。"""

    def __init__(self, cfg: TreeConfig):
        self.cfg = cfg
        self.objective = ObjectiveEvaluator(eps=cfg.epsilon)
        self.vectorizer = DenseVectorizer(cfg.embedding_model, cfg.device)
        self.vector_store: dict[str, np.ndarray] = {}
        self.node_seq = 0
        self.rng = np.random.default_rng(cfg.random_state)

        # 初始化空 root 节点
        self.root = self._new_node(level=0, data_ids=[])

    def _new_node(self, level: int, data_ids: list[str]) -> CapabilityNode:
        node_id = f"N{self.node_seq}"
        self.node_seq += 1
        node = CapabilityNode(node_id=node_id, center=None, data_ids=list(dict.fromkeys(data_ids)), level=level)
        node.recompute_center(self.vector_store)
        return node

    def _cluster_matrix(self, node: CapabilityNode) -> np.ndarray:
        """将节点样本转换为矩阵，供目标函数计算。"""
        return np.stack([self.vector_store[did] for did in node.data_ids], axis=0)

    def _children_objective(self, node: CapabilityNode) -> float:
        """计算某节点 children 层的 J 值。"""
        valid_children = [c for c in node.children if c.data_ids]
        if len(valid_children) <= 1:
            return 0.0
        clusters = [self._cluster_matrix(c) for c in valid_children]
        return self.objective.evaluate(clusters)

    def _global_objective(self) -> float:
        """全局 J：按叶子节点视为最终簇集合。"""
        leaves = [n for n in self._iter_nodes() if not n.children and n.data_ids]
        if len(leaves) <= 1:
            return 0.0
        clusters = [self._cluster_matrix(leaf) for leaf in leaves]
        return self.objective.evaluate(clusters)

    def _iter_nodes(self) -> list[CapabilityNode]:
        out: list[CapabilityNode] = []
        q: deque[CapabilityNode] = deque([self.root])
        while q:
            cur = q.popleft()
            out.append(cur)
            q.extend(cur.children)
        return out

    def _tree_depth(self) -> int:
        return max((n.level for n in self._iter_nodes()), default=0)

    def _level_counts(self) -> dict[int, int]:
        counter: Counter[int] = Counter(n.level for n in self._iter_nodes())
        return dict(sorted(counter.items(), key=lambda kv: kv[0]))

    def _soft_route_children(self, node: CapabilityNode, vector: np.ndarray) -> list[CapabilityNode]:
        """软路由：基于动态阈值 D（节点半径）筛选候选子节点；若为空则 fallback Top-K 最近。"""
        if not node.children:
            return []

        dist_pairs: list[tuple[CapabilityNode, float]] = []
        for child in node.children:
            if child.center is None:
                continue
            dist = float(np.linalg.norm(vector - child.center))
            dist_pairs.append((child, dist))
        dist_pairs.sort(key=lambda x: x[1])

        selected: list[tuple[CapabilityNode, float]] = []
        for child, dist in dist_pairs:
            radius = child.radius(self.vector_store)
            threshold = max(radius * self.cfg.route_radius_scale, 1e-8)
            if dist <= threshold:
                selected.append((child, dist))

        # 若动态阈值没有命中，则兜底选择最近 Top-K，确保样本不会丢失。
        if not selected:
            selected = dist_pairs[: max(1, self.cfg.route_top_k)]
        else:
            selected = selected[: max(1, self.cfg.route_top_k)]

        return [c for c, _ in selected]

    def _two_means(
        self,
        mat: np.ndarray,
        trials: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """轻量 K-Means(K=2)。

        仅用 numpy 实现，避免额外依赖。返回两个簇索引数组；若失败返回 None。
        """
        n = mat.shape[0]
        if n < 2:
            return None

        best_sse = float("inf")
        best_assign: np.ndarray | None = None

        for _ in range(max(1, trials)):
            # 随机选两个不同样本作为初始中心
            seeds = self.rng.choice(n, size=2, replace=False)
            c0, c1 = mat[seeds[0]].copy(), mat[seeds[1]].copy()
            # 使用 -1 作为未初始化标签，避免首轮“全 0”时被误判收敛。
            assign = np.full(n, -1, dtype=np.int64)

            for _iter in range(20):
                d0 = np.linalg.norm(mat - c0.reshape(1, -1), axis=1)
                d1 = np.linalg.norm(mat - c1.reshape(1, -1), axis=1)
                new_assign = (d1 < d0).astype(np.int64)
                if np.array_equal(new_assign, assign):
                    break
                assign = new_assign
                idx0 = np.where(assign == 0)[0]
                idx1 = np.where(assign == 1)[0]
                if len(idx0) == 0 or len(idx1) == 0:
                    break
                c0 = mat[idx0].mean(axis=0)
                c1 = mat[idx1].mean(axis=0)

            idx0 = np.where(assign == 0)[0]
            idx1 = np.where(assign == 1)[0]
            if len(idx0) == 0 or len(idx1) == 0:
                continue

            sse = float(np.sum((mat[idx0] - c0.reshape(1, -1)) ** 2) + np.sum((mat[idx1] - c1.reshape(1, -1)) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_assign = assign.copy()

        if best_assign is None:
            return None
        return np.where(best_assign == 0)[0], np.where(best_assign == 1)[0]

    def _median_bisect(self, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """KMeans 失败时的稳健兜底：按主方向投影后以中位数一刀切分。"""
        n = mat.shape[0]
        if n < 2:
            return None

        centered = mat - mat.mean(axis=0, keepdims=True)
        # 取方差最大的维度作为近似主方向，避免 SVD 额外开销。
        axis = int(np.argmax(np.var(centered, axis=0)))
        proj = centered[:, axis]
        order = np.argsort(proj)
        mid = n // 2
        idx0 = order[:mid]
        idx1 = order[mid:]
        if len(idx0) == 0 or len(idx1) == 0:
            return None
        return idx0, idx1

    def _simulate_split(
        self,
        node: CapabilityNode,
        target_child: CapabilityNode,
    ) -> tuple[float, list[CapabilityNode]] | None:
        """构造状态 B：将 target_child 用 K=2 拆成两个新子节点（同层替换）。"""
        if len(target_child.data_ids) < self.cfg.split_min_size:
            return None

        mat = self._cluster_matrix(target_child)
        density = self.objective.density(mat)
        if density < self.cfg.split_density_threshold:
            return None

        split_idx = self._two_means(mat, trials=self.cfg.split_trials)
        if split_idx is None:
            split_idx = self._median_bisect(mat)
        if split_idx is None:
            return None

        idx0, idx1 = split_idx
        ids0 = [target_child.data_ids[i] for i in idx0.tolist()]
        ids1 = [target_child.data_ids[i] for i in idx1.tolist()]
        if not ids0 or not ids1:
            return None

        left = self._new_node(level=node.level + 1, data_ids=ids0)
        right = self._new_node(level=node.level + 1, data_ids=ids1)

        simulated = [c for c in node.children if c.node_id != target_child.node_id] + [left, right]
        if len(simulated) <= 1:
            return 0.0, simulated

        j_split = self.objective.evaluate([self._cluster_matrix(c) for c in simulated if c.data_ids])
        return j_split, simulated

    def _simulate_merge(self, node: CapabilityNode) -> tuple[float, list[CapabilityNode]] | None:
        """构造状态 C：合并当前 children 中中心最近的两个节点。"""
        if len(node.children) < 2:
            return None

        min_pair: tuple[int, int] | None = None
        min_dist = float("inf")
        for i in range(len(node.children)):
            for j in range(i + 1, len(node.children)):
                ci, cj = node.children[i], node.children[j]
                if ci.center is None or cj.center is None:
                    continue
                d = float(np.linalg.norm(ci.center - cj.center))
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)

        if min_pair is None:
            return None

        i, j = min_pair
        ci, cj = node.children[i], node.children[j]
        merged_ids = list(dict.fromkeys(ci.data_ids + cj.data_ids))
        merged = self._new_node(level=node.level + 1, data_ids=merged_ids)
        simulated = [c for idx, c in enumerate(node.children) if idx not in (i, j)] + [merged]
        if len(simulated) <= 1:
            return 0.0, simulated

        j_merge = self.objective.evaluate([self._cluster_matrix(c) for c in simulated if c.data_ids])
        return j_merge, simulated

    def _init_children_from_leaf_if_needed(self, node: CapabilityNode) -> None:
        """当节点是叶子且拥挤时，允许其“自分裂”生长出第一层子节点。"""
        if node.children:
            return
        if len(node.data_ids) < self.cfg.split_min_size:
            return

        mat = self._cluster_matrix(node)
        density = self.objective.density(mat)
        if density < self.cfg.split_density_threshold:
            return

        split_idx = self._two_means(mat, trials=self.cfg.split_trials)
        if split_idx is None:
            split_idx = self._median_bisect(mat)
        if split_idx is None:
            logging.debug(
                "Leaf split skipped at %s: split failed (size=%d, density=%.6f).",
                node.node_id,
                len(node.data_ids),
                density,
            )
            return
        idx0, idx1 = split_idx
        ids0 = [node.data_ids[i] for i in idx0.tolist()]
        ids1 = [node.data_ids[i] for i in idx1.tolist()]
        if not ids0 or not ids1:
            return

        node.children = [
            self._new_node(level=node.level + 1, data_ids=ids0),
            self._new_node(level=node.level + 1, data_ids=ids1),
        ]

    def _topology_evolution(self, node: CapabilityNode, routed_children: list[CapabilityNode]) -> None:
        """Step2 + Step3：在 node.children 上做 absorb/split/merge 比较并贪心执行。"""
        if len(node.children) <= 1:
            return

        j_absorb = self._children_objective(node)
        best_state = "absorb"
        best_j = j_absorb
        best_children = node.children

        # 状态 B：优先尝试在“本次被路由命中”的子节点上做 split（更贴近增量变化触发）。
        for child in routed_children:
            split_result = self._simulate_split(node, child)
            if split_result is None:
                continue
            j_split, simulated_children = split_result
            if j_split > best_j:
                best_state = "split"
                best_j = j_split
                best_children = simulated_children

        # 状态 C：尝试 merge 最近中心的两簇。
        merge_result = self._simulate_merge(node)
        if merge_result is not None:
            j_merge, simulated_children = merge_result
            if j_merge > best_j:
                best_state = "merge"
                best_j = j_merge
                best_children = simulated_children

        if best_state != "absorb":
            node.children = best_children
            logging.debug(
                "Topology evolution at %s(level=%d): %s (J %.6f -> %.6f)",
                node.node_id,
                node.level,
                best_state,
                j_absorb,
                best_j,
            )

    def insert(self, node: CapabilityNode, vector: np.ndarray, data_id: str) -> None:
        """递归插入入口。

        对应题目流程：
        Step1 软路由
        Step2 计算三种状态的 J
        Step3 贪心选择并更新拓扑
        """
        # 无论当前是否叶子，都先把样本挂到当前节点，保证祖先节点能追踪其覆盖样本集合。
        node.add_data_id(data_id)
        node.recompute_center(self.vector_store)

        if not node.children:
            # 叶子节点先吸收数据，再判断是否需要“生长出 children”。
            self._init_children_from_leaf_if_needed(node)
            return

        # Step1: 软分配路由，可能命中多个子节点，从而形成重叠簇。
        candidates = self._soft_route_children(node, vector)
        for child in candidates:
            self.insert(child, vector, data_id)

        # 当前节点中心可能因子树更新而变化，再次刷新。
        node.recompute_center(self.vector_store)

        # Step2 + Step3: 局部同层拓扑演化（absorb/split/merge）。
        self._topology_evolution(node, routed_children=candidates)

    def insert_one(self, data_id: str, text: str) -> None:
        """向 root 插入一条样本。"""
        vector = self.vectorizer.encode_one(text)
        self.vector_store[data_id] = vector
        self.insert(self.root, vector, data_id)

    def ascii_tree(self, max_lines: int) -> str:
        """生成 ASCII 树字符串。"""
        lines: list[str] = []

        def _dfs(cur: CapabilityNode, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            center_norm = float(np.linalg.norm(cur.center)) if cur.center is not None else 0.0
            lines.append(
                f"{prefix}{connector}{cur.node_id} "
                f"(L{cur.level}, size={len(cur.data_ids)}, children={len(cur.children)}, |center|={center_norm:.4f})"
            )
            if len(lines) >= max_lines:
                return
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(cur.children):
                _dfs(child, new_prefix, i == len(cur.children) - 1)
                if len(lines) >= max_lines:
                    return

        lines.append(f"{self.root.node_id} (L0, size={len(self.root.data_ids)}, children={len(self.root.children)})")
        for i, child in enumerate(self.root.children):
            _dfs(child, "", i == len(self.root.children) - 1)
            if len(lines) >= max_lines:
                break

        if len(lines) >= max_lines:
            lines.append("... (tree truncated)")
        return "\n".join(lines)

    def log_snapshot(self, processed: int) -> None:
        depth = self._tree_depth()
        level_counts = self._level_counts()
        global_j = self._global_objective()
        logging.info("Processed=%d | depth=%d | level_counts=%s | global_J=%.6f", processed, depth, level_counts, global_j)
        logging.info("Capability Tree:\n%s", self.ascii_tree(self.cfg.max_tree_print_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Overlapping Incremental Hierarchical Clustering")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl"),
        help="Stage1 输出文件，需包含 CDT_description 字段。",
    )
    parser.add_argument("--max-samples", type=int, default=1000, help="仅处理前 N 条；-1 表示全量。")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--route-top-k", type=int, default=2, help="软路由最多命中子节点数量。")
    parser.add_argument("--route-radius-scale", type=float, default=1.0, help="动态阈值系数：threshold = D * scale")
    parser.add_argument("--split-min-size", type=int, default=8, help="子簇最少样本数，低于该值不尝试 split。")
    parser.add_argument(
        "--split-density-threshold",
        type=float,
        default=3.0,
        help="当 Density 高于该阈值才尝试 split。",
    )
    parser.add_argument("--split-trials", type=int, default=5, help="K=2 初始化重启次数。")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=1e-5, help="目标函数防除零常数。")
    parser.add_argument("--max-tree-print-lines", type=int, default=120)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def stream_rows(path: Path, max_samples: int | None):
    """流式读取 JSONL，仅提取 (data_id, CDT_description) 并逐条产出。"""
    with path.open("r", encoding="utf-8") as f:
        yielded = 0
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            data_id = str(row.get("id", idx))
            text = str(row.get("CDT_description", "")).strip()
            if not text:
                # 若字段缺失，降级兼容 T_description
                text = str(row.get("T_description", "")).strip()
            yield data_id, text
            yielded += 1
            if max_samples is not None and yielded >= max_samples:
                break


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    max_samples = None if args.max_samples == -1 else max(0, args.max_samples)
    cfg = TreeConfig(
        input_jsonl=args.input_jsonl,
        max_samples=max_samples,
        embedding_model=args.embedding_model,
        device=args.device,
        log_every=max(1, args.log_every),
        route_top_k=max(1, args.route_top_k),
        route_radius_scale=max(0.1, args.route_radius_scale),
        split_min_size=max(2, args.split_min_size),
        split_density_threshold=max(0.0, args.split_density_threshold),
        split_trials=max(1, args.split_trials),
        random_state=args.random_state,
        epsilon=max(1e-12, args.epsilon),
        max_tree_print_lines=max(20, args.max_tree_print_lines),
    )

    if not cfg.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {cfg.input_jsonl}")

    engine = OverlappingIncrementalHierarchy(cfg)
    logging.info("Start streaming insert from %s", cfg.input_jsonl)

    processed = 0
    for i, (data_id, text) in enumerate(stream_rows(cfg.input_jsonl, cfg.max_samples), start=1):
        if not text:
            # 空文本跳过向量化，避免 embedding 模型报错
            continue
        engine.insert_one(data_id=data_id, text=text)
        processed = i
        if i % cfg.log_every == 0:
            engine.log_snapshot(i)

    if processed == 0:
        raise ValueError("No valid rows found in input file.")

    # 收尾再打印一次最终结构
    engine.log_snapshot(processed)


if __name__ == "__main__":
    main()
