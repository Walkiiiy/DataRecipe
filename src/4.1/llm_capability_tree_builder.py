"""LLM-driven incremental capability tree builder on raw instruction data.

This script drops CDT descriptions and builds the tree directly from raw samples
(instruction + input + output/response).

Core loop for each new sample:
1) Ask LLM to route: assign to an existing node, or create a new node under a parent.
2) If assign, compare two local parent-level objectives (J):
   - Add: add sample into target node.
   - Split: bisect target node with KMeans(k=2), replacing target by two new nodes.
   Choose split only when J_split > J_add.
3) If split is chosen, ask LLM to generate two contrasted new cluster names
   based on old name + representative samples nearest each new centroid.

Outputs:
- capability_tree_final.json (named tree)
- capability_tree_summary.json (run summary)
- capability_tree_decisions.jsonl (per-sample decisions)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from clustering_objective import ObjectiveEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "https://api.deepseek.com"

try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    torch = None
    SentenceTransformer = None

try:
    from sklearn.cluster import KMeans
except Exception:  # noqa: BLE001
    KMeans = None


ROUTING_SYSTEM_PROMPT = (
    "You are a capability-tree routing engine.\n"
    "Given one new sample and the current named tree, decide ONE action:\n"
    "1) assign: sample belongs to an existing node\n"
    "2) create: sample does not fit existing nodes; create a new node under a parent\n"
    "Return strict JSON only."
)

NEW_NAME_SYSTEM_PROMPT = (
    "You are an expert capability taxonomy namer.\n"
    "Generate one highly specific cluster name for a new node from the given sample and parent context.\n"
    "Name should clearly reflect the sample's unique data characteristic.\n"
    "Return strict JSON only."
)

SPLIT_NAME_SYSTEM_PROMPT = (
    "You are an expert capability taxonomy namer for split clusters.\n"
    "Given old cluster name and two representative samples from two new subclusters,\n"
    "produce two contrasted new names that preserve shared core while highlighting differences.\n"
    "Return strict JSON only."
)


@dataclass
class SampleRecord:
    sample_id: str
    row: dict[str, Any]
    raw_text: str


@dataclass
class Config:
    input_jsonl: Path
    output_tree_json: Path
    output_summary_json: Path
    output_decisions_jsonl: Path
    embedding_model: str
    device: str
    max_samples: int | None
    shuffle: bool
    shuffle_seed: int | None
    api_key: str
    base_url: str
    model: str
    timeout: int
    max_retries: int
    retry_base_delay: float
    route_max_tokens: int
    name_max_tokens: int
    temperature: float
    log_every: int
    print_tree_every: int
    tree_print_max_lines: int
    patience_structure_stable: int
    max_leaf_nodes: int
    min_node_size_for_split: int
    stale_singleton_rounds: int
    max_sample_chars_for_prompt: int
    max_tree_chars_for_prompt: int
    log_level: str


@dataclass
class TreeNode:
    node_id: str
    cluster_name: str = ""
    data_ids: list[str] = field(default_factory=list)
    children: list["TreeNode"] = field(default_factory=list)
    center: np.ndarray | None = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class DenseVectorizer:
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


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.calls = 0

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        body_bytes = json.dumps(payload).encode("utf-8")

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                self.calls += 1
                req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:  # noqa: S310
                    body = resp.read().decode("utf-8", errors="replace")
                    status = int(getattr(resp, "status", 200))
                    if status in {429, 500, 502, 503, 504}:
                        raise RuntimeError(f"retryable HTTP {status}: {body[:300]}")
                    if status >= 400:
                        raise RuntimeError(f"non-retryable HTTP {status}: {body[:300]}")
                    data = json.loads(body)
                    return str(data["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as exc:
                status = int(getattr(exc, "code", 0))
                detail = exc.read().decode("utf-8", errors="replace")
                if status in {429, 500, 502, 503, 504} and attempt < self.cfg.max_retries:
                    backoff = self.cfg.retry_base_delay * (2 ** (attempt - 1))
                    jitter = random.uniform(0, 0.25 * backoff)
                    time.sleep(backoff + jitter)
                    continue
                raise RuntimeError(f"HTTP {status}: {detail[:300]}") from exc
            except Exception as exc:  # noqa: BLE001
                if attempt == self.cfg.max_retries:
                    raise RuntimeError(f"LLM request failed after retries: {exc}") from exc
                backoff = self.cfg.retry_base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.25 * backoff)
                time.sleep(backoff + jitter)
        raise RuntimeError("Unreachable")


class LLMCapabilityTreeBuilder:
    def __init__(self, cfg: Config):
        if KMeans is None:
            raise ImportError("Missing scikit-learn. Please install scikit-learn first.")

        self.cfg = cfg
        self.objective = ObjectiveEvaluator(eps=1e-5)
        self.vectorizer = DenseVectorizer(cfg.embedding_model, cfg.device)

        self.node_seq = 0
        self.root = self._new_node(cluster_name="")
        self.nodes: dict[str, TreeNode] = {self.root.node_id: self.root}
        self.parent: dict[str, str | None] = {self.root.node_id: None}

        self.vector_store: dict[str, np.ndarray] = {}
        self.text_store: dict[str, str] = {}
        self.row_store: dict[str, dict[str, Any]] = {}

        self.stats: dict[str, int] = {
            "create_initial": 0,
            "create_by_router": 0,
            "assign_add": 0,
            "assign_add_skip_split_small_node": 0,
            "assign_split": 0,
            "router_invalid_fallback": 0,
            "prune_stale_singleton": 0,
        }

        # Track node growth for optional stale singleton pruning (all non-root nodes).
        self._node_prev_counts: dict[str, int] = {}
        self._node_idle_rounds: dict[str, int] = {}

    # -----------------
    # Node helpers
    # -----------------
    def _new_node_id(self) -> str:
        node_id = f"N{self.node_seq}"
        self.node_seq += 1
        return node_id

    def _new_node(self, cluster_name: str, data_ids: list[str] | None = None) -> TreeNode:
        return TreeNode(node_id=self._new_node_id(), cluster_name=cluster_name.strip(), data_ids=data_ids or [])

    def _register_subtree(self, node: TreeNode, parent_id: str | None) -> None:
        self.nodes[node.node_id] = node
        self.parent[node.node_id] = parent_id
        for child in node.children:
            self._register_subtree(child, node.node_id)

    def _unregister_subtree(self, node: TreeNode) -> None:
        for child in node.children:
            self._unregister_subtree(child)
        self.nodes.pop(node.node_id, None)
        self.parent.pop(node.node_id, None)

    def _collect_subtree_ids(self, node: TreeNode) -> list[str]:
        ids = list(node.data_ids)
        for child in node.children:
            ids.extend(self._collect_subtree_ids(child))
        return list(dict.fromkeys(ids))

    def _refresh_center(self, node: TreeNode) -> None:
        ids = self._collect_subtree_ids(node)
        if not ids:
            node.center = None
            return
        mat = np.stack([self.vector_store[sid] for sid in ids], axis=0)
        node.center = mat.mean(axis=0)

    def _refresh_upward(self, from_node_id: str) -> None:
        cur: str | None = from_node_id
        while cur is not None:
            self._refresh_center(self.nodes[cur])
            cur = self.parent.get(cur)

    def _node_depth(self, node_id: str) -> int:
        depth = 0
        cur = node_id
        while True:
            parent_id = self.parent.get(cur)
            if parent_id is None:
                return depth
            depth += 1
            cur = parent_id

    @staticmethod
    def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _subtree_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        def _count(node: TreeNode) -> int:
            total = len(node.data_ids)
            for child in node.children:
                total += _count(child)
            counts[node.node_id] = total
            return total

        _count(self.root)
        return counts

    @staticmethod
    def _label_with_counts(node: TreeNode, subtree_count: int) -> str:
        suffix = f"[{node.node_id} | data={subtree_count}, direct={len(node.data_ids)}]"
        if node.cluster_name.strip():
            return f"{node.cluster_name} {suffix}"
        return suffix

    def _build_named_tree_lines(self) -> list[str]:
        counts = self._subtree_counts()
        lines = [
            "=== Capability Tree (Named Clusters) ===",
            self._label_with_counts(self.root, counts.get(self.root.node_id, 0)),
        ]

        def walk(node: TreeNode, prefix: str) -> None:
            for idx, child in enumerate(node.children):
                is_last = idx == len(node.children) - 1
                connector = "└── " if is_last else "├── "
                label = self._label_with_counts(child, counts.get(child.node_id, 0))
                lines.append(f"{prefix}{connector}{label}")
                walk(child, prefix + ("    " if is_last else "│   "))

        walk(self.root, "")
        return lines

    def render_named_tree_for_prompt(self) -> str:
        tree_text = "\n".join(self._build_named_tree_lines())
        max_chars = int(self.cfg.max_tree_chars_for_prompt)
        if max_chars > 0 and len(tree_text) > max_chars:
            keep_head = int(max_chars * 0.7)
            keep_tail = max_chars - keep_head
            tree_text = f"{tree_text[:keep_head]}\n...\n{tree_text[-keep_tail:]}"
        return tree_text

    def render_named_tree_for_cli(self) -> str:
        lines = self._build_named_tree_lines()
        max_lines = max(0, self.cfg.tree_print_max_lines)
        if max_lines > 0 and len(lines) > max_lines:
            keep_head = max(2, int(max_lines * 0.7))
            keep_tail = max(1, max_lines - keep_head - 1)
            lines = lines[:keep_head] + ["..."] + lines[-keep_tail:]
        return "\n".join(lines)

    def _iter_nodes(self, node: TreeNode | None = None) -> list[TreeNode]:
        if node is None:
            node = self.root
        out = [node]
        for child in node.children:
            out.extend(self._iter_nodes(child))
        return out

    def _iter_leaf_nodes_under(self, node: TreeNode) -> list[TreeNode]:
        return [n for n in self._iter_nodes(node) if n.node_id != self.root.node_id and n.is_leaf]

    def depth(self) -> int:
        def _depth(n: TreeNode) -> int:
            if not n.children:
                return 0
            return 1 + max(_depth(c) for c in n.children)

        return _depth(self.root)

    def level_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}

        def _walk(n: TreeNode, lv: int) -> None:
            counts[lv] = counts.get(lv, 0) + 1
            for c in n.children:
                _walk(c, lv + 1)

        _walk(self.root, 0)
        return dict(sorted(counts.items(), key=lambda kv: kv[0]))

    def leaf_count(self) -> int:
        return len(self._iter_leaf_nodes_under(self.root))

    def recompute_all_centers(self) -> None:
        # Refresh bottom-up so parent centers are always based on latest children/data.
        for node in reversed(self._iter_nodes(self.root)):
            self._refresh_center(node)

    def _update_node_stale_tracking(self) -> dict[str, int]:
        current_counts: dict[str, int] = {}
        for node in self._iter_nodes(self.root):
            if node.node_id == self.root.node_id:
                continue
            current_counts[node.node_id] = self._subtree_size(node)

        current_ids = set(current_counts.keys())
        for node_id, count in current_counts.items():
            prev_count = self._node_prev_counts.get(node_id)
            if prev_count is None or count > prev_count:
                self._node_idle_rounds[node_id] = 0
            else:
                self._node_idle_rounds[node_id] = self._node_idle_rounds.get(node_id, 0) + 1
            self._node_prev_counts[node_id] = count

        removed_ids = set(self._node_prev_counts.keys()) - current_ids
        for node_id in removed_ids:
            self._node_prev_counts.pop(node_id, None)
            self._node_idle_rounds.pop(node_id, None)

        return current_counts

    def prune_stale_singleton_nodes(self, stale_rounds: int) -> list[dict[str, Any]]:
        """Prune non-root nodes stale at singleton size and merge data ids into parent."""
        if stale_rounds <= 0:
            return []

        current_counts = self._update_node_stale_tracking()
        candidate_ids = [
            node_id
            for node_id, count in current_counts.items()
            if int(count) == 1 and int(self._node_idle_rounds.get(node_id, 0)) >= stale_rounds
        ]
        if not candidate_ids:
            return []

        # Deep-first pruning keeps parent pointers stable while removing descendants.
        candidate_ids.sort(key=lambda nid: self._node_depth(nid), reverse=True)
        pruned: list[dict[str, Any]] = []

        for node_id in candidate_ids:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            parent_id = self.parent.get(node_id)
            if parent_id is None:
                continue
            parent_node = self.nodes.get(parent_id)
            if parent_node is None:
                continue

            # Re-check against current tree state.
            if self._subtree_size(node) != 1:
                continue
            if int(self._node_idle_rounds.get(node_id, 0)) < stale_rounds:
                continue

            moved_sample_ids = self._collect_subtree_ids(node)
            removed_node_ids = [n.node_id for n in self._iter_nodes(node)]
            for sid in moved_sample_ids:
                if sid not in parent_node.data_ids:
                    parent_node.data_ids.append(sid)

            parent_node.children = [c for c in parent_node.children if c.node_id != node_id]
            self._unregister_subtree(node)
            self._refresh_upward(parent_id)

            for rid in removed_node_ids:
                self._node_prev_counts.pop(rid, None)
                self._node_idle_rounds.pop(rid, None)

            pruned.append(
                {
                    "node_id": node_id,
                    "parent_node_id": parent_id,
                    "moved_sample_ids": moved_sample_ids,
                }
            )

        if pruned:
            self.stats["prune_stale_singleton"] += len(pruned)
        return pruned

    # -----------------
    # Objective + split
    # -----------------
    def _cluster_matrix(self, ids: list[str]) -> np.ndarray:
        mat = np.stack([self.vector_store[sid] for sid in ids], axis=0)
        return mat

    def _evaluate_parent_objective_add(self, parent_node: TreeNode, target_node_id: str, new_sample_id: str) -> float:
        clusters: list[np.ndarray] = []
        for child in parent_node.children:
            ids = self._collect_subtree_ids(child)
            if child.node_id == target_node_id:
                ids = list(dict.fromkeys(ids + [new_sample_id]))
            if ids:
                clusters.append(self._cluster_matrix(ids))
        if len(clusters) <= 1:
            return 0.0
        return float(self.objective.evaluate(clusters))

    def _bisect_ids(self, ids: list[str]) -> tuple[list[str], list[str]] | None:
        if len(ids) < 2:
            return None
        mat = self._cluster_matrix(ids)
        if mat.shape[0] < 2:
            return None

        try:
            km = KMeans(n_clusters=2, random_state=42, n_init="auto")
        except TypeError:
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(mat)

        g0 = [ids[i] for i, lb in enumerate(labels) if int(lb) == 0]
        g1 = [ids[i] for i, lb in enumerate(labels) if int(lb) == 1]
        if not g0 or not g1:
            return None
        return g0, g1

    def _evaluate_parent_objective_split(
        self,
        parent_node: TreeNode,
        target_node_id: str,
        split_groups: tuple[list[str], list[str]],
    ) -> float:
        clusters: list[np.ndarray] = []
        g0, g1 = split_groups
        for child in parent_node.children:
            if child.node_id == target_node_id:
                clusters.append(self._cluster_matrix(g0))
                clusters.append(self._cluster_matrix(g1))
            else:
                ids = self._collect_subtree_ids(child)
                if ids:
                    clusters.append(self._cluster_matrix(ids))
        if len(clusters) <= 1:
            return 0.0
        return float(self.objective.evaluate(clusters))

    def _best_match(self, vector: np.ndarray) -> tuple[str | None, float, float]:
        best_id: str | None = None
        best_dist = float("inf")
        best_cos = -1.0
        for node in self._iter_assignable_nodes():
            if node.center is None:
                self._refresh_center(node)
            if node.center is None:
                continue
            dist = float(np.linalg.norm(vector - node.center))
            cos = self._safe_cosine(vector, node.center)
            if dist < best_dist:
                best_dist = dist
                best_cos = cos
                best_id = node.node_id
        return best_id, best_dist, best_cos

    def _find_nearest_node_id(self, vector: np.ndarray) -> str | None:
        node_id, _dist, _cos = self._best_match(vector)
        return node_id

    def _find_nearest_leaf_under(self, node: TreeNode, vector: np.ndarray) -> TreeNode | None:
        leaves = self._iter_leaf_nodes_under(node)
        if not leaves:
            return None
        best_node: TreeNode | None = None
        best_dist = float("inf")
        for leaf in leaves:
            if leaf.center is None:
                self._refresh_center(leaf)
            if leaf.center is None:
                continue
            dist = float(np.linalg.norm(vector - leaf.center))
            if dist < best_dist:
                best_dist = dist
                best_node = leaf
        return best_node

    def _iter_assignable_nodes(self) -> list[TreeNode]:
        # Route assignment targets to leaf nodes only, which avoids collapsing
        # previously deeper subtrees when a later split is applied.
        return [n for n in self._iter_nodes(self.root) if n.node_id != self.root.node_id and n.is_leaf]

    def _top_route_candidates(self, vector: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        counts = self._subtree_counts()
        for node in self._iter_assignable_nodes():
            if node.center is None:
                self._refresh_center(node)
            if node.center is None:
                continue
            rows.append(
                {
                    "node_id": node.node_id,
                    "depth": self._node_depth(node.node_id),
                    "cluster_name": node.cluster_name,
                    "data_count": counts.get(node.node_id, 0),
                    "cosine": self._safe_cosine(vector, node.center),
                    "distance": float(np.linalg.norm(vector - node.center)),
                }
            )
        rows.sort(key=lambda x: x["distance"])
        return rows[: max(1, k)]

    def _parent_options_for_create(self, candidate_node_ids: list[str], max_options: int = 12) -> list[dict[str, Any]]:
        counts = self._subtree_counts()
        option_ids: set[str] = {self.root.node_id}

        # Include candidate nodes and a few ancestor hops for placement flexibility.
        for nid in candidate_node_ids:
            cur: str | None = nid
            hops = 0
            while cur is not None and hops < 4:
                option_ids.add(cur)
                cur = self.parent.get(cur)
                hops += 1

        # Also include some broad top-level branches.
        for child in self.root.children[:6]:
            option_ids.add(child.node_id)

        rows: list[dict[str, Any]] = []
        for nid in option_ids:
            node = self.nodes.get(nid)
            if node is None:
                continue
            rows.append(
                {
                    "node_id": nid,
                    "depth": self._node_depth(nid),
                    "cluster_name": node.cluster_name,
                    "data_count": counts.get(nid, 0),
                }
            )

        rows.sort(key=lambda x: (x["depth"], -int(x["data_count"])))
        return rows[: max(1, max_options)]

    def _pick_representative_id(self, ids: list[str]) -> str:
        mat = self._cluster_matrix(ids)
        centroid = mat.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(mat - centroid, axis=1)
        idx = int(np.argmin(dists))
        return ids[idx]

    # -----------------
    # LLM interaction
    # -----------------
    def _route_with_llm(self, sample: SampleRecord, vector: np.ndarray, client: LLMClient) -> dict[str, Any]:
        sample_text = sample.raw_text[: self.cfg.max_sample_chars_for_prompt]
        tree_text = self.render_named_tree_for_prompt()
        candidates = self._top_route_candidates(vector, k=5)
        candidate_ids = [str(c.get("node_id", "")) for c in candidates]
        parent_options = self._parent_options_for_create(candidate_ids, max_options=12)
        candidate_lines: list[str] = []
        for i, c in enumerate(candidates, start=1):
            name = str(c.get("cluster_name", "")).strip() or "[unnamed]"
            candidate_lines.append(
                f"{i}. {c['node_id']} | depth={c['depth']} | cosine={c['cosine']:.4f} | "
                f"data={c['data_count']} | name={name}"
            )
        candidate_text = "\n".join(candidate_lines) if candidate_lines else "(no candidate nodes)"
        parent_lines: list[str] = []
        for i, p in enumerate(parent_options, start=1):
            pname = str(p.get("cluster_name", "")).strip() or "[unnamed]"
            parent_lines.append(
                f"{i}. {p['node_id']} | depth={p['depth']} | data={p['data_count']} | name={pname}"
            )
        parent_text = "\n".join(parent_lines) if parent_lines else "(no parent options)"
        user_prompt = (
            "请判断这条数据属于哪一个簇，如果实在没有合适的簇加入就新建一个节点。\n"
            "你只需要做最小决策，不要输出树结构。\n\n"
            "当前整棵树（带节点ID与计数）:\n"
            f"{tree_text}\n\n"
            "新数据（原始样本文本）:\n"
            f"{sample_text}\n\n"
            "可加入候选叶子簇（按embedding距离由近到远）:\n"
            f"{candidate_text}\n\n"
            "如果必须新建，可选父节点位置:\n"
            f"{parent_text}\n\n"
            "请严格返回JSON（二选一）:\n"
            "A) 加入已有簇:\n"
            '{"decision":"assign","target_node_id":"Nxx","reason":"..."}\n'
            "B) 新建簇:\n"
            '{"decision":"create","parent_node_id":"Nxx","new_cluster_name":"...","reason":"..."}\n'
            "规则:\n"
            "- 只输出一个JSON对象，不要输出任何树结构文本。\n"
            "- target_node_id 必须从候选叶子簇里选（只能加入叶子节点）。\n"
            "- parent_node_id 必须从可选父节点里选。\n"
            "- 优先加入(assign)；新建(create)是例外。\n"
            "- 只有当所有候选簇都明显不合适时才允许新建。\n"
            "- 若候选第一名与第二名都高度相关，必须加入，不要新建。\n"
            "- 对上层分支，禁止因为措辞细差异而新建分支。\n"
            "- 若新建，new_cluster_name必须可复用、不过细，不要与现有簇仅做同义改写。\n"
            "- 新簇名要“刚好体现差异”：只表达与最相近已有簇的最小必要区别，不夸大差异。\n"
            "- 若parent不是ROOT，新簇名必须避免重复parent里的公共前缀信息（例如parent为intent_classification时，子簇不要再以intent_classification开头）。"
        )
        raw = client.chat(ROUTING_SYSTEM_PROMPT, user_prompt, max_tokens=self.cfg.route_max_tokens)
        obj = extract_json_object(raw) or {}
        obj["_raw"] = raw
        obj["_candidate_ids"] = candidate_ids
        obj["_parent_option_ids"] = [str(x.get("node_id", "")) for x in parent_options]
        obj["_candidate_infos"] = candidates
        return obj

    @staticmethod
    def _compact_name_part(parent_part: str, child_part: str) -> str:
        p = parent_part.strip()
        c = child_part.strip()
        if not p or not c:
            return c

        p_low = p.lower()
        c_low = c.lower()
        if c_low.startswith(p_low):
            rest = c[len(p) :].strip(" _-:/|")
            if rest:
                return rest

        p_tokens = [t for t in re.split(r"[_\-\s/:]+", p_low) if t]
        c_tokens_low = [t for t in re.split(r"[_\-\s/:]+", c_low) if t]
        c_tokens_raw = [t for t in re.split(r"[_\-\s/:]+", c) if t]
        if not p_tokens or not c_tokens_low or len(c_tokens_low) != len(c_tokens_raw):
            return c

        prefix_len = 0
        while (
            prefix_len < len(p_tokens)
            and prefix_len < len(c_tokens_low)
            and p_tokens[prefix_len] == c_tokens_low[prefix_len]
        ):
            prefix_len += 1

        if prefix_len >= 2 and prefix_len < len(c_tokens_raw):
            sep = "_" if "_" in c else " "
            compact = sep.join(c_tokens_raw[prefix_len:]).strip()
            if compact:
                return compact
        return c

    def _compact_child_name(self, parent_name: str, child_name: str) -> str:
        parent = parent_name.strip()
        child = child_name.strip()
        if not child:
            return child
        if not parent or parent.upper() == "ROOT":
            return child

        if "|" in child:
            child_parts = [x.strip() for x in child.split("|")]
            parent_parts = [x.strip() for x in parent.split("|")] if "|" in parent else [parent]
            out_parts: list[str] = []
            for i, cpart in enumerate(child_parts):
                ppart = parent_parts[i] if i < len(parent_parts) else parent_parts[-1]
                compact_part = self._compact_name_part(ppart, cpart)
                out_parts.append(compact_part if compact_part else cpart)
            compact = " | ".join(out_parts).strip(" |")
            return compact or child

        compact = self._compact_name_part(parent, child)
        return compact or child

    def _name_new_node(
        self,
        sample: SampleRecord,
        parent_node: TreeNode,
        client: LLMClient,
        candidate_infos: list[dict[str, Any]] | None = None,
    ) -> str:
        parent_name = parent_node.cluster_name if parent_node.cluster_name else "ROOT"
        parent_depth = self._node_depth(parent_node.node_id)
        if parent_depth <= 0:
            granularity = (
                "COARSE level: broad reusable capability family. "
                "Do NOT use overly specific phrasing tied to one narrow intent."
            )
        elif parent_depth <= 2:
            granularity = (
                "MID level: moderately specific but still reusable across many samples under this branch."
            )
        else:
            granularity = (
                "FINE level: can be more specific, but still summarize a pattern rather than a single phrasing."
            )
        contrast_lines: list[str] = []
        for i, info in enumerate(candidate_infos or [], start=1):
            if i > 2:
                break
            name = str(info.get("cluster_name", "")).strip() or "[unnamed]"
            contrast_lines.append(
                f"{i}. {info.get('node_id','')} | cosine={float(info.get('cosine',0.0)):.4f} | name={name}"
            )
        contrast_text = "\n".join(contrast_lines) if contrast_lines else "(none)"
        sample_text = sample.raw_text[: self.cfg.max_sample_chars_for_prompt]
        user_prompt = (
            "你需要为“新建节点”命名。\n"
            "命名目标：在可复用前提下，刚好体现与最近已有簇的差异程度，不要过度细分。\n\n"
            "Parent context name:\n"
            f"{parent_name}\n\n"
            f"Parent depth: {parent_depth}\n"
            f"Granularity policy: {granularity}\n\n"
            "Nearest existing clusters for contrast:\n"
            f"{contrast_text}\n\n"
            "Sample text:\n"
            f"{sample_text}\n\n"
            "Return strict JSON:\n"
            '{"cluster_name":"..."}\n'
            "要求:\n"
            "- 新簇名要体现“最小必要差异”，不要把轻微措辞差异写成新能力。\n"
            "- 与最近已有簇保持共性，只写关键区分点。\n"
            "- 名称可复用，不能只适配单条样本。\n"
            "- 子节点名称不要重复父节点中已表达的公共前缀信息。"
        )
        raw = client.chat(NEW_NAME_SYSTEM_PROMPT, user_prompt, max_tokens=self.cfg.name_max_tokens)
        obj = extract_json_object(raw) or {}
        name = str(obj.get("cluster_name", "")).strip()
        name = name or "未命名新簇 | Unnamed New Cluster"
        return self._compact_child_name(parent_name, name)

    def _name_split_children(
        self,
        old_name: str,
        group_a: list[str],
        group_b: list[str],
        client: LLMClient,
    ) -> tuple[str, str]:
        rep_a = self._pick_representative_id(group_a)
        rep_b = self._pick_representative_id(group_b)
        text_a = self.text_store.get(rep_a, "")[: self.cfg.max_sample_chars_for_prompt]
        text_b = self.text_store.get(rep_b, "")[: self.cfg.max_sample_chars_for_prompt]

        user_prompt = (
            "Old cluster name:\n"
            f"{old_name or '[unnamed cluster]'}\n\n"
            "Representative sample A (nearest to centroid of split A):\n"
            f"{text_a}\n\n"
            "Representative sample B (nearest to centroid of split B):\n"
            f"{text_b}\n\n"
            "Return strict JSON:\n"
            '{"name_a":"...","name_b":"..."}\n'
            "Requirements:\n"
            "- Preserve shared core implied by old cluster.\n"
            "- Express only the minimum necessary discriminative difference between A and B.\n"
            "- Do not over-specialize due to minor wording differences.\n"
            "- Names should stay reusable for future similar samples.\n"
            "- Each child name should be relative under old cluster and avoid repeating old cluster prefix."
        )
        raw = client.chat(SPLIT_NAME_SYSTEM_PROMPT, user_prompt, max_tokens=self.cfg.name_max_tokens)
        obj = extract_json_object(raw) or {}

        base = old_name or "能力簇 | Capability Cluster"
        name_a = str(obj.get("name_a", "")).strip() or f"{base} - 子簇A | Subcluster A"
        name_b = str(obj.get("name_b", "")).strip() or f"{base} - 子簇B | Subcluster B"
        name_a = self._compact_child_name(base, name_a)
        name_b = self._compact_child_name(base, name_b)
        return name_a, name_b

    # -----------------
    # Tree mutation
    # -----------------
    def _append_new_leaf(self, parent_node: TreeNode, cluster_name: str, sample_id: str) -> TreeNode:
        node = self._new_node(cluster_name=cluster_name, data_ids=[sample_id])
        node.center = self.vector_store[sample_id].copy()
        parent_node.children.append(node)
        self._register_subtree(node, parent_node.node_id)
        self._refresh_upward(parent_node.node_id)
        return node

    def process_one(
        self,
        sample: SampleRecord,
        client: LLMClient,
    ) -> dict[str, Any]:
        vec = self.vectorizer.encode_one(sample.raw_text)
        self.vector_store[sample.sample_id] = vec
        self.text_store[sample.sample_id] = sample.raw_text
        self.row_store[sample.sample_id] = sample.row

        if not self.root.children:
            init_name = self._name_new_node(sample, self.root, client, candidate_infos=None)
            new_node = self._append_new_leaf(self.root, init_name, sample.sample_id)
            self.stats["create_initial"] += 1
            return {
                "sample_id": sample.sample_id,
                "route_decision": "create_initial",
                "target_node_id": new_node.node_id,
                "parent_node_id": self.root.node_id,
                "new_cluster_name": init_name,
                "j_add": None,
                "j_split": None,
                "chosen": "create",
                "structure_changed": True,
            }

        route_obj = self._route_with_llm(sample, vec, client)
        decision = str(route_obj.get("decision", "")).strip().lower()
        chosen = ""

        # Fallback if invalid LLM output.
        if decision not in {"assign", "create"}:
            self.stats["router_invalid_fallback"] += 1
            nearest = self._find_nearest_node_id(vec)
            if nearest is None:
                nearest = self.root.node_id
            route_obj = {
                "decision": "assign",
                "target_node_id": nearest,
                "reason": "fallback_invalid_llm_output",
                "_raw": route_obj.get("_raw", ""),
            }
            decision = "assign"

        allowed_candidate_ids = set(str(x) for x in route_obj.get("_candidate_ids", []) if str(x).strip())
        allowed_parent_ids = set(str(x) for x in route_obj.get("_parent_option_ids", []) if str(x).strip())

        if decision == "assign":
            target_id_local = str(route_obj.get("target_node_id", "")).strip()
            if allowed_candidate_ids and target_id_local not in allowed_candidate_ids:
                self.stats["router_invalid_fallback"] += 1
                nearest = self._find_nearest_node_id(vec) or self.root.node_id
                route_obj["target_node_id"] = nearest
                prev_reason = str(route_obj.get("reason", "")).strip()
                route_obj["reason"] = (
                    f"{prev_reason}; target out of allowed candidates, fallback to nearest={nearest}"
                ).strip("; ")

        if decision == "create":
            parent_id = str(route_obj.get("parent_node_id", "")).strip()
            if allowed_parent_ids and parent_id not in allowed_parent_ids:
                self.stats["router_invalid_fallback"] += 1
                parent_id = self.root.node_id
                prev_reason = str(route_obj.get("reason", "")).strip()
                route_obj["reason"] = (
                    f"{prev_reason}; parent out of allowed options, fallback to root"
                ).strip("; ")
            parent_node = self.nodes.get(parent_id) or self.root
            cluster_name = str(route_obj.get("new_cluster_name", "")).strip()
            if not cluster_name:
                cluster_name = self._name_new_node(
                    sample,
                    parent_node,
                    client,
                    candidate_infos=route_obj.get("_candidate_infos", []),
                )
            cluster_name = self._compact_child_name(parent_node.cluster_name, cluster_name)

            created = self._append_new_leaf(parent_node, cluster_name, sample.sample_id)
            self.stats["create_by_router"] += 1
            chosen = "create"
            return {
                "sample_id": sample.sample_id,
                "route_decision": "create",
                "target_node_id": created.node_id,
                "parent_node_id": parent_node.node_id,
                "new_cluster_name": cluster_name,
                "j_add": None,
                "j_split": None,
                "chosen": chosen,
                "structure_changed": True,
                "router_reason": route_obj.get("reason", ""),
                "router_raw": route_obj.get("_raw", ""),
            }

        # Assign path
        target_id = str(route_obj.get("target_node_id", "")).strip()
        target_node = self.nodes.get(target_id)
        if target_node is None:
            nearest = self._find_nearest_node_id(vec)
            if nearest is None:
                nearest = self.root.node_id
            target_node = self.nodes[nearest]
            self.stats["router_invalid_fallback"] += 1

        # Hard constraint: data can only be added to leaf nodes.
        if not target_node.is_leaf:
            fallback_leaf = self._find_nearest_leaf_under(target_node, vec)
            if fallback_leaf is None:
                nearest_leaf_id = self._find_nearest_node_id(vec)
                if nearest_leaf_id is not None:
                    fallback_leaf = self.nodes.get(nearest_leaf_id)
            if fallback_leaf is not None:
                target_node = fallback_leaf
                self.stats["router_invalid_fallback"] += 1

        # Assigning to root is degenerate; force create under root.
        if target_node.node_id == self.root.node_id:
            cluster_name = self._name_new_node(
                sample,
                self.root,
                client,
                candidate_infos=route_obj.get("_candidate_infos", []),
            )
            created = self._append_new_leaf(self.root, cluster_name, sample.sample_id)
            self.stats["create_by_router"] += 1
            return {
                "sample_id": sample.sample_id,
                "route_decision": "assign_root_fallback_create",
                "target_node_id": created.node_id,
                "parent_node_id": self.root.node_id,
                "new_cluster_name": cluster_name,
                "j_add": None,
                "j_split": None,
                "chosen": "create",
                "structure_changed": True,
                "router_reason": route_obj.get("reason", ""),
                "router_raw": route_obj.get("_raw", ""),
            }

        parent_id = self.parent[target_node.node_id]
        if parent_id is None:
            raise RuntimeError("Non-root target node unexpectedly has no parent.")
        parent_node = self.nodes[parent_id]

        target_size_before_add = len(self._collect_subtree_ids(target_node))
        if target_size_before_add <= self.cfg.min_node_size_for_split:
            if sample.sample_id not in target_node.data_ids:
                target_node.data_ids.append(sample.sample_id)
            self._refresh_upward(target_node.node_id)

            self.stats["assign_add"] += 1
            self.stats["assign_add_skip_split_small_node"] += 1
            chosen = "add"
            return {
                "sample_id": sample.sample_id,
                "route_decision": "assign",
                "target_node_id": target_node.node_id,
                "parent_node_id": parent_node.node_id,
                "target_size_before_add": target_size_before_add,
                "split_min_size_threshold_m": self.cfg.min_node_size_for_split,
                "split_evaluation_skipped": True,
                "j_add": None,
                "j_split": None,
                "objective_scope": "local_parent_level",
                "objective_computed_locally": False,
                "chosen": chosen,
                "structure_changed": False,
                "router_reason": route_obj.get("reason", ""),
                "router_raw": route_obj.get("_raw", ""),
            }

        target_ids_plus = list(dict.fromkeys(self._collect_subtree_ids(target_node) + [sample.sample_id]))
        j_add = self._evaluate_parent_objective_add(parent_node, target_node.node_id, sample.sample_id)

        split_groups = self._bisect_ids(target_ids_plus)
        if split_groups is None:
            j_split = float("-inf")
        else:
            j_split = self._evaluate_parent_objective_split(parent_node, target_node.node_id, split_groups)

        if split_groups is not None and j_split > j_add:
            g0, g1 = split_groups
            old_name = target_node.cluster_name
            name_a, name_b = self._name_split_children(old_name, g0, g1, client)

            new_a = self._new_node(cluster_name=name_a, data_ids=g0)
            new_b = self._new_node(cluster_name=name_b, data_ids=g1)
            self._refresh_center(new_a)
            self._refresh_center(new_b)

            # Build a true hierarchical split: insert an intermediate parent node.
            split_parent_name = old_name.strip() if old_name and old_name.strip() else "Refined Capability Cluster"
            split_parent = self._new_node(cluster_name=split_parent_name, data_ids=[])
            split_parent.children = [new_a, new_b]
            self._refresh_center(split_parent)

            idx = next(i for i, c in enumerate(parent_node.children) if c.node_id == target_node.node_id)
            self._unregister_subtree(target_node)
            parent_node.children[idx : idx + 1] = [split_parent]
            self._register_subtree(split_parent, parent_node.node_id)
            self._refresh_upward(parent_node.node_id)

            self.stats["assign_split"] += 1
            chosen = "split"
            return {
                "sample_id": sample.sample_id,
                "route_decision": "assign",
                "target_node_id": target_node.node_id,
                "parent_node_id": parent_node.node_id,
                "split_parent_node_id": split_parent.node_id,
                "new_nodes": [new_a.node_id, new_b.node_id],
                "split_parent_name": split_parent_name,
                "new_cluster_names": [name_a, name_b],
                "target_size_before_add": target_size_before_add,
                "split_min_size_threshold_m": self.cfg.min_node_size_for_split,
                "split_evaluation_skipped": False,
                "j_add": j_add,
                "j_split": j_split,
                "objective_scope": "local_parent_level",
                "objective_computed_locally": True,
                "chosen": chosen,
                "structure_changed": True,
                "router_reason": route_obj.get("reason", ""),
                "router_raw": route_obj.get("_raw", ""),
            }

        if sample.sample_id not in target_node.data_ids:
            target_node.data_ids.append(sample.sample_id)
        self._refresh_upward(target_node.node_id)

        self.stats["assign_add"] += 1
        chosen = "add"
        return {
            "sample_id": sample.sample_id,
            "route_decision": "assign",
            "target_node_id": target_node.node_id,
            "parent_node_id": parent_node.node_id,
            "target_size_before_add": target_size_before_add,
            "split_min_size_threshold_m": self.cfg.min_node_size_for_split,
            "split_evaluation_skipped": False,
            "j_add": j_add,
            "j_split": None if split_groups is None else j_split,
            "objective_scope": "local_parent_level",
            "objective_computed_locally": True,
            "chosen": chosen,
            "structure_changed": False,
            "router_reason": route_obj.get("reason", ""),
            "router_raw": route_obj.get("_raw", ""),
        }

    # -----------------
    # Export
    # -----------------
    def _subtree_size(self, node: TreeNode) -> int:
        return len(self._collect_subtree_ids(node))

    def export_tree_dict(self, node: TreeNode | None = None) -> dict[str, Any]:
        if node is None:
            node = self.root
        subtree_count = self._subtree_size(node)
        leaf_center_vector = (
            [float(x) for x in node.center.tolist()] if node.is_leaf and node.center is not None else None
        )
        return {
            "node_id": node.node_id,
            "cluster_name": node.cluster_name,
            "data_count": subtree_count,
            "direct_data_count": len(node.data_ids),
            "subtree_size": subtree_count,
            "leaf_payload_size": len(node.data_ids),
            "children_count": len(node.children),
            "center_norm": float(np.linalg.norm(node.center)) if node.center is not None else 0.0,
            "leaf_center_vector": leaf_center_vector,
            "data_ids": list(node.data_ids),
            "children": [self.export_tree_dict(c) for c in node.children],
        }


def clean_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        txt = json.dumps(v, ensure_ascii=False)
    else:
        txt = str(v)
    return " ".join(txt.strip().split())


def compose_raw_text(row: dict[str, Any]) -> str:
    instruction = clean_text(row.get("instruction", ""))
    input_text = clean_text(row.get("input", ""))
    output_text = clean_text(row.get("output", row.get("response", "")))

    parts: list[str] = []
    if instruction:
        parts.append(f"Instruction: {instruction}")
    if input_text:
        parts.append(f"Input: {input_text}")
    if output_text:
        parts.append(f"Output: {output_text}")
    return "\n".join(parts).strip()


def extract_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    candidates = [text]
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        candidates.append(m.group(0))

    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:  # noqa: BLE001
            continue
    return None


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def default_output_paths(input_jsonl: Path) -> tuple[Path, Path, Path]:
    base = input_jsonl.parent
    return (
        base / "capability_tree_final.json",
        base / "capability_tree_summary.json",
        base / "capability_tree_decisions.jsonl",
    )


def load_samples(path: Path, max_samples: int | None, shuffle: bool, shuffle_seed: int | None) -> list[SampleRecord]:
    rows: list[SampleRecord] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = compose_raw_text(row)
            if not text:
                continue

            sid = str(row.get("id", idx))
            if not sid:
                sid = str(idx)
            if sid in seen:
                suffix = 1
                base = sid
                while f"{base}__{suffix}" in seen:
                    suffix += 1
                sid = f"{base}__{suffix}"
            seen.add(sid)
            rows.append(SampleRecord(sample_id=sid, row=row, raw_text=text))

    if shuffle:
        rng = random.Random(shuffle_seed)
        rng.shuffle(rows)

    if max_samples is not None:
        rows = rows[:max_samples]

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-driven incremental capability tree builder (raw data only)")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/alpaca-gpt4-data-en/train.jsonl"))
    parser.add_argument("--output-tree-json", type=Path, default=None)
    parser.add_argument("--output-summary-json", type=Path, default=None)
    parser.add_argument("--output-decisions-jsonl", type=Path, default=None)

    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--max-samples", type=int, default=1000, help="-1 for full dataset")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to shuffle samples before insertion (default: true).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Seed for shuffle. Omit for non-deterministic random sampling.",
    )

    parser.add_argument("--api-key", type=str, default=os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--route-max-tokens", type=int, default=512)
    parser.add_argument("--name-max-tokens", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--print-tree-every",
        type=int,
        default=20,
        help="Print a CLI tree snapshot every N processed samples; 0 disables periodic tree snapshots.",
    )
    parser.add_argument(
        "--tree-print-max-lines",
        type=int,
        default=120,
        help="Maximum CLI tree snapshot lines; 0 means no line truncation.",
    )
    parser.add_argument(
        "--patience-structure-stable",
        type=int,
        default=0,
        help="Early stop when no structure change (no create/split) for K consecutive samples; 0 disables.",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=0,
        help="Early stop when leaf node count is > m; 0 disables.",
    )
    parser.add_argument(
        "--min-node-size-for-split",
        type=int,
        default=3,
        help="Only evaluate local objective J for split when target node size is > m.",
    )
    parser.add_argument(
        "--stale-singleton-rounds",
        "--prune-first-layer-singleton-stale-rounds",
        dest="stale_singleton_rounds",
        type=int,
        default=0,
        help=(
            "If >0, remove any non-root node when its subtree size remains 1 and has not increased "
            "for N consecutive processed samples; move its data id(s) into parent."
        ),
    )
    parser.add_argument("--max-sample-chars-for-prompt", type=int, default=2400)
    parser.add_argument(
        "--max-tree-chars-for-prompt",
        type=int,
        default=0,
        help="Max chars for tree text in routing prompt; 0 means no truncation (full tree).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def run(cfg: Config) -> None:
    if not cfg.input_jsonl.exists():
        raise FileNotFoundError(f"Input not found: {cfg.input_jsonl}")

    samples = load_samples(cfg.input_jsonl, cfg.max_samples, cfg.shuffle, cfg.shuffle_seed)
    if not samples:
        raise ValueError("No valid samples found in input JSONL.")

    builder = LLMCapabilityTreeBuilder(cfg)
    logging.info(
        "Start building tree | samples=%d | shuffle=%s | shuffle_seed=%s | min_node_size_for_split=%d | stale_singleton_rounds=%d | max_leaf_nodes=%d | log_every=%d | print_tree_every=%d | patience_structure_stable=%d",
        len(samples),
        cfg.shuffle,
        str(cfg.shuffle_seed),
        cfg.min_node_size_for_split,
        cfg.stale_singleton_rounds,
        cfg.max_leaf_nodes,
        cfg.log_every,
        cfg.print_tree_every,
        cfg.patience_structure_stable,
    )

    cfg.output_tree_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_decisions_jsonl.parent.mkdir(parents=True, exist_ok=True)

    llm_client = LLMClient(cfg)
    processed = 0
    stable_streak = 0
    converged_early = False
    leaf_limit_triggered = False
    interrupted = False
    runtime_error: Exception | None = None
    try:
        with cfg.output_decisions_jsonl.open("w", encoding="utf-8", buffering=1) as decision_writer:
            for idx, sample in enumerate(samples, start=1):
                decision = builder.process_one(sample, llm_client)

                pruned_nodes: list[dict[str, Any]] = []
                if cfg.stale_singleton_rounds > 0:
                    pruned_nodes = builder.prune_stale_singleton_nodes(
                        cfg.stale_singleton_rounds
                    )
                    if pruned_nodes:
                        decision["pruned_stale_singleton_nodes"] = pruned_nodes
                        decision["structure_changed"] = True
                        pruned_ids = [str(x.get("node_id", "")) for x in pruned_nodes]
                        logging.info(
                            "Pruned stale singleton nodes at processed=%d: %s",
                            idx,
                            ",".join(pruned_ids),
                        )

                decision_writer.write(json.dumps(decision, ensure_ascii=False) + "\n")
                decision_writer.flush()
                processed = idx

                if bool(decision.get("structure_changed", False)):
                    stable_streak = 0
                else:
                    stable_streak += 1

                if idx % cfg.log_every == 0:
                    logging.info(
                        "Processed=%d | decision=%s | target=%s | depth=%d | nodes=%d | leaves=%d | stable_streak=%d | level_counts=%s | stats=%s",
                        idx,
                        str(decision.get("chosen", "")),
                        str(decision.get("target_node_id", "")),
                        builder.depth(),
                        len(builder._iter_nodes(builder.root)),
                        builder.leaf_count(),
                        stable_streak,
                        builder.level_counts(),
                        builder.stats,
                    )

                if cfg.print_tree_every > 0 and idx % cfg.print_tree_every == 0:
                    logging.info("Tree snapshot at processed=%d\n%s", idx, builder.render_named_tree_for_cli())

                if cfg.patience_structure_stable > 0 and stable_streak >= cfg.patience_structure_stable:
                    converged_early = True
                    logging.info(
                        "Early convergence triggered: structure stable for %d consecutive samples at processed=%d.",
                        stable_streak,
                        idx,
                    )
                    break

                if cfg.max_leaf_nodes > 0:
                    leaf_count = builder.leaf_count()
                    if leaf_count > cfg.max_leaf_nodes:
                        leaf_limit_triggered = True
                        logging.info(
                            "Leaf-node limit triggered: leaves=%d > max_leaf_nodes=%d at processed=%d.",
                            leaf_count,
                            cfg.max_leaf_nodes,
                            idx,
                        )
                        break
    except KeyboardInterrupt:
        interrupted = True
        logging.warning("KeyboardInterrupt received at processed=%d. Saving latest outputs.", processed)
    except Exception as exc:  # noqa: BLE001
        runtime_error = exc
        logging.exception("Runtime error at processed=%d. Saving latest outputs before re-raising.", processed)
    finally:
        builder.recompute_all_centers()
        tree_payload = builder.export_tree_dict()
        with cfg.output_tree_json.open("w", encoding="utf-8") as f:
            json.dump(tree_payload, f, ensure_ascii=False, indent=2)

        logging.info("Final tree snapshot\n%s", builder.render_named_tree_for_cli())

        summary = {
            "input_jsonl": str(cfg.input_jsonl),
            "output_tree_json": str(cfg.output_tree_json),
            "output_decisions_jsonl": str(cfg.output_decisions_jsonl),
            "processed": processed,
            "total_nodes": len(builder._iter_nodes(builder.root)),
            "depth": builder.depth(),
            "level_counts": builder.level_counts(),
            "stats": builder.stats,
            "termination": {
                "interrupted": interrupted,
                "leaf_limit_triggered": leaf_limit_triggered,
                "runtime_error": "" if runtime_error is None else str(runtime_error),
            },
            "convergence": {
                "patience_structure_stable": cfg.patience_structure_stable,
                "final_stable_streak": stable_streak,
                "converged_early": converged_early,
                "max_leaf_nodes": cfg.max_leaf_nodes,
                "final_leaf_count": builder.leaf_count(),
            },
            "objective": {
                "split_decision_scope": "local_parent_level",
                "computed_locally": True,
            },
            "pruning": {
                "stale_singleton_rounds": cfg.stale_singleton_rounds,
                "pruned_stale_singleton_total": builder.stats.get("prune_stale_singleton", 0),
            },
            "llm_calls": {
                "total": int(getattr(llm_client, "calls", 0)),
                "model": cfg.model,
                "base_url": cfg.base_url,
            },
            "config": {
                "embedding_model": cfg.embedding_model,
                "device": cfg.device,
                "shuffle": cfg.shuffle,
                "shuffle_seed": cfg.shuffle_seed,
                "max_leaf_nodes": cfg.max_leaf_nodes,
                "min_node_size_for_split": cfg.min_node_size_for_split,
                "stale_singleton_rounds": cfg.stale_singleton_rounds,
                "route_max_tokens": cfg.route_max_tokens,
                "name_max_tokens": cfg.name_max_tokens,
                "temperature": cfg.temperature,
            },
        }
        with cfg.output_summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logging.info("Saved tree: %s", cfg.output_tree_json)
        logging.info("Saved summary: %s", cfg.output_summary_json)
        logging.info("Saved decisions: %s", cfg.output_decisions_jsonl)

    if runtime_error is not None:
        raise runtime_error


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.api_key:
        raise ValueError("Missing API key. Set --api-key or DEEPSEEK_API_KEY/OPENAI_API_KEY.")

    max_samples = None if args.max_samples == -1 else max(0, args.max_samples)
    input_jsonl = resolve_repo_path(args.input_jsonl)
    default_tree_json, default_summary_json, default_decisions_jsonl = default_output_paths(input_jsonl)

    cfg = Config(
        input_jsonl=input_jsonl,
        output_tree_json=(
            resolve_repo_path(args.output_tree_json) if args.output_tree_json else default_tree_json
        ),
        output_summary_json=(
            resolve_repo_path(args.output_summary_json) if args.output_summary_json else default_summary_json
        ),
        output_decisions_jsonl=(
            resolve_repo_path(args.output_decisions_jsonl)
            if args.output_decisions_jsonl
            else default_decisions_jsonl
        ),
        embedding_model=args.embedding_model,
        device=args.device,
        max_samples=max_samples,
        shuffle=bool(args.shuffle),
        shuffle_seed=args.shuffle_seed,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=max(1, args.timeout),
        max_retries=max(1, args.max_retries),
        retry_base_delay=max(0.1, args.retry_base_delay),
        route_max_tokens=max(64, args.route_max_tokens),
        name_max_tokens=max(64, args.name_max_tokens),
        temperature=max(0.0, float(args.temperature)),
        log_every=max(1, args.log_every),
        print_tree_every=max(0, args.print_tree_every),
        tree_print_max_lines=max(0, args.tree_print_max_lines),
        patience_structure_stable=max(0, args.patience_structure_stable),
        max_leaf_nodes=max(0, args.max_leaf_nodes),
        min_node_size_for_split=max(0, args.min_node_size_for_split),
        stale_singleton_rounds=max(0, args.stale_singleton_rounds),
        max_sample_chars_for_prompt=max(200, args.max_sample_chars_for_prompt),
        max_tree_chars_for_prompt=max(0, args.max_tree_chars_for_prompt),
        log_level=args.log_level,
    )

    run(cfg)


if __name__ == "__main__":
    main()
