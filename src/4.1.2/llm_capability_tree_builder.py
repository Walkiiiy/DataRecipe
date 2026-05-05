"""LLM-driven incremental capability tree builder on raw instruction data.

This script drops CDT descriptions and builds the tree directly from raw samples
(instruction + input + output/response).

Core loop for each new sample:
1) Optionally warm-start broad root anchors with embedding KMeans over raw sample text.
2) Route with LLM in llm/hybrid modes, or by nearest root embedding anchor in deterministic mode.
3) If assign, build local KMeans candidates for k=2..max_split_k and choose the
   valid candidate with the best Silhouette score.
4) Evaluate only that best local topology with the Lemma-1 global split gate:
   split iff structural-risk penalty is lower than expression-gain bound.
5) If split is chosen, ask LLM to generate contrasted child names based on the
   old name + representative samples nearest each new centroid.

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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from clustering_objective import CapabilitySplitEvaluator, LemmaSplitDecision

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
    from sklearn.metrics import silhouette_score
except Exception:  # noqa: BLE001
    KMeans = None
    silhouette_score = None


ROUTING_SYSTEM_PROMPT = (
    "You are a conservative capability-tree routing engine.\n"
    "A capability is a reusable task/skill pattern, not a topic, named entity, or one-off fact.\n"
    "Given one new sample and the current named tree, decide ONE action:\n"
    "1) assign: sample fits an existing reusable capability node\n"
    "2) create: sample requires a genuinely new reusable capability under a parent\n"
    "Prefer assign whenever the task operation, reasoning pattern, or output format matches an existing node.\n"
    "Return strict JSON only."
)

NEW_NAME_SYSTEM_PROMPT = (
    "You are an expert capability taxonomy namer.\n"
    "Generate a reusable capability name from samples and parent context.\n"
    "Name task operations, reasoning patterns, domains, and output formats; avoid one-off topics/entities.\n"
    "Return strict JSON only."
)

SPLIT_NAME_SYSTEM_PROMPT = (
    "You are an expert capability taxonomy namer for split clusters.\n"
    "Given old cluster name and representative samples from new subclusters,\n"
    "produce reusable contrasted names that preserve shared core while highlighting capability differences.\n"
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
    min_processed_before_early_stop: int
    max_leaf_nodes: int
    capability_text_mode: str
    warm_start_samples: int
    min_node_size_for_split: int
    max_split_k: int
    min_child_size_for_split: int
    intrinsic_dim: float
    split_gain_scale: float
    split_random_seed: int
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
        self.nonfinite_sanitized = 0
        self.zero_norm_sanitized = 0

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
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(arr)):
            self.nonfinite_sanitized += 1
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        norm = float(np.linalg.norm(arr))
        if not np.isfinite(norm) or norm <= 1e-12:
            self.zero_norm_sanitized += 1
            return np.zeros_like(arr, dtype=np.float64)
        return arr / norm


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
        if KMeans is None or silhouette_score is None:
            raise ImportError("Missing scikit-learn. Please install scikit-learn first.")

        self.cfg = cfg
        self.split_evaluator = CapabilitySplitEvaluator()
        self.vectorizer = DenseVectorizer(cfg.embedding_model, cfg.device)

        self.node_seq = 0
        self.root = self._new_node(cluster_name="")
        self.nodes: dict[str, TreeNode] = {self.root.node_id: self.root}
        self.parent: dict[str, str | None] = {self.root.node_id: None}

        self.vector_store: dict[str, np.ndarray] = {}
        self.text_store: dict[str, str] = {}
        self.row_store: dict[str, dict[str, Any]] = {}

        self.stats: dict[str, int] = {
            "create_by_router": 0,
            "assign_add": 0,
            "assign_add_skip_split_small_node": 0,
            "assign_split": 0,
            "assign_add_no_valid_silhouette_split": 0,
            "assign_add_lemma_reject": 0,
            "warm_start_create": 0,
            "warm_start_add": 0,
            "nonfinite_cluster_matrix_sanitized": 0,
            "split_candidate_numeric_warning": 0,
            "split_candidate_nonfinite_silhouette": 0,
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
            moved_sample_ids = self._collect_subtree_ids(node)
            if len(moved_sample_ids) != 1:
                continue
            if int(self._node_idle_rounds.get(node_id, 0)) < stale_rounds:
                continue
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
    def _sanitize_matrix_for_clustering(self, mat: np.ndarray) -> np.ndarray:
        arr = np.asarray(mat, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            self.stats["nonfinite_cluster_matrix_sanitized"] += 1
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _cluster_matrix(self, ids: list[str]) -> np.ndarray:
        mat = np.stack([self.vector_store[sid] for sid in ids], axis=0)
        return self._sanitize_matrix_for_clustering(mat)

    def _make_kmeans(self, n_clusters: int) -> KMeans:
        try:
            return KMeans(n_clusters=n_clusters, random_state=self.cfg.split_random_seed, n_init="auto")
        except TypeError:
            return KMeans(n_clusters=n_clusters, random_state=self.cfg.split_random_seed, n_init=10)

    def _cluster_ids_by_k(self, ids: list[str], k: int) -> tuple[list[list[str]], np.ndarray] | None:
        if len(ids) < k:
            return None
        mat = self._cluster_matrix(ids)
        if mat.shape[0] < k:
            return None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            labels = self._make_kmeans(k).fit_predict(mat)
        runtime_warnings = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        if runtime_warnings:
            self.stats["split_candidate_numeric_warning"] += 1
        unique_labels = sorted(set(int(x) for x in labels.tolist()))
        if len(unique_labels) != k:
            return None

        groups: list[list[str]] = []
        for label in unique_labels:
            group = [ids[i] for i, lb in enumerate(labels) if int(lb) == label]
            if not group:
                return None
            groups.append(group)
        return groups, labels

    def _best_silhouette_split(self, ids: list[str]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Choose the locally best topology by Silhouette only.

        Lemma-1 acceptance is intentionally evaluated only after this function
        picks the best local split candidate.
        """
        n = len(ids)
        if n < 3:
            return None, []

        mat = self._cluster_matrix(ids)
        max_k = min(int(self.cfg.max_split_k), n - 1)
        min_child_size = max(1, int(self.cfg.min_child_size_for_split))
        candidate_summaries: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for k in range(2, max_k + 1):
            if k * min_child_size > n:
                candidate_summaries.append(
                    {
                        "k": k,
                        "status": "skipped_min_child_size_capacity",
                        "min_child_size": min_child_size,
                    }
                )
                continue

            clustered = self._cluster_ids_by_k(ids, k)
            if clustered is None:
                candidate_summaries.append({"k": k, "status": "skipped_empty_missing_or_numeric_warning"})
                continue

            groups, labels = clustered
            child_sizes = [len(g) for g in groups]
            min_observed_child_size = min(child_sizes)
            if min_observed_child_size < min_child_size:
                candidate_summaries.append(
                    {
                        "k": k,
                        "status": "skipped_min_child_size",
                        "child_sizes": child_sizes,
                        "min_child_size": min_child_size,
                    }
                )
                continue

            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    silhouette = float(silhouette_score(mat, labels, metric="euclidean"))
                runtime_warnings = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
                if runtime_warnings:
                    self.stats["split_candidate_numeric_warning"] += 1
            except Exception as exc:  # noqa: BLE001
                candidate_summaries.append(
                    {
                        "k": k,
                        "status": "skipped_silhouette_error",
                        "error": str(exc),
                    }
                )
                continue

            if not np.isfinite(silhouette):
                self.stats["split_candidate_nonfinite_silhouette"] += 1
                candidate_summaries.append(
                    {
                        "k": k,
                        "status": "skipped_nonfinite_silhouette",
                        "silhouette": None,
                    }
                )
                continue

            summary = {
                "k": k,
                "status": "valid",
                "silhouette": silhouette,
                "child_sizes": child_sizes,
                "min_child_size": min_child_size,
            }
            candidate_summaries.append(summary)
            candidate = {
                "k": k,
                "silhouette": silhouette,
                "child_sizes": child_sizes,
                "groups": groups,
            }
            if best is None:
                best = candidate
            else:
                # Silhouette is primary; lower k is the deterministic tie-breaker.
                if silhouette > float(best["silhouette"]) or (
                    np.isclose(silhouette, float(best["silhouette"])) and k < int(best["k"])
                ):
                    best = candidate
        return best, candidate_summaries

    def _evaluate_lemma_split_gate(
        self,
        parent_ids: list[str],
        split_groups: list[list[str]],
    ) -> LemmaSplitDecision | None:
        if not split_groups:
            return None
        parent_vectors = self._cluster_matrix(parent_ids)
        child_vectors = [self._cluster_matrix(g) for g in split_groups]
        return self.split_evaluator.evaluate_lemma_gate(
            parent_vectors=parent_vectors,
            child_vectors=child_vectors,
            basis_count_before=self.leaf_count(),
            intrinsic_dim=self.cfg.intrinsic_dim,
            split_gain_scale=self.cfg.split_gain_scale,
        )

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

    def _route_with_strategy(
        self,
        sample: SampleRecord,
        vector: np.ndarray,
        client: LLMClient,
    ) -> dict[str, Any]:
        route_obj = self._route_with_llm(sample, vector, client)
        route_obj["_route_source"] = "llm"
        return route_obj

    def _pick_representative_id(self, ids: list[str]) -> str:
        mat = self._cluster_matrix(ids)
        centroid = mat.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(mat - centroid, axis=1)
        idx = int(np.argmin(dists))
        return ids[idx]

    def _pick_representative_ids(self, ids: list[str], count: int = 3) -> list[str]:
        if not ids:
            return []
        mat = self._cluster_matrix(ids)
        centroid = mat.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(mat - centroid, axis=1)
        order = np.argsort(dists)
        reps: list[str] = []
        for idx in order[: max(1, min(count, len(ids)))]:
            reps.append(ids[int(idx)])
        return reps

    @staticmethod
    def _fallback_cluster_name_from_row(row: dict[str, Any]) -> str:
        instruction = clean_text(row.get("instruction", ""))
        words = [w for w in re.split(r"[^A-Za-z0-9]+", instruction) if w]
        if words:
            return " ".join(words[:6]).title()

        input_text = clean_text(row.get("input", ""))
        words = [w for w in re.split(r"[^A-Za-z0-9]+", input_text) if w]
        if words:
            return "Input " + " ".join(words[:5]).title()

        output_text = clean_text(row.get("output", row.get("response", "")))
        words = [w for w in re.split(r"[^A-Za-z0-9]+", output_text) if w]
        if words:
            return "Response " + " ".join(words[:5]).title()
        return "General Instruction Following"

    def _append_leaf_with_ids(self, parent_node: TreeNode, cluster_name: str, sample_ids: list[str]) -> TreeNode:
        node = self._new_node(cluster_name=cluster_name, data_ids=list(dict.fromkeys(sample_ids)))
        self._refresh_center(node)
        parent_node.children.append(node)
        self._register_subtree(node, parent_node.node_id)
        self._refresh_upward(parent_node.node_id)
        return node

    def _name_warm_start_groups(
        self,
        groups: list[list[str]],
        sample_by_id: dict[str, SampleRecord],
        client: LLMClient,
    ) -> list[str]:
        if not groups:
            return []

        group_blocks: list[str] = []
        representative_ids: list[str] = []
        for idx, group in enumerate(groups, start=1):
            rep_ids = self._pick_representative_ids(group, count=3)
            representative_ids.append(rep_ids[0])
            samples_text = []
            for rep_idx, rep_id in enumerate(rep_ids, start=1):
                rep_text = self.text_store.get(rep_id, "")[: self.cfg.max_sample_chars_for_prompt]
                samples_text.append(f"Representative {rep_idx}:\n{rep_text}")
            group_blocks.append(f"Group {idx} | size={len(group)}\n" + "\n\n".join(samples_text))

        fallback_names = [
            self._fallback_cluster_name_from_row(sample_by_id[rep_id].row)
            for rep_id in representative_ids
        ]
        user_prompt = (
            "你需要为一组 embedding warm start 初始簇命名。\n"
            "这些簇来自样本文本的语义聚类，不提供也不允许使用数据集自带类别字段。\n\n"
            "命名原则：\n"
            "- 根层名字必须是粗粒度、可复用的 capability family。\n"
            "- 优先描述任务操作/认知过程/输出形式，例如 classification, explanation, extraction, planning, recommendation, rewriting, QA。\n"
            "- 不要用单条样本里的地点、人名、年份、物品、故事情节、冷门事实给根节点命名。\n"
            "- 如果一个簇内样本主题分散，请抽象成共同任务形式，不要挑某个代表样本当标题。\n\n"
            "Initial groups:\n"
            f"{chr(10).join(group_blocks)}\n\n"
            "Return strict JSON:\n"
            '{"names":["...","..."]}\n'
            "Requirements:\n"
            f"- Return exactly {len(groups)} names in the same order as the groups.\n"
            "- Names should be broad, reusable capability families, not sample titles.\n"
            "- Use only the representative sample text and group size shown above.\n"
            "- Do not invent dataset-specific category labels."
        )
        raw = client.chat(NEW_NAME_SYSTEM_PROMPT, user_prompt, max_tokens=self.cfg.name_max_tokens)
        obj = extract_json_object(raw) or {}
        raw_names = obj.get("names", [])
        if not isinstance(raw_names, list):
            raw_names = []

        names: list[str] = []
        for idx, fallback in enumerate(fallback_names):
            name = str(raw_names[idx]).strip() if idx < len(raw_names) else ""
            names.append(name or fallback or f"Warm Start Capability {idx + 1}")
        return names

    def warm_start_from_embeddings(
        self,
        samples: list[SampleRecord],
        client: LLMClient,
    ) -> list[dict[str, Any]]:
        """Initialize broad root leaves via silhouette-optimal KMeans on leading samples."""
        warm_start_n = min(max(0, int(self.cfg.warm_start_samples)), len(samples))
        if warm_start_n <= 0:
            return []

        seed_samples = samples[:warm_start_n]
        sample_by_id = {sample.sample_id: sample for sample in seed_samples}
        sample_ids: list[str] = []
        for sample in seed_samples:
            vec = self.vectorizer.encode_one(sample.raw_text)
            self.vector_store[sample.sample_id] = vec
            self.text_store[sample.sample_id] = sample.raw_text
            self.row_store[sample.sample_id] = sample.row
            sample_ids.append(sample.sample_id)

        best_candidate, _ = self._best_silhouette_split(sample_ids)
        if best_candidate is not None:
            groups = best_candidate["groups"]
            warm_start_k = best_candidate["k"]
        else:
            groups = [sample_ids]
            warm_start_k = 1

        decisions: list[dict[str, Any]] = []
        cluster_names = self._name_warm_start_groups(groups, sample_by_id, client)
        for group, cluster_name in zip(groups, cluster_names):
            node = self._append_leaf_with_ids(self.root, cluster_name, group)
            self.stats["warm_start_create"] += 1
            self.stats["warm_start_add"] += len(group)
            representative_id = self._pick_representative_id(group)
            decisions.append(
                {
                    "sample_ids": group,
                    "route_decision": "warm_start_embedding",
                    "target_node_id": node.node_id,
                    "parent_node_id": self.root.node_id,
                    "new_cluster_name": cluster_name,
                    "warm_start_k": warm_start_k,
                    "representative_sample_id": representative_id,
                    "chosen": "create",
                    "structure_changed": True,
                    "data_count": len(group),
                }
            )
        return decisions

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
            hint = " | recommended_assign_target" if i == 1 else ""
            candidate_lines.append(
                f"{i}. {c['node_id']} | depth={c['depth']} | cosine={c['cosine']:.4f} | "
                f"data={c['data_count']} | name={name}{hint}"
            )
        candidate_text = "\n".join(candidate_lines) if candidate_lines else "(no candidate nodes)"
        parent_lines: list[str] = []
        for i, p in enumerate(parent_options, start=1):
            pname = str(p.get("cluster_name", "")).strip() or "[unnamed]"
            policy = ""
            if str(p.get("node_id", "")) == self.root.node_id:
                policy = " | ROOT: use only for a genuinely new top-level capability family"
            elif int(p.get("depth", 0)) > 0:
                policy = " | preferred parent for narrower refinements"
            parent_lines.append(
                f"{i}. {p['node_id']} | depth={p['depth']} | data={p['data_count']} | name={pname}{policy}"
            )
        parent_text = "\n".join(parent_lines) if parent_lines else "(no parent options)"
        user_prompt = (
            "请判断这条数据应该加入哪个已有 capability 节点；只有在确实需要一个新的、可复用 capability 时才新建节点。\n"
            "你只需要做最小路由决策，不要输出树结构。\n\n"
            "重要定义：\n"
            "- capability 不是 topic index，不按具体人物、地点、年份、物品、故事情节或冷门事实建节点。\n"
            "- capability 应描述任务操作/认知过程/输出形式/稳定领域，例如分类、解释、信息抽取、推理、规划、推荐、改写、开放问答。\n"
            "- 如果新样本只是主题不同，但任务操作与已有簇相同，应 assign。\n"
            "- 只有当现有候选在任务操作、推理模式、输出形式上都明显不适合时，才 create。\n\n"
            "层级规则：\n"
            "- ROOT 只放少量顶层 capability family；不要把单条样本、细 topic、细领域直接挂到 ROOT。\n"
            "- 如果必须 create，优先把新节点挂在最相近的已有 capability 下，作为更细粒度子能力。\n"
            "- 只有当新样本代表一个与所有已有顶层分支都不同的粗粒度能力族时，parent_node_id 才能是 ROOT。\n"
            "- 如果 new_cluster_name 会与现有节点同名、近义或只是更窄 topic，必须 assign 到那个已有节点，不要 create。\n\n"
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
            "- 强烈优先加入(assign)；新建(create)是少数例外。\n"
            "- 不要因为样本出现新的实体、地点、历史人物、奖项、年份、医学名词、礼物对象等 topic 差异而 create。\n"
            "- 不要因为候选簇名称不完美或偏窄而 create；只要其任务形式相近，就 assign 到最接近的叶子。\n"
            "- 只有当所有候选簇在任务操作/推理模式/输出形式上都明显不合适时才允许新建。\n"
            "- create 时 parent_node_id 通常应选择最相近的非 ROOT 父节点；选择 ROOT 需要在 reason 中说明它为何是新的顶层能力族。\n"
            "- 若候选第一名与第二名都高度相关，必须加入，不要新建。\n"
            "- 对上层分支，禁止因为措辞细差异而新建分支。\n"
            "- 若新建，new_cluster_name必须是可复用的 capability 名，不要写成样本标题或具体 topic。\n"
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
            "命名目标：在可复用前提下，刚好体现与最近已有簇的 capability 差异，不要过度细分。\n\n"
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
            "- 新簇名要体现“最小必要 capability 差异”，不要把轻微措辞差异写成新能力。\n"
            "- 名称要描述任务操作/认知过程/输出形式/稳定领域，不要写成样本标题。\n"
            "- 避免具体人名、地点、年份、单个物品、故事事件、奖项名称等一次性 topic，除非它们代表稳定领域。\n"
            "- 与最近已有簇保持共性，只写关键区分点。\n"
            "- 名称可复用，必须能容纳多条未来相似任务，不能只适配单条样本。\n"
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
        split_groups: list[list[str]],
        client: LLMClient,
    ) -> list[str]:
        group_blocks: list[str] = []
        for idx, group in enumerate(split_groups, start=1):
            rep_ids = self._pick_representative_ids(group, count=3)
            samples_text = []
            for rep_idx, rep_id in enumerate(rep_ids, start=1):
                text = self.text_store.get(rep_id, "")[: self.cfg.max_sample_chars_for_prompt]
                samples_text.append(f"Representative {rep_idx}:\n{text}")
            group_blocks.append(f"Group {idx} | size={len(group)}\n" + "\n\n".join(samples_text))
        group_text = "\n\n".join(group_blocks)
        expected_n = len(split_groups)

        user_prompt = (
            "Old cluster name:\n"
            f"{old_name or '[unnamed cluster]'}\n\n"
            "Split groups:\n"
            f"{group_text}\n\n"
            "Return strict JSON:\n"
            '{"names":["...","..."]}\n'
            "Requirements:\n"
            f"- Return exactly {expected_n} names in the same order as the groups.\n"
            "- Preserve shared core implied by old cluster.\n"
            "- Express only the minimum necessary capability difference among groups.\n"
            "- Prefer task-operation and output-form distinctions over topic/entity distinctions.\n"
            "- Do not over-specialize due to minor wording differences.\n"
            "- Avoid one-off names based on specific people, places, dates, objects, or story details.\n"
            "- Names should stay reusable for future similar samples, not describe a single representative.\n"
            "- Each child name should be relative under old cluster and avoid repeating old cluster prefix."
        )
        raw = client.chat(SPLIT_NAME_SYSTEM_PROMPT, user_prompt, max_tokens=self.cfg.name_max_tokens)
        obj = extract_json_object(raw) or {}

        base = old_name or "能力簇 | Capability Cluster"
        raw_names = obj.get("names", [])
        if not isinstance(raw_names, list):
            raw_names = []

        names: list[str] = []
        for idx in range(expected_n):
            fallback = f"{base} - 子簇{idx + 1} | Subcluster {idx + 1}"
            raw_name = raw_names[idx] if idx < len(raw_names) else fallback
            name = str(raw_name).strip() or fallback
            names.append(self._compact_child_name(base, name))
        return names

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

        route_obj = self._route_with_strategy(sample, vec, client)
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
                "_route_source": "fallback",
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
                "objective_scope": "silhouette_local_topology_global_lemma_gate",
                "objective_computed_locally": False,
                "chosen": chosen,
                "structure_changed": False,
                "router_reason": route_obj.get("reason", ""),
                "router_raw": route_obj.get("_raw", ""),
            }

        target_ids_plus = list(dict.fromkeys(self._collect_subtree_ids(target_node) + [sample.sample_id]))
        split_candidate, split_candidates = self._best_silhouette_split(target_ids_plus)

        if split_candidate is not None:
            split_groups = split_candidate["groups"]
            lemma_decision = self._evaluate_lemma_split_gate(target_ids_plus, split_groups)
        else:
            split_groups = None
            lemma_decision = None

        if split_candidate is not None and lemma_decision is not None and lemma_decision.accepted:
            old_name = target_node.cluster_name
            child_names = self._name_split_children(old_name, split_groups, client)

            new_nodes: list[TreeNode] = []
            for child_name, group in zip(child_names, split_groups):
                child_node = self._new_node(cluster_name=child_name, data_ids=group)
                self._refresh_center(child_node)
                new_nodes.append(child_node)

            # Build a true hierarchical split: insert an intermediate parent node.
            split_parent_name = old_name.strip() if old_name and old_name.strip() else "Refined Capability Cluster"
            split_parent = self._new_node(cluster_name=split_parent_name, data_ids=[])
            split_parent.children = new_nodes
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
                "new_nodes": [n.node_id for n in new_nodes],
                "split_parent_name": split_parent_name,
                "new_cluster_names": child_names,
                "target_size_before_add": target_size_before_add,
                "split_min_size_threshold_m": self.cfg.min_node_size_for_split,
                "split_evaluation_skipped": False,
                "split_strategy": "silhouette_best_then_lemma_gate",
                "split_candidate_count": len(split_candidates),
                "split_candidates": split_candidates,
                "best_split_k": int(split_candidate["k"]),
                "best_split_silhouette": float(split_candidate["silhouette"]),
                "best_split_child_sizes": list(split_candidate["child_sizes"]),
                "lemma_split_decision": lemma_decision.to_dict(),
                "j_add": None,
                "j_split": None,
                "objective_scope": "silhouette_local_topology_global_lemma_gate",
                "objective_computed_locally": True,
                "chosen": chosen,
                "structure_changed": True,
                "router_reason": route_obj.get("reason", ""),
                "router_raw": route_obj.get("_raw", ""),
            }

        if split_candidate is None:
            self.stats["assign_add_no_valid_silhouette_split"] += 1
            split_reject_reason = "no_valid_silhouette_candidate"
            lemma_payload: dict[str, Any] | None = None
            best_split_k: int | None = None
            best_split_silhouette: float | None = None
            best_split_child_sizes: list[int] | None = None
        else:
            self.stats["assign_add_lemma_reject"] += 1
            split_reject_reason = "lemma_gate_rejected"
            lemma_payload = lemma_decision.to_dict() if lemma_decision is not None else None
            best_split_k = int(split_candidate["k"])
            best_split_silhouette = float(split_candidate["silhouette"])
            best_split_child_sizes = list(split_candidate["child_sizes"])

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
            "split_strategy": "silhouette_best_then_lemma_gate",
            "split_reject_reason": split_reject_reason,
            "split_candidate_count": len(split_candidates),
            "split_candidates": split_candidates,
            "best_split_k": best_split_k,
            "best_split_silhouette": best_split_silhouette,
            "best_split_child_sizes": best_split_child_sizes,
            "lemma_split_decision": lemma_payload,
            "j_add": None,
            "j_split": None,
            "objective_scope": "silhouette_local_topology_global_lemma_gate",
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


def normalize_capability_text_mode(mode: str) -> str:
    normalized = str(mode or "raw").strip().lower().replace("-", "_")
    aliases = {
        "default": "raw",
        "full": "raw",
        "io": "raw",
        "prompt_response": "raw",
        "instruction_only": "instruction",
        "input_instruction": "instruction_input",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"raw", "instruction", "instruction_input"}:
        raise ValueError(
            f"Unsupported capability_text_mode={mode!r}; choose raw, instruction, or instruction_input."
        )
    return normalized


def compose_raw_text(row: dict[str, Any], capability_text_mode: str = "raw") -> str:
    mode = normalize_capability_text_mode(capability_text_mode)
    instruction = clean_text(row.get("instruction", ""))
    input_text = clean_text(row.get("input", ""))
    output_text = clean_text(row.get("output", row.get("response", "")))

    parts: list[str] = []
    if instruction:
        parts.append(f"Instruction: {instruction}")
    if input_text and mode in {"raw", "instruction_input"}:
        parts.append(f"Input: {input_text}")
    if output_text and mode == "raw":
        parts.append(f"Output: {output_text}")
    text = "\n".join(parts).strip()
    if text:
        return text
    # Fallback to the old broad composition when the selected mode has no content.
    fallback_parts: list[str] = []
    if instruction:
        fallback_parts.append(f"Instruction: {instruction}")
    if input_text:
        fallback_parts.append(f"Input: {input_text}")
    if output_text:
        fallback_parts.append(f"Output: {output_text}")
    return "\n".join(fallback_parts).strip()


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


def load_samples(
    path: Path,
    max_samples: int | None,
    shuffle: bool,
    shuffle_seed: int | None,
    capability_text_mode: str,
) -> list[SampleRecord]:
    rows: list[SampleRecord] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = compose_raw_text(row, capability_text_mode=capability_text_mode)
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
        "--min-processed-before-early-stop",
        type=int,
        default=0,
        help="Do not apply patience-based early stop until at least this many samples have been consumed.",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=0,
        help="Early stop when leaf node count is > m; 0 disables.",
    )
    parser.add_argument(
        "--capability-text-mode",
        type=str,
        choices=["raw", "instruction", "instruction_input"],
        default="raw",
        help="Text fields used for embedding/routing.",
    )
    parser.add_argument(
        "--warm-start-samples",
        type=int,
        default=50,
        help="Number of leading samples used to build silhouette-optimal warm-start root anchors.",
    )
    parser.add_argument(
        "--min-node-size-for-split",
        type=int,
        default=3,
        help="Only evaluate split candidates when target node size before add is > m.",
    )
    parser.add_argument(
        "--max-split-k",
        type=int,
        default=6,
        help="Maximum KMeans k considered for one local split candidate search (also used for warm-start).",
    )
    parser.add_argument(
        "--min-child-size-for-split",
        type=int,
        default=2,
        help="Reject split candidates whose smallest child cluster has fewer samples than this.",
    )
    parser.add_argument(
        "--intrinsic-dim",
        type=float,
        default=16.0,
        help="Intrinsic dimension d used in the Lemma-1 expression-gain term.",
    )
    parser.add_argument(
        "--split-gain-scale",
        type=float,
        default=5.0,
        help="Beta = K_d / L^2 scale for the Lemma-1 expression-gain term.",
    )
    parser.add_argument(
        "--split-random-seed",
        type=int,
        default=42,
        help="Random seed for KMeans split candidate generation.",
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

    samples = load_samples(
        cfg.input_jsonl,
        cfg.max_samples,
        cfg.shuffle,
        cfg.shuffle_seed,
        cfg.capability_text_mode,
    )
    if not samples:
        raise ValueError("No valid samples found in input JSONL.")

    builder = LLMCapabilityTreeBuilder(cfg)
    logging.info(
        "Start building tree | samples=%d | shuffle=%s | shuffle_seed=%s | capability_text_mode=%s | warm_start_samples=%d | min_node_size_for_split=%d | max_split_k=%d | min_child_size_for_split=%d | intrinsic_dim=%.4f | split_gain_scale=%.4f | stale_singleton_rounds=%d | max_leaf_nodes=%d | log_every=%d | print_tree_every=%d | patience_structure_stable=%d",
        len(samples),
        cfg.shuffle,
        str(cfg.shuffle_seed),
        cfg.capability_text_mode,
        cfg.warm_start_samples,
        cfg.min_node_size_for_split,
        cfg.max_split_k,
        cfg.min_child_size_for_split,
        cfg.intrinsic_dim,
        cfg.split_gain_scale,
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
            warm_start_decisions = builder.warm_start_from_embeddings(samples, llm_client)
            warm_started_count = 0
            for decision in warm_start_decisions:
                decision_writer.write(json.dumps(decision, ensure_ascii=False) + "\n")
                decision_writer.flush()
                warm_started_count += len(decision.get("sample_ids", []))
            if warm_start_decisions:
                processed = warm_started_count
                logging.info(
                    "Warm start completed | consumed=%d | root_anchors=%d | stats=%s",
                    warm_started_count,
                    len(builder.root.children),
                    builder.stats,
                )

            for idx, sample in enumerate(samples[warm_started_count:], start=warm_started_count + 1):
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

                if (
                    cfg.patience_structure_stable > 0
                    and idx >= cfg.min_processed_before_early_stop
                    and stable_streak >= cfg.patience_structure_stable
                ):
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
                "min_processed_before_early_stop": cfg.min_processed_before_early_stop,
                "final_stable_streak": stable_streak,
                "converged_early": converged_early,
                "max_leaf_nodes": cfg.max_leaf_nodes,
                "final_leaf_count": builder.leaf_count(),
            },
            "objective": {
                "split_decision_scope": "silhouette_local_topology_global_lemma_gate",
                "topology_selector": "best_silhouette_over_candidate_k",
                "lemma_gate": "risk_delta_lt_bias_gain",
                "risk_density": "mean_squared_radius_divided_by_sample_count",
                "basis_count": "active_leaf_count",
                "computed_locally": False,
            },
            "embedding_sanitization": {
                "nonfinite_sanitized": int(getattr(builder.vectorizer, "nonfinite_sanitized", 0)),
                "zero_norm_sanitized": int(getattr(builder.vectorizer, "zero_norm_sanitized", 0)),
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
                "capability_text_mode": cfg.capability_text_mode,
                "warm_start_samples": cfg.warm_start_samples,
                "shuffle": cfg.shuffle,
                "shuffle_seed": cfg.shuffle_seed,
                "max_leaf_nodes": cfg.max_leaf_nodes,
                "min_node_size_for_split": cfg.min_node_size_for_split,
                "max_split_k": cfg.max_split_k,
                "min_child_size_for_split": cfg.min_child_size_for_split,
                "intrinsic_dim": cfg.intrinsic_dim,
                "split_gain_scale": cfg.split_gain_scale,
                "split_random_seed": cfg.split_random_seed,
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
    capability_text_mode = normalize_capability_text_mode(args.capability_text_mode)

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
        min_processed_before_early_stop=max(0, args.min_processed_before_early_stop),
        max_leaf_nodes=max(0, args.max_leaf_nodes),
        capability_text_mode=capability_text_mode,
        warm_start_samples=max(1, args.warm_start_samples),
        min_node_size_for_split=max(0, args.min_node_size_for_split),
        max_split_k=max(2, args.max_split_k),
        min_child_size_for_split=max(1, args.min_child_size_for_split),
        intrinsic_dim=max(1e-6, float(args.intrinsic_dim)),
        split_gain_scale=max(0.0, float(args.split_gain_scale)),
        split_random_seed=int(args.split_random_seed),
        stale_singleton_rounds=max(0, args.stale_singleton_rounds),
        max_sample_chars_for_prompt=max(200, args.max_sample_chars_for_prompt),
        max_tree_chars_for_prompt=max(0, args.max_tree_chars_for_prompt),
        log_level=args.log_level,
    )

    run(cfg)


if __name__ == "__main__":
    main()
