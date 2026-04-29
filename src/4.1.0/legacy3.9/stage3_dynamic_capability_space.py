"""Stage 3: Dynamic Capability Space Evolution.

This script implements:
1) Dense vectorization for each sample's T_description
2) Density-based split and separation-based merge loop
3) Logging of convergence trajectory in each iteration
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


@dataclass
class EvolutionConfig:
    profile_jsonl: Path
    cluster_dict_json: Path
    output_cluster_dict_json: Path
    output_metrics_json: Path
    embedding_model: str
    embedding_batch_size: int
    device: str
    normalize_embeddings: bool
    target_text: str
    tau: float
    epsilon: float
    merge_cos_threshold: float
    max_iter: int
    random_state: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Dynamic Capability Space Evolution")
    parser.add_argument(
        "--profile-jsonl",
        type=str,
        default="data/alpaca-gpt4-data-en/alpaca_capability_profile.jsonl",
        help="Input capability profile JSONL (must contain T_description).",
    )
    parser.add_argument(
        "--cluster-dict-json",
        type=str,
        required=True,
        help="Initial cluster dictionary JSON: cluster_id -> [data_id, ...].",
    )
    parser.add_argument(
        "--output-cluster-dict-json",
        type=str,
        default="data/alpaca-gpt4-data-en/cluster_dict_evolved.json",
        help="Output evolved cluster dictionary JSON.",
    )
    parser.add_argument(
        "--output-metrics-json",
        type=str,
        default="data/alpaca-gpt4-data-en/cluster_metrics_evolved.json",
        help="Output cluster metrics JSON for final state.",
    )
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", help="auto / cuda / cpu")
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable L2 normalization in sentence-transformers encoding.",
    )
    parser.add_argument(
        "--target-text",
        type=str,
        default="High-quality instruction following, logical reasoning, and helpful assistance",
        help="Global target text for alignment.",
    )
    parser.add_argument("--tau", type=float, default=1.0, help="Density threshold for split.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Alignment-gap threshold for merge.")
    parser.add_argument(
        "--merge-cos-threshold",
        type=float,
        default=0.85,
        help="Center cosine similarity threshold for merge.",
    )
    parser.add_argument("--max-iter", type=int, default=20, help="Maximum evolution iterations.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity for rows in a against rows in b using matrix ops."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_safe = a / np.clip(a_norm, 1e-12, None)
    b_safe = b / np.clip(b_norm, 1e-12, None)
    return a_safe @ b_safe.T


class DynamicCapabilitySpace:
    """Dynamic evolution over capability clusters using density and separation criteria."""

    def __init__(self, cfg: EvolutionConfig):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg.embedding_model, device=cfg.device)

        self.rows: list[dict[str, Any]] = []
        self.data_ids: list[str] = []
        self.id_to_index: dict[str, int] = {}
        self.vectors: np.ndarray | None = None
        self.v_target: np.ndarray | None = None

        self.cluster_dict: dict[str, list[str]] = {}
        self._cluster_seq = 0

    def load_profiles(self) -> None:
        rows: list[dict[str, Any]] = []
        with self.cfg.profile_jsonl.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
                rows.append(row)

        if not rows:
            raise ValueError("Empty profile file.")

        data_ids: list[str] = []
        for i, row in enumerate(rows):
            data_id = row.get("id", i)
            data_ids.append(str(data_id))

        # Ensure uniqueness for robust mapping.
        if len(set(data_ids)) != len(data_ids):
            logging.warning("Duplicate data ids found; fallback to row index IDs.")
            data_ids = [str(i) for i in range(len(rows))]

        self.rows = rows
        self.data_ids = data_ids
        self.id_to_index = {did: i for i, did in enumerate(self.data_ids)}
        logging.info("Loaded %d samples from %s", len(self.rows), self.cfg.profile_jsonl)

    def load_initial_clusters(self) -> None:
        with self.cfg.cluster_dict_json.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError("cluster_dict_json must be a JSON object.")

        cluster_dict = self._normalize_cluster_dict(raw)

        if not cluster_dict:
            raise ValueError("No valid clusters found after ID filtering.")

        self.cluster_dict = cluster_dict
        self._cluster_seq = self._infer_next_cluster_seq(cluster_dict)
        logging.info("Loaded %d initial clusters from %s", len(cluster_dict), self.cfg.cluster_dict_json)

    def _normalize_cluster_dict(self, raw: dict[str, Any]) -> dict[str, list[str]]:
        # Mode A: cluster_id -> [data_id, ...]
        direct: dict[str, list[str]] = {}
        for cid, members in raw.items():
            if not isinstance(members, list):
                continue
            valid_members = [str(m) for m in members if str(m) in self.id_to_index]
            if valid_members:
                direct[str(cid)] = sorted(set(valid_members))

        # Mode B: sample_id -> [cluster_id, ...], invert it.
        # We detect this mode if keys look like sample IDs.
        key_is_data_id = sum(1 for k in raw.keys() if str(k) in self.id_to_index)
        looks_like_sample_to_clusters = key_is_data_id > max(1, len(raw) // 2)
        if looks_like_sample_to_clusters:
            inverted: dict[str, list[str]] = {}
            for sid, cids in raw.items():
                sid_s = str(sid)
                if sid_s not in self.id_to_index or not isinstance(cids, list):
                    continue
                for cid in cids:
                    cid_s = str(cid)
                    inverted.setdefault(cid_s, []).append(sid_s)
            inverted = {k: sorted(set(v)) for k, v in inverted.items() if v}
            if inverted:
                logging.info("Detected sample->clusters input format; converted to cluster->samples.")
                return inverted

        return direct

    @staticmethod
    def _infer_next_cluster_seq(cluster_dict: dict[str, list[str]]) -> int:
        max_int = -1
        for cid in cluster_dict:
            try:
                max_int = max(max_int, int(cid))
            except ValueError:
                continue
        return max_int + 1

    def _new_cluster_id(self) -> str:
        cid = str(self._cluster_seq)
        self._cluster_seq += 1
        return cid

    def encode_vectors(self) -> None:
        descriptions = [str(r.get("T_description", "")).strip() for r in self.rows]
        if any(not d for d in descriptions):
            logging.warning("Some rows have empty T_description; embedding empty text for those rows.")

        vectors = self.model.encode(
            descriptions,
            batch_size=self.cfg.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize_embeddings,
        )
        self.vectors = np.asarray(vectors, dtype=np.float32)

        v_target = self.model.encode(
            [self.cfg.target_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize_embeddings,
        )
        self.v_target = np.asarray(v_target[0], dtype=np.float32)
        logging.info("Vectorized T_description matrix shape: %s", tuple(self.vectors.shape))

    def _cluster_indices(self, members: list[str]) -> np.ndarray:
        return np.asarray([self.id_to_index[m] for m in members], dtype=np.int64)

    def compute_cluster_stats(self, cluster_dict: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
        if self.vectors is None or self.v_target is None:
            raise RuntimeError("Vectors are not initialized.")

        target = self.v_target.reshape(1, -1)
        stats: dict[str, dict[str, Any]] = {}

        for cid, members in cluster_dict.items():
            idx = self._cluster_indices(members)
            cluster_vecs = self.vectors[idx]
            center = cluster_vecs.mean(axis=0)

            # A_k: cosine(center, target), clipped to [0, 1]
            a_raw = float(cosine_similarity_matrix(center.reshape(1, -1), target)[0, 0])
            a_k = float(np.clip(a_raw, 0.0, 1.0))

            # r_k: mean euclidean distance to center
            diff = cluster_vecs - center.reshape(1, -1)
            r_k = float(np.linalg.norm(diff, axis=1).mean())

            rho_k = float(a_k / (r_k + 1e-5))
            stats[cid] = {
                "size": int(len(members)),
                "center": center,
                "A_k": a_k,
                "r_k": r_k,
                "rho_k": rho_k,
            }
        return stats

    def split_low_density_clusters(self, cluster_dict: dict[str, list[str]], stats: dict[str, dict[str, Any]]) -> tuple[dict[str, list[str]], int]:
        if self.vectors is None:
            raise RuntimeError("Vectors are not initialized.")

        updated = dict(cluster_dict)
        split_count = 0

        for cid in list(cluster_dict.keys()):
            members = cluster_dict[cid]
            rho_k = float(stats[cid]["rho_k"])
            if rho_k >= self.cfg.tau:
                continue
            if len(members) < 2:
                # Cannot split a singleton cluster.
                continue

            idx = self._cluster_indices(members)
            x = self.vectors[idx]
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=self.cfg.random_state)
            labels = kmeans.fit_predict(x)

            group_a = [members[i] for i, lb in enumerate(labels) if lb == 0]
            group_b = [members[i] for i, lb in enumerate(labels) if lb == 1]
            if not group_a or not group_b:
                continue

            del updated[cid]
            new_a = self._new_cluster_id()
            new_b = self._new_cluster_id()
            updated[new_a] = group_a
            updated[new_b] = group_b
            split_count += 1

        return updated, split_count

    def _merge_once(self, cluster_dict: dict[str, list[str]], stats: dict[str, dict[str, Any]]) -> tuple[dict[str, list[str]], bool]:
        if not cluster_dict:
            return cluster_dict, False

        cids = list(cluster_dict.keys())
        centers = np.stack([stats[cid]["center"] for cid in cids], axis=0)
        a_values = np.asarray([stats[cid]["A_k"] for cid in cids], dtype=np.float32)

        # delta_ij = |A_i - A_j|
        delta = np.abs(a_values[:, None] - a_values[None, :])
        # cosine(center_i, center_j)
        center_cos = cosine_similarity_matrix(centers, centers)

        n = len(cids)
        best_pair: tuple[int, int] | None = None
        best_delta = float("inf")

        for i in range(n):
            for j in range(i + 1, n):
                if delta[i, j] < self.cfg.epsilon and center_cos[i, j] > self.cfg.merge_cos_threshold:
                    if delta[i, j] < best_delta:
                        best_delta = float(delta[i, j])
                        best_pair = (i, j)

        if best_pair is None:
            return cluster_dict, False

        i, j = best_pair
        ci, cj = cids[i], cids[j]
        merged_members = sorted(set(cluster_dict[ci] + cluster_dict[cj]))

        updated = dict(cluster_dict)
        del updated[ci]
        del updated[cj]
        new_cid = self._new_cluster_id()
        updated[new_cid] = merged_members
        return updated, True

    def merge_redundant_clusters(self, cluster_dict: dict[str, list[str]]) -> tuple[dict[str, list[str]], int]:
        merged_count = 0
        updated = dict(cluster_dict)

        while True:
            if len(updated) < 2:
                break
            stats = self.compute_cluster_stats(updated)
            updated_next, merged = self._merge_once(updated, stats)
            if not merged:
                break
            updated = updated_next
            merged_count += 1

        return updated, merged_count

    def evolve(self) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
        current = dict(self.cluster_dict)
        logging.info(
            "Start evolution: clusters=%d, tau=%.4f, epsilon=%.4f, merge_cos_threshold=%.4f, max_iter=%d",
            len(current),
            self.cfg.tau,
            self.cfg.epsilon,
            self.cfg.merge_cos_threshold,
            self.cfg.max_iter,
        )

        for iteration in range(1, self.cfg.max_iter + 1):
            before_n = len(current)
            stable = True

            # Step 1: density-based split
            stats_before = self.compute_cluster_stats(current)
            current, split_count = self.split_low_density_clusters(current, stats_before)
            if split_count > 0:
                stable = False

            # Step 2: separation-based merge
            current, merge_count = self.merge_redundant_clusters(current)
            if merge_count > 0:
                stable = False

            after_n = len(current)
            logging.info(
                "[Iter %d] clusters: %d -> %d | splits=%d | merges=%d",
                iteration,
                before_n,
                after_n,
                split_count,
                merge_count,
            )

            if stable:
                logging.info("Converged at iteration %d (no split/merge).", iteration)
                break
        else:
            logging.info("Reached max_iter=%d before full stability.", self.cfg.max_iter)

        final_stats_raw = self.compute_cluster_stats(current)
        final_stats: dict[str, dict[str, Any]] = {}
        for cid, s in final_stats_raw.items():
            final_stats[cid] = {
                "size": int(s["size"]),
                "A_k": float(s["A_k"]),
                "r_k": float(s["r_k"]),
                "rho_k": float(s["rho_k"]),
            }

        return current, final_stats

    def run(self) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
        self.load_profiles()
        self.load_initial_clusters()
        self.encode_vectors()
        return self.evolve()


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = EvolutionConfig(
        profile_jsonl=Path(args.profile_jsonl),
        cluster_dict_json=Path(args.cluster_dict_json),
        output_cluster_dict_json=Path(args.output_cluster_dict_json),
        output_metrics_json=Path(args.output_metrics_json),
        embedding_model=args.embedding_model,
        embedding_batch_size=max(1, args.embedding_batch_size),
        device=resolve_device(args.device),
        normalize_embeddings=not args.no_normalize_embeddings,
        target_text=args.target_text,
        tau=float(args.tau),
        epsilon=float(args.epsilon),
        merge_cos_threshold=float(args.merge_cos_threshold),
        max_iter=max(1, int(args.max_iter)),
        random_state=int(args.random_state),
    )

    dcs = DynamicCapabilitySpace(cfg)
    final_clusters, final_metrics = dcs.run()

    save_json(final_clusters, cfg.output_cluster_dict_json)
    save_json(final_metrics, cfg.output_metrics_json)
    logging.info("Saved evolved clusters to: %s", cfg.output_cluster_dict_json.resolve())
    logging.info("Saved final cluster metrics to: %s", cfg.output_metrics_json.resolve())
    logging.info("Final cluster count: %d", len(final_clusters))


if __name__ == "__main__":
    main()
