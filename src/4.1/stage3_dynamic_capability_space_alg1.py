"""ALG1: Dynamic capability space evolution (split by density, merge by separation)."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm


def parse_tags_field(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, list):
        items = [str(t).strip() for t in tags]
    else:
        items = [p.strip() for p in re.split(r"[,，;/\\|]+", str(tags))]
    uniq: list[str] = []
    seen: set[str] = set()
    for tag in items:
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(tag)
    return uniq


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def to_0_1(cos_sim: float) -> float:
    return (cos_sim + 1.0) / 2.0


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    return rows


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class ClusterMetrics:
    center: np.ndarray
    alignment: float
    radius: float
    density: float


class DynamicCapabilitySpace:
    def __init__(
        self,
        tag_to_vector: dict[str, np.ndarray],
        initial_tag_to_cluster: dict[str, int],
        target_vector: np.ndarray,
        tau: float = 1.0,
        gamma: float = 0.05,
        epsilon: float = 1e-5,
        radius_mode: str = "mean",
        radius_quantile: float = 0.9,
        merge_center_cosine_threshold: float = 0.85,
        random_state: int = 42,
    ):
        self.tag_to_vector = tag_to_vector
        self.target_vector = target_vector
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.radius_mode = radius_mode
        self.radius_quantile = radius_quantile
        self.merge_center_cosine_threshold = merge_center_cosine_threshold
        self.random_state = random_state

        clusters: dict[int, list[str]] = {}
        for tag, cid in initial_tag_to_cluster.items():
            clusters.setdefault(int(cid), []).append(tag)
        self.clusters = {cid: sorted(tags) for cid, tags in clusters.items() if tags}
        self.next_cluster_id = (max(self.clusters.keys()) + 1) if self.clusters else 0

        self.history: list[dict[str, Any]] = []

    def _cluster_vectors(self, cid: int) -> np.ndarray:
        tags = self.clusters[cid]
        return np.vstack([self.tag_to_vector[t] for t in tags])

    def _radius(self, vectors: np.ndarray, center: np.ndarray) -> float:
        dists = np.linalg.norm(vectors - center[None, :], axis=1)
        if len(dists) == 0:
            return 0.0
        if self.radius_mode == "p90":
            return float(np.quantile(dists, self.radius_quantile))
        return float(np.mean(dists))

    def _metrics_for_cluster(self, cid: int) -> ClusterMetrics:
        vectors = self._cluster_vectors(cid)
        center = np.mean(vectors, axis=0)
        align_cos = cosine_similarity(center, self.target_vector)
        align = to_0_1(align_cos)
        radius = self._radius(vectors, center)
        density = align / (radius + self.epsilon)
        return ClusterMetrics(center=center, alignment=align, radius=radius, density=density)

    def _all_metrics(self) -> dict[int, ClusterMetrics]:
        return {cid: self._metrics_for_cluster(cid) for cid in self.clusters}

    def _split_low_density_clusters(self, metrics: dict[int, ClusterMetrics]) -> int:
        split_count = 0
        for cid in sorted(list(self.clusters.keys())):
            if cid not in self.clusters:
                continue
            tags = self.clusters[cid]
            if len(tags) < 2:
                continue
            if metrics[cid].density >= self.tau:
                continue

            vectors = self._cluster_vectors(cid)
            kmeans = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(vectors)

            child_a = [tag for tag, lab in zip(tags, labels) if lab == 0]
            child_b = [tag for tag, lab in zip(tags, labels) if lab == 1]
            if not child_a or not child_b:
                continue

            self.clusters[cid] = sorted(child_a)
            new_cid = self.next_cluster_id
            self.next_cluster_id += 1
            self.clusters[new_cid] = sorted(child_b)
            split_count += 1
            logging.info(
                "Split cluster %s (density=%.4f) -> %s(%d), %s(%d)",
                cid,
                metrics[cid].density,
                cid,
                len(child_a),
                new_cid,
                len(child_b),
            )
        return split_count

    def _best_merge_pair(self, metrics: dict[int, ClusterMetrics]) -> tuple[int, int] | None:
        cids = sorted(self.clusters.keys())
        best_pair: tuple[int, int] | None = None
        best_delta = float("inf")

        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                ci, cj = cids[i], cids[j]
                delta = abs(metrics[ci].alignment - metrics[cj].alignment)
                if delta >= self.gamma:
                    continue
                center_cos = cosine_similarity(metrics[ci].center, metrics[cj].center)
                if center_cos <= self.merge_center_cosine_threshold:
                    continue
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (ci, cj)
        return best_pair

    def _merge_once(self) -> bool:
        metrics = self._all_metrics()
        pair = self._best_merge_pair(metrics)
        if pair is None:
            return False

        a, b = pair
        keep, remove = (a, b) if a < b else (b, a)
        merged = sorted(set(self.clusters[keep] + self.clusters[remove]))
        self.clusters[keep] = merged
        del self.clusters[remove]
        logging.info("Merged clusters %s and %s -> %s(%d tags)", a, b, keep, len(merged))
        return True

    def evolve(self, max_iters: int = 100) -> dict[str, int]:
        for it in range(1, max_iters + 1):
            stable = True
            metrics_before = self._all_metrics()
            split_count = self._split_low_density_clusters(metrics_before)
            if split_count > 0:
                stable = False

            merge_count = 0
            while self._merge_once():
                merge_count += 1
                stable = False

            self.history.append(
                {
                    "iteration": it,
                    "num_clusters": len(self.clusters),
                    "num_splits": split_count,
                    "num_merges": merge_count,
                }
            )
            logging.info(
                "Iteration %d finished | clusters=%d | splits=%d | merges=%d",
                it,
                len(self.clusters),
                split_count,
                merge_count,
            )

            if stable:
                logging.info("Stable reached at iteration %d", it)
                break
        else:
            logging.warning("Stopped at max_iters=%d before convergence.", max_iters)

        return self.export_tag_to_cluster()

    def export_tag_to_cluster(self) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for cid, tags in self.clusters.items():
            for t in tags:
                mapping[t] = int(cid)
        return mapping

    def export_cluster_metrics(self) -> dict[str, dict[str, float]]:
        metrics = self._all_metrics()
        return {
            str(cid): {
                "size": len(self.clusters[cid]),
                "alignment": m.alignment,
                "radius": m.radius,
                "density": m.density,
            }
            for cid, m in metrics.items()
        }


def map_samples_with_evolved_clusters(
    rows: list[dict[str, Any]],
    evolved_tag_to_cluster: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    mapped_rows: list[dict[str, Any]] = []
    sample_map: dict[str, list[int]] = {}
    for idx, row in enumerate(tqdm(rows, desc="Map samples to evolved clusters")):
        tags = parse_tags_field(row.get("tags"))
        cluster_ids = sorted({evolved_tag_to_cluster[t] for t in tags if t in evolved_tag_to_cluster})
        out = dict(row)
        out["evolved_cluster_ids"] = cluster_ids
        mapped_rows.append(out)
        sample_id = row.get("id", idx)
        sample_map[str(sample_id)] = cluster_ids
    return mapped_rows, sample_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALG1: Dynamic capability space evolution")
    parser.add_argument(
        "--input-tag-cluster-json",
        type=str,
        default="data/tag_to_cluster.json",
        help="Stage-2 output: tag -> initial_cluster_id",
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="",
        help="Optional Stage-1/2 JSONL to map samples to evolved clusters.",
    )
    parser.add_argument(
        "--target-text",
        type=str,
        default="Advanced Python coding and logical reasoning",
        help="Target task text T.",
    )
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--tau", type=float, default=1.0, help="Split threshold for density rho.")
    parser.add_argument("--gamma", type=float, default=0.05, help="Merge threshold for separation delta.")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--radius-mode",
        type=str,
        default="mean",
        choices=["mean", "p90"],
        help="Radius proxy: mean distance or p90 quantile distance.",
    )
    parser.add_argument("--radius-quantile", type=float, default=0.9)
    parser.add_argument(
        "--merge-center-cosine-threshold",
        type=float,
        default=0.85,
        help="Merge only when center cosine similarity is above this value.",
    )
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-evolved-tag-cluster-json",
        type=str,
        default="data/evolved_tag_to_cluster_alg1.json",
    )
    parser.add_argument(
        "--output-cluster-metrics-json",
        type=str,
        default="data/evolved_cluster_metrics_alg1.json",
    )
    parser.add_argument(
        "--output-history-json",
        type=str,
        default="data/evolution_history_alg1.json",
    )
    parser.add_argument(
        "--output-sample-map-json",
        type=str,
        default="data/sample_to_evolved_clusters_alg1.json",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="data/alpaca_with_evolved_clusters_alg1.jsonl",
        help="Output JSONL with evolved cluster IDs (if --input-jsonl is provided).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = resolve_device(args.device)
    if not (0.0 < args.radius_quantile <= 1.0):
        raise ValueError("--radius-quantile must be in (0, 1].")

    input_tag_cluster_path = Path(args.input_tag_cluster_json)
    initial_tag_to_cluster_raw = load_json(input_tag_cluster_path)
    initial_tag_to_cluster = {str(k): int(v) for k, v in initial_tag_to_cluster_raw.items()}
    tags = sorted(initial_tag_to_cluster.keys())
    if not tags:
        raise ValueError("No tags found in input-tag-cluster-json.")

    logging.info(
        "Loaded initial clusters from %s | tags=%d | initial_clusters=%d",
        input_tag_cluster_path,
        len(tags),
        len(set(initial_tag_to_cluster.values())),
    )
    logging.info("Embedding model=%s | device=%s", args.embedding_model, device)
    encoder = SentenceTransformer(args.embedding_model, device=device)

    tag_vectors_mat = encoder.encode(
        tags,
        batch_size=max(1, args.embedding_batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    target_vec = encoder.encode(
        [args.target_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    tag_to_vector = {t: v.astype(np.float32) for t, v in zip(tags, tag_vectors_mat)}

    space = DynamicCapabilitySpace(
        tag_to_vector=tag_to_vector,
        initial_tag_to_cluster=initial_tag_to_cluster,
        target_vector=target_vec.astype(np.float32),
        tau=args.tau,
        gamma=args.gamma,
        epsilon=args.epsilon,
        radius_mode=args.radius_mode,
        radius_quantile=args.radius_quantile,
        merge_center_cosine_threshold=args.merge_center_cosine_threshold,
        random_state=args.random_state,
    )

    evolved_tag_to_cluster = space.evolve(max_iters=max(1, args.max_iters))
    metrics = space.export_cluster_metrics()

    print("\nFinal cluster size distribution:")
    size_dist = sorted(
        ((cid, len(tags_)) for cid, tags_ in space.clusters.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    for cid, size in size_dist:
        print(f"  cluster_{cid}: {size}")

    save_json(evolved_tag_to_cluster, Path(args.output_evolved_tag_cluster_json))
    save_json(metrics, Path(args.output_cluster_metrics_json))
    save_json(space.history, Path(args.output_history_json))
    logging.info("Saved evolved tag->cluster map: %s", Path(args.output_evolved_tag_cluster_json).resolve())
    logging.info("Saved evolved cluster metrics: %s", Path(args.output_cluster_metrics_json).resolve())
    logging.info("Saved evolution history: %s", Path(args.output_history_json).resolve())

    if args.input_jsonl:
        rows = load_jsonl(Path(args.input_jsonl))
        mapped_rows, sample_map = map_samples_with_evolved_clusters(rows, evolved_tag_to_cluster)
        save_jsonl(mapped_rows, Path(args.output_jsonl))
        save_json(sample_map, Path(args.output_sample_map_json))
        logging.info("Saved sample->evolved_clusters map: %s", Path(args.output_sample_map_json).resolve())
        logging.info("Saved mapped JSONL: %s", Path(args.output_jsonl).resolve())


if __name__ == "__main__":
    main()
