"""Stage 2: Tag vectorization and initial clustering.

Pipeline:
1) Load tagged Alpaca JSONL from Stage 1
2) Build unique atomic skill set S from all tags
3) Vectorize tags with sentence-transformers
4) Cluster tag embeddings with AgglomerativeClustering
5) Map each sample to initial cluster IDs and save outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


@dataclass
class Config:
    input_jsonl: Path
    output_jsonl: Path
    output_mapping_json: Path
    output_tag_cluster_json: Path
    embedding_model: str
    embedding_batch_size: int
    device: str
    normalize_embeddings: bool
    n_clusters: int | None
    distance_threshold: float | None
    linkage: str
    metric: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: Tag vectorization and initial clustering")
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="data/alpaca_with_tags.jsonl",
        help="Stage-1 JSONL with a `tags` field.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="data/alpaca_with_initial_clusters.jsonl",
        help="Output JSONL with appended initial cluster IDs for each sample.",
    )
    parser.add_argument(
        "--output-mapping-json",
        type=str,
        default="data/sample_to_initial_clusters.json",
        help="Output JSON map: sample_id -> list[cluster_id].",
    )
    parser.add_argument(
        "--output-tag-cluster-json",
        type=str,
        default="data/tag_to_cluster.json",
        help="Output JSON map: tag -> cluster_id.",
    )
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="`auto`, `cuda`, or `cpu`.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable embedding L2 normalization.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Fixed number of initial clusters K. If set, distance-threshold is ignored.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.7,
        help="Agglomerative distance threshold when n-clusters is not set.",
    )
    parser.add_argument(
        "--linkage",
        type=str,
        default="average",
        choices=["ward", "complete", "average", "single"],
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["euclidean", "l1", "l2", "manhattan", "cosine"],
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def build_atomic_skill_set(rows: list[dict[str, Any]]) -> list[str]:
    skills: set[str] = set()
    for row in rows:
        for tag in parse_tags_field(row.get("tags")):
            skills.add(tag)
    return sorted(skills)


class TagVectorizer:
    def __init__(self, model_name: str, device: str, batch_size: int, normalize_embeddings: bool):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    def encode(self, tags: list[str]) -> np.ndarray:
        if not tags:
            raise ValueError("No tags to encode.")
        vectors = self.model.encode(
            tags,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(vectors, dtype=np.float32)


class AgglomerativeTagClusterer:
    def __init__(
        self,
        n_clusters: int | None,
        distance_threshold: float | None,
        linkage: str,
        metric: str,
    ):
        if n_clusters is None and distance_threshold is None:
            raise ValueError("Either n_clusters or distance_threshold must be provided.")
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.metric = metric

    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "linkage": self.linkage,
            "metric": self.metric,
        }
        if self.n_clusters is not None:
            kwargs["n_clusters"] = self.n_clusters
            kwargs["distance_threshold"] = None
        else:
            kwargs["n_clusters"] = None
            kwargs["distance_threshold"] = self.distance_threshold
            kwargs["compute_full_tree"] = True

        if self.linkage == "ward":
            # sklearn constraint: ward only supports euclidean metric.
            kwargs["metric"] = "euclidean"

        try:
            model = AgglomerativeClustering(**kwargs)
        except TypeError:
            # Backward compatibility for old sklearn versions that use `affinity`.
            legacy_kwargs = dict(kwargs)
            legacy_kwargs["affinity"] = legacy_kwargs.pop("metric")
            model = AgglomerativeClustering(**legacy_kwargs)
        return model.fit_predict(vectors)


def map_samples_to_clusters(
    rows: list[dict[str, Any]],
    tag_to_cluster: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    mapped_rows: list[dict[str, Any]] = []
    sample_to_clusters: dict[str, list[int]] = {}

    for idx, row in enumerate(tqdm(rows, desc="Mapping samples")):
        tags = parse_tags_field(row.get("tags"))
        cluster_ids = sorted({tag_to_cluster[tag] for tag in tags if tag in tag_to_cluster})
        row_out = dict(row)
        row_out["initial_cluster_ids"] = cluster_ids
        mapped_rows.append(row_out)
        sample_id = row.get("id", idx)
        sample_to_clusters[str(sample_id)] = cluster_ids

    return mapped_rows, sample_to_clusters


def cluster_distribution(sample_to_clusters: dict[str, list[int]]) -> Counter[int]:
    # Multi-label counting: one sample contributes to each cluster it touches.
    counter: Counter[int] = Counter()
    for cluster_ids in sample_to_clusters.values():
        for cid in cluster_ids:
            counter[cid] += 1
    return counter


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run(cfg: Config) -> None:
    logging.info("Loading Stage-1 file: %s", cfg.input_jsonl)
    rows = load_jsonl(cfg.input_jsonl)
    logging.info("Loaded %d samples", len(rows))

    atomic_skills = build_atomic_skill_set(rows)
    logging.info("Atomic skill set |S| = %d", len(atomic_skills))
    if not atomic_skills:
        raise ValueError("No tags found in input data.")

    logging.info(
        "Embedding model: %s | device: %s | normalize_embeddings=%s",
        cfg.embedding_model,
        cfg.device,
        cfg.normalize_embeddings,
    )
    vectorizer = TagVectorizer(
        model_name=cfg.embedding_model,
        device=cfg.device,
        batch_size=cfg.embedding_batch_size,
        normalize_embeddings=cfg.normalize_embeddings,
    )
    vectors = vectorizer.encode(atomic_skills)
    logging.info("Embedding matrix shape: %s", tuple(vectors.shape))

    if cfg.n_clusters is not None:
        logging.info("Clustering with fixed K=%d", cfg.n_clusters)
    else:
        logging.info("Clustering with distance_threshold=%.4f", cfg.distance_threshold)

    clusterer = AgglomerativeTagClusterer(
        n_clusters=cfg.n_clusters,
        distance_threshold=cfg.distance_threshold,
        linkage=cfg.linkage,
        metric=cfg.metric,
    )
    labels = clusterer.fit_predict(vectors)

    tag_to_cluster = {tag: int(label) for tag, label in zip(atomic_skills, labels)}
    mapped_rows, sample_to_clusters = map_samples_to_clusters(rows, tag_to_cluster)

    dist = cluster_distribution(sample_to_clusters)
    sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    logging.info("Cluster count: %d", len(set(labels)))
    print("\nCluster sample distribution (multi-label counts):")
    for cid, cnt in sorted_dist:
        print(f"  cluster_{cid}: {cnt}")

    save_jsonl(mapped_rows, cfg.output_jsonl)
    save_json(sample_to_clusters, cfg.output_mapping_json)
    save_json(tag_to_cluster, cfg.output_tag_cluster_json)

    logging.info("Saved mapped JSONL: %s", cfg.output_jsonl.resolve())
    logging.info("Saved sample->clusters map: %s", cfg.output_mapping_json.resolve())
    logging.info("Saved tag->cluster map: %s", cfg.output_tag_cluster_json.resolve())


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    device = resolve_device(args.device)
    normalize_embeddings = not args.no_normalize_embeddings
    distance_threshold = None if args.n_clusters is not None else args.distance_threshold

    cfg = Config(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        output_mapping_json=Path(args.output_mapping_json),
        output_tag_cluster_json=Path(args.output_tag_cluster_json),
        embedding_model=args.embedding_model,
        embedding_batch_size=max(1, args.embedding_batch_size),
        device=device,
        normalize_embeddings=normalize_embeddings,
        n_clusters=args.n_clusters if args.n_clusters is None else max(1, args.n_clusters),
        distance_threshold=distance_threshold,
        linkage=args.linkage,
        metric=args.metric,
    )
    run(cfg)


if __name__ == "__main__":
    main()
