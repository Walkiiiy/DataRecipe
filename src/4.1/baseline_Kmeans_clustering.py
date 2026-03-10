"""Baseline: K-Means clustering with silhouette peak search and uniform sampling.

Pipeline:
1) Load Alpaca JSONL and concatenate instruction/input/output into plain text.
2) Encode text with sentence-transformers (GPU supported).
3) Dynamically search optimal K by silhouette score with early stopping.
4) Refit K-Means using best K, then uniformly sample each cluster by ratio.
5) Save sampled subset as JSONL.
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
from sklearn.metrics import silhouette_score


@dataclass
class BaselineConfig:
    input_jsonl: Path
    output_jsonl: Path
    embedding_model: str
    embedding_batch_size: int
    device: str
    normalize_embeddings: bool
    random_state: int
    start_k: int
    step: int
    max_k: int | None
    patience: int
    silhouette_sample_size: int
    sample_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline K-Means with silhouette peak search and uniform sampling"
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="data/alpaca-gpt4-data-en/alpaca_1000.jsonl",
        help="Input Alpaca JSONL file.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="data/alpaca-gpt4-data-en/alpaca_kmeans_peak_baseline.jsonl",
        help="Output sampled JSONL file.",
    )
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", help="auto / cuda / cpu")
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable embedding normalization.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--start-k", type=int, default=10)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Optional max K bound. Default: n_samples - 1.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stop when score is below best score for consecutive steps.",
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=10000,
        help="Sample size for silhouette_score acceleration.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.1,
        help="Uniform sampling ratio in each cluster.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_config(cfg: BaselineConfig) -> None:
    if cfg.start_k < 2:
        raise ValueError("start_k must be >= 2.")
    if cfg.step < 1:
        raise ValueError("step must be >= 1.")
    if cfg.patience < 1:
        raise ValueError("patience must be >= 1.")
    if not (0 < cfg.sample_ratio <= 1):
        raise ValueError("sample_ratio must be in (0, 1].")
    if cfg.silhouette_sample_size < 2:
        raise ValueError("silhouette_sample_size must be >= 2.")


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
    if not rows:
        raise ValueError(f"Input file is empty: {path}")
    return rows


def build_plain_text(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output_text = str(row.get("output", "")).strip()
    # Structured join preserves fields while keeping a pure text representation.
    return "\n".join(
        [
            f"Instruction: {instruction}",
            f"Input: {input_text}",
            f"Output: {output_text}",
        ]
    ).strip()


def encode_texts(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(vectors, dtype=np.float32)


def fit_kmeans(vectors: np.ndarray, k: int, random_state: int) -> np.ndarray:
    # Keep compatibility with both old/new sklearn versions.
    try:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    except TypeError:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(vectors)


def silhouette_with_cap(
    vectors: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    random_state: int,
) -> float:
    n = vectors.shape[0]
    sample_size_eff = min(sample_size, n)
    return float(
        silhouette_score(
            vectors,
            labels,
            sample_size=sample_size_eff,
            random_state=random_state,
        )
    )


def search_best_k_by_peak(
    vectors: np.ndarray,
    start_k: int,
    step: int,
    max_k: int,
    patience: int,
    silhouette_sample_size: int,
    random_state: int,
) -> tuple[int, float]:
    n_samples = vectors.shape[0]
    upper = min(max_k, n_samples - 1)
    if start_k > upper:
        raise ValueError(f"start_k={start_k} is larger than search upper bound={upper}.")

    best_k = start_k
    best_score = float("-inf")
    consecutive_no_improve = 0

    k = start_k
    while k <= upper:
        labels = fit_kmeans(vectors, k=k, random_state=random_state)
        score = silhouette_with_cap(
            vectors=vectors,
            labels=labels,
            sample_size=silhouette_sample_size,
            random_state=random_state,
        )
        logging.info("Search step: k=%d, silhouette=%.6f", k, score)

        if score > best_score:
            best_score = score
            best_k = k
            consecutive_no_improve = 0
            logging.info("New best found: best_k=%d, best_score=%.6f", best_k, best_score)
        else:
            consecutive_no_improve += 1
            logging.info(
                "No improvement count: %d/%d (current=%.6f, best=%.6f)",
                consecutive_no_improve,
                patience,
                score,
                best_score,
            )
            if consecutive_no_improve >= patience:
                logging.info(
                    "Peak detected at k=%d with score=%.6f, stopping search",
                    best_k,
                    best_score,
                )
                break
        k += step

    return best_k, best_score


def uniform_sample_by_cluster(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    sample_ratio: float,
    random_state: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    sampled_indices: list[int] = []
    labels = np.asarray(labels)

    for cid in np.unique(labels):
        cluster_indices = np.where(labels == cid)[0]
        cluster_size = len(cluster_indices)
        # Ensure every non-empty cluster contributes at least one sample.
        take = max(1, int(np.floor(cluster_size * sample_ratio)))
        take = min(take, cluster_size)
        chosen = rng.choice(cluster_indices, size=take, replace=False)
        sampled_indices.extend(chosen.tolist())
        logging.info("Cluster %d: size=%d, sampled=%d", int(cid), cluster_size, take)

    sampled_indices = sorted(set(sampled_indices))
    sampled_rows = [rows[i] for i in sampled_indices]
    logging.info("Total sampled rows: %d / %d", len(sampled_rows), len(rows))
    return sampled_rows


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    cfg = BaselineConfig(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        device=resolve_device(args.device),
        normalize_embeddings=not args.no_normalize_embeddings,
        random_state=args.random_state,
        start_k=args.start_k,
        step=args.step,
        max_k=args.max_k,
        patience=args.patience,
        silhouette_sample_size=args.silhouette_sample_size,
        sample_ratio=args.sample_ratio,
    )
    validate_config(cfg)

    rows = load_jsonl(cfg.input_jsonl)
    texts = [build_plain_text(r) for r in rows]
    logging.info("Loaded %d rows from %s", len(rows), cfg.input_jsonl)
    logging.info("Embedding model: %s | device: %s", cfg.embedding_model, cfg.device)

    vectors = encode_texts(
        texts=texts,
        model_name=cfg.embedding_model,
        device=cfg.device,
        batch_size=cfg.embedding_batch_size,
        normalize_embeddings=cfg.normalize_embeddings,
    )
    logging.info("Embedding matrix shape: %s", tuple(vectors.shape))

    max_k = cfg.max_k if cfg.max_k is not None else (len(rows) - 1)
    best_k, best_score = search_best_k_by_peak(
        vectors=vectors,
        start_k=cfg.start_k,
        step=cfg.step,
        max_k=max_k,
        patience=cfg.patience,
        silhouette_sample_size=cfg.silhouette_sample_size,
        random_state=cfg.random_state,
    )
    logging.info("Best K selected: %d (silhouette=%.6f)", best_k, best_score)

    final_labels = fit_kmeans(vectors, k=best_k, random_state=cfg.random_state)
    sampled_rows = uniform_sample_by_cluster(
        rows=rows,
        labels=final_labels,
        sample_ratio=cfg.sample_ratio,
        random_state=cfg.random_state,
    )
    save_jsonl(sampled_rows, cfg.output_jsonl)
    logging.info("Saved sampled baseline dataset to: %s", cfg.output_jsonl)


if __name__ == "__main__":
    main()
