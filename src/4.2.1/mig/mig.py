#!/usr/bin/env python3
"""MIG scoring/prep script (decoupled from sampling).

This stage builds MIG-ready scoring data by:
1) loading InSTAG tags;
2) loading delta quality scores;
3) merging semantically similar tags via DBSCAN (eps=0.05 by default);
4) emitting a score-aligned json/jsonl used by `mig_sample.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    SentenceTransformer = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TagCluster:
    cluster_id: int
    name: str
    members: List[str]


def choose_row_id(row: Dict[str, Any], fallback_idx: int) -> Any:
    for key in ("id", "data_id", "uid", "idx", "index"):
        if key in row and row[key] is not None:
            return row[key]
    return fallback_idx


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def load_json_or_jsonl(path: str) -> Tuple[List[Dict[str, Any]], str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return [], "jsonl"

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data, "json"
        raise ValueError("Input JSON must be a list of objects.")
    except json.JSONDecodeError:
        lines = [line for line in text.splitlines() if line.strip()]
        return [json.loads(line) for line in lines], "jsonl"


def write_json_or_jsonl(path: str, rows: List[Dict[str, Any]], prefer_jsonl: bool) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_is_jsonl = prefer_jsonl or path.lower().endswith(".jsonl")
    with open(path, "w", encoding="utf-8") as f:
        if output_is_jsonl:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(rows, f, ensure_ascii=False, indent=2)


def dedup_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        k = str(item).strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def normalize_tag(tag: str) -> str:
    return str(tag or "").strip().lower()


def _hash_embed_one(text: str, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    s = f" {str(text).lower().strip()} "
    if not s.strip():
        return vec

    grams: List[str] = []
    if len(s) >= 3:
        grams = [s[i : i + 3] for i in range(len(s) - 2)]
    else:
        grams = [s]

    for gram in grams:
        idx = zlib.crc32(gram.encode("utf-8")) % dim
        vec[idx] += 1.0

    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


def encode_texts(
    texts: List[str],
    *,
    embedding_backend: str,
    embedding_model: str,
    embedding_cache: str,
) -> Tuple[np.ndarray, str]:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32), "empty"

    backend = embedding_backend.strip().lower()
    if backend not in {"auto", "sentence_transformers", "hash"}:
        raise ValueError("--embedding_backend must be one of: auto, sentence_transformers, hash")

    if backend in {"auto", "sentence_transformers"}:
        if SentenceTransformer is None:
            if backend == "sentence_transformers":
                raise ModuleNotFoundError(
                    "sentence-transformers is not installed. Install it or use --embedding_backend hash"
                )
            logger.warning("sentence-transformers unavailable, fallback to hash embeddings.")
        else:
            try:
                cache = embedding_cache.strip() or None
                model_name = embedding_model.strip() or "sentence-transformers/all-MiniLM-L6-v2"
                model = SentenceTransformer(model_name, cache_folder=cache)
                arr = model.encode(
                    texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim != 2:
                    raise ValueError(f"Unexpected embedding shape: {arr.shape}")
                return arr, "sentence_transformers"
            except Exception as exc:  # noqa: BLE001
                if backend == "sentence_transformers":
                    raise
                logger.warning("sentence-transformers encode failed (%s), fallback to hash embeddings.", exc)

    # fallback hash embeddings
    emb = np.stack([_hash_embed_one(t) for t in texts], axis=0)
    return emb, "hash"


def pairwise_cosine_distance(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    x = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms

    sim = x @ x.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    dist = np.clip(dist, 0.0, 2.0)
    return dist.astype(np.float32)


def dbscan_precomputed(distance: np.ndarray, eps: float, min_samples: int = 1) -> np.ndarray:
    """Simple DBSCAN implementation on a precomputed distance matrix."""
    n = int(distance.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    neighbors: List[np.ndarray] = [np.where(distance[i] <= eps)[0] for i in range(n)]

    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neigh = neighbors[i]
        if len(neigh) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seeds = set(int(x) for x in neigh.tolist() if int(x) != i)

        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                neigh_j = neighbors[j]
                if len(neigh_j) >= min_samples:
                    seeds.update(int(x) for x in neigh_j.tolist())

            if labels[j] == -1:
                labels[j] = cluster_id
            elif labels[j] < 0:
                labels[j] = cluster_id

        cluster_id += 1

    # For robustness: assign any remaining noise to singleton clusters.
    for i in range(n):
        if labels[i] < 0:
            labels[i] = cluster_id
            cluster_id += 1

    return labels


def choose_cluster_name(member_indices: List[int], tags: List[str], dist: np.ndarray) -> str:
    if not member_indices:
        return ""
    if len(member_indices) == 1:
        return tags[member_indices[0]]

    sub = dist[np.ix_(member_indices, member_indices)]
    avg_dist = sub.mean(axis=1)
    medoid_local = int(np.argmin(avg_dist))
    return tags[member_indices[medoid_local]]


def build_tag_clusters(
    tags: List[str],
    *,
    eps: float,
    min_samples: int,
    embedding_backend: str,
    embedding_model: str,
    embedding_cache: str,
) -> Tuple[List[TagCluster], Dict[str, str], str]:
    if not tags:
        return [], {}, "empty"

    embeddings, backend_used = encode_texts(
        tags,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_cache=embedding_cache,
    )
    dist = pairwise_cosine_distance(embeddings)
    labels = dbscan_precomputed(dist, eps=eps, min_samples=min_samples)

    cluster_to_members: Dict[int, List[int]] = {}
    for idx, cid in enumerate(labels.tolist()):
        cluster_to_members.setdefault(int(cid), []).append(idx)

    clusters: List[TagCluster] = []
    tag_to_cluster_name: Dict[str, str] = {}

    for cid in sorted(cluster_to_members):
        member_indices = cluster_to_members[cid]
        name = choose_cluster_name(member_indices, tags, dist)
        members = sorted(tags[i] for i in member_indices)
        cluster = TagCluster(cluster_id=cid, name=name, members=members)
        clusters.append(cluster)

        for t in members:
            tag_to_cluster_name[t] = name

    return clusters, tag_to_cluster_name, backend_used


def extract_delta_scalar(row: Dict[str, Any], mode: str = "auto") -> float:
    mode = mode.strip().lower()

    if "delta_scalar" in row:
        return to_float(row.get("delta_scalar"), 0.0)

    for key in ("score_scalar", "quality_score", "deita_score"):
        if key in row:
            return to_float(row.get(key), 0.0)

    score_val = row.get("score")
    mapped = row.get("mapped_vector")

    vec: Optional[List[float]] = None
    if isinstance(score_val, list):
        vec = [to_float(x, 0.0) for x in score_val]
    elif isinstance(mapped, list):
        vec = [to_float(x, 0.0) for x in mapped]

    if vec is None:
        if isinstance(score_val, (int, float)):
            return float(score_val)
        return 0.0

    if not vec:
        return 0.0

    nonzero = [x for x in vec if abs(x) > 1e-12]
    topk = row.get("top_k_indices")
    topk_len = len(topk) if isinstance(topk, list) and topk else 0

    if mode == "sum":
        return float(sum(vec))
    if mode == "max":
        return float(max(vec))
    if mode == "mean_nonzero":
        return float(sum(nonzero) / max(1, len(nonzero)))
    if mode == "mean_topk":
        denom = max(1, topk_len)
        return float(sum(vec) / denom)

    # auto mode
    score_type = str(row.get("score_type", "")).lower()
    if "delta_array" in score_type:
        if topk_len > 0:
            return float(sum(vec) / topk_len)
        if nonzero:
            return float(sum(nonzero) / len(nonzero))
        return 0.0

    # origin-like mapped vectors are weighted by top-k probabilities and sum to scalar.
    return float(sum(vec))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIG scoring stage: connect instag+delta, merge tags by DBSCAN, and output score-aligned data."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Raw dataset path (JSON or JSONL).")
    parser.add_argument("--instag_path", type=str, required=True, help="InSTAG tag output path.")
    parser.add_argument("--delta_path", type=str, required=True, help="Delta output path.")
    parser.add_argument("--output_path", type=str, required=True, help="MIG score output path.")
    parser.add_argument(
        "--cluster_output_path",
        type=str,
        default="",
        help="Optional path to save DBSCAN tag-cluster metadata json.",
    )
    parser.add_argument(
        "--valid_tag_output_path",
        type=str,
        default="",
        help="Optional path to save final merged tag list json.",
    )

    parser.add_argument("--starting_sample", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)

    parser.add_argument("--tag_merge_eps", type=float, default=0.05, help="DBSCAN eps on cosine distance.")
    parser.add_argument("--tag_merge_min_samples", type=int, default=1)
    parser.add_argument(
        "--embedding_backend",
        type=str,
        default="auto",
        choices=["auto", "sentence_transformers", "hash"],
        help="Embedding backend for semantic tag merge.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name/path when enabled.",
    )
    parser.add_argument("--embedding_cache", type=str, default="")

    parser.add_argument(
        "--delta_scalar_mode",
        type=str,
        default="auto",
        choices=["auto", "sum", "mean_nonzero", "mean_topk", "max"],
        help="How to derive scalar quality from delta row when delta_scalar is absent.",
    )
    parser.add_argument(
        "--include_raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to keep original row fields in MIG score output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_rows, data_fmt = load_json_or_jsonl(args.data_path)
    instag_rows, _ = load_json_or_jsonl(args.instag_path)
    delta_rows, _ = load_json_or_jsonl(args.delta_path)

    logger.info("Loaded data rows=%d from %s", len(data_rows), args.data_path)
    logger.info("Loaded instag rows=%d from %s", len(instag_rows), args.instag_path)
    logger.info("Loaded delta rows=%d from %s", len(delta_rows), args.delta_path)

    start = max(0, int(args.starting_sample))
    if start >= len(data_rows):
        raise ValueError(f"starting_sample={start} out of range for dataset size={len(data_rows)}")
    if int(args.max_samples) < 0:
        end = len(data_rows)
    else:
        end = min(len(data_rows), start + int(args.max_samples))

    selected_rows = data_rows[start:end]
    logger.info("Build MIG score rows [%d, %d) -> %d", start, end, len(selected_rows))

    # id -> instag tags
    instag_map: Dict[str, List[str]] = {}
    for idx, row in enumerate(instag_rows):
        rid = choose_row_id(row, idx)
        tags_raw = row.get("tags", row.get("labels", row.get("instag_tags", [])))
        tags: List[str] = []
        if isinstance(tags_raw, list):
            tags = [normalize_tag(t) for t in tags_raw if normalize_tag(str(t))]
        instag_map[str(rid)] = dedup_keep_order(tags)

    # id -> delta scalar
    delta_map: Dict[str, float] = {}
    for idx, row in enumerate(delta_rows):
        rid = choose_row_id(row, idx)
        delta_map[str(rid)] = extract_delta_scalar(row, mode=args.delta_scalar_mode)

    # Build global tag vocabulary from selected rows only.
    all_tags: List[str] = []
    missing_instag = 0
    for local_idx, row in enumerate(selected_rows):
        sid = choose_row_id(row, start + local_idx)
        tags = instag_map.get(str(sid), [])
        if not tags:
            missing_instag += 1
        all_tags.extend(tags)
    all_tags = dedup_keep_order(all_tags)

    logger.info(
        "Selected rows missing instag tags: %d / %d | unique raw tags=%d",
        missing_instag,
        len(selected_rows),
        len(all_tags),
    )

    clusters, tag_to_cluster_name, backend_used = build_tag_clusters(
        all_tags,
        eps=float(args.tag_merge_eps),
        min_samples=max(1, int(args.tag_merge_min_samples)),
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_cache=args.embedding_cache,
    )

    merged_tag_list = sorted({c.name for c in clusters if c.name})
    logger.info(
        "Tag clustering done | backend=%s | eps=%.4f | clusters=%d | merged_tags=%d",
        backend_used,
        float(args.tag_merge_eps),
        len(clusters),
        len(merged_tag_list),
    )

    out_rows: List[Dict[str, Any]] = []
    missing_delta = 0

    for local_idx, row in enumerate(selected_rows):
        sid = choose_row_id(row, start + local_idx)
        sid_key = str(sid)

        raw_tags = instag_map.get(sid_key, [])
        merged_labels = dedup_keep_order(tag_to_cluster_name.get(t, t) for t in raw_tags)

        if sid_key not in delta_map:
            missing_delta += 1
        delta_scalar = float(delta_map.get(sid_key, 0.0))

        out: Dict[str, Any] = {}
        if args.include_raw:
            out.update(row)

        out["id"] = sid
        out["labels"] = merged_labels
        out["score"] = delta_scalar
        out["score_type"] = "mig_point_score"
        out["instag_tags"] = raw_tags
        out["instag_tag_count"] = len(raw_tags)
        out["mig_labels"] = merged_labels
        out["mig_label_count"] = len(merged_labels)
        out["delta_scalar"] = delta_scalar
        out_rows.append(out)

    write_json_or_jsonl(args.output_path, out_rows, prefer_jsonl=(data_fmt == "jsonl"))
    logger.info(
        "Saved MIG score rows=%d to %s | missing delta=%d",
        len(out_rows),
        args.output_path,
        missing_delta,
    )

    if args.cluster_output_path:
        clusters_json = [
            {
                "cluster_id": c.cluster_id,
                "name": c.name,
                "members": c.members,
                "size": len(c.members),
            }
            for c in clusters
        ]
        meta = {
            "embedding_backend": backend_used,
            "embedding_model": args.embedding_model,
            "tag_merge_eps": float(args.tag_merge_eps),
            "tag_merge_min_samples": max(1, int(args.tag_merge_min_samples)),
            "unique_raw_tags": len(all_tags),
            "num_clusters": len(clusters_json),
            "num_merged_tags": len(merged_tag_list),
            "clusters": clusters_json,
            "tag_to_cluster_name": tag_to_cluster_name,
        }
        out_dir = os.path.dirname(args.cluster_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.cluster_output_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("Saved MIG tag-cluster metadata to %s", args.cluster_output_path)

    if args.valid_tag_output_path:
        out_dir = os.path.dirname(args.valid_tag_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.valid_tag_output_path, "w", encoding="utf-8") as f:
            json.dump(merged_tag_list, f, ensure_ascii=False, indent=2)
        logger.info("Saved merged valid tag list to %s", args.valid_tag_output_path)


if __name__ == "__main__":
    main()
