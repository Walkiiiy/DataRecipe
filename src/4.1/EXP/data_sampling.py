"""4.1 EXP - 数据采样脚本

目标：
1) 从 capability_tree_final.json 解析叶子簇，筛选 size > min_valid_cluster_size 的有效簇。
2) 计算预算 N = 所有有效簇样本量之和。
3) 生成三套严格等量（均为 N）的训练子集：
   - dataset_ours.jsonl: 基于能力树有效簇采样（按比例/均匀）
   - dataset_kmeans.jsonl: KMeans(K=8) 均匀采样
   - dataset_random.jsonl: 全量随机无放回采样

说明：
- 脚本只负责数据子集构造，不做训练。
- 为保证论文可复现，采样过程支持固定随机种子。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception:  # noqa: BLE001
    torch = None
    SentenceTransformer = None

from sklearn.cluster import KMeans


@dataclass
class SamplingConfig:
    tree_json: Path
    profile_jsonl: Path
    out_dir: Path
    min_valid_cluster_size: int
    ours_mode: str
    kmeans_k: int
    embedding_model: str
    device: str
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data sampling for controlled recipe comparison")
    parser.add_argument(
        "--tree-json",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/capability_tree_final.json"),
        help="Stage3 输出的能力树 JSON。",
    )
    parser.add_argument(
        "--profile-jsonl",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl"),
        help="原始带 CDT 的样本文件。",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/alpaca-gpt4-data-en/exp"))
    parser.add_argument(
        "--min-valid-cluster-size",
        type=int,
        default=10,
        help="有效簇最小样本数阈值（严格大于该值）。",
    )
    parser.add_argument(
        "--ours-mode",
        type=str,
        choices=["proportional", "uniform"],
        default="proportional",
        help="Ours 采样模式：按原簇比例或簇间均匀。",
    )
    parser.add_argument("--kmeans-k", type=int, default=8)
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Tree JSON must be object: {path}")
    return data


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
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def iter_leaf_nodes(tree: dict[str, Any]) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []

    def _walk(node: dict[str, Any]) -> None:
        children = node.get("children", [])
        if not children:
            leaves.append(node)
            return
        for child in children:
            _walk(child)

    _walk(tree)
    return leaves


def collect_valid_clusters(tree: dict[str, Any], min_valid_cluster_size: int) -> list[list[str]]:
    """提取有效簇（叶子）并返回每簇 data_ids。"""
    leaves = iter_leaf_nodes(tree)
    valid: list[list[str]] = []
    for leaf in leaves:
        ids = [str(x) for x in leaf.get("data_ids", [])]
        if len(ids) > min_valid_cluster_size:
            # 去重，保证后续无放回采样合法
            valid.append(list(dict.fromkeys(ids)))
    return valid


def sample_from_clusters(
    clusters: list[list[str]],
    budget_n: int,
    mode: str,
    rng: random.Random,
) -> list[str]:
    """从簇集合中采样总量为 N 的样本 ID。"""
    if not clusters:
        return []
    if budget_n <= 0:
        return []

    # 全部候选池（用于后续补齐）
    full_pool = list(dict.fromkeys([x for c in clusters for x in c]))
    if budget_n > len(full_pool):
        raise ValueError(f"Budget N={budget_n} exceeds unique candidate size={len(full_pool)}")

    picked: list[str] = []

    if mode == "proportional":
        total = sum(len(c) for c in clusters)
        quotas_float = [budget_n * len(c) / total for c in clusters]
        quotas = [int(math.floor(q)) for q in quotas_float]
        # 把余数按小数部分最大优先补齐，确保总和严格为 N
        remain = budget_n - sum(quotas)
        frac_rank = sorted(range(len(clusters)), key=lambda i: quotas_float[i] - quotas[i], reverse=True)
        for i in frac_rank[:remain]:
            quotas[i] += 1
    else:
        # uniform: 簇间尽量均匀，再考虑簇容量上限
        k = len(clusters)
        quotas = [budget_n // k] * k
        for i in range(budget_n % k):
            quotas[i] += 1
        # 若某些簇容量不够，先截断，再在剩余池补齐
        for i in range(k):
            quotas[i] = min(quotas[i], len(clusters[i]))

    for c, q in zip(clusters, quotas):
        if q <= 0:
            continue
        picked.extend(rng.sample(c, q))

    # 不足时，从剩余候选中补齐
    picked = list(dict.fromkeys(picked))
    if len(picked) < budget_n:
        remain_pool = [x for x in full_pool if x not in set(picked)]
        need = budget_n - len(picked)
        picked.extend(rng.sample(remain_pool, need))

    if len(picked) > budget_n:
        picked = rng.sample(picked, budget_n)

    if len(picked) != budget_n:
        raise RuntimeError(f"Ours sampling failed to meet budget: got {len(picked)} expected {budget_n}")
    return picked


def build_embeddings(rows: list[dict[str, Any]], model_name: str, device: str) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError("Missing sentence-transformers/torch. Install before running kmeans baseline.")
    model = SentenceTransformer(model_name, device=resolve_device(device))
    texts = [str(r.get("CDT_description") or r.get("T_description") or "").strip() for r in rows]
    vectors = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def sample_kmeans_uniform(
    row_ids: list[str],
    vectors: np.ndarray,
    budget_n: int,
    k: int,
    rng: random.Random,
    random_seed: int,
) -> list[str]:
    if budget_n > len(row_ids):
        raise ValueError(f"Budget N={budget_n} exceeds dataset size={len(row_ids)}")
    if k <= 1:
        raise ValueError("kmeans_k must be >= 2")

    model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    labels = model.fit_predict(vectors)
    buckets: dict[int, list[str]] = defaultdict(list)
    for rid, lb in zip(row_ids, labels):
        buckets[int(lb)].append(rid)

    # 均匀配额
    quotas = [budget_n // k] * k
    for i in range(budget_n % k):
        quotas[i] += 1
    for i in range(k):
        quotas[i] = min(quotas[i], len(buckets.get(i, [])))

    picked: list[str] = []
    for i in range(k):
        q = quotas[i]
        if q <= 0:
            continue
        picked.extend(rng.sample(buckets[i], q))

    # 容量不足补齐
    picked = list(dict.fromkeys(picked))
    if len(picked) < budget_n:
        remain_pool = [rid for rid in row_ids if rid not in set(picked)]
        need = budget_n - len(picked)
        picked.extend(rng.sample(remain_pool, need))

    if len(picked) > budget_n:
        picked = rng.sample(picked, budget_n)

    if len(picked) != budget_n:
        raise RuntimeError(f"KMeans sampling failed to meet budget: got {len(picked)} expected {budget_n}")
    return picked


def write_subset_jsonl(rows_by_id: dict[str, dict[str, Any]], picked_ids: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rid in picked_ids:
            row = rows_by_id[rid]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = SamplingConfig(
        tree_json=args.tree_json,
        profile_jsonl=args.profile_jsonl,
        out_dir=args.out_dir,
        min_valid_cluster_size=max(0, args.min_valid_cluster_size),
        ours_mode=args.ours_mode,
        kmeans_k=max(2, args.kmeans_k),
        embedding_model=args.embedding_model,
        device=args.device,
        random_seed=args.random_seed,
    )
    rng = random.Random(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    if not cfg.tree_json.exists():
        raise FileNotFoundError(f"Tree json not found: {cfg.tree_json}")
    if not cfg.profile_jsonl.exists():
        raise FileNotFoundError(f"Profile jsonl not found: {cfg.profile_jsonl}")

    tree = load_json(cfg.tree_json)
    rows = load_jsonl(cfg.profile_jsonl)
    if not rows:
        raise ValueError("Profile JSONL is empty.")

    # 建立 ID 索引，便于精确抽样和写回
    rows_by_id: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        rid = str(row.get("id", i))
        # 若出现重复 ID，后者覆盖前者；并在日志中给出提示
        if rid in rows_by_id:
            logging.warning("Duplicate sample id found: %s; overwrite with later row.", rid)
        rows_by_id[rid] = row
    all_ids = list(rows_by_id.keys())

    valid_clusters = collect_valid_clusters(tree, cfg.min_valid_cluster_size)
    # 过滤掉树里存在但数据集中找不到的 ID
    valid_clusters = [[sid for sid in c if sid in rows_by_id] for c in valid_clusters]
    valid_clusters = [c for c in valid_clusters if c]

    budget_n = sum(len(c) for c in valid_clusters)
    if budget_n <= 0:
        raise ValueError("No valid clusters found. Please check tree quality or lower min_valid_cluster_size.")
    if budget_n > len(all_ids):
        raise ValueError(f"Budget N={budget_n} exceeds available dataset size={len(all_ids)}.")

    logging.info(
        "Valid clusters=%d, budget N=%d, dataset size=%d",
        len(valid_clusters),
        budget_n,
        len(all_ids),
    )

    ours_ids = sample_from_clusters(valid_clusters, budget_n, cfg.ours_mode, rng)
    random_ids = rng.sample(all_ids, budget_n)
    vectors = build_embeddings(rows, cfg.embedding_model, cfg.device)
    row_ids_in_order = [str(r.get("id", i)) for i, r in enumerate(rows)]
    kmeans_ids = sample_kmeans_uniform(
        row_ids=row_ids_in_order,
        vectors=vectors,
        budget_n=budget_n,
        k=cfg.kmeans_k,
        rng=rng,
        random_seed=cfg.random_seed,
    )

    out_ours = cfg.out_dir / "dataset_ours.jsonl"
    out_kmeans = cfg.out_dir / "dataset_kmeans.jsonl"
    out_random = cfg.out_dir / "dataset_random.jsonl"
    write_subset_jsonl(rows_by_id, ours_ids, out_ours)
    write_subset_jsonl(rows_by_id, kmeans_ids, out_kmeans)
    write_subset_jsonl(rows_by_id, random_ids, out_random)

    meta = {
        "budget_n": budget_n,
        "valid_cluster_count": len(valid_clusters),
        "min_valid_cluster_size": cfg.min_valid_cluster_size,
        "ours_mode": cfg.ours_mode,
        "kmeans_k": cfg.kmeans_k,
        "random_seed": cfg.random_seed,
        "output": {
            "ours": str(out_ours),
            "kmeans": str(out_kmeans),
            "random": str(out_random),
        },
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.out_dir / "sampling_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("Sampling completed. Meta saved to %s", cfg.out_dir / "sampling_meta.json")


if __name__ == "__main__":
    main()
