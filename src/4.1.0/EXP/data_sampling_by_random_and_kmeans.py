"""4.1 EXP - Random + KMeans 采样脚本

目标：
1) 使用统一预算 N（由 --budget-n 指定）。
2) 生成两套严格等量（均为 N）的训练子集：
   - dataset_kmeans.jsonl: KMeans(K=k) 均匀采样
   - dataset_random.jsonl: 全量随机无放回采样

说明：
- 脚本只负责数据子集构造，不做训练。
- 为保证论文可复现，采样过程支持固定随机种子。
"""

from __future__ import annotations

import argparse
import json
import logging
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
    profile_jsonl: Path
    out_dir: Path
    budget_n: int
    kmeans_k: int
    embedding_model: str
    device: str
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random + KMeans data sampling")
    parser.add_argument(
        "--profile-jsonl",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl"),
        help="原始带 CDT 的样本文件（用于 KMeans 向量化与最终写出）。",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/alpaca-gpt4-data-en/exp"))
    parser.add_argument("--budget-n", type=int, required=True, help="统一采样预算 N。")
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


def build_embeddings(rows: list[dict[str, Any]], model_name: str, device: str) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError("Missing sentence-transformers/torch. Install before running kmeans baseline.")
    model = SentenceTransformer(model_name, device=resolve_device(device))
    texts = [str(r.get("instruction") or "").strip() for r in rows]
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
        profile_jsonl=args.profile_jsonl,
        out_dir=args.out_dir,
        budget_n=max(1, args.budget_n),
        kmeans_k=max(2, args.kmeans_k),
        embedding_model=args.embedding_model,
        device=args.device,
        random_seed=args.random_seed,
    )
    rng = random.Random(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    if not cfg.profile_jsonl.exists():
        raise FileNotFoundError(f"Profile jsonl not found: {cfg.profile_jsonl}")

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

    budget_n = cfg.budget_n
    if budget_n <= 0:
        raise ValueError("budget_n must be > 0.")
    if budget_n > len(all_ids):
        raise ValueError(f"Budget N={budget_n} exceeds available dataset size={len(all_ids)}.")

    logging.info(
        "Budget N=%d, dataset size=%d, kmeans_k=%d",
        budget_n,
        len(all_ids),
        cfg.kmeans_k,
    )

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

    out_kmeans = cfg.out_dir / "dataset_kmeans.jsonl"
    out_random = cfg.out_dir / "dataset_random.jsonl"
    write_subset_jsonl(rows_by_id, kmeans_ids, out_kmeans)
    write_subset_jsonl(rows_by_id, random_ids, out_random)

    meta = {
        "budget_n": budget_n,
        "kmeans_k": cfg.kmeans_k,
        "random_seed": cfg.random_seed,
        "output": {
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
