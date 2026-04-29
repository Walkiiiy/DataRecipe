"""4.1 EXP - 基于数据集自带 category 的采样脚本

目标：
1) 从输入 JSONL 的 category 字段构建类别分组。
2) 筛选 size > min_valid_category_size 的有效类别。
3) 默认预算 N 为所有有效类别样本量之和（也可手动指定）。
4) 按类别比例或类别均匀方式采样，输出严格等量 N 的子集：
   - dataset_category.jsonl
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


@dataclass
class SamplingConfig:
    profile_jsonl: Path
    category_jsonl: Path | None
    out_dir: Path
    category_field: str
    min_valid_category_size: int
    category_mode: str
    budget_n: int | None
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Category-based data sampling")
    parser.add_argument(
        "--profile-jsonl",
        type=Path,
        default=Path("data/dolly-15k/dolly-15k_cdt_profile.jsonl"),
        help="输出子集的数据来源 JSONL（保持与现有 dataset_ours/kmeans/random 一致的结构）。",
    )
    parser.add_argument(
        "--category-jsonl",
        type=Path,
        default=Path("data/dolly-15k/train.jsonl"),
        help="提供 category 字段的 JSONL；若不提供则尝试直接从 profile-jsonl 读取 category。",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/dolly-15k/exp"))
    parser.add_argument(
        "--category-field",
        type=str,
        default="category",
        help="用于分组的类别字段名。",
    )
    parser.add_argument(
        "--min-valid-category-size",
        type=int,
        default=10,
        help="有效类别最小样本数阈值（严格大于该值）。",
    )
    parser.add_argument(
        "--category-mode",
        type=str,
        choices=["proportional", "uniform"],
        default="proportional",
        help="类别采样模式：按原类别比例或类别均匀。",
    )
    parser.add_argument(
        "--budget-n",
        type=int,
        default=None,
        help="可选手动预算 N。默认使用所有有效类别样本量之和。",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


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


def collect_valid_categories(
    row_ids_in_order: list[str],
    category_by_id: dict[str, str],
    min_valid_category_size: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for rid in row_ids_in_order:
        category = str(category_by_id.get(rid, "")).strip()
        if not category:
            category = "__MISSING_CATEGORY__"
        grouped[category].append(rid)

    all_category_sizes = {name: len(ids) for name, ids in grouped.items()}
    valid = {name: ids for name, ids in grouped.items() if len(ids) > min_valid_category_size}
    return valid, all_category_sizes


def sample_from_groups(
    groups: list[list[str]],
    budget_n: int,
    mode: str,
    rng: random.Random,
) -> list[str]:
    if not groups or budget_n <= 0:
        return []

    full_pool = list(dict.fromkeys([x for g in groups for x in g]))
    if budget_n > len(full_pool):
        raise ValueError(f"Budget N={budget_n} exceeds unique candidate size={len(full_pool)}")

    if mode == "proportional":
        total = sum(len(g) for g in groups)
        quotas_float = [budget_n * len(g) / total for g in groups]
        quotas = [int(math.floor(q)) for q in quotas_float]
        remain = budget_n - sum(quotas)
        frac_rank = sorted(range(len(groups)), key=lambda i: quotas_float[i] - quotas[i], reverse=True)
        for i in frac_rank[:remain]:
            quotas[i] += 1
    else:
        k = len(groups)
        quotas = [budget_n // k] * k
        for i in range(budget_n % k):
            quotas[i] += 1
        for i in range(k):
            quotas[i] = min(quotas[i], len(groups[i]))

    picked: list[str] = []
    for g, q in zip(groups, quotas):
        if q > 0:
            picked.extend(rng.sample(g, q))

    picked = list(dict.fromkeys(picked))
    if len(picked) < budget_n:
        picked_set = set(picked)
        remain_pool = [x for x in full_pool if x not in picked_set]
        need = budget_n - len(picked)
        picked.extend(rng.sample(remain_pool, need))

    if len(picked) > budget_n:
        picked = rng.sample(picked, budget_n)

    if len(picked) != budget_n:
        raise RuntimeError(f"Category sampling failed to meet budget: got {len(picked)} expected {budget_n}")
    return picked


def write_subset_jsonl(rows_by_id: dict[str, dict[str, Any]], picked_ids: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rid in picked_ids:
            f.write(json.dumps(rows_by_id[rid], ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = SamplingConfig(
        profile_jsonl=args.profile_jsonl,
        category_jsonl=args.category_jsonl,
        out_dir=args.out_dir,
        category_field=args.category_field,
        min_valid_category_size=max(0, args.min_valid_category_size),
        category_mode=args.category_mode,
        budget_n=args.budget_n,
        random_seed=args.random_seed,
    )
    rng = random.Random(cfg.random_seed)

    if not cfg.profile_jsonl.exists():
        raise FileNotFoundError(f"Profile jsonl not found: {cfg.profile_jsonl}")
    if cfg.category_jsonl is not None and not cfg.category_jsonl.exists():
        raise FileNotFoundError(f"Category jsonl not found: {cfg.category_jsonl}")

    rows = load_jsonl(cfg.profile_jsonl)
    if not rows:
        raise ValueError("Profile JSONL is empty.")

    rows_by_id: dict[str, dict[str, Any]] = {}
    row_ids_in_order: list[str] = []
    for i, row in enumerate(rows):
        rid = str(row.get("id", i))
        if rid in rows_by_id:
            logging.warning("Duplicate sample id found: %s; overwrite with later row.", rid)
        rows_by_id[rid] = row
        row_ids_in_order.append(rid)

    category_by_id: dict[str, str] = {}
    if cfg.category_jsonl is not None:
        category_rows = load_jsonl(cfg.category_jsonl)
        for i, row in enumerate(category_rows):
            rid = str(row.get("id", i))
            category_by_id[rid] = str(row.get(cfg.category_field, "")).strip()

    # 回退：若 category_jsonl 未提供或缺失映射，尝试从 profile 自身读取 category。
    fallback_count = 0
    for rid, row in zip(row_ids_in_order, rows):
        if rid not in category_by_id or not category_by_id[rid]:
            category_by_id[rid] = str(row.get(cfg.category_field, "")).strip()
            fallback_count += 1
    if fallback_count > 0:
        logging.info(
            "Category fallback to profile rows for %d samples (field=%s).",
            fallback_count,
            cfg.category_field,
        )

    valid_categories, all_category_sizes = collect_valid_categories(
        row_ids_in_order=row_ids_in_order,
        category_by_id=category_by_id,
        min_valid_category_size=cfg.min_valid_category_size,
    )
    valid_groups = [ids for _, ids in sorted(valid_categories.items(), key=lambda x: x[0])]

    budget_n = sum(len(g) for g in valid_groups) if cfg.budget_n is None else cfg.budget_n
    if budget_n <= 0:
        raise ValueError("No valid categories found or budget_n <= 0.")
    candidate_size = len(list(dict.fromkeys([x for g in valid_groups for x in g])))
    if budget_n > candidate_size:
        raise ValueError(f"Budget N={budget_n} exceeds valid-category candidate size={candidate_size}.")

    logging.info(
        "Total categories=%d, valid categories=%d, budget N=%d, dataset size=%d",
        len(all_category_sizes),
        len(valid_groups),
        budget_n,
        len(rows_by_id),
    )

    picked_ids = sample_from_groups(valid_groups, budget_n, cfg.category_mode, rng)
    out_category = cfg.out_dir / "dataset_category.jsonl"
    write_subset_jsonl(rows_by_id, picked_ids, out_category)

    meta = {
        "budget_n": budget_n,
        "category_field": cfg.category_field,
        "category_mode": cfg.category_mode,
        "min_valid_category_size": cfg.min_valid_category_size,
        "random_seed": cfg.random_seed,
        "total_category_count": len(all_category_sizes),
        "valid_category_count": len(valid_groups),
        "all_category_sizes": dict(sorted(all_category_sizes.items(), key=lambda x: x[0])),
        "valid_category_sizes": {
            k: len(v) for k, v in sorted(valid_categories.items(), key=lambda x: x[0])
        },
        "output": {"category": str(out_category)},
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cfg.out_dir / "sampling_meta_category.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("Category sampling completed. Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()
