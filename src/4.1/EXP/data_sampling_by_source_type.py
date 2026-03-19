"""4.1 EXP - 按 source_type 均匀采样脚本。

目标：
1) 按 source_type 分组构建采样桶。
2) 过滤有效分组（size > min_valid_source_size）。
3) 默认预算 N=所有有效分组样本总量（可手动指定 --budget-n）。
4) 进行分组间均匀采样，输出：
   - dataset_source_type.jsonl
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


@dataclass
class SamplingConfig:
    profile_jsonl: Path
    source_jsonl: Path | None
    out_dir: Path
    source_field: str
    min_valid_source_size: int
    budget_n: int | None
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Uniform sampling by source_type")
    parser.add_argument(
        "--profile-jsonl",
        type=Path,
        default=Path("data/data_ablation_full59K/train.jsonl"),
        help="输出子集的数据来源 JSONL（结构保持不变）。",
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=None,
        help="可选：提供 source_type 映射来源；默认与 profile-jsonl 相同。",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/data_ablation_full59K/exp"))
    parser.add_argument("--source-field", type=str, default="source_type")
    parser.add_argument(
        "--min-valid-source-size",
        type=int,
        default=0,
        help="有效 source_type 最小样本数阈值（严格大于该值）。",
    )
    parser.add_argument(
        "--budget-n",
        type=int,
        default=None,
        help="可选手动预算 N。默认使用所有有效分组样本总量。",
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


def collect_valid_groups(
    row_ids_in_order: list[str],
    source_by_id: dict[str, str],
    min_valid_source_size: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for rid in row_ids_in_order:
        source_type = str(source_by_id.get(rid, "")).strip()
        if not source_type:
            source_type = "__MISSING_SOURCE_TYPE__"
        grouped[source_type].append(rid)

    all_sizes = {name: len(ids) for name, ids in grouped.items()}
    valid = {name: ids for name, ids in grouped.items() if len(ids) > min_valid_source_size}
    return valid, all_sizes


def sample_uniform_from_groups(groups: list[list[str]], budget_n: int, rng: random.Random) -> list[str]:
    if not groups or budget_n <= 0:
        return []

    full_pool = list(dict.fromkeys([x for g in groups for x in g]))
    if budget_n > len(full_pool):
        raise ValueError(f"Budget N={budget_n} exceeds unique candidate size={len(full_pool)}")

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
        raise RuntimeError(f"Uniform source_type sampling failed: got {len(picked)} expected {budget_n}")
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
        source_jsonl=args.source_jsonl,
        out_dir=args.out_dir,
        source_field=args.source_field,
        min_valid_source_size=max(0, args.min_valid_source_size),
        budget_n=args.budget_n,
        random_seed=args.random_seed,
    )
    rng = random.Random(cfg.random_seed)

    if not cfg.profile_jsonl.exists():
        raise FileNotFoundError(f"Profile jsonl not found: {cfg.profile_jsonl}")
    if cfg.source_jsonl is not None and not cfg.source_jsonl.exists():
        raise FileNotFoundError(f"Source jsonl not found: {cfg.source_jsonl}")

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

    source_by_id: dict[str, str] = {}
    source_rows = rows if cfg.source_jsonl is None else load_jsonl(cfg.source_jsonl)
    for i, row in enumerate(source_rows):
        rid = str(row.get("id", i))
        source_by_id[rid] = str(row.get(cfg.source_field, "")).strip()

    fallback_count = 0
    for rid, row in zip(row_ids_in_order, rows):
        if rid not in source_by_id or not source_by_id[rid]:
            source_by_id[rid] = str(row.get(cfg.source_field, "")).strip()
            fallback_count += 1
    if fallback_count > 0:
        logging.info(
            "Source fallback to profile rows for %d samples (field=%s).",
            fallback_count,
            cfg.source_field,
        )

    valid_groups_map, all_sizes = collect_valid_groups(
        row_ids_in_order=row_ids_in_order,
        source_by_id=source_by_id,
        min_valid_source_size=cfg.min_valid_source_size,
    )
    valid_groups = [ids for _, ids in sorted(valid_groups_map.items(), key=lambda x: x[0])]

    budget_n = sum(len(g) for g in valid_groups) if cfg.budget_n is None else int(cfg.budget_n)
    if budget_n <= 0:
        raise ValueError("No valid source_type groups found or budget_n <= 0.")
    candidate_size = len(list(dict.fromkeys([x for g in valid_groups for x in g])))
    if budget_n > candidate_size:
        raise ValueError(f"Budget N={budget_n} exceeds candidate size={candidate_size}.")

    logging.info(
        "Total source_types=%d, valid source_types=%d, budget N=%d, dataset size=%d",
        len(all_sizes),
        len(valid_groups),
        budget_n,
        len(rows_by_id),
    )

    picked_ids = sample_uniform_from_groups(valid_groups, budget_n, rng)
    out_path = cfg.out_dir / "dataset_source_type.jsonl"
    write_subset_jsonl(rows_by_id, picked_ids, out_path)

    meta = {
        "budget_n": budget_n,
        "source_field": cfg.source_field,
        "sampling_mode": "uniform",
        "min_valid_source_size": cfg.min_valid_source_size,
        "random_seed": cfg.random_seed,
        "total_source_type_count": len(all_sizes),
        "valid_source_type_count": len(valid_groups),
        "all_source_type_sizes": dict(sorted(all_sizes.items(), key=lambda x: x[0])),
        "valid_source_type_sizes": {
            k: len(v) for k, v in sorted(valid_groups_map.items(), key=lambda x: x[0])
        },
        "output": {"source_type_uniform": str(out_path)},
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cfg.out_dir / "sampling_meta_source_type.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("Source_type uniform sampling completed. Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()
