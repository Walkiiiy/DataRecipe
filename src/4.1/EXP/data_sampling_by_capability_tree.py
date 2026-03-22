"""4.1 EXP - 能力树采样（v5：仅用树内数据 + 按树内比例采样）。

设计目标：
1) 仅使用能力树自身携带的 data_ids 作为候选池；
2) 若树内候选总量小于预算 N，直接报错；
3) 按树内分组（节点 data_ids）大小比例分配配额并采样；
4) 输出 dataset_ours.jsonl 与采样统计 meta。

输出：
- dataset_ours.jsonl
- sampling_meta_ours_tree_v3.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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


@dataclass
class SamplingConfig:
    tree_json: Path
    profile_jsonl: Path
    out_dir: Path
    min_valid_cluster_size: int
    budget_n: int
    assignment_distance_quantile: float
    assignment_distance_scale: float
    near_ratio: float
    min_level_quota: int
    # Deprecated args kept for backward compatibility in CLI.
    radius_quantile: float
    radius_scale: float
    allow_loose_fill: bool
    embedding_model: str
    device: str
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capability-tree sampling v5 (tree-only proportional)")
    parser.add_argument(
        "--tree-json",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/capability_tree_final_pruned.json"),
        help="Stage3/Stage4 输出的能力树 JSON。",
    )
    parser.add_argument(
        "--profile-jsonl",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/alpaca_cdt_profile.jsonl"),
        help="带 CDT_description 的训练集 JSONL（最终输出结构来源）。",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/alpaca-gpt4-data-en/exp"))
    parser.add_argument(
        "--min-valid-cluster-size",
        type=int,
        default=10,
        help="有效叶簇最小样本数阈值（严格大于该值）。",
    )
    parser.add_argument(
        "--budget-n",
        type=int,
        required=True,
        help="统一采样预算 N（与 kmeans/random 对齐，必填）。",
    )
    parser.add_argument(
        "--assignment-distance-quantile",
        type=float,
        default=0.99,
        help="样本到最近叶簇中心距离的分位数门限（<1 时启用过滤）。",
    )
    parser.add_argument(
        "--assignment-distance-scale",
        type=float,
        default=1.0,
        help="距离门限缩放系数，threshold = quantile * scale。",
    )
    parser.add_argument(
        "--near-ratio",
        type=float,
        default=0.7,
        help="簇内近心样本比例（其余为边界/困难样本）。范围 [0,1]。",
    )
    parser.add_argument(
        "--min-level-quota",
        type=int,
        default=1,
        help="每个有容量层的最小保底配额。",
    )
    # Deprecated args (v3 不使用，但保留参数兼容旧命令行)
    parser.add_argument(
        "--radius-quantile",
        type=float,
        default=0.9,
        help="[Deprecated in v3] 保留兼容，无实际作用。",
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="[Deprecated in v3] 保留兼容，无实际作用。",
    )
    parser.add_argument(
        "--allow-loose-fill",
        action="store_true",
        help="[Deprecated in v3] 保留兼容，无实际作用。",
    )
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


def iter_leaf_nodes_with_depth(tree: dict[str, Any]) -> list[tuple[dict[str, Any], int]]:
    leaves: list[tuple[dict[str, Any], int]] = []

    def _walk(node: dict[str, Any], depth: int) -> None:
        children = node.get("children", [])
        if not children:
            leaves.append((node, depth))
            return
        for child in children:
            _walk(child, depth + 1)

    _walk(tree, 0)
    return leaves


def collect_valid_leaf_specs(tree: dict[str, Any], min_valid_cluster_size: int) -> list[dict[str, Any]]:
    leaf_pairs = iter_leaf_nodes_with_depth(tree)
    specs: list[dict[str, Any]] = []
    for leaf, depth in leaf_pairs:
        ids = [str(x) for x in leaf.get("data_ids", [])]
        ids = list(dict.fromkeys(ids))
        if len(ids) > min_valid_cluster_size:
            specs.append(
                {
                    "node_id": str(leaf.get("node_id", f"LEAF_{len(specs)}")),
                    "depth": int(depth),
                    "core_ids": ids,
                }
            )
    return specs


def build_embeddings(rows: list[dict[str, Any]], model_name: str, device: str) -> np.ndarray:
    if SentenceTransformer is None:
        raise ImportError("Missing sentence-transformers/torch. Install before running tree sampling.")
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


def write_subset_jsonl(rows_by_id: dict[str, dict[str, Any]], picked_ids: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rid in picked_ids:
            f.write(json.dumps(rows_by_id[rid], ensure_ascii=False) + "\n")


def collect_tree_sampling_groups(tree: dict[str, Any]) -> tuple[list[dict[str, Any]], set[str]]:
    """从能力树收集采样分组（仅基于树自身 data_ids）。

    规则：
    - 每个节点自身的 data_ids 作为一个分组；
    - 采用 DFS 遍历；
    - 全局去重：同一个 id 仅归属到首次遇到的分组，避免重复采样。
    """
    groups: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    def _walk(node: dict[str, Any], depth: int) -> None:
        node_id = str(node.get("node_id", f"N_DEPTH_{depth}"))
        raw_ids = [str(x) for x in node.get("data_ids", [])]
        unique_ids: list[str] = []
        for rid in raw_ids:
            if rid not in used_ids:
                used_ids.add(rid)
                unique_ids.append(rid)
        if unique_ids:
            groups.append(
                {
                    "group_id": node_id,
                    "depth": int(depth),
                    "ids": unique_ids,
                }
            )
        for child in node.get("children", []) or []:
            _walk(child, depth + 1)

    _walk(tree, 0)
    return groups, used_ids


def allocate_group_quotas_by_size(group_sizes: list[int], budget_n: int) -> list[int]:
    """按组大小比例分配预算（Hamilton 最大余数法）。"""
    total = int(sum(group_sizes))
    if total <= 0:
        return [0] * len(group_sizes)

    # 理论配额 -> 向下取整
    exact = [budget_n * s / total for s in group_sizes]
    quotas = [min(int(x), group_sizes[i]) for i, x in enumerate(exact)]

    remain = int(budget_n - sum(quotas))
    remainders = sorted(
        [(exact[i] - int(exact[i]), i) for i in range(len(group_sizes))],
        key=lambda x: x[0],
        reverse=True,
    )
    while remain > 0:
        progressed = False
        for _, idx in remainders:
            if quotas[idx] < group_sizes[idx]:
                quotas[idx] += 1
                remain -= 1
                progressed = True
                if remain == 0:
                    break
        if not progressed:
            break
    return quotas


def allocate_level_quotas_by_capacity(
    levels: list[int],
    level_capacities: dict[int, int],
    budget_n: int,
    min_level_quota: int,
) -> dict[int, int]:
    """按层容量比例分配预算，并支持每层最小保底。"""
    quotas = {lv: 0 for lv in levels}
    if not levels or budget_n <= 0:
        return quotas

    active_levels = [lv for lv in levels if level_capacities.get(lv, 0) > 0]
    if not active_levels:
        return quotas

    remain_budget = int(budget_n)

    # 先给每个有容量层最小保底，避免深层被完全饿死。
    if min_level_quota > 0:
        for lv in active_levels:
            if remain_budget <= 0:
                break
            give = min(int(min_level_quota), int(level_capacities[lv]), remain_budget)
            quotas[lv] += give
            remain_budget -= give

    residual_caps = {lv: max(0, int(level_capacities[lv] - quotas[lv])) for lv in active_levels}
    total_residual_cap = sum(residual_caps.values())
    if remain_budget <= 0 or total_residual_cap <= 0:
        return quotas

    # 再按剩余容量比例分配剩余预算。
    base_added: dict[int, int] = {}
    frac_parts: list[tuple[float, int]] = []
    for lv in active_levels:
        cap = residual_caps[lv]
        if cap <= 0:
            base_added[lv] = 0
            continue
        exact = remain_budget * cap / total_residual_cap
        base = min(int(exact), cap)
        base_added[lv] = base
        frac_parts.append((exact - base, lv))

    used = sum(base_added.values())
    for lv, add in base_added.items():
        quotas[lv] += add
    leftover = remain_budget - used

    # 最后用小数部分从大到小补齐剩余名额（同时受容量上限约束）。
    frac_parts.sort(reverse=True, key=lambda x: x[0])
    while leftover > 0:
        progressed = False
        for _, lv in frac_parts:
            if quotas[lv] < level_capacities[lv]:
                quotas[lv] += 1
                leftover -= 1
                progressed = True
                if leftover == 0:
                    break
        if not progressed:
            break
    return quotas


def build_mixed_queue(pairs_sorted_asc: list[tuple[str, float]], near_ratio: float) -> list[str]:
    """将单簇样本按“近心 + 边界”混合成可顺序抽取队列。

    - near_ratio=1: 纯近心优先；
    - near_ratio=0: 纯边界优先；
    - 中间值：按近/远交替近似比例混合。
    """
    if not pairs_sorted_asc:
        return []
    if near_ratio >= 1.0:
        return [rid for rid, _ in pairs_sorted_asc]
    if near_ratio <= 0.0:
        return [rid for rid, _ in reversed(pairs_sorted_asc)]

    n = len(pairs_sorted_asc)
    near_count = max(1, min(n, int(round(n * near_ratio))))
    near_part = pairs_sorted_asc[:near_count]
    far_part = list(reversed(pairs_sorted_asc[near_count:]))  # 边界样本：距离更远优先

    near_per_far = max(1, int(round(near_ratio / max(1e-6, 1.0 - near_ratio))))
    out: list[str] = []
    i = 0
    j = 0
    while i < len(near_part) or j < len(far_part):
        for _ in range(near_per_far):
            if i < len(near_part):
                out.append(near_part[i][0])
                i += 1
        if j < len(far_part):
            out.append(far_part[j][0])
            j += 1
        if i >= len(near_part) and j < len(far_part):
            out.extend([rid for rid, _ in far_part[j:]])
            break
        if j >= len(far_part) and i < len(near_part):
            out.extend([rid for rid, _ in near_part[i:]])
            break
    return out


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
        budget_n=max(1, int(args.budget_n)),
        assignment_distance_quantile=min(1.0, max(0.0, float(args.assignment_distance_quantile))),
        assignment_distance_scale=max(1e-8, float(args.assignment_distance_scale)),
        near_ratio=min(1.0, max(0.0, float(args.near_ratio))),
        min_level_quota=max(0, int(args.min_level_quota)),
        radius_quantile=min(1.0, max(1e-6, float(args.radius_quantile))),
        radius_scale=max(1e-6, float(args.radius_scale)),
        allow_loose_fill=bool(args.allow_loose_fill),
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
    rows_raw = load_jsonl(cfg.profile_jsonl)
    if not rows_raw:
        raise ValueError("Profile JSONL is empty.")

    # 建立唯一 ID 索引（后者覆盖前者）。
    rows_by_id: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows_raw):
        rid = str(row.get("id", i))
        if rid in rows_by_id:
            logging.warning("Duplicate sample id found: %s; overwrite with later row.", rid)
        rows_by_id[rid] = row

    # 新策略：仅使用能力树自身数据作为采样池（不再从全量 profile 做最近簇重分配）。
    groups_raw, tree_ids_unique = collect_tree_sampling_groups(tree)
    if not groups_raw:
        raise ValueError("No data_ids found in tree. Cannot sample from tree itself.")

    groups: list[dict[str, Any]] = []
    missing_in_profile_count = 0
    for g in groups_raw:
        ids_in_profile = [rid for rid in g["ids"] if rid in rows_by_id]
        missing_in_profile_count += len(g["ids"]) - len(ids_in_profile)
        if ids_in_profile:
            groups.append(
                {
                    "group_id": g["group_id"],
                    "depth": g["depth"],
                    "ids": ids_in_profile,
                }
            )

    if not groups:
        raise ValueError("All tree ids are missing in profile jsonl. Cannot write sampled dataset.")

    group_sizes = [len(g["ids"]) for g in groups]
    candidate_count = int(sum(group_sizes))
    if cfg.budget_n > candidate_count:
        raise ValueError(
            f"Tree candidate size={candidate_count} is smaller than budget_n={cfg.budget_n}. "
            "Please lower budget or rebuild tree with more samples."
        )

    quotas = allocate_group_quotas_by_size(group_sizes, cfg.budget_n)
    picked: list[str] = []
    selected_per_group: list[int] = []
    for i, g in enumerate(groups):
        q = int(quotas[i])
        if q <= 0:
            selected_per_group.append(0)
            continue
        chosen = rng.sample(g["ids"], q)
        picked.extend(chosen)
        selected_per_group.append(len(chosen))

    # 理论上 Hamilton 配额应达到预算；此处做健壮性补齐。
    picked = list(dict.fromkeys(picked))
    if len(picked) < cfg.budget_n:
        remain_pool = [rid for g in groups for rid in g["ids"] if rid not in set(picked)]
        need = cfg.budget_n - len(picked)
        if need > len(remain_pool):
            raise RuntimeError(
                f"Not enough remaining ids for fill: need={need}, remain_pool={len(remain_pool)}"
            )
        logging.warning(
            "Proportional allocation underfilled by %d; fallback global fill from tree pool.",
            need,
        )
        picked.extend(rng.sample(remain_pool, need))

    if len(picked) > cfg.budget_n:
        picked = rng.sample(picked, cfg.budget_n)

    if len(picked) != cfg.budget_n:
        raise RuntimeError(f"Final picked size mismatch: got={len(picked)} expected={cfg.budget_n}")
    if len(set(picked)) != len(picked):
        raise RuntimeError("Picked ids contain duplicates, which should not happen.")

    rng.shuffle(picked)

    out_ours = cfg.out_dir / "dataset_ours.jsonl"
    write_subset_jsonl(rows_by_id, picked, out_ours)

    groups_by_depth: dict[int, int] = {}
    selected_by_depth: dict[int, int] = {}
    for i, g in enumerate(groups):
        d = int(g["depth"])
        groups_by_depth[d] = groups_by_depth.get(d, 0) + 1
        selected_by_depth[d] = selected_by_depth.get(d, 0) + int(selected_per_group[i])

    meta = {
        "strategy": "tree_internal_ids_proportional_sampling_v5",
        "budget_n": cfg.budget_n,
        "tree_unique_ids_count": len(tree_ids_unique),
        "candidate_count_in_profile": candidate_count,
        "missing_in_profile_count": missing_in_profile_count,
        "group_count": len(groups),
        "group_count_by_depth": dict(sorted(groups_by_depth.items(), key=lambda kv: kv[0])),
        "selected_count_by_depth": dict(sorted(selected_by_depth.items(), key=lambda kv: kv[0])),
        "group_size_min": int(min(group_sizes)),
        "group_size_mean": float(np.mean(group_sizes)),
        "group_size_max": int(max(group_sizes)),
        "selected_per_group_min": int(min(selected_per_group)),
        "selected_per_group_mean": float(np.mean(selected_per_group)),
        "selected_per_group_max": int(max(selected_per_group)),
        "random_seed": cfg.random_seed,
        "output": {"ours": str(out_ours)},
    }

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cfg.out_dir / "sampling_meta_ours_tree_v3.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("Tree sampling v5 completed (tree-only proportional). Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()
