"""4.1 EXP - 能力树采样（v2：保留 tree core + 低噪扩展）。

设计目标：
1) 有效簇（size > min_valid_cluster_size）中的 tree 样本全部保留。
2) 如需补充预算，使用 CDT_description 嵌入做“最近簇单分配”扩展：
   - 每个有效簇中心由其 tree core 计算；
   - 半径阈值使用簇内距离分位数（默认 P90），而非 max，降低离群点放大效应；
   - 每个树外样本最多归入一个最近簇，避免多簇重复吸入。
3) 扩展优先做簇间平衡补充，不足时按全局最近继续补齐，确保达到预算。

输出：
- dataset_ours.jsonl
- sampling_meta_ours_tree_v2.json
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
    budget_n: int | None
    radius_quantile: float
    radius_scale: float
    allow_loose_fill: bool
    embedding_model: str
    device: str
    random_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capability-tree sampling v2 (keep core + low-noise expansion)")
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
        help="有效簇最小样本数阈值（严格大于该值）。",
    )
    parser.add_argument(
        "--budget-n",
        type=int,
        default=None,
        help="目标预算 N。默认等于有效簇 tree core 的去重总量（全部保留）。",
    )
    parser.add_argument(
        "--radius-quantile",
        type=float,
        default=0.9,
        help="扩展阈值半径分位数 q（0<q<=1），默认 0.9 对应 P90。",
    )
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="半径缩放系数（>0）。",
    )
    parser.add_argument(
        "--allow-loose-fill",
        action="store_true",
        help="严格阈值候选不足时，允许按全局最近距离继续补齐。",
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
    leaves = iter_leaf_nodes(tree)
    valid: list[list[str]] = []
    for leaf in leaves:
        ids = [str(x) for x in leaf.get("data_ids", [])]
        ids = list(dict.fromkeys(ids))
        if len(ids) > min_valid_cluster_size:
            valid.append(ids)
    return valid


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
        budget_n=args.budget_n,
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
    all_ids = list(rows_by_id.keys())

    valid_clusters = collect_valid_clusters(tree, cfg.min_valid_cluster_size)
    valid_clusters = [[sid for sid in c if sid in rows_by_id] for c in valid_clusters]
    valid_clusters = [c for c in valid_clusters if c]
    if not valid_clusters:
        raise ValueError("No valid clusters found. Please check tree quality or lower min_valid_cluster_size.")

    k = len(valid_clusters)
    core_ids = set(sid for c in valid_clusters for sid in c)
    core_ids_in_order = [rid for rid in row_ids_in_order if rid in core_ids]
    core_n = len(core_ids_in_order)

    budget_n = core_n if cfg.budget_n is None else int(cfg.budget_n)
    if budget_n < core_n:
        raise ValueError(f"budget_n={budget_n} < core_n={core_n}. v2 策略要求保留全部 tree core。")
    if budget_n > len(all_ids):
        raise ValueError(f"Budget N={budget_n} exceeds dataset size={len(all_ids)}.")

    extra_needed = budget_n - core_n
    logging.info(
        "Valid clusters=%d, core_n=%d, budget_n=%d, extra_needed=%d",
        k,
        core_n,
        budget_n,
        extra_needed,
    )

    selected_extra: list[str] = []
    strict_selected_n = 0
    loose_selected_n = 0
    fallback_random_n = 0
    radius_stats: dict[str, float] = {}

    if extra_needed > 0:
        vectors = build_embeddings(rows, cfg.embedding_model, cfg.device)
        id_to_index = {rid: i for i, rid in enumerate(row_ids_in_order)}

        # 每个有效簇：center + quantile radius
        centers: list[np.ndarray] = []
        radii: list[float] = []
        cluster_sizes: list[int] = []
        for c in valid_clusters:
            member_ids = [sid for sid in c if sid in id_to_index]
            member_vecs = vectors[[id_to_index[sid] for sid in member_ids]]
            center = member_vecs.mean(axis=0)
            dist = np.linalg.norm(member_vecs - center, axis=1)
            radius_q = float(np.quantile(dist, cfg.radius_quantile)) if len(dist) > 0 else 0.0
            radius = max(radius_q * cfg.radius_scale, 1e-8)
            centers.append(center.astype(np.float32))
            radii.append(radius)
            cluster_sizes.append(len(member_ids))

        centers_arr = np.stack(centers, axis=0)
        radii_arr = np.asarray(radii, dtype=np.float32)

        radius_stats = {
            "min": float(radii_arr.min()),
            "mean": float(radii_arr.mean()),
            "max": float(radii_arr.max()),
        }

        # 树外样本 -> 最近簇单分配
        core_set = set(core_ids_in_order)
        non_tree_ids = [rid for rid in row_ids_in_order if rid not in core_set]

        strict_candidates_by_cluster: list[list[tuple[float, str]]] = [[] for _ in range(k)]
        global_candidates: list[tuple[float, str, int]] = []  # (normalized_dist, rid, cluster_idx)
        for rid in non_tree_ids:
            vec = vectors[id_to_index[rid]]
            dist = np.linalg.norm(centers_arr - vec, axis=1)
            best_idx = int(dist.argmin())
            best_dist = float(dist[best_idx])
            norm_dist = best_dist / float(radii_arr[best_idx])
            global_candidates.append((norm_dist, rid, best_idx))
            if best_dist <= float(radii_arr[best_idx]):
                strict_candidates_by_cluster[best_idx].append((best_dist, rid))

        for arr in strict_candidates_by_cluster:
            arr.sort(key=lambda x: x[0])
        global_candidates.sort(key=lambda x: x[0])

        # 平衡补充：优先给当前规模较小簇补入最近样本
        selected_set: set[str] = set()
        cursor = [0] * k
        selected_count_by_cluster = [0] * k
        remaining = extra_needed
        while remaining > 0:
            progress = False
            order = sorted(
                range(k),
                key=lambda i: cluster_sizes[i] + selected_count_by_cluster[i],
            )
            for ci in order:
                arr = strict_candidates_by_cluster[ci]
                p = cursor[ci]
                while p < len(arr) and arr[p][1] in selected_set:
                    p += 1
                cursor[ci] = p
                if p >= len(arr):
                    continue
                rid = arr[p][1]
                selected_set.add(rid)
                selected_count_by_cluster[ci] += 1
                cursor[ci] += 1
                remaining -= 1
                progress = True
                if remaining == 0:
                    break
            if not progress:
                break
        strict_selected_n = len(selected_set)

        # 允许放宽时，按全局最近继续补齐
        if remaining > 0 and cfg.allow_loose_fill:
            for _, rid, ci in global_candidates:
                if rid in selected_set:
                    continue
                selected_set.add(rid)
                selected_count_by_cluster[ci] += 1
                remaining -= 1
                if remaining == 0:
                    break
            loose_selected_n = len(selected_set) - strict_selected_n

        # 最终兜底：随机补齐
        if remaining > 0:
            remain_pool = [rid for rid in non_tree_ids if rid not in selected_set]
            if remaining > len(remain_pool):
                raise RuntimeError(
                    f"Cannot fill budget: remaining={remaining}, remain_pool={len(remain_pool)}"
                )
            picks = rng.sample(remain_pool, remaining)
            selected_set.update(picks)
            fallback_random_n = len(picks)
            remaining = 0

        selected_extra = list(selected_set)
        rng.shuffle(selected_extra)

    picked = core_ids_in_order + selected_extra
    if len(picked) != budget_n:
        raise RuntimeError(f"Final picked size mismatch: got={len(picked)} expected={budget_n}")

    # 打散顺序，降低原始文件顺序偏置
    rng.shuffle(picked)

    out_ours = cfg.out_dir / "dataset_ours.jsonl"
    write_subset_jsonl(rows_by_id, picked, out_ours)

    meta = {
        "strategy": "tree_core_keep_all_v2",
        "min_valid_cluster_size": cfg.min_valid_cluster_size,
        "effective_cluster_count": k,
        "core_n": core_n,
        "budget_n": budget_n,
        "extra_needed": extra_needed,
        "radius_quantile": cfg.radius_quantile,
        "radius_scale": cfg.radius_scale,
        "allow_loose_fill": cfg.allow_loose_fill,
        "radius_stats": radius_stats,
        "strict_selected_n": strict_selected_n,
        "loose_selected_n": loose_selected_n,
        "fallback_random_n": fallback_random_n,
        "random_seed": cfg.random_seed,
        "output": {"ours": str(out_ours)},
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cfg.out_dir / "sampling_meta_ours_tree_v2.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("Tree sampling v2 completed. Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()
