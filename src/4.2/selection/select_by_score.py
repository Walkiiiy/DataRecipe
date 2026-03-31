#!/usr/bin/env python3
"""Unified score-based selection for 4.2 methods.

Supports two selection modes:
1) scalar_topk: sort by scalar score and keep top-k
2) vector_greedy: greedy maximize sum_i log(1 + total_i)
   where total_i is cumulative score on capability dimension i.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SCALAR_FALLBACK_FIELDS = [
    "score",
    "delta_scalar",
    "alpagasus_scalar",
    "judge_score",
    "mig_score",
    "datawhisperer_scalar",
]

VECTOR_FALLBACK_FIELDS = [
    "mapped_vector",
    "score",
]


@dataclass(frozen=True)
class Candidate:
    row_index: int
    row_id: Any
    row_id_key: str
    raw_row: Dict[str, Any]
    vector: Dict[int, float]
    scalar: Optional[float]


def choose_row_id(row: Dict[str, Any], fallback_idx: int, preferred_key: str = "id") -> Any:
    if preferred_key and preferred_key in row and row.get(preferred_key) is not None:
        return row.get(preferred_key)
    for key in ("id", "data_id", "uid", "idx", "index"):
        if key in row and row.get(key) is not None:
            return row.get(key)
    return fallback_idx


def to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value).strip()
        if not s:
            return None
        return float(s)
    except Exception:  # noqa: BLE001
        return None


def to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        s = str(value).strip()
        if not s:
            return None
        return int(s)
    except Exception:  # noqa: BLE001
        return None


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
        # NOTE:
        # Do not use splitlines() here. Some datasets may contain U+0085 (NEL) or
        # other Unicode line separators inside JSON string values; splitlines()
        # would incorrectly split one JSON object into multiple fragments.
        rows: List[Dict[str, Any]] = []
        for line_no, raw in enumerate(text.split("\n"), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL object at {path}:{line_no}: expected dict.")
            rows.append(row)
        return rows, "jsonl"


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


def parse_score_vector(value: Any) -> Dict[int, float]:
    """Parse dense/sparse vector into sparse dict(index->positive score)."""
    out: Dict[int, float] = {}
    if isinstance(value, list):
        for idx, v in enumerate(value):
            fv = to_float_or_none(v)
            if fv is None or fv <= 0.0:
                continue
            out[int(idx)] = float(fv)
        return out

    if isinstance(value, dict):
        for k, v in value.items():
            idx = to_int_or_none(k)
            if idx is None or idx < 0:
                continue
            fv = to_float_or_none(v)
            if fv is None or fv <= 0.0:
                continue
            out[int(idx)] = float(fv)
        return out

    return out


def extract_vector(
    row: Dict[str, Any],
    preferred_field: Optional[str],
    allow_fallback: bool,
) -> Dict[int, float]:
    if preferred_field:
        vec = parse_score_vector(row.get(preferred_field))
        if vec:
            return vec
        if not allow_fallback:
            return {}

    for field in VECTOR_FALLBACK_FIELDS:
        if preferred_field and field == preferred_field:
            continue
        vec = parse_score_vector(row.get(field))
        if vec:
            return vec
    return {}


def extract_scalar(
    row: Dict[str, Any],
    preferred_field: Optional[str],
    allow_fallback: bool,
) -> Optional[float]:
    if preferred_field:
        scalar = to_float_or_none(row.get(preferred_field))
        if scalar is not None:
            return scalar
        if not allow_fallback:
            return None

    for field in SCALAR_FALLBACK_FIELDS:
        if preferred_field and field == preferred_field:
            continue
        scalar = to_float_or_none(row.get(field))
        if scalar is not None:
            return scalar
    return None


def build_candidates(
    rows: List[Dict[str, Any]],
    *,
    id_field: str,
    vector_field: Optional[str],
    scalar_field: Optional[str],
    allow_fallback_fields: bool,
) -> List[Candidate]:
    out: List[Candidate] = []
    for idx, row in enumerate(rows):
        row_id = choose_row_id(row, idx, preferred_key=id_field)
        out.append(
            Candidate(
                row_index=idx,
                row_id=row_id,
                row_id_key=str(row_id),
                raw_row=row,
                vector=extract_vector(row, vector_field, allow_fallback_fields),
                scalar=extract_scalar(row, scalar_field, allow_fallback_fields),
            )
        )
    return out


def infer_vector_dimensions(vectors: Iterable[Dict[int, float]]) -> int:
    max_idx = -1
    for vec in vectors:
        for idx in vec.keys():
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0


def select_scalar_topk(candidates: List[Candidate], num_samples: int) -> List[int]:
    ranking = sorted(
        range(len(candidates)),
        key=lambda i: (
            float(candidates[i].scalar) if candidates[i].scalar is not None else float("-inf"),
            -candidates[i].row_index,
        ),
        reverse=True,
    )
    k = min(max(0, int(num_samples)), len(ranking))
    return ranking[:k]


def select_vector_greedy(
    candidates: List[Candidate],
    num_samples: int,
) -> Tuple[List[int], List[float]]:
    """Greedy maximize sum_i log(1 + total_i) over capability dimensions."""
    n = len(candidates)
    k = min(max(0, int(num_samples)), n)
    if k == 0:
        return [], []

    selected: List[int] = []
    selected_mask = [False] * n
    gains: List[float] = []

    current_totals: Dict[int, float] = {}
    vector_sums = [float(sum(c.vector.values())) for c in candidates]
    nonzero_count = sum(1 for c in candidates if c.vector)
    logger.info("Vector-greedy candidates=%d, nonzero_vectors=%d, budget=%d", n, nonzero_count, k)

    for step in range(k):
        best_idx = -1
        best_gain = -1.0
        best_sum = -1.0

        for i, cand in enumerate(candidates):
            if selected_mask[i]:
                continue

            if not cand.vector:
                gain = 0.0
            else:
                gain = 0.0
                for dim, val in cand.vector.items():
                    before = float(current_totals.get(dim, 0.0))
                    gain += math.log1p(before + val) - math.log1p(before)

            v_sum = vector_sums[i]
            if (
                gain > best_gain + 1e-12
                or (abs(gain - best_gain) <= 1e-12 and v_sum > best_sum + 1e-12)
                or (
                    abs(gain - best_gain) <= 1e-12
                    and abs(v_sum - best_sum) <= 1e-12
                    and (best_idx < 0 or cand.row_index < candidates[best_idx].row_index)
                )
            ):
                best_idx = i
                best_gain = gain
                best_sum = v_sum

        if best_idx < 0:
            break

        selected.append(best_idx)
        selected_mask[best_idx] = True
        gains.append(float(best_gain))

        for dim, val in candidates[best_idx].vector.items():
            current_totals[dim] = float(current_totals.get(dim, 0.0) + val)

        if (step + 1) % 100 == 0 or step + 1 == k:
            logger.info(
                "Vector-greedy progress: %d/%d selected, last_gain=%.6f",
                step + 1,
                k,
                float(best_gain),
            )

    return selected, gains


def build_data_map(data_rows: List[Dict[str, Any]], id_field: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for idx, row in enumerate(data_rows):
        rid = str(choose_row_id(row, idx, preferred_key=id_field))
        out[rid] = row
    return out


def materialize_rows(
    candidates: List[Candidate],
    selected_indices: List[int],
    *,
    strategy: str,
    gains: Optional[List[float]],
    data_map: Optional[Dict[str, Dict[str, Any]]],
    annotate_selection: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    missing_in_data_map = 0

    for rank, cand_idx in enumerate(selected_indices):
        cand = candidates[cand_idx]
        base = None
        if data_map is not None:
            base = data_map.get(cand.row_id_key)
            if base is None:
                missing_in_data_map += 1

        row_out = dict(base) if base is not None else dict(cand.raw_row)
        if row_out.get("id") is None:
            row_out["id"] = cand.row_id

        if annotate_selection:
            row_out["selection_rank"] = int(rank)
            row_out["selection_strategy"] = strategy
            if gains is not None and rank < len(gains):
                row_out["selection_gain"] = float(gains[rank])
            if cand.scalar is not None:
                row_out["selection_scalar"] = float(cand.scalar)
            if cand.vector:
                row_out["selection_vector_sum"] = float(sum(cand.vector.values()))

        out.append(row_out)

    if data_map is not None and missing_in_data_map > 0:
        logger.warning(
            "Selected rows missing in data_path by id: %d (fallback to score rows).",
            missing_in_data_map,
        )
    return out


def auto_detect_strategy(candidates: List[Candidate]) -> str:
    has_vector = any(bool(c.vector) for c in candidates)
    if has_vector:
        return "vector_greedy"
    return "scalar_topk"


def summarize_dimension_totals(candidates: List[Candidate], selected_indices: List[int]) -> List[float]:
    m = infer_vector_dimensions(c.vector for c in candidates)
    if m <= 0:
        return []
    totals = [0.0] * m
    for idx in selected_indices:
        for dim, val in candidates[idx].vector.items():
            if 0 <= dim < m:
                totals[dim] += float(val)
    return totals


def run_selection(
    *,
    score_path: str,
    output_path: str,
    num_samples: int,
    strategy: str,
    data_path: Optional[str] = None,
    id_field: str = "id",
    vector_field: Optional[str] = None,
    scalar_field: Optional[str] = None,
    allow_fallback_fields: bool = True,
    annotate_selection: bool = False,
    meta_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    if int(num_samples) <= 0:
        raise ValueError("--num_samples must be > 0.")

    score_rows, score_fmt = load_json_or_jsonl(score_path)
    if not score_rows:
        raise ValueError(f"No rows found in score_path: {score_path}")

    candidates = build_candidates(
        score_rows,
        id_field=id_field,
        vector_field=vector_field,
        scalar_field=scalar_field,
        allow_fallback_fields=allow_fallback_fields,
    )

    chosen_strategy = strategy
    if strategy == "auto":
        chosen_strategy = auto_detect_strategy(candidates)
        logger.info("Auto-detected strategy: %s", chosen_strategy)

    if chosen_strategy not in {"vector_greedy", "scalar_topk"}:
        raise ValueError(f"Unsupported strategy: {chosen_strategy}")

    if chosen_strategy == "vector_greedy":
        valid_vec = sum(1 for c in candidates if c.vector)
        if valid_vec == 0:
            raise ValueError(
                "No valid vector score found for vector_greedy. "
                "Check --vector_field or the score file format."
            )
        if valid_vec < len(candidates):
            logger.warning("Rows with empty vector score: %d / %d", len(candidates) - valid_vec, len(candidates))
        selected_indices, gains = select_vector_greedy(candidates, int(num_samples))
    else:
        valid_scalar = sum(1 for c in candidates if c.scalar is not None)
        if valid_scalar == 0:
            raise ValueError(
                "No valid scalar score found for scalar_topk. "
                "Check --scalar_field or the score file format."
            )
        if valid_scalar < len(candidates):
            logger.warning("Rows with empty scalar score: %d / %d", len(candidates) - valid_scalar, len(candidates))
        selected_indices = select_scalar_topk(candidates, int(num_samples))
        gains = None

    data_map: Optional[Dict[str, Dict[str, Any]]] = None
    if data_path:
        data_rows, _ = load_json_or_jsonl(data_path)
        data_map = build_data_map(data_rows, id_field=id_field)
        logger.info("Loaded original data rows=%d from %s", len(data_rows), data_path)

    selected_rows = materialize_rows(
        candidates,
        selected_indices,
        strategy=chosen_strategy,
        gains=gains,
        data_map=data_map,
        annotate_selection=annotate_selection,
    )

    write_json_or_jsonl(output_path, selected_rows, prefer_jsonl=(score_fmt == "jsonl"))
    logger.info("Saved selected rows=%d to %s", len(selected_rows), output_path)

    dim_totals = summarize_dimension_totals(candidates, selected_indices)
    meta = {
        "score_path": score_path,
        "data_path": data_path,
        "output_path": output_path,
        "strategy": chosen_strategy,
        "requested_num_samples": int(num_samples),
        "selected_num_samples": len(selected_rows),
        "num_score_rows": len(score_rows),
        "dimension_count": len(dim_totals),
        "dimension_totals": dim_totals,
        "vector_field": vector_field,
        "scalar_field": scalar_field,
    }

    if meta_output_path:
        out_dir = os.path.dirname(meta_output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(meta_output_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info("Saved selection meta to %s", meta_output_path)

    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select dataset by score rows (scalar top-k or vector greedy).")
    parser.add_argument("--score_path", type=str, required=True, help="Score file path (JSON or JSONL).")
    parser.add_argument("--data_path", type=str, default=None, help="Optional raw dataset path for id-based row restore.")
    parser.add_argument("--output_path", type=str, required=True, help="Selected dataset output path.")
    parser.add_argument("--meta_output_path", type=str, default=None, help="Optional meta summary output path.")
    parser.add_argument("--num_samples", type=int, required=True, help="Data budget (number of selected rows).")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["auto", "vector_greedy", "scalar_topk"],
        default="auto",
        help="Selection strategy.",
    )
    parser.add_argument("--id_field", type=str, default="id", help="Preferred row id field.")
    parser.add_argument("--vector_field", type=str, default=None, help="Preferred vector field name.")
    parser.add_argument("--scalar_field", type=str, default=None, help="Preferred scalar field name.")
    parser.add_argument(
        "--allow_fallback_fields",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to fallback to known score fields if preferred field is missing.",
    )
    parser.add_argument(
        "--annotate_selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to add selection metadata fields into each output row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_selection(
        score_path=args.score_path,
        data_path=args.data_path,
        output_path=args.output_path,
        meta_output_path=args.meta_output_path,
        num_samples=int(args.num_samples),
        strategy=args.strategy,
        id_field=args.id_field,
        vector_field=args.vector_field,
        scalar_field=args.scalar_field,
        allow_fallback_fields=bool(args.allow_fallback_fields),
        annotate_selection=bool(args.annotate_selection),
    )


if __name__ == "__main__":
    main()
