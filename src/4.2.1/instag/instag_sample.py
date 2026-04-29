#!/usr/bin/env python3
"""InSTAG sampling script (decoupled from tag scoring)."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def choose_row_id(row: Dict[str, Any], fallback_idx: int) -> Any:
    for key in ("id", "data_id", "uid", "idx", "index"):
        if key in row and row[key] is not None:
            return row[key]
    return fallback_idx


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


def complexity_first_diverse_sampling(dataset: List[Dict[str, Any]], target_size: int) -> List[Dict[str, Any]]:
    """Algorithm 1: Complexity-first Diverse Sampling.

    Dataset format: [{"id":..., "tags":[...], "raw": {...}}, ...]
    """
    if target_size <= 0:
        return []

    sorted_dataset = sorted(dataset, key=lambda x: len(x.get("tags", [])), reverse=True)
    sampled_data: List[Dict[str, Any]] = []

    while len(sampled_data) < target_size and sorted_dataset:
        current_tag_set = set()
        progressed = False

        for item in list(sorted_dataset):
            item_tags = set(item.get("tags", []))
            if len(current_tag_set.union(item_tags)) > len(current_tag_set):
                sampled_data.append(item)
                current_tag_set.update(item_tags)
                sorted_dataset.remove(item)
                progressed = True
                if len(sampled_data) == target_size:
                    return sampled_data

        if not progressed:
            break

    while len(sampled_data) < target_size and sorted_dataset:
        sampled_data.append(sorted_dataset.pop(0))

    return sampled_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InSTAG sampling script (uses precomputed tag file).")
    parser.add_argument("--data_path", type=str, required=True, help="Original dataset path (JSON or JSONL).")
    parser.add_argument("--tag_path", type=str, required=True, help="InSTAG tag-score file path.")
    parser.add_argument("--output_path", type=str, required=True, help="Sampled output path (JSON or JSONL).")
    parser.add_argument("--num_samples", type=int, required=True, help="Target number of sampled rows.")
    parser.add_argument(
        "--max_pool_samples",
        type=int,
        default=-1,
        help="Optional cap on input pool size before sampling; -1 means use all.",
    )
    parser.add_argument(
        "--attach_tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include instag tag fields in sampled outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_rows, data_fmt = load_json_or_jsonl(args.data_path)
    tag_rows, _ = load_json_or_jsonl(args.tag_path)

    if args.max_pool_samples >= 0:
        data_rows = data_rows[: args.max_pool_samples]

    logger.info("Loaded data rows=%d from %s", len(data_rows), args.data_path)
    logger.info("Loaded tag rows=%d from %s", len(tag_rows), args.tag_path)

    tag_map: Dict[str, List[str]] = {}
    for idx, row in enumerate(tag_rows):
        sid = choose_row_id(row, idx)
        tags_raw = row.get("tags", [])
        tags = [str(t).strip().lower() for t in tags_raw if str(t).strip()]
        tag_map[str(sid)] = list(dict.fromkeys(tags))

    enriched: List[Dict[str, Any]] = []
    missing_tag_cnt = 0
    for idx, row in enumerate(data_rows):
        sid = choose_row_id(row, idx)
        tags = tag_map.get(str(sid), [])
        if not tags:
            missing_tag_cnt += 1
        enriched.append({"id": sid, "tags": tags, "raw": row})

    logger.info("Rows without tags: %d / %d", missing_tag_cnt, len(enriched))

    sampled = complexity_first_diverse_sampling(enriched, target_size=int(args.num_samples))
    logger.info("Sampled %d rows (target=%d)", len(sampled), int(args.num_samples))

    output_rows: List[Dict[str, Any]] = []
    for item in sampled:
        out = dict(item["raw"])
        out["id"] = item["id"]
        if args.attach_tags:
            out["instag_tags"] = item["tags"]
            out["instag_tag_count"] = len(item["tags"])
        output_rows.append(out)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_is_jsonl = data_fmt == "jsonl" or args.output_path.lower().endswith(".jsonl")
    with open(args.output_path, "w", encoding="utf-8") as f:
        if output_is_jsonl:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(output_rows, f, ensure_ascii=False, indent=2)

    logger.info("Saved sampled rows to %s", args.output_path)


if __name__ == "__main__":
    main()
