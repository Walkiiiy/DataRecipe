#!/usr/bin/env python3
"""Randomly sample a subset from train.jsonl."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly sample a subset from train.jsonl")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Input train.jsonl path.")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Output sampled subset jsonl path.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to keep. If set, overrides --sample-ratio.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Ratio in (0, 1]. Used when --sample-size is not set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--keep-original-order",
        action="store_true",
        help="Keep sampled rows in original input order (otherwise shuffled sample order).",
    )
    parser.add_argument(
        "--output-rest-jsonl",
        type=Path,
        default=None,
        help="Optional path to save the non-sampled remainder.",
    )
    parser.add_argument("--meta-json", type=Path, default=None, help="Optional output meta json path.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    if args.sample_size is None and args.sample_ratio is None:
        parser.error("Either --sample-size or --sample-ratio must be provided.")
    if args.sample_size is not None and args.sample_size <= 0:
        parser.error("--sample-size must be > 0.")
    if args.sample_ratio is not None and not (0.0 < args.sample_ratio <= 1.0):
        parser.error("--sample-ratio must be in (0, 1].")
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(row).__name__}.")
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")

    rows = load_jsonl(args.input_jsonl)
    total = len(rows)
    if total == 0:
        raise ValueError("Input jsonl is empty.")

    if args.sample_size is not None:
        target_n = min(total, args.sample_size)
        sampling_mode = "size"
    else:
        target_n = max(1, int(total * args.sample_ratio))
        target_n = min(total, target_n)
        sampling_mode = "ratio"

    rng = random.Random(args.seed)
    all_idx = list(range(total))
    sampled_idx = rng.sample(all_idx, target_n)
    sampled_idx_set = set(sampled_idx)
    if args.keep_original_order:
        sampled_idx = sorted(sampled_idx)

    sampled_rows = [rows[i] for i in sampled_idx]
    rest_rows = [rows[i] for i in all_idx if i not in sampled_idx_set]

    write_jsonl(sampled_rows, args.output_jsonl)
    logging.info("Saved sampled subset to %s (%d/%d)", args.output_jsonl, len(sampled_rows), total)

    if args.output_rest_jsonl is not None:
        write_jsonl(rest_rows, args.output_rest_jsonl)
        logging.info("Saved remainder subset to %s (%d)", args.output_rest_jsonl, len(rest_rows))

    meta_path: Path
    if args.meta_json is not None:
        meta_path = args.meta_json
    else:
        meta_path = args.output_jsonl.with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "output_rest_jsonl": str(args.output_rest_jsonl) if args.output_rest_jsonl else None,
        "rows_total": total,
        "rows_sampled": len(sampled_rows),
        "rows_rest": len(rest_rows),
        "sampling_mode": sampling_mode,
        "sample_size": args.sample_size,
        "sample_ratio": args.sample_ratio,
        "seed": args.seed,
        "keep_original_order": bool(args.keep_original_order),
        "sampled_indices": sampled_idx,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.info("Saved meta to %s", meta_path)


if __name__ == "__main__":
    main()
