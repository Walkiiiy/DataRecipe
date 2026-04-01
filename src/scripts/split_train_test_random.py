#!/usr/bin/env python3
"""Split train.jsonl into train+test by random sampling and overwrite train."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample a portion from train.jsonl to test.jsonl and remove it from train.jsonl."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True, help="Input train.jsonl path to be updated in-place.")
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=None,
        help="Output test.jsonl path. Default: sibling test.jsonl next to train-jsonl.",
    )
    parser.add_argument("--ratio", type=float, default=0.1, help="Test split ratio in (0,1). Default: 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--backup-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to backup original train.jsonl before overwrite.",
    )
    parser.add_argument("--backup-suffix", type=str, default=".bak", help="Backup suffix for original train file.")
    parser.add_argument(
        "--keep-original-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep rows in original order for both updated train and test outputs.",
    )
    parser.add_argument("--meta-json", type=Path, default=None, help="Optional output meta json path.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    if not (0.0 < float(args.ratio) < 1.0):
        parser.error("--ratio must be in (0, 1).")
    if not str(args.backup_suffix):
        parser.error("--backup-suffix must be non-empty.")
    return args


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}, got {type(row).__name__}.")
            rows.append(row)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    train_path = args.train_jsonl
    if not train_path.exists():
        raise FileNotFoundError(f"train jsonl not found: {train_path}")

    test_path = args.test_jsonl if args.test_jsonl is not None else train_path.with_name("test.jsonl")
    rows = load_jsonl(train_path)
    total = len(rows)
    if total < 2:
        raise ValueError(f"Need at least 2 rows to split, got {total}.")

    test_n = max(1, int(total * float(args.ratio)))
    test_n = min(test_n, total - 1)

    rng = random.Random(int(args.seed))
    all_idx = list(range(total))
    test_idx = rng.sample(all_idx, test_n)
    test_idx_set = set(test_idx)

    if bool(args.keep_original_order):
        test_idx = sorted(test_idx)

    test_rows = [rows[i] for i in test_idx]
    train_rows = [rows[i] for i in all_idx if i not in test_idx_set]

    if bool(args.backup_train):
        backup_path = train_path.with_name(train_path.name + str(args.backup_suffix))
        shutil.copy2(train_path, backup_path)
        logging.info("Backed up original train file to %s", backup_path)

    write_jsonl(test_rows, test_path)
    write_jsonl(train_rows, train_path)

    logging.info("Split done: total=%d, test=%d, train=%d", total, len(test_rows), len(train_rows))
    logging.info("Updated train: %s", train_path)
    logging.info("Saved test: %s", test_path)

    if args.meta_json is not None:
        meta_path = args.meta_json
    else:
        meta_path = test_path.with_suffix(".split_meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "train_jsonl": str(train_path),
        "test_jsonl": str(test_path),
        "rows_total_before": total,
        "rows_test": len(test_rows),
        "rows_train_after": len(train_rows),
        "ratio": float(args.ratio),
        "seed": int(args.seed),
        "keep_original_order": bool(args.keep_original_order),
        "sampled_test_indices": test_idx,
        "backup_train": bool(args.backup_train),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logging.info("Saved split meta: %s", meta_path)


if __name__ == "__main__":
    main()

