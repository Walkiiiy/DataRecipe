#!/usr/bin/env python3
"""将 DialogSum 的 train/validation/test 合并并转换为标准 train.jsonl。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DialogSum to standard jsonl format")
    parser.add_argument("--input-dir", type=Path, default=Path("data/dialogsum"))
    parser.add_argument(
        "--splits",
        type=str,
        default="train,validation,test",
        help="逗号分隔的 CSV split 名称，不带 .csv 后缀。",
    )
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/dialogsum/train.jsonl"))
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def build_alpaca_text(instruction: str, input_text: str | None, output: str) -> str:
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes "
            "the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{output}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{output}"
    )


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    dialogue = str(row.get("dialogue", "")).strip()
    summary = str(row.get("summary", "")).strip()
    topic = str(row.get("topic", "")).strip() or "__MISSING_TOPIC__"

    instruction = "Summarize the following dialogue."
    input_text = dialogue if dialogue else None
    output = summary
    text = build_alpaca_text(instruction, input_text, output)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": text,
        "category": topic,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    if not splits:
        raise ValueError("No valid splits provided.")

    merged_rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split in splits:
        csv_path = args.input_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_path}")
        count = 0
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merged_rows.append(normalize_row(row))
                count += 1
        split_counts[split] = count

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "input_dir": str(args.input_dir),
        "splits": splits,
        "split_counts": split_counts,
        "total_rows": len(merged_rows),
        "output_jsonl": str(args.output_jsonl),
        "category_from": "topic",
    }
    meta_path = args.output_jsonl.parent / "prepare_dialogsum_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("DialogSum prepared. rows=%d, output=%s", len(merged_rows), args.output_jsonl)
    logging.info("Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()
