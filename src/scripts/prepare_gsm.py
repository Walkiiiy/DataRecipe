#!/usr/bin/env python3
"""将 GSM parquet 转为标准 train.jsonl（dataset -> catelog）。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GSM to aligned train.jsonl format")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("data/gsm/data/train-00000-of-00001.parquet"),
        help="输入 parquet 路径。",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/gsm/train.jsonl"),
        help="输出 train.jsonl 路径。",
    )
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
    messages = row.get("messages", []) or []
    user_content = ""
    for m in messages:
        if isinstance(m, dict) and str(m.get("role", "")).strip().lower() == "user":
            user_content = str(m.get("content", "")).strip()
            break
    if not user_content and messages:
        first = messages[0]
        if isinstance(first, dict):
            user_content = str(first.get("content", "")).strip()

    instruction = user_content
    input_text = None
    output = str(row.get("ground_truth", "")).strip()
    catelog = str(row.get("dataset", "")).strip() or "__MISSING_DATASET__"

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": build_alpaca_text(instruction, input_text, output),
        "catelog": catelog,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input_parquet}")

    table = pq.read_table(args.input_parquet)
    rows = table.to_pylist()
    if not rows:
        raise ValueError("Input parquet is empty.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    catelog_count: dict[str, int] = {}
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            out = normalize_row(row)
            cat = out["catelog"]
            catelog_count[cat] = catelog_count.get(cat, 0) + 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    meta = {
        "input_parquet": str(args.input_parquet),
        "output_jsonl": str(args.output_jsonl),
        "rows": len(rows),
        "catelog_count": catelog_count,
    }
    meta_path = args.output_jsonl.parent / "prepare_gsm_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("GSM prepared. rows=%d, output=%s", len(rows), args.output_jsonl)
    logging.info("Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()
