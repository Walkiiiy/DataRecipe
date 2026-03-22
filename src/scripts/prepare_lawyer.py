#!/usr/bin/env python3
"""将 lawyer_llama_data.jsonl 转为标准 train.jsonl。"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare lawyer dataset to aligned train.jsonl format")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/lawyer/lawyer_llama_data.jsonl"),
        help="输入 JSONL 文件。",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/lawyer/train.jsonl"),
        help="输出 train.jsonl 路径。",
    )
    parser.add_argument(
        "--fallback-category",
        type=str,
        default="lawyer_llama",
        help="当 source 缺失或无效时使用的 category。",
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


def normalize_category(raw_source: Any, fallback_category: str) -> str:
    text = str(raw_source).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return fallback_category
    return text


def normalize_row(row: dict[str, Any], fallback_category: str) -> dict[str, Any]:
    instruction = str(row.get("instruction", "")).strip()
    input_raw = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    input_text = input_raw if input_raw else None
    category = normalize_category(row.get("source"), fallback_category)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": build_alpaca_text(instruction, input_text, output),
        "category": category,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0
    skipped = 0
    category_count: Counter[str] = Counter()
    with args.input_jsonl.open("r", encoding="utf-8") as fin, args.output_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                skipped += 1
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                skipped += 1
                continue
            out = normalize_row(row, args.fallback_category)
            category_count[out["category"]] += 1
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    meta = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "rows_total": total,
        "rows_written": written,
        "rows_skipped": skipped,
        "category_from": "source",
        "fallback_category": args.fallback_category,
        "category_count": dict(category_count),
    }
    meta_path = args.output_jsonl.parent / "prepare_lawyer_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("Lawyer prepared. total=%d, written=%d, skipped=%d", total, written, skipped)
    logging.info("Output saved to %s", args.output_jsonl)
    logging.info("Meta saved to %s", meta_path)
    logging.info("Category count=%d (top10=%s)", len(category_count), category_count.most_common(10))


if __name__ == "__main__":
    main()
