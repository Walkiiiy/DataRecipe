#!/usr/bin/env python3
"""将 MMLU parquet 转为 Dolly 风格 train.jsonl。

输出字段：
- instruction
- input
- output
- text
- category
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MMLU parquet to Dolly-like train.jsonl")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=Path("data/mmlu/all/auxiliary_train-00000-of-00001.parquet"),
        help="输入 MMLU parquet 文件。",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/mmlu/train.jsonl"),
        help="输出 train.jsonl 路径。",
    )
    parser.add_argument(
        "--category-field",
        type=str,
        default="subject",
        help="用于 category 的字段名（通常是 subject）。",
    )
    parser.add_argument(
        "--fallback-category",
        type=str,
        default="mmlu_auxiliary_train",
        help="当 category-field 为空时使用的默认 category。",
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


def format_options(choices: list[str]) -> str:
    labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    lines: list[str] = []
    for i, ch in enumerate(choices):
        label = labels[i] if i < len(labels) else f"Option{i+1}"
        lines.append(f"{label}. {ch}")
    return "\n".join(lines)


def convert_row(row: dict[str, Any], category_field: str, fallback_category: str) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    choices = row.get("choices", []) or []
    answer_idx = row.get("answer", None)

    if not isinstance(choices, list):
        choices = list(choices)
    choices = [str(x) for x in choices]

    if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
        label = chr(ord("A") + answer_idx) if answer_idx < 26 else str(answer_idx)
        output = f"{label}. {choices[answer_idx]}"
    else:
        output = ""

    instruction = question
    input_text = format_options(choices) if choices else None

    category = str(row.get(category_field, "")).strip()
    if not category:
        category = fallback_category

    return {
        "instruction": instruction,
        "input": input_text if input_text else None,
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

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input_parquet}")

    table = pq.read_table(args.input_parquet)
    rows = table.to_pylist()
    if not rows:
        raise ValueError("Input parquet is empty.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    category_count: dict[str, int] = {}
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            out = convert_row(row, args.category_field, args.fallback_category)
            c = out["category"]
            category_count[c] = category_count.get(c, 0) + 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    logging.info("Converted rows=%d -> %s", len(rows), args.output_jsonl)
    logging.info("Category count=%d (top10=%s)", len(category_count), sorted(category_count.items())[:10])


if __name__ == "__main__":
    main()
