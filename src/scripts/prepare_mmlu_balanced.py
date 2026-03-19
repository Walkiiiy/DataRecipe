#!/usr/bin/env python3
"""将 MMLU 按 category 均衡采样后转为 Dolly 风格 train.jsonl。"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced MMLU train.jsonl by category")
    parser.add_argument("--input-root", type=Path, default=Path("data/mmlu"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/mmlu/train.jsonl"))
    parser.add_argument(
        "--splits",
        type=str,
        default="dev,validation,test",
        help="逗号分隔，例如 dev,validation,test",
    )
    parser.add_argument("--category-field", type=str, default="subject")
    parser.add_argument("--per-category", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
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
    out: list[str] = []
    for i, ch in enumerate(choices):
        label = labels[i] if i < len(labels) else f"Option{i+1}"
        out.append(f"{label}. {ch}")
    return "\n".join(out)


def convert_row(row: dict[str, Any], category_field: str, fallback_category: str) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    choices = row.get("choices", []) or []
    if not isinstance(choices, list):
        choices = list(choices)
    choices = [str(x) for x in choices]

    answer_idx = row.get("answer", None)
    output = ""
    if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
        label = chr(ord("A") + answer_idx) if answer_idx < 26 else str(answer_idx)
        output = f"{label}. {choices[answer_idx]}"

    input_text = format_options(choices) if choices else None
    category = str(row.get(category_field, "")).strip() or fallback_category

    return {
        "instruction": question,
        "input": input_text if input_text else None,
        "output": output,
        "text": build_alpaca_text(question, input_text, output),
        "category": category,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    rng = random.Random(args.random_seed)

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    if not splits:
        raise ValueError("No valid splits specified.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned_files = 0
    scanned_rows = 0

    for subj_dir in sorted(args.input_root.iterdir()):
        if not subj_dir.is_dir() or subj_dir.name in {"all", "auxiliary_train"}:
            continue
        fallback_category = subj_dir.name
        for split in splits:
            fp = subj_dir / f"{split}-00000-of-00001.parquet"
            if not fp.exists():
                continue
            table = pq.read_table(fp)
            rows = table.to_pylist()
            scanned_files += 1
            scanned_rows += len(rows)
            for row in rows:
                out = convert_row(row, args.category_field, fallback_category)
                grouped[out["category"]].append(out)

    if not grouped:
        raise ValueError("No MMLU rows loaded. Check input root and splits.")

    picked: list[dict[str, Any]] = []
    sampled_sizes: dict[str, int] = {}
    source_sizes: dict[str, int] = {}
    for cat, rows in sorted(grouped.items(), key=lambda x: x[0]):
        source_sizes[cat] = len(rows)
        take = min(args.per_category, len(rows))
        sampled_sizes[cat] = take
        if take > 0:
            picked.extend(rng.sample(rows, take))

    rng.shuffle(picked)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "input_root": str(args.input_root),
        "splits": splits,
        "category_field": args.category_field,
        "per_category": args.per_category,
        "random_seed": args.random_seed,
        "scanned_files": scanned_files,
        "scanned_rows": scanned_rows,
        "category_count": len(grouped),
        "source_sizes": source_sizes,
        "sampled_sizes": sampled_sizes,
        "output_jsonl": str(args.output_jsonl),
        "output_rows": len(picked),
    }
    meta_path = args.output_jsonl.parent / "prepare_mmlu_balanced_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("Balanced MMLU built: rows=%d, categories=%d", len(picked), len(grouped))
    logging.info("Saved train: %s", args.output_jsonl)
    logging.info("Saved meta: %s", meta_path)


if __name__ == "__main__":
    main()
