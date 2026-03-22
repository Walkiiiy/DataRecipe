#!/usr/bin/env python3
"""将 Evol-Instruct JSONL 转为标准 train.jsonl。"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare evolinstruct to aligned train.jsonl format")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/evolinstruct/WizardLM_evol_instruct_V2_143k.jsonl"),
        help="输入 JSONL 文件路径。",
    )
    parser.add_argument(
        "--output-full-jsonl",
        type=Path,
        default=Path("data/evolinstruct/train_full.jsonl"),
        help="输出全量 train_full.jsonl 路径。",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/evolinstruct/train.jsonl"),
        help="输出采样 train.jsonl 路径。",
    )
    parser.add_argument("--sample-size", type=int, default=4000, help="当全量大于该值时随机抽样数量。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--category",
        type=str,
        default="evolinstruct",
        help="category 固定值（该数据集无原生类别字段）。",
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


def _turn_text(turn: dict[str, Any]) -> str:
    value = turn.get("value")
    if value is None:
        return ""
    return str(value).strip()


def normalize_row(row: dict[str, Any], category: str) -> dict[str, Any] | None:
    conv = row.get("conversations")
    if not isinstance(conv, list) or not conv:
        return None

    user_idx: int | None = None
    instruction = ""
    for i, turn in enumerate(conv):
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip().lower()
        text = _turn_text(turn)
        if role in {"human", "user"} and text:
            user_idx = i
            instruction = text
            break
    if user_idx is None or not instruction:
        return None

    output = ""
    for turn in conv[user_idx + 1 :]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip().lower()
        if role in {"human", "user", "system"}:
            continue
        text = _turn_text(turn)
        if text:
            output = text
            break
    if not output:
        return None

    sample_id = str(row.get("idx", "")).strip()
    out = {
        "instruction": instruction,
        "input": None,
        "output": output,
        "text": build_alpaca_text(instruction, None, output),
        "category": category,
    }
    if sample_id:
        out["id"] = sample_id
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input_jsonl}")

    args.output_full_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    valid = 0
    lines: list[str] = []
    with args.input_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            out = normalize_row(row, args.category)
            if out is None:
                continue
            lines.append(json.dumps(out, ensure_ascii=False))
            valid += 1

    with args.output_full_jsonl.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    sampled_n = valid
    if valid > args.sample_size:
        random.seed(args.seed)
        picked = sorted(random.sample(range(valid), args.sample_size))
        sampled_n = args.sample_size
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for i in picked:
                f.write(lines[i] + "\n")
    else:
        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    meta = {
        "input_jsonl": str(args.input_jsonl),
        "output_full_jsonl": str(args.output_full_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "rows_total": total,
        "rows_valid": valid,
        "sample_size": args.sample_size,
        "rows_output": sampled_n,
        "seed": args.seed,
        "category_from": "fixed_dataset_name",
        "category_value": args.category,
    }
    meta_path = args.output_jsonl.parent / "prepare_evolinstruct_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("EvolInstruct prepared. total=%d, valid=%d, sampled=%d", total, valid, sampled_n)
    logging.info("Saved full to %s", args.output_full_jsonl)
    logging.info("Saved sampled to %s", args.output_jsonl)
    logging.info("Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()
