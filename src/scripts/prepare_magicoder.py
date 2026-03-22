#!/usr/bin/env python3
"""将 Magicoder-Evol-Instruct-Clean 对齐为 train_full/train。"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any


LANG_PATTERN = re.compile(r"```[\t ]*([a-zA-Z0-9_+#.\-]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare magicoder to aligned train jsonl")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/magicoder/Magicoder-Evol-Instruct-Clean.json"),
        help="输入文件路径（虽然扩展名是 .json，但内容为 JSONL）。",
    )
    parser.add_argument("--output-full-jsonl", type=Path, default=Path("data/magicoder/train_full.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/magicoder/train.jsonl"))
    parser.add_argument("--sample-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fallback-category", type=str, default="magicoder_unknown")
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


def normalize_lang(raw_lang: str) -> str:
    lang = raw_lang.strip().lower()
    alias = {
        "py": "python",
        "python3": "python",
        "js": "javascript",
        "ts": "typescript",
        "c++": "cpp",
        "cs": "csharp",
        "sh": "bash",
    }
    return alias.get(lang, lang)


def pick_category(target: str, fallback_category: str) -> str:
    m = LANG_PATTERN.search(target)
    if not m:
        return fallback_category
    lang = normalize_lang(m.group(1))
    if not lang:
        return fallback_category
    return f"code_{lang}"


def normalize_row(row: dict[str, Any], fallback_category: str) -> dict[str, Any] | None:
    instruction = str(row.get("input", "")).strip()
    output = str(row.get("target", "")).strip()
    if not instruction or not output:
        return None

    category = pick_category(output, fallback_category)
    out = {
        "instruction": instruction,
        "input": None,
        "output": output,
        "text": build_alpaca_text(instruction, None, output),
        "category": category,
    }
    sim = row.get("max_similarity")
    if sim is not None:
        out["max_similarity"] = sim
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_jsonl}")

    args.output_full_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    total = 0
    valid = 0
    with args.input_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            out = normalize_row(row, args.fallback_category)
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
        "rows_output": sampled_n,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "category_from": "first_code_fence_language_in_target",
        "fallback_category": args.fallback_category,
    }
    meta_path = args.output_jsonl.parent / "prepare_magicoder_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("Magicoder prepared. total=%d, valid=%d, sampled=%d", total, valid, sampled_n)
    logging.info("Saved full to %s", args.output_full_jsonl)
    logging.info("Saved sampled to %s", args.output_jsonl)
    logging.info("Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()
