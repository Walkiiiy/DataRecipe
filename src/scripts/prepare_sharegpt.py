#!/usr/bin/env python3
"""将 ShareGPT JSON 转为标准 train.jsonl。"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ShareGPT to standard jsonl format")
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"),
        help="输入 ShareGPT JSON 路径。",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/sharegpt/train.jsonl"),
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


def _turn_text(turn: dict[str, Any]) -> str:
    for key in ("value", "text"):
        value = turn.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def normalize_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    conv = row.get("conversations")
    if not isinstance(conv, list) or not conv:
        return None, "invalid_conversations"

    system_contexts: list[str] = []
    first_user_idx: int | None = None
    instruction = ""

    for idx, turn in enumerate(conv):
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip().lower()
        text = _turn_text(turn)
        if role == "system" and text:
            system_contexts.append(text)
            continue
        if role in {"human", "user"} and text:
            first_user_idx = idx
            instruction = text
            break

    if first_user_idx is None or not instruction:
        return None, "no_user_instruction"

    output = ""
    assistant_role = ""
    for turn in conv[first_user_idx + 1 :]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip().lower()
        if role in {"human", "user", "system"}:
            continue
        text = _turn_text(turn)
        if not text:
            continue
        assistant_role = role if role else "gpt"
        output = text
        break

    if not output:
        return None, "no_assistant_output"

    input_text = "\n\n".join(system_contexts) if system_contexts else None
    category = assistant_role or "gpt"
    sample_id = str(row.get("id", "")).strip()

    out: dict[str, Any] = {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": build_alpaca_text(instruction, input_text, output),
        "category": category,
    }
    if sample_id:
        out["id"] = sample_id
    return out, None


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    with args.input_json.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input JSON must be a non-empty list.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    category_count: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            if not isinstance(row, dict):
                skip_reasons["invalid_row_type"] += 1
                continue
            out, reason = normalize_row(row)
            if out is None:
                skip_reasons[reason or "unknown"] += 1
                continue
            category = str(out.get("category", "")).strip() or "__MISSING_CATEGORY__"
            out["category"] = category
            category_count[category] += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

    meta = {
        "input_json": str(args.input_json),
        "output_jsonl": str(args.output_jsonl),
        "input_rows": len(rows),
        "output_rows": written,
        "category_from": "first_assistant_from",
        "category_count": dict(category_count),
        "skip_reasons": dict(skip_reasons),
    }
    meta_path = args.output_jsonl.parent / "prepare_sharegpt_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("ShareGPT prepared. input=%d, output=%d, output_path=%s", len(rows), written, args.output_jsonl)
    logging.info("Category count=%d (top10=%s)", len(category_count), category_count.most_common(10))
    if skip_reasons:
        logging.info("Skipped rows=%s", dict(skip_reasons))
    logging.info("Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()
