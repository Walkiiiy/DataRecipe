#!/usr/bin/env python3
"""Download CLINC150 and convert it to aligned train.jsonl format."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import urlopen


DEFAULT_SOURCE_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CLINC150 to aligned train.jsonl format")
    parser.add_argument("--source-url", type=str, default=DEFAULT_SOURCE_URL)
    parser.add_argument("--raw-json", type=Path, default=Path("data/SciQ/clinc150_data_full.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/SciQ/train.jsonl"))
    parser.add_argument(
        "--instruction",
        type=str,
        default="Classify the user utterance into the correct intent category.",
    )
    parser.add_argument(
        "--input-prefix",
        type=str,
        default="you are an intent router. category the following user utterance",
    )
    parser.add_argument(
        "--include-oos-train",
        action="store_true",
        help="Include oos_train split samples in output train.jsonl.",
    )
    parser.add_argument("--force-download", action="store_true")
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


def download_json(url: str, output_path: Path, force: bool = False) -> None:
    if output_path.exists() and not force:
        logging.info("Raw file already exists, skip download: %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Downloading CLINC150 from %s", url)
    with urlopen(url) as resp:
        content = resp.read()
    output_path.write_bytes(content)
    logging.info("Downloaded %d bytes to %s", len(content), output_path)


def load_raw(raw_json: Path) -> dict[str, Any]:
    with raw_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Raw CLINC150 file must be a JSON object.")
    return obj


def build_rows(
    raw_obj: dict[str, Any],
    instruction: str,
    input_prefix: str,
    include_oos_train: bool,
) -> tuple[list[dict[str, Any]], dict[str, int], Counter[str]]:
    splits = ["train"]
    if include_oos_train:
        splits.append("oos_train")

    samples: list[list[str]] = []
    for split in splits:
        part = raw_obj.get(split)
        if part is None:
            raise KeyError(f"Missing split in raw CLINC150 JSON: {split}")
        if not isinstance(part, list):
            raise ValueError(f"Split {split} must be a list.")
        samples.extend(part)

    rows: list[dict[str, Any]] = []
    intent_to_id: dict[str, int] = {}
    intent_counter: Counter[str] = Counter()

    for idx, item in enumerate(samples):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Unexpected sample format at index {idx}: {item!r}")
        utterance_raw, intent_raw = item
        utterance = str(utterance_raw).strip()
        intent = str(intent_raw).strip()
        if not intent:
            raise ValueError(f"Empty intent at sample index {idx}")

        if intent not in intent_to_id:
            intent_to_id[intent] = len(intent_to_id)

        input_text = utterance
        if input_prefix:
            input_text = f"{input_prefix}\n{utterance}"

        output = intent
        text = build_alpaca_text(instruction, input_text, output)
        category = intent_to_id[intent]
        intent_counter[intent] += 1

        rows.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "text": text,
                "category": category,
            }
        )

    return rows, intent_to_id, intent_counter


def write_jsonl(rows: list[dict[str, Any]], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    download_json(args.source_url, args.raw_json, force=args.force_download)
    raw_obj = load_raw(args.raw_json)
    rows, intent_to_id, intent_counter = build_rows(
        raw_obj=raw_obj,
        instruction=args.instruction,
        input_prefix=args.input_prefix,
        include_oos_train=args.include_oos_train,
    )
    write_jsonl(rows, args.output_jsonl)

    meta = {
        "source_url": args.source_url,
        "raw_json": str(args.raw_json),
        "output_jsonl": str(args.output_jsonl),
        "include_oos_train": args.include_oos_train,
        "rows": len(rows),
        "num_intents": len(intent_to_id),
        "intent_to_id": intent_to_id,
        "intent_counts": dict(intent_counter),
        "splits_used": ["train", "oos_train"] if args.include_oos_train else ["train"],
    }
    meta_path = args.output_jsonl.parent / "prepare_clinc150_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("CLINC150 prepared. rows=%d output=%s", len(rows), args.output_jsonl)
    logging.info("Meta saved to %s", meta_path)


if __name__ == "__main__":
    main()
