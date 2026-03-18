#!/usr/bin/env python3
"""下载并预处理 Dolly-15k 为与 Alpaca 兼容的 JSONL 格式。

输出字段：
- instruction
- input
- output
- text
- category

其中 category 为 Dolly 自带的技能/任务类别字段，原样保留。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import requests


DEFAULT_DOLLY_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess Dolly-15k dataset")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_DOLLY_URL,
        help="Dolly-15k 原始 JSONL 下载地址。",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/dolly-15k"),
        help="输出目录。",
    )
    parser.add_argument(
        "--raw-name",
        type=str,
        default="databricks-dolly-15k.raw.jsonl",
        help="下载的原始文件名。",
    )
    parser.add_argument(
        "--train-name",
        type=str,
        default="train.jsonl",
        help="预处理后的训练集文件名。",
    )
    parser.add_argument("--timeout", type=int, default=60, help="单次请求超时时间（秒）。")
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


def download_file(url: str, output_path: Path, timeout: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def normalize_record(row: dict[str, Any]) -> dict[str, Any]:
    instruction = str(row.get("instruction", "")).strip()
    input_text = row.get("context")
    if input_text is not None:
        input_text = str(input_text).strip()
        if input_text == "":
            input_text = None
    output = str(row.get("response", "")).strip()
    category = str(row.get("category", "")).strip()

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": build_alpaca_text(instruction, input_text, output),
        "category": category,
    }


def preprocess_jsonl(raw_path: Path, train_path: Path) -> dict[str, Any]:
    count = 0
    categories: dict[str, int] = {}

    with raw_path.open("r", encoding="utf-8") as fin, train_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            normalized = normalize_record(row)
            category = normalized["category"]
            categories[category] = categories.get(category, 0) + 1
            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    summary = {
        "num_samples": count,
        "num_categories": len(categories),
        "categories": dict(sorted(categories.items())),
        "raw_path": str(raw_path),
        "train_path": str(train_path),
    }
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out_dir = args.out_dir
    raw_path = out_dir / args.raw_name
    train_path = out_dir / args.train_name
    summary_path = out_dir / "dataset_meta.json"

    logging.info("Downloading Dolly-15k from %s", args.url)
    download_file(args.url, raw_path, args.timeout)
    logging.info("Downloaded raw file to %s", raw_path)

    summary = preprocess_jsonl(raw_path, train_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Saved normalized train set to %s", train_path)
    logging.info("Saved dataset metadata to %s", summary_path)
    logging.info("Samples=%d Categories=%d", summary["num_samples"], summary["num_categories"])


if __name__ == "__main__":
    main()
