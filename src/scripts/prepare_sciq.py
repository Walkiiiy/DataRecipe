#!/usr/bin/env python3
"""Download allenai/sciq and convert it to aligned train.jsonl format."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import urlopen


ROWS_API = "https://datasets-server.huggingface.co/rows"
SIZE_API = "https://datasets-server.huggingface.co/size"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SciQ to aligned train.jsonl format")
    parser.add_argument("--dataset", type=str, default="allenai/sciq")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--inter-page-sleep", type=float, default=0.2)
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/sciq/train.jsonl"))
    parser.add_argument("--meta-json", type=Path, default=Path("data/sciq/prepare_sciq_meta.json"))
    parser.add_argument(
        "--instruction",
        type=str,
        default="Answer the science question using the provided context.",
    )
    parser.add_argument(
        "--input-prefix",
        type=str,
        default="You are a science QA assistant. Read the context and answer the question.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--shuffle-seed", type=int, default=20260327)
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


def fetch_json(url: str, timeout: int, max_retries: int, retry_sleep: float) -> Any:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with urlopen(url, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt == max_retries:
                break
            sleep_s = retry_sleep * attempt
            if isinstance(err, HTTPError) and err.code == 429:
                retry_after = err.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = max(float(retry_after), sleep_s)
                    except ValueError:
                        sleep_s = max(5.0, sleep_s)
                else:
                    sleep_s = max(5.0, sleep_s)
            logging.warning(
                "Fetch failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt,
                max_retries,
                err,
                sleep_s,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch JSON from {url}") from last_err


def get_split_size(
    dataset: str,
    config: str,
    split: str,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> int:
    qs = urlencode({"dataset": dataset})
    url = f"{SIZE_API}?{qs}"
    payload = fetch_json(url, timeout=timeout, max_retries=max_retries, retry_sleep=retry_sleep)

    for item in payload.get("size", {}).get("splits", []):
        if item.get("config") == config and item.get("split") == split:
            return int(item["num_rows"])
    raise ValueError(f"Split not found in size API response: dataset={dataset}, config={config}, split={split}")


def fetch_rows_page(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> list[dict[str, Any]]:
    qs = urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{ROWS_API}?{qs}"
    payload = fetch_json(url, timeout=timeout, max_retries=max_retries, retry_sleep=retry_sleep)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected rows payload at offset={offset}: {type(rows)!r}")
    return rows


def normalize_sample(
    sample: dict[str, Any],
    instruction: str,
    input_prefix: str,
    rng: random.Random,
) -> dict[str, Any]:
    question = str(sample.get("question", "")).strip()
    support = str(sample.get("support", "")).strip()
    distractor1 = str(sample.get("distractor1", "")).strip()
    distractor2 = str(sample.get("distractor2", "")).strip()
    distractor3 = str(sample.get("distractor3", "")).strip()
    correct_answer = str(sample.get("correct_answer", "")).strip()

    options = [
        (distractor1, False),
        (distractor2, False),
        (distractor3, False),
        (correct_answer, True),
    ]
    rng.shuffle(options)

    labels = ["A", "B", "C", "D"]
    option_lines: list[str] = []
    correct_label = None
    for i, (option_text, is_correct) in enumerate(options):
        label = labels[i]
        option_lines.append(f"{label}. {option_text}")
        if is_correct:
            correct_label = label
    if correct_label is None:
        raise ValueError("Failed to locate correct option after shuffling.")

    input_text = (
        f"{input_prefix}\n"
        f"Context:\n{support}\n\n"
        f"Question:\n{question}\n\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\n\nReturn only the correct answer text."
    )
    output = correct_answer
    text = build_alpaca_text(instruction, input_text, output)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": text,
        "category": correct_label,
    }


def append_jsonl(rows: list[dict[str, Any]], output_jsonl: Path) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.page_size <= 0:
        raise ValueError("--page-size must be > 0")

    total = get_split_size(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    logging.info("Split size: %s/%s/%s -> %d rows", args.dataset, args.config, args.split, total)
    shuffle_rng = random.Random(args.shuffle_seed)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists():
        with args.output_jsonl.open("r", encoding="utf-8") as f:
            written = sum(1 for _ in f)
        if written > total:
            raise ValueError(
                f"Existing output has more rows than source split: existing={written}, total={total}"
            )
        if written == total:
            logging.info("Output already complete, skip downloading: %s", args.output_jsonl)
    else:
        args.output_jsonl.write_text("", encoding="utf-8")
        written = 0

    for offset in range(written, total, args.page_size):
        page_len = min(args.page_size, total - offset)
        rows = fetch_rows_page(
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            offset=offset,
            length=page_len,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        page_rows: list[dict[str, Any]] = []
        for row in rows:
            sample = row.get("row")
            if not isinstance(sample, dict):
                raise ValueError(f"Unexpected row format at offset={offset}: {row!r}")
            page_rows.append(
                normalize_sample(
                    sample=sample,
                    instruction=args.instruction,
                    input_prefix=args.input_prefix,
                    rng=shuffle_rng,
                )
            )
        append_jsonl(page_rows, args.output_jsonl)
        written += len(page_rows)
        logging.info("Downloaded and converted rows: %d/%d", written, total)
        if args.inter_page_sleep > 0:
            time.sleep(args.inter_page_sleep)

    meta = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "rows": written,
        "page_size": args.page_size,
        "output_jsonl": str(args.output_jsonl),
        "instruction": args.instruction,
        "input_prefix": args.input_prefix,
        "support_in_input": True,
        "inter_page_sleep": args.inter_page_sleep,
        "shuffle_seed": args.shuffle_seed,
        "category_definition": "Correct option letter after shuffling options",
    }
    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    with args.meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logging.info("SciQ prepared. rows=%d output=%s", written, args.output_jsonl)
    logging.info("Meta saved to %s", args.meta_json)


if __name__ == "__main__":
    main()
