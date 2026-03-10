"""Download full alpaca-gpt4-data-en split and save to local JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelscope.msdatasets import MsDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="alpaca-gpt4-data-en",
        help="ModelScope dataset name.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="AI-ModelScope",
        help="ModelScope dataset namespace.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/alpaca-gpt4-data-en/train.jsonl",
        help="Output JSONL file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = MsDataset.load(
        args.dataset_name,
        namespace=args.namespace,
        split=args.split,
    )

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Downloaded {count} samples to {output_path.resolve()}")


if __name__ == "__main__":
    main()
