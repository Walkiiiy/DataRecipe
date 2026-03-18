#!/usr/bin/env python3
"""将本地 S1-reasoning parquet 分片转换为 Alpaca 风格 JSONL。

输入数据：
- data/data_ablation_full59K/data/train-*.parquet

输出字段：
- instruction
- input
- output
- text

并保留原始字段：
- question
- solution
- cot_type
- source_type
- metadata
- cot
- thinking_trajectories
- attempt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except Exception as exc:  # noqa: BLE001
    pq = None
    PYARROW_IMPORT_ERROR = exc
else:
    PYARROW_IMPORT_ERROR = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local S1-reasoning parquet shards to Alpaca-style JSONL")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/data_ablation_full59K/data"),
        help="包含 parquet 分片的目录。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/data_ablation_full59K/train.jsonl"),
        help="输出的标准化 JSONL 文件。",
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=Path("data/data_ablation_full59K/dataset_meta.json"),
        help="输出的数据集统计信息 JSON。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="每次从 parquet 读取的记录数。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="可选，仅转换前 N 条样本，便于测试。",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def ensure_pyarrow() -> None:
    if pq is None:
        raise ImportError(
            "pyarrow is required to read parquet shards. "
            "Please install pyarrow first."
        ) from PYARROW_IMPORT_ERROR


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


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    instruction = str(row.get("question", "") or "").strip()
    output = str(row.get("solution", "") or "").strip()

    # S1 原始数据没有单独的 context/input 字段，这里与 Alpaca 对齐为 null。
    input_text = None

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "text": build_alpaca_text(instruction, input_text, output),
        "question": row.get("question"),
        "solution": row.get("solution"),
        "cot_type": row.get("cot_type"),
        "source_type": row.get("source_type"),
        "metadata": row.get("metadata"),
        "cot": row.get("cot"),
        "thinking_trajectories": row.get("thinking_trajectories"),
        "attempt": row.get("attempt"),
    }


def collect_parquet_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {input_dir}")
    return files


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ensure_pyarrow()

    parquet_files = collect_parquet_files(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.meta_output.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    cot_type_distribution: dict[str, int] = {}
    source_type_distribution: dict[str, int] = {}

    with args.output.open("w", encoding="utf-8") as fout:
        for parquet_path in parquet_files:
            logging.info("Reading shard: %s", parquet_path)
            parquet_file = pq.ParquetFile(parquet_path)

            for batch in parquet_file.iter_batches(batch_size=args.batch_size):
                for row in batch.to_pylist():
                    normalized = normalize_row(row)
                    cot_type = str(normalized.get("cot_type") or "")
                    source_type = str(normalized.get("source_type") or "")
                    cot_type_distribution[cot_type] = cot_type_distribution.get(cot_type, 0) + 1
                    source_type_distribution[source_type] = source_type_distribution.get(source_type, 0) + 1

                    fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    total_samples += 1

                    if args.max_samples is not None and total_samples >= args.max_samples:
                        break

                if args.max_samples is not None and total_samples >= args.max_samples:
                    break

            logging.info("Converted %d samples so far", total_samples)
            if args.max_samples is not None and total_samples >= args.max_samples:
                break

    summary = {
        "dataset_name": "simplescaling/data_ablation_full59K",
        "num_samples": total_samples,
        "num_parquet_shards": len(parquet_files),
        "input_dir": str(args.input_dir),
        "output": str(args.output),
        "cot_type_distribution": dict(sorted(cot_type_distribution.items())),
        "source_type_distribution": dict(sorted(source_type_distribution.items())),
    }
    with args.meta_output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Saved normalized dataset to %s", args.output)
    logging.info("Saved metadata to %s", args.meta_output)


if __name__ == "__main__":
    main()
