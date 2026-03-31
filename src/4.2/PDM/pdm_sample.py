#!/usr/bin/env python3
"""Selection for PDM outputs (capability-vector greedy)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


THIS_DIR = Path(__file__).resolve().parent
SELECT_DIR = THIS_DIR.parent / "selection"
if str(SELECT_DIR) not in sys.path:
    sys.path.insert(0, str(SELECT_DIR))

from select_by_score import run_selection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select data by PDM mapped_vector score (greedy).")
    parser.add_argument("--score_path", type=str, required=True, help="PDM score file path.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional raw data path for id-based restore.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Selected dataset output path.")
    parser.add_argument("--meta_output_path", type=str, default=None, help="Optional selection meta output path.")
    parser.add_argument("--num_samples", type=int, required=True, help="Selection budget.")
    parser.add_argument("--id_field", type=str, default="id")
    parser.add_argument(
        "--annotate_selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to append selection metadata into output rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_selection(
        score_path=args.score_path,
        data_path=args.data_path,
        output_path=args.output_path,
        meta_output_path=args.meta_output_path,
        num_samples=int(args.num_samples),
        strategy="vector_greedy",
        id_field=args.id_field,
        vector_field="mapped_vector",
        allow_fallback_fields=True,
        annotate_selection=bool(args.annotate_selection),
    )


if __name__ == "__main__":
    main()

