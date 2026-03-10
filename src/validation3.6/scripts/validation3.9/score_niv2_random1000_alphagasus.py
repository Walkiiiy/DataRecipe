"""Batch-score niv2_capability_data_random1000 with AlphaGasus multi-aspect mapping.

Usage example:
python /home/walkiiiy/DataRecipe/src/scripts/score_niv2_random1000_alphagasus.py \
  --api_key YOUR_DEEPSEEK_KEY
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple
import sys

# Ensure project root is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.CapMapping.AlphaGasusMapping import AlphaGasusAccuracyMappingService, LLMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/walkiiiy/DataRecipe/data/flan/niv2_capability_data_random1000",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/walkiiiy/DataRecipe/data/flan/niv2_capability_data_random1000_alphagasus_scored",
    )
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com/chat/completions")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=None)

    parser.add_argument(
        "--anchor_dir",
        type=str,
        default=(
            "/home/walkiiiy/DataRecipe/data/flan/"
            "niv2_capability_data_ramdom1000_preprocessed/capability_anchors"
        ),
    )
    parser.add_argument("--anchor_vector_dim", type=int, default=8192)

    parser.add_argument("--sleep_sec", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_per_file",
        type=int,
        default=100,
        help="Maximum number of samples to process per input file.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=20,
        help="Print sample-level progress every N processed samples within each file.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield idx, json.loads(line)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_input_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    err_log = output_dir / "errors.jsonl"

    print("[Stage 1/4] Build scorer config...", flush=True)
    cfg = LLMConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        anchor_dir=args.anchor_dir,
        anchor_vector_dim=args.anchor_vector_dim,
    )
    print("[Stage 2/4] Initialize AlphaGasus scorer (includes anchor retriever)...", flush=True)
    scorer = AlphaGasusAccuracyMappingService(config=cfg)
    print("[Stage 2/4] Scorer ready.", flush=True)

    files = sorted(input_dir.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No jsonl files found in {input_dir}")

    print(f"[Stage 3/4] Found {len(files)} jsonl files under: {input_dir}", flush=True)

    total_target_lines = 0
    for fp in files:
        total_target_lines += min(count_input_nonempty_lines(fp), args.max_per_file)
    if args.max_samples is not None:
        total_target_lines = min(total_target_lines, args.max_samples)
    print(
        f"[Stage 3/4] Total target samples to score (upper bound): {total_target_lines}",
        flush=True,
    )

    total_processed = 0
    total_errors = 0
    file_done = 0

    print("[Stage 4/4] Start scoring loop...", flush=True)
    for in_file in files:
        file_done += 1
        rel = in_file.relative_to(input_dir)
        out_file = output_dir / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)

        in_total = count_input_nonempty_lines(in_file)
        start_line = count_lines(out_file) if args.resume else 0
        mode = "a" if args.resume else "w"
        file_target = min(in_total, args.max_per_file)
        estimated_to_process = max(0, file_target - start_line)
        print(
            f"[File {file_done}/{len(files)}] {rel} | input={in_total} resume_skip={start_line} "
            f"to_process~={estimated_to_process}",
            flush=True,
        )

        file_processed = 0
        file_errors = 0
        with out_file.open(mode, encoding="utf-8") as fw:
            for line_no, ex in iter_jsonl(in_file):
                if line_no > file_target:
                    break
                if line_no <= start_line:
                    continue

                if args.max_samples is not None and total_processed >= args.max_samples:
                    print(
                        f"Reached max_samples={args.max_samples}. "
                        f"processed={total_processed}, errors={total_errors}"
                    )
                    return

                try:
                    scores = scorer.score(ex)
                    ex["相关度"] = scores["相关度"]
                    ex["准确性"] = scores["准确性"]
                    ex["能力多样性"] = scores["能力多样性"]
                    ex["难度"] = scores["难度"]
                    ex["能力锚点属性"] = scores["能力锚点"]

                    fw.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    total_processed += 1
                    file_processed += 1

                    if file_processed % max(1, args.progress_every) == 0:
                        pct_total = (
                            100.0 * total_processed / total_target_lines
                            if total_target_lines > 0
                            else 0.0
                        )
                        print(
                            f"  [Progress] file_processed={file_processed} file_errors={file_errors} "
                            f"global_processed={total_processed} global_errors={total_errors} "
                            f"global_pct={pct_total:.2f}%",
                            flush=True,
                        )

                except Exception as exc:  # noqa: BLE001
                    total_errors += 1
                    file_errors += 1
                    with err_log.open("a", encoding="utf-8") as fe:
                        fe.write(
                            json.dumps(
                                {
                                    "file": str(in_file),
                                    "line_no": line_no,
                                    "error": str(exc),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    # keep alignment with input if needed
                    ex["相关度"] = ""
                    ex["准确性"] = ""
                    ex["能力多样性"] = ""
                    ex["难度"] = ""
                    ex["能力锚点属性"] = ""
                    fw.write(json.dumps(ex, ensure_ascii=False) + "\n")

                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

        print(
            f"[File Done] {rel} | processed={file_processed} errors={file_errors} "
            f"global_processed={total_processed} global_errors={total_errors}",
            flush=True,
        )

    print(
        f"Done. output_dir={output_dir} processed={total_processed} errors={total_errors}",
        flush=True,
    )


if __name__ == "__main__":
    main()
