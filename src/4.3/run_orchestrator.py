#!/usr/bin/env python3
"""Module 3: Orchestrator for decoupled Recipe Evolution training.

Workflow:
1) Split `train_raw.jsonl` into N chunks.
2) For each chunk i:
   - run module 2 (`update_beta_anchors.py`) with latest checkpoint
   - run module 1 (`train_dynamic_chunk.py`) on chunk i
   - append selected rows into `selected_epoch1.jsonl`
3) After epoch-1 dynamic selection, call original
   `src/4.1/EXP/sft_lora_train_shared_eval.py` for static E-1 epochs.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from recipe_common import (
    append_jsonl,
    choose_row_id,
    load_jsonl,
    parse_mapper_item,
    resolve_model_path,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recipe Evolution decoupled orchestrator")

    parser.add_argument("--train-raw-jsonl", type=Path, required=True)
    parser.add_argument("--anchor-jsonl", type=Path, required=True)
    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--score-path", type=str, action="append", required=True)

    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--model-source", type=str, choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--modelscope-cache-dir", type=Path, default=None)
    parser.add_argument("--torch-dtype", type=str, choices=["auto", "fp32", "fp16", "bf16"], default="auto")

    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--total-epochs", type=int, default=4)
    parser.add_argument("--shuffle-before-chunk", type=int, default=1)
    parser.add_argument(
        "--epoch1-target-size",
        type=int,
        default=0,
        help="Hard cap for epoch-1 selected dataset size. 0 means no cap (use all kept samples).",
    )

    # module 2 params
    parser.add_argument("--capability-key", type=str, default="auto")
    parser.add_argument("--anchor-size-per-cap", type=int, default=4)
    parser.add_argument("--anchor-batch-size", type=int, default=2)
    parser.add_argument("--anchor-max-seq-length", type=int, default=512)
    parser.add_argument("--max-anchor-capabilities", type=int, default=0)
    parser.add_argument("--beta-ema", type=float, default=0.0)
    parser.add_argument(
        "--focus-anchor-by-chunk",
        type=int,
        default=1,
        help="1: module2 only refresh capabilities appearing in current chunk top-k.",
    )
    parser.add_argument(
        "--beta-grad-param-mode",
        type=str,
        choices=["lora", "lm_head", "all_trainable", "name_contains"],
        default="lora",
    )
    parser.add_argument("--beta-grad-param-filter", type=str, default="")

    # module 1 params
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--keep-policy", type=str, choices=["threshold", "softmax"], default="threshold")
    parser.add_argument("--keep-threshold", type=float, default=0.5)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--score-ema-momentum", type=float, default=0.02)
    parser.add_argument("--min-keep-prob", type=float, default=0.05)
    parser.add_argument("--keep-every-n", type=int, default=32)
    parser.add_argument("--reward-weight", type=float, default=0.5)
    parser.add_argument("--data-weight", type=float, default=0.5)
    parser.add_argument("--alpha-ema", type=float, default=0.1)
    parser.add_argument("--alpha-temperature", type=float, default=1.0)
    parser.add_argument("--min-response-tokens", type=int, default=4)
    parser.add_argument(
        "--chunk-grad-param-mode",
        type=str,
        choices=["lora", "lm_head", "all_trainable", "name_contains"],
        default="lora",
    )
    parser.add_argument("--chunk-grad-param-filter", type=str, default="")
    parser.add_argument("--grad-ckpt", type=int, default=1)
    parser.add_argument(
        "--max-grad-dim",
        type=int,
        default=65536,
        help="Truncate flattened gradients in module1/2 for memory safety. 0 means full dim.",
    )

    # static stage config
    parser.add_argument("--run-static-stage", type=int, default=1)
    parser.add_argument("--shared-eval-jsonl", type=Path, default=None)
    parser.add_argument("--eval-source-jsonl", type=Path, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--static-output-root", type=Path, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def run_cmd(cmd: list[str]) -> None:
    logging.info("RUN: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def chunk_rows(rows: list[dict[str, Any]], n_chunks: int) -> list[list[dict[str, Any]]]:
    n = len(rows)
    c = max(1, n_chunks)
    chunk_size = int(math.ceil(n / float(c)))
    out = []
    for i in range(c):
        st = i * chunk_size
        ed = min(n, (i + 1) * chunk_size)
        part = rows[st:ed]
        if part:
            out.append(part)
    return out


def ensure_row_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        obj = dict(row)
        if obj.get("id") is None:
            obj["id"] = i
        out.append(obj)
    return out


def append_selected_dedup(
    selected_file: Path,
    chunk_selected_rows: list[dict[str, Any]],
    seen_ids: set[str],
    max_append: int = 0,
) -> int:
    filtered: list[dict[str, Any]] = []
    for i, row in enumerate(chunk_selected_rows):
        if max_append > 0 and len(filtered) >= max_append:
            break
        rid = choose_row_id(row, i)
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        filtered.append(row)

    if filtered:
        append_jsonl(filtered, selected_file)
    return len(filtered)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    random.seed(args.seed)

    if not args.train_raw_jsonl.exists():
        raise FileNotFoundError(f"train_raw_jsonl not found: {args.train_raw_jsonl}")
    if not args.anchor_jsonl.exists():
        raise FileNotFoundError(f"anchor_jsonl not found: {args.anchor_jsonl}")
    if not args.topk_jsonl.exists():
        raise FileNotFoundError(f"topk_jsonl not found: {args.topk_jsonl}")

    mapper_inputs = [parse_mapper_item(x) for x in args.score_path]
    for mapper in mapper_inputs:
        if not mapper.path.exists():
            raise FileNotFoundError(f"score file not found: {mapper.path}")

    # build paths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    state_dir = args.output_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    chunk_run_root = args.output_dir / "chunk_runs"
    chunk_run_root.mkdir(parents=True, exist_ok=True)

    selected_epoch1 = args.output_dir / "selected_epoch1.jsonl"
    write_jsonl([], selected_epoch1)

    alpha_json = state_dir / "alpha_state.json"
    beta_json = state_dir / "current_beta.json"
    anchor_grads = state_dir / "anchor_grads.pt"

    # split chunks
    rows = ensure_row_ids(load_jsonl(args.train_raw_jsonl))
    if int(args.shuffle_before_chunk) == 1:
        random.shuffle(rows)

    chunked = chunk_rows(rows, n_chunks=max(1, args.num_chunks))
    chunk_files: list[Path] = []
    for i, part in enumerate(chunked):
        p = chunks_dir / f"chunk_{i}.jsonl"
        write_jsonl(part, p)
        chunk_files.append(p)

    logging.info("Prepared %d chunks from %d samples", len(chunk_files), len(rows))

    epoch1_target_size = 0
    if int(args.epoch1_target_size) > 0:
        epoch1_target_size = min(max(1, int(args.epoch1_target_size)), len(rows))
        logging.info("Epoch-1 hard target size enabled: %d", epoch1_target_size)

    py = sys.executable
    module2 = Path(__file__).resolve().parent / "update_beta_anchors.py"
    module1 = Path(__file__).resolve().parent / "train_dynamic_chunk.py"
    static_sft = Path(__file__).resolve().parents[1] / "4.1" / "EXP" / "sft_lora_train_shared_eval.py"

    # Resolve base checkpoint once.
    # In modelscope mode this becomes a local snapshot path and prevents
    # first-step fallback to HF model id.
    current_checkpoint = resolve_model_path(
        base_model=args.base_model,
        model_source=args.model_source,
        cache_dir=args.modelscope_cache_dir,
    )
    logging.info("Initial checkpoint resolved to: %s", current_checkpoint)
    seen_selected_ids: set[str] = set()
    chunk_stats: list[dict[str, Any]] = []

    for i, chunk_file in enumerate(chunk_files):
        if epoch1_target_size > 0 and len(seen_selected_ids) >= epoch1_target_size:
            logging.info(
                "Epoch-1 target reached (%d/%d). Stop scheduling remaining chunks.",
                len(seen_selected_ids),
                epoch1_target_size,
            )
            break

        logging.info("[Chunk %d/%d] start", i + 1, len(chunk_files))

        # module 2: update beta + anchor grads
        cmd_m2 = [
            py,
            str(module2),
            "--checkpoint-path",
            str(current_checkpoint),
            "--base-model",
            args.base_model,
            "--model-source",
            args.model_source,
            "--torch-dtype",
            args.torch_dtype,
            "--anchor-jsonl",
            str(args.anchor_jsonl),
            "--topk-jsonl",
            str(args.topk_jsonl),
            "--capability-key",
            args.capability_key,
            "--anchor-size-per-cap",
            str(args.anchor_size_per_cap),
            "--anchor-batch-size",
            str(args.anchor_batch_size),
            "--max-seq-length",
            str(args.anchor_max_seq_length),
            "--max-capabilities",
            str(args.max_anchor_capabilities),
            "--beta-ema",
            str(args.beta_ema),
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--max-grad-dim",
            str(args.max_grad_dim),
            "--output-beta-json",
            str(beta_json),
            "--output-anchor-grads",
            str(anchor_grads),
            "--grad-param-mode",
            args.beta_grad_param_mode,
            "--grad-param-filter",
            args.beta_grad_param_filter,
            "--seed",
            str(args.seed),
            "--log-level",
            args.log_level,
        ]
        if args.modelscope_cache_dir is not None:
            cmd_m2.extend(["--modelscope-cache-dir", str(args.modelscope_cache_dir)])
        if beta_json.exists():
            cmd_m2.extend(["--prev-beta-json", str(beta_json)])
        if int(args.focus_anchor_by_chunk) == 1:
            cmd_m2.extend(["--focus-jsonl", str(chunk_file)])

        run_cmd(cmd_m2)

        # module 1: dynamic train current chunk
        chunk_out = chunk_run_root / f"chunk_{i}"
        chunk_out.mkdir(parents=True, exist_ok=True)

        cmd_m1 = [
            py,
            str(module1),
            "--chunk-jsonl",
            str(chunk_file),
            "--checkpoint-path",
            str(current_checkpoint),
            "--base-model",
            args.base_model,
            "--model-source",
            args.model_source,
            "--torch-dtype",
            args.torch_dtype,
            "--topk-jsonl",
            str(args.topk_jsonl),
            "--beta-json",
            str(beta_json),
            "--anchor-grads",
            str(anchor_grads),
            "--alpha-json",
            str(alpha_json),
            "--output-dir",
            str(chunk_out),
            "--max-seq-length",
            str(args.max_seq_length),
            "--learning-rate",
            str(args.learning_rate),
            "--train-batch-size",
            str(args.train_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--num-train-epochs",
            "1.0",
            "--weight-decay",
            str(args.weight_decay),
            "--warmup-ratio",
            str(args.warmup_ratio),
            "--logging-steps",
            str(args.logging_steps),
            "--save-steps",
            str(args.save_steps),
            "--lora-r",
            str(args.lora_r),
            "--lora-alpha",
            str(args.lora_alpha),
            "--lora-dropout",
            str(args.lora_dropout),
            "--keep-policy",
            args.keep_policy,
            "--keep-threshold",
            str(args.keep_threshold),
            "--keep-ratio",
            str(args.keep_ratio),
            "--softmax-temperature",
            str(args.softmax_temperature),
            "--score-ema-momentum",
            str(args.score_ema_momentum),
            "--min-keep-prob",
            str(args.min_keep_prob),
            "--keep-every-n",
            str(args.keep_every_n),
            "--reward-weight",
            str(args.reward_weight),
            "--data-weight",
            str(args.data_weight),
            "--alpha-ema",
            str(args.alpha_ema),
            "--alpha-temperature",
            str(args.alpha_temperature),
            "--min-response-tokens",
            str(args.min_response_tokens),
            "--grad-param-mode",
            args.chunk_grad_param_mode,
            "--grad-param-filter",
            args.chunk_grad_param_filter,
            "--max-grad-dim",
            str(args.max_grad_dim),
            "--grad-ckpt",
            str(args.grad_ckpt),
            "--seed",
            str(args.seed),
            "--log-level",
            args.log_level,
        ]
        if args.modelscope_cache_dir is not None:
            cmd_m1.extend(["--modelscope-cache-dir", str(args.modelscope_cache_dir)])
        for mapper in mapper_inputs:
            cmd_m1.extend(["--score-path", f"{mapper.name}::{mapper.path}"])

        run_cmd(cmd_m1)

        chunk_result_path = chunk_out / "chunk_result.json"
        if not chunk_result_path.exists():
            raise FileNotFoundError(f"chunk result not found: {chunk_result_path}")

        with chunk_result_path.open("r", encoding="utf-8") as f:
            chunk_result = json.load(f)

        next_ckpt = chunk_result.get("next_checkpoint")
        if not next_ckpt:
            raise ValueError(f"chunk result missing next_checkpoint: {chunk_result_path}")
        current_checkpoint = str(next_ckpt)

        chunk_selected_path = Path(chunk_result.get("selected_jsonl", ""))
        if chunk_selected_path.exists():
            chunk_selected_rows = load_jsonl(chunk_selected_path)
        else:
            chunk_selected_rows = []

        remaining = 0
        if epoch1_target_size > 0:
            remaining = max(0, epoch1_target_size - len(seen_selected_ids))

        appended = append_selected_dedup(
            selected_file=selected_epoch1,
            chunk_selected_rows=chunk_selected_rows,
            seen_ids=seen_selected_ids,
            max_append=remaining,
        )

        chunk_stats.append(
            {
                "chunk_index": i,
                "chunk_file": str(chunk_file),
                "chunk_size": int(len(load_jsonl(chunk_file))),
                "selected_from_chunk": int(appended),
                "selected_running_total": int(len(seen_selected_ids)),
                "chunk_result": str(chunk_result_path),
                "next_checkpoint": str(current_checkpoint),
            }
        )

    static_stage_info: dict[str, Any] = {
        "ran": False,
        "reason": "total_epochs<=1 or disabled",
        "command": None,
    }

    # static stage (E-1 epochs)
    static_epochs = int(args.total_epochs) - 1
    if int(args.run_static_stage) == 1 and static_epochs > 0:
        static_output_root = args.static_output_root or (args.output_dir / "static_stage")
        static_output_root.mkdir(parents=True, exist_ok=True)
        static_run_dir = static_output_root / "run_recipe_selected"
        resume_adapter_ckpt: str | None = None
        ckpt_path = Path(str(current_checkpoint))
        if ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
            resume_adapter_ckpt = str(ckpt_path)
            logging.info("Static stage will resume from epoch-1 adapter: %s", resume_adapter_ckpt)
        else:
            logging.info("Static stage starts from base model (no adapter checkpoint found at %s).", current_checkpoint)

        cmd_static = [
            py,
            str(static_sft),
            "--run",
            f"recipe_selected::{selected_epoch1}::{static_run_dir}",
            "--num_train_epochs",
            str(static_epochs),
            "--seed",
            str(args.seed),
            "--base_model",
            args.base_model,
            "--model_source",
            args.model_source,
            "--output-root",
            str(static_output_root),
            "--train_batch_size",
            str(args.train_batch_size),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--learning_rate",
            str(args.learning_rate),
            "--max_seq_length",
            str(args.max_seq_length),
            "--warmup_ratio",
            str(args.warmup_ratio),
            "--weight_decay",
            str(args.weight_decay),
            "--logging_steps",
            str(args.logging_steps),
            "--save_steps",
            str(max(1, args.save_steps) if args.save_steps > 0 else 100),
            "--eval-ratio",
            str(args.eval_ratio),
        ]
        if resume_adapter_ckpt is not None:
            cmd_static.extend(["--init_adapter_checkpoint", resume_adapter_ckpt])

        if args.modelscope_cache_dir is not None:
            cmd_static.extend(["--modelscope_cache_dir", str(args.modelscope_cache_dir)])

        if args.shared_eval_jsonl is not None:
            cmd_static.extend(["--shared-eval-jsonl", str(args.shared_eval_jsonl)])
        elif args.eval_source_jsonl is not None:
            cmd_static.extend(["--eval-source-jsonl", str(args.eval_source_jsonl)])
        else:
            static_stage_info = {
                "ran": False,
                "reason": "missing --shared-eval-jsonl or --eval-source-jsonl",
                "command": cmd_static,
            }
            cmd_static = []

        if cmd_static:
            run_cmd(cmd_static)
            static_stage_info = {
                "ran": True,
                "reason": "ok",
                "command": cmd_static,
                "static_epochs": static_epochs,
                "resume_adapter_checkpoint": resume_adapter_ckpt,
                "static_output_root": str(static_output_root),
            }

    summary = {
        "train_raw_jsonl": str(args.train_raw_jsonl),
        "anchor_jsonl": str(args.anchor_jsonl),
        "topk_jsonl": str(args.topk_jsonl),
        "score_paths": [str(parse_mapper_item(x).path) for x in args.score_path],
        "output_dir": str(args.output_dir),
        "num_chunks": int(len(chunk_files)),
        "total_epochs": int(args.total_epochs),
        "selected_epoch1_jsonl": str(selected_epoch1),
        "selected_epoch1_size": int(len(seen_selected_ids)),
        "epoch1_target_size": int(epoch1_target_size),
        "final_checkpoint_after_epoch1": str(current_checkpoint),
        "chunk_stats": chunk_stats,
        "static_stage": static_stage_info,
    }

    summary_path = args.output_dir / "orchestrator_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Done. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
