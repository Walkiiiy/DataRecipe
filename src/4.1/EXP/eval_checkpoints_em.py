"""Evaluate one or more checkpoints on a sampled set with EM metric.

Highlights:
1) Supports pluggable repeated ``--run`` arguments:
   ``--run <name>::<checkpoint_path>``
2) Builds eval set by random sampling from ``train.jsonl``.
3) Computes Exact Match (EM) and exports a final CSV summary.
4) Supports both LoRA adapter checkpoints and full model checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import string
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from modelscope import snapshot_download
except Exception:  # noqa: BLE001
    snapshot_download = None

RESPONSE_MARKER = "### Response:\n"


@dataclass
class EvalRun:
    name: str
    checkpoint_path: Path


@dataclass
class EvalConfig:
    runs: list[EvalRun]
    eval_source_jsonl: Path
    eval_size: int | None
    eval_ratio: float | None
    seed: int
    output_dir: Path
    output_csv: Path
    base_model: str | None
    model_source: str
    modelscope_cache_dir: Path | None
    batch_size: int
    max_seq_length: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    em_normalize: str
    save_per_run_jsonl: bool
    continue_on_error: bool
    device: str


def parse_run_item(text: str) -> EvalRun:
    parts = text.split("::")
    if len(parts) != 2:
        raise ValueError(f"Invalid --run format: {text}. Expected <name>::<checkpoint_path>")
    name = parts[0].strip()
    checkpoint_path = Path(parts[1].strip())
    if not name:
        raise ValueError(f"Invalid run name: {text}")
    return EvalRun(name=name, checkpoint_path=checkpoint_path)


def parse_runs(run_items: list[str]) -> list[EvalRun]:
    runs = [parse_run_item(x) for x in run_items]
    seen = set()
    for run in runs:
        if run.name in seen:
            raise ValueError(f"Duplicate run name: {run.name}")
        seen.add(run.name)
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple checkpoints on a sampled set with EM")
    parser.add_argument(
        "--run",
        type=str,
        action="append",
        required=True,
        help="Repeatable: <name>::<checkpoint_path>",
    )
    parser.add_argument(
        "--eval-source-jsonl",
        type=Path,
        required=True,
        help="Source train.jsonl (or any jsonl with instruction/input/output).",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=None,
        help="Sample size for eval set. If omitted, use --eval-ratio.",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.05,
        help="Sample ratio for eval set when --eval-size is not set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/exp/checkpoint_em_eval"))
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model path or model id (required for LoRA if adapter config base path is unusable).",
    )
    parser.add_argument("--model_source", type=str, choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--modelscope_cache_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--em-normalize",
        type=str,
        choices=["none", "strip", "lower_strip", "squad"],
        default="strip",
        help="Text normalization mode before EM.",
    )
    parser.add_argument("--save-per-run-jsonl", action="store_true", help="Save per-sample predictions per run.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue other runs if one run fails.")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_eval_rows(
    source_rows: list[dict[str, Any]],
    eval_size: int | None,
    eval_ratio: float | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[int]]:
    if len(source_rows) < 1:
        raise ValueError("Eval source is empty.")

    if eval_size is not None:
        n_eval = max(1, min(eval_size, len(source_rows)))
    else:
        if eval_ratio is None:
            raise ValueError("Either --eval-size or --eval-ratio must be provided.")
        ratio = max(1e-6, min(1.0, eval_ratio))
        n_eval = max(1, int(len(source_rows) * ratio))
        n_eval = min(n_eval, len(source_rows))

    idx = list(range(len(source_rows)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    chosen_idx = idx[:n_eval]
    eval_rows = [source_rows[i] for i in chosen_idx]
    return eval_rows, chosen_idx


def build_prompt_without_output(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    inp = str(row.get("input", "")).strip()
    if instruction:
        if inp:
            return (
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Input:\n"
                f"{inp}\n\n"
                "### Response:\n"
            )
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )

    text = str(row.get("text", "")).strip()
    if text:
        if RESPONSE_MARKER in text:
            return text.split(RESPONSE_MARKER, 1)[0] + RESPONSE_MARKER
        return text
    raise ValueError("Row has no usable instruction/text for prompt building.")


def extract_gold_output(row: dict[str, Any]) -> str:
    output = str(row.get("output", "")).strip()
    if output:
        return output
    text = str(row.get("text", "")).strip()
    if RESPONSE_MARKER in text:
        return text.split(RESPONSE_MARKER, 1)[1].strip()
    return ""


def normalize_for_em(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "strip":
        return text.strip()
    if mode == "lower_strip":
        return " ".join(text.lower().strip().split())
    if mode == "squad":
        text = text.lower()
        exclude = set(string.punctuation)
        text = "".join(ch for ch in text if ch not in exclude)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = " ".join(text.strip().split())
        return text
    raise ValueError(f"Unknown normalization mode: {mode}")


def resolve_model_path(model_name_or_path: str, model_source: str, cache_dir: Path | None) -> str:
    model_path = Path(model_name_or_path)
    if model_path.exists():
        return str(model_path)
    if model_source == "hf":
        return model_name_or_path
    if snapshot_download is None:
        raise ImportError("ModelScope is not installed. Please install: pip install modelscope")
    return snapshot_download(
        model_id=model_name_or_path,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_causal_lm(model_path: str, dtype: torch.dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )


def detect_checkpoint_type(checkpoint_path: Path) -> str:
    has_adapter_cfg = (checkpoint_path / "adapter_config.json").exists()
    has_adapter_weight = (checkpoint_path / "adapter_model.safetensors").exists() or (
        checkpoint_path / "adapter_model.bin"
    ).exists()
    if has_adapter_cfg and has_adapter_weight:
        return "lora_adapter"
    return "full_model"


def read_adapter_base_model(checkpoint_path: Path) -> str | None:
    cfg_path = checkpoint_path / "adapter_config.json"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path", None)
    if base is None:
        return None
    base = str(base).strip()
    return base if base else None


def load_tokenizer(primary: str, fallback: str | None) -> Any:
    last_err: Exception | None = None
    for candidate in [primary, fallback]:
        if candidate is None:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif tokenizer.unk_token is not None:
                    tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"
            return tokenizer
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if last_err is not None:
        raise last_err
    raise RuntimeError("Unable to load tokenizer from both primary and fallback locations.")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def postprocess_prediction(text: str) -> str:
    pred = text.strip()
    if RESPONSE_MARKER in pred:
        pred = pred.split(RESPONSE_MARKER, 1)[1].strip()
    return pred


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def evaluate_run(
    cfg: EvalConfig,
    run: EvalRun,
    eval_items: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not run.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {run.checkpoint_path}")

    t0 = time.time()
    checkpoint_type = detect_checkpoint_type(run.checkpoint_path)
    base_model_used: str | None = None

    if checkpoint_type == "lora_adapter":
        adapter_base = read_adapter_base_model(run.checkpoint_path)
        requested_base = cfg.base_model if cfg.base_model else adapter_base
        if requested_base is None:
            raise ValueError(
                f"{run.name}: LoRA adapter detected, but base model is unknown. "
                "Please pass --base_model."
            )
        base_model_path = resolve_model_path(
            requested_base,
            cfg.model_source,
            cfg.modelscope_cache_dir,
        )
        base_model_used = base_model_path
        tokenizer = load_tokenizer(str(run.checkpoint_path), base_model_path)
        model = load_causal_lm(base_model_path, get_dtype())
        model = PeftModel.from_pretrained(model, str(run.checkpoint_path))
    else:
        checkpoint_model_path = str(run.checkpoint_path)
        fallback_base = None
        if cfg.base_model:
            fallback_base = resolve_model_path(cfg.base_model, cfg.model_source, cfg.modelscope_cache_dir)
        tokenizer = load_tokenizer(checkpoint_model_path, fallback_base)
        model = load_causal_lm(checkpoint_model_path, get_dtype())
        base_model_used = checkpoint_model_path

    if tokenizer.pad_token_id is None:
        raise ValueError(f"{run.name}: tokenizer pad_token_id is None and cannot be inferred.")
    model.to(device)
    model.eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = cfg.do_sample
        if cfg.do_sample:
            model.generation_config.temperature = cfg.temperature
            model.generation_config.top_p = cfg.top_p
        else:
            # Reset to common greedy defaults to avoid irrelevant sampling warnings.
            model.generation_config.temperature = 1.0
            model.generation_config.top_p = 1.0
            model.generation_config.top_k = 50

    details: list[dict[str, Any]] = []
    prompts = [x["prompt"] for x in eval_items]
    preds: list[str] = []

    for start, batch_prompts in batched(prompts, cfg.batch_size):
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_seq_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if cfg.do_sample:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p

        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The following generation flags are not valid and may be ignored:.*",
                )
                output_ids = model.generate(**encoded, **gen_kwargs)

        input_len = int(encoded["input_ids"].shape[1])
        for i in range(output_ids.shape[0]):
            gen_ids = output_ids[i, input_len:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
            preds.append(postprocess_prediction(pred))

        if (start // cfg.batch_size) % 10 == 0:
            logging.info(
                "[%s] generated %d/%d",
                run.name,
                min(start + len(batch_prompts), len(prompts)),
                len(prompts),
            )

    correct = 0
    for i, (item, pred) in enumerate(zip(eval_items, preds, strict=True)):
        gold = item["gold"]
        norm_pred = normalize_for_em(pred, cfg.em_normalize)
        norm_gold = normalize_for_em(gold, cfg.em_normalize)
        matched = int(norm_pred == norm_gold)
        correct += matched
        details.append(
            {
                "sample_id": i,
                "source_index": item["source_index"],
                "id": item.get("id"),
                "gold": gold,
                "prediction": pred,
                "norm_gold": norm_gold,
                "norm_prediction": norm_pred,
                "exact_match": matched,
            }
        )

    total = len(eval_items)
    em = float(correct / total) if total > 0 else float("nan")
    elapsed = time.time() - t0

    summary = {
        "method": run.name,
        "checkpoint_path": str(run.checkpoint_path),
        "checkpoint_type": checkpoint_type,
        "base_model_used": base_model_used,
        "eval_size": total,
        "correct": correct,
        "em": em,
        "em_percent": em * 100.0,
        "elapsed_sec": elapsed,
        "error": "",
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, details


def safe_name(text: str) -> str:
    out = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("_")
    return out if out else "run"


def save_summary_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "checkpoint_path",
        "checkpoint_type",
        "base_model_used",
        "eval_size",
        "correct",
        "em",
        "em_percent",
        "elapsed_sec",
        "error",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    runs = parse_runs(args.run)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv if args.output_csv is not None else (output_dir / "checkpoint_em_results.csv")

    cfg = EvalConfig(
        runs=runs,
        eval_source_jsonl=args.eval_source_jsonl,
        eval_size=(max(1, args.eval_size) if args.eval_size is not None else None),
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        output_dir=output_dir,
        output_csv=output_csv,
        base_model=args.base_model,
        model_source=args.model_source,
        modelscope_cache_dir=args.modelscope_cache_dir,
        batch_size=max(1, args.batch_size),
        max_seq_length=max(32, args.max_seq_length),
        max_new_tokens=max(1, args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=max(1e-5, float(args.temperature)),
        top_p=max(1e-5, min(1.0, float(args.top_p))),
        em_normalize=args.em_normalize,
        save_per_run_jsonl=bool(args.save_per_run_jsonl),
        continue_on_error=bool(args.continue_on_error),
        device=args.device,
    )

    if not cfg.eval_source_jsonl.exists():
        raise FileNotFoundError(f"Eval source jsonl not found: {cfg.eval_source_jsonl}")

    source_rows = load_jsonl(cfg.eval_source_jsonl)
    sampled_rows, sampled_indices = sample_eval_rows(
        source_rows=source_rows,
        eval_size=cfg.eval_size,
        eval_ratio=cfg.eval_ratio,
        seed=cfg.seed,
    )
    sampled_eval_path = cfg.output_dir / "sampled_eval.jsonl"
    write_jsonl(sampled_rows, sampled_eval_path)
    with (cfg.output_dir / "sampled_eval_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_source_jsonl": str(cfg.eval_source_jsonl),
                "eval_size": len(sampled_rows),
                "seed": cfg.seed,
                "indices": sampled_indices,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logging.info("Sampled eval set saved: %s (size=%d)", sampled_eval_path, len(sampled_rows))

    eval_items: list[dict[str, Any]] = []
    skipped = 0
    for source_index, row in zip(sampled_indices, sampled_rows, strict=True):
        gold = extract_gold_output(row)
        if not gold:
            skipped += 1
            continue
        prompt = build_prompt_without_output(row)
        eval_items.append(
            {
                "source_index": source_index,
                "id": row.get("id", None),
                "prompt": prompt,
                "gold": gold,
            }
        )
    if skipped:
        logging.warning("Skipped %d sampled rows due to empty gold output.", skipped)
    if not eval_items:
        raise ValueError("No valid eval items after filtering empty gold outputs.")

    device = resolve_device(cfg.device)
    logging.info("Evaluation device: %s", device)
    logging.info("Runs=%d | eval_items=%d | normalize=%s", len(cfg.runs), len(eval_items), cfg.em_normalize)

    summaries: list[dict[str, Any]] = []
    for run in cfg.runs:
        logging.info("=== Evaluating run: %s | checkpoint=%s ===", run.name, run.checkpoint_path)
        try:
            summary, details = evaluate_run(cfg, run, eval_items, device)
            summaries.append(summary)
            logging.info(
                "[%s] EM=%.4f (%d/%d), elapsed=%.2fs",
                run.name,
                summary["em"],
                summary["correct"],
                summary["eval_size"],
                summary["elapsed_sec"],
            )
            if cfg.save_per_run_jsonl:
                pred_path = cfg.output_dir / f"predictions_{safe_name(run.name)}.jsonl"
                write_jsonl(details, pred_path)
                logging.info("[%s] saved predictions: %s", run.name, pred_path)
        except Exception as exc:  # noqa: BLE001
            if not cfg.continue_on_error:
                raise
            logging.exception("Run failed: %s", run.name)
            summaries.append(
                {
                    "method": run.name,
                    "checkpoint_path": str(run.checkpoint_path),
                    "checkpoint_type": "",
                    "base_model_used": "",
                    "eval_size": len(eval_items),
                    "correct": "",
                    "em": "",
                    "em_percent": "",
                    "elapsed_sec": "",
                    "error": str(exc),
                }
            )

    save_summary_csv(summaries, cfg.output_csv)
    with (cfg.output_dir / "checkpoint_em_results.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    logging.info("Saved summary csv: %s", cfg.output_csv)
    logging.info("Done.")


if __name__ == "__main__":
    main()
