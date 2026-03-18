"""4.1 EXP - 三组共享同一验证集的 LoRA 训练脚本。

用途：
1) 顺序训练 ours / kmeans / random（可选 category）四组实验。
2) 所有实验共享同一份 eval 数据，保证验证口径一致。
3) 自动从每个训练集剔除与共享 eval 重叠的样本（按 id，回退 instruction+output）。
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

try:
    from modelscope import snapshot_download
except Exception:  # noqa: BLE001
    snapshot_download = None


@dataclass
class MethodJob:
    name: str
    dataset_path: Path
    output_dir: Path


@dataclass
class TrainConfig:
    # 数据输入
    ours_dataset_path: Path
    kmeans_dataset_path: Path
    random_dataset_path: Path
    category_dataset_path: Path | None
    ours_output_dir: Path
    kmeans_output_dir: Path
    random_output_dir: Path
    category_output_dir: Path | None
    shared_eval_jsonl: Path | None
    eval_source_jsonl: Path | None
    eval_ratio: float
    seed: int
    output_root: Path
    # 模型与训练超参
    base_model: str
    model_source: str
    modelscope_cache_dir: Path | None
    max_seq_length: int
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiple LoRA runs with one shared eval set")
    parser.add_argument("--ours-dataset-path", type=Path, required=True)
    parser.add_argument("--kmeans-dataset-path", type=Path, required=True)
    parser.add_argument("--random-dataset-path", type=Path, required=True)
    parser.add_argument("--category-dataset-path", type=Path, default=None)

    parser.add_argument("--ours-output-dir", type=Path, required=True)
    parser.add_argument("--kmeans-output-dir", type=Path, required=True)
    parser.add_argument("--random-output-dir", type=Path, required=True)
    parser.add_argument("--category-output-dir", type=Path, default=None)

    parser.add_argument(
        "--shared-eval-jsonl",
        type=Path,
        default=None,
        help="已准备好的共享验证集。若未提供，则必须提供 --eval-source-jsonl 自动切分。",
    )
    parser.add_argument(
        "--eval-source-jsonl",
        type=Path,
        default=None,
        help="用于自动切分共享验证集的源数据（例如原始 train.jsonl）。",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("data/alpaca-gpt4-data-en/exp/shared_eval_runs"))

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--model_source", type=str, choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--modelscope_cache_dir", type=Path, default=None)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
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
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sample_id(row: dict[str, Any], fallback_idx: int) -> str | None:
    sid = row.get("id", None)
    if sid is not None and str(sid) != "":
        return str(sid)
    # 无 id 时回退到 index 仅用于日志，不参与跨集匹配
    _ = fallback_idx
    return None


def sample_io_signature(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    inp = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    return f"{instruction}||{inp}||{output}"


def build_prompt(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    inp = str(row.get("input", "")).strip()
    output = str(row.get("output", "")).strip()
    if inp:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{inp}\n\n"
            "### Response:\n"
            f"{output}"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{output}"
    )


def to_text_dataset(rows: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list([{"text": build_prompt(r)} for r in rows])


def build_or_load_shared_eval(
    shared_eval_jsonl: Path | None,
    eval_source_jsonl: Path | None,
    eval_ratio: float,
    seed: int,
    output_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    if shared_eval_jsonl is not None:
        if not shared_eval_jsonl.exists():
            raise FileNotFoundError(f"Shared eval jsonl not found: {shared_eval_jsonl}")
        rows = load_jsonl(shared_eval_jsonl)
        if not rows:
            raise ValueError("Shared eval jsonl is empty.")
        return shared_eval_jsonl, rows

    if eval_source_jsonl is None:
        raise ValueError("Either --shared-eval-jsonl or --eval-source-jsonl must be provided.")
    if not eval_source_jsonl.exists():
        raise FileNotFoundError(f"Eval source jsonl not found: {eval_source_jsonl}")

    source_rows = load_jsonl(eval_source_jsonl)
    if len(source_rows) < 2:
        raise ValueError("Eval source data is too small to split.")

    rng = random.Random(seed)
    idx = list(range(len(source_rows)))
    rng.shuffle(idx)
    n_eval = max(1, int(len(source_rows) * eval_ratio))
    n_eval = min(n_eval, len(source_rows))
    eval_rows = [source_rows[i] for i in idx[:n_eval]]

    out_path = output_root / "shared_eval.jsonl"
    write_jsonl(eval_rows, out_path)
    return out_path, eval_rows


def filter_train_rows_by_eval(
    rows: list[dict[str, Any]],
    eval_id_set: set[str],
    eval_io_set: set[str],
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0
    for i, row in enumerate(rows):
        sid = sample_id(row, i)
        io_sig = sample_io_signature(row)
        if (sid is not None and sid in eval_id_set) or (io_sig in eval_io_set):
            removed += 1
            continue
        kept.append(row)
    return kept, removed


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_model_path(base_model: str, model_source: str, cache_dir: Path | None) -> str:
    if model_source == "hf":
        return base_model
    if snapshot_download is None:
        raise ImportError("ModelScope is not installed. Please install: pip install modelscope")
    return snapshot_download(
        model_id=base_model,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


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


class EvalLossCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: D401
        if metrics and "eval_loss" in metrics:
            logging.info("Eval step=%s eval_loss=%.6f", state.global_step, float(metrics["eval_loss"]))


def export_logs(log_history: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_json = out_dir / "log_history.json"
    with raw_json.open("w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "train_eval_log.csv"
    fields = ["step", "epoch", "loss", "eval_loss", "learning_rate"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in log_history:
            writer.writerow(
                {
                    "step": item.get("step", ""),
                    "epoch": item.get("epoch", ""),
                    "loss": item.get("loss", ""),
                    "eval_loss": item.get("eval_loss", ""),
                    "learning_rate": item.get("learning_rate", ""),
                }
            )


def build_training_args(cfg: TrainConfig, output_dir: Path) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": cfg.learning_rate,
        "per_device_train_batch_size": cfg.train_batch_size,
        "per_device_eval_batch_size": cfg.eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_train_epochs": cfg.num_train_epochs,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "logging_steps": cfg.logging_steps,
        "eval_steps": cfg.eval_steps,
        "save_steps": cfg.save_steps,
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "report_to": [],
        "fp16": (get_dtype() == torch.float16),
        "bf16": (get_dtype() == torch.bfloat16),
        "gradient_checkpointing": True,
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "load_best_model_at_end": False,
    }
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        raise RuntimeError("Neither `evaluation_strategy` nor `eval_strategy` exists in TrainingArguments.")
    return TrainingArguments(**kwargs)


def build_sft_trainer(
    model,
    tokenizer,
    train_ds: Dataset,
    eval_ds: Dataset,
    peft_cfg: LoraConfig,
    train_args: TrainingArguments,
    max_seq_length: int,
):
    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters

    kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "peft_config": peft_cfg,
        "args": train_args,
        "callbacks": [EvalLossCallback()],
    }
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        kwargs["processing_class"] = tokenizer

    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in params:
        kwargs["formatting_func"] = lambda x: x["text"]

    if "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq_length
    if "packing" in params:
        kwargs["packing"] = False
    return SFTTrainer(**kwargs)


def train_one_method(
    cfg: TrainConfig,
    job: MethodJob,
    eval_ds: Dataset,
    eval_id_set: set[str],
    eval_io_set: set[str],
    model_path: str,
    tokenizer,
) -> dict[str, Any]:
    if not job.dataset_path.exists():
        raise FileNotFoundError(f"{job.name} dataset not found: {job.dataset_path}")
    job.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(job.dataset_path)
    kept_rows, removed = filter_train_rows_by_eval(rows, eval_id_set, eval_io_set)
    if len(kept_rows) < 2:
        raise ValueError(f"{job.name} train set too small after removing eval overlap: {len(kept_rows)}")
    train_ds = to_text_dataset(kept_rows)

    logging.info(
        "[%s] train_raw=%d, removed_overlap=%d, train_final=%d, eval_shared=%d",
        job.name,
        len(rows),
        removed,
        len(train_ds),
        len(eval_ds),
    )

    model = load_causal_lm(model_path, get_dtype())
    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_args = build_training_args(cfg, job.output_dir)
    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        peft_cfg=peft_cfg,
        train_args=train_args,
        max_seq_length=cfg.max_seq_length,
    )

    trainer.train()
    final_metrics = trainer.evaluate()
    trainer.save_model(str(job.output_dir / "final_checkpoint"))
    with (job.output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    export_logs(trainer.state.log_history, job.output_dir)
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "method": job.name,
        "dataset_path": str(job.dataset_path),
        "output_dir": str(job.output_dir),
        "train_raw_size": len(rows),
        "removed_overlap": removed,
        "train_final_size": len(train_ds),
        "eval_size": len(eval_ds),
        "final_metrics": final_metrics,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = TrainConfig(
        ours_dataset_path=args.ours_dataset_path,
        kmeans_dataset_path=args.kmeans_dataset_path,
        random_dataset_path=args.random_dataset_path,
        category_dataset_path=args.category_dataset_path,
        ours_output_dir=args.ours_output_dir,
        kmeans_output_dir=args.kmeans_output_dir,
        random_output_dir=args.random_output_dir,
        category_output_dir=args.category_output_dir,
        shared_eval_jsonl=args.shared_eval_jsonl,
        eval_source_jsonl=args.eval_source_jsonl,
        eval_ratio=max(1e-4, min(0.5, args.eval_ratio)),
        seed=args.seed,
        output_root=args.output_root,
        base_model=args.base_model,
        model_source=args.model_source,
        modelscope_cache_dir=args.modelscope_cache_dir,
        max_seq_length=max(128, args.max_seq_length),
        learning_rate=args.learning_rate,
        train_batch_size=max(1, args.train_batch_size),
        eval_batch_size=max(1, args.eval_batch_size),
        gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
        num_train_epochs=max(0.1, args.num_train_epochs),
        weight_decay=max(0.0, args.weight_decay),
        warmup_ratio=max(0.0, args.warmup_ratio),
        logging_steps=max(1, args.logging_steps),
        eval_steps=max(1, args.eval_steps),
        save_steps=max(1, args.save_steps),
        lora_r=max(1, args.lora_r),
        lora_alpha=max(1, args.lora_alpha),
        lora_dropout=max(0.0, args.lora_dropout),
    )

    if cfg.category_dataset_path is not None and cfg.category_output_dir is None:
        raise ValueError("When --category-dataset-path is provided, --category-output-dir is required.")

    set_seed(cfg.seed)
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    shared_eval_path, eval_rows = build_or_load_shared_eval(
        shared_eval_jsonl=cfg.shared_eval_jsonl,
        eval_source_jsonl=cfg.eval_source_jsonl,
        eval_ratio=cfg.eval_ratio,
        seed=cfg.seed,
        output_root=cfg.output_root,
    )
    eval_ds = to_text_dataset(eval_rows)
    eval_id_set = set()
    eval_io_set = set()
    for i, r in enumerate(eval_rows):
        sid = sample_id(r, i)
        if sid is not None:
            eval_id_set.add(sid)
        eval_io_set.add(sample_io_signature(r))
    logging.info("Shared eval path=%s, size=%d", shared_eval_path, len(eval_ds))

    model_path = resolve_model_path(cfg.base_model, cfg.model_source, cfg.modelscope_cache_dir)
    logging.info("Model resolved from %s: %s", cfg.model_source, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    jobs = [
        MethodJob("ours", cfg.ours_dataset_path, cfg.ours_output_dir),
        MethodJob("kmeans", cfg.kmeans_dataset_path, cfg.kmeans_output_dir),
        MethodJob("random", cfg.random_dataset_path, cfg.random_output_dir),
    ]
    if cfg.category_dataset_path is not None and cfg.category_output_dir is not None:
        jobs.append(MethodJob("category", cfg.category_dataset_path, cfg.category_output_dir))

    summary: list[dict[str, Any]] = []
    for job in jobs:
        summary.append(
            train_one_method(
                cfg=cfg,
                job=job,
                eval_ds=eval_ds,
                eval_id_set=eval_id_set,
                eval_io_set=eval_io_set,
                model_path=model_path,
                tokenizer=tokenizer,
            )
        )

    summary_path = cfg.output_root / "shared_eval_train_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "shared_eval_path": str(shared_eval_path),
                "eval_size": len(eval_ds),
                "seed": cfg.seed,
                "eval_ratio": cfg.eval_ratio,
                "jobs": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logging.info("All runs finished. Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
