"""4.1 EXP - 多数据集共享验证集 LoRA 训练脚本（通用版）。

特点：
1) 不写死对比方法数量与名称；通过可重复参数 --run 动态指定。
2) 所有 run 共享同一份 eval 数据，保证验证口径一致。
3) 自动从每个训练集剔除与共享 eval 重叠样本（按 id，回退 instruction+input+output）。

--run 格式：
  --run <name>::<dataset_jsonl>::<output_dir>
示例：
  --run ours::data/dolly-15k/exp/dataset_ours.jsonl::data/dolly-15k/exp/run_ours_shared_eval
  --run kmeans::data/dolly-15k/exp/dataset_kmeans.jsonl::data/dolly-15k/exp/run_kmeans_shared_eval
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
from peft import LoraConfig, PeftModel
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

RESPONSE_MARKER = "### Response:\n"


@dataclass
class MethodJob:
    name: str
    dataset_path: Path
    output_dir: Path


@dataclass
class TrainConfig:
    runs: list[MethodJob]
    shared_eval_jsonl: Path | None
    eval_source_jsonl: Path | None
    eval_ratio: float
    seed: int
    output_root: Path
    base_model: str
    model_source: str
    modelscope_cache_dir: Path | None
    init_adapter_checkpoint: Path | None
    max_seq_length: int
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    eval_steps: int | None
    eval_steps_mode: str
    target_eval_count: int
    min_eval_steps: int
    save_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float


def parse_run_item(text: str) -> MethodJob:
    parts = text.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --run format: {text}. Expected <name>::<dataset_jsonl>::<output_dir>"
        )
    name = parts[0].strip()
    dataset_path = Path(parts[1].strip())
    output_dir = Path(parts[2].strip())
    if not name:
        raise ValueError(f"Invalid --run name: {text}")
    return MethodJob(name=name, dataset_path=dataset_path, output_dir=output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiple LoRA runs with one shared eval set")
    parser.add_argument(
        "--run",
        type=str,
        action="append",
        required=True,
        help="可重复参数：<name>::<dataset_jsonl>::<output_dir>",
    )
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
    parser.add_argument("--output-root", type=Path, default=Path("data/exp/shared_eval_runs"))
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model_source", type=str, choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--modelscope_cache_dir", type=Path, default=None)
    parser.add_argument(
        "--init_adapter_checkpoint",
        type=Path,
        default=None,
        help="可选：LoRA adapter checkpoint。若提供，则在 base model 上加载该 adapter 并继续训练。",
    )
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="手动指定 eval 间隔步数；若不传则按数据集大小自动计算。",
    )
    parser.add_argument(
        "--eval-steps-mode",
        type=str,
        choices=["auto", "manual"],
        default="auto",
        help="eval_steps 模式：auto=按数据集规模自动计算；manual=使用 --eval_steps。",
    )
    parser.add_argument(
        "--target-eval-count",
        type=int,
        default=12,
        help="auto 模式下，目标评估次数（全训练过程）。",
    )
    parser.add_argument(
        "--min-eval-steps",
        type=int,
        default=10,
        help="auto 模式下 eval_steps 的最小值。",
    )
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


def _find_subsequence(seq: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(seq):
        return -1
    end = len(seq) - len(pattern) + 1
    for i in range(end):
        if seq[i : i + len(pattern)] == pattern:
            return i
    return -1


class ResponseOnlyCollator:
    """只对 Response 段计算 labels（prompt 段全部 -100）。"""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        variants = [RESPONSE_MARKER, f"\n{RESPONSE_MARKER}", f"\n\n{RESPONSE_MARKER}"]
        self.marker_token_variants: list[list[int]] = []
        for text in variants:
            marker_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if marker_ids and marker_ids not in self.marker_token_variants:
                self.marker_token_variants.append(marker_ids)
        if not self.marker_token_variants:
            raise ValueError("Failed to build response marker token ids.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        core_features = []
        for f in features:
            item: dict[str, Any] = {"input_ids": f["input_ids"]}
            # Some TRL/Transformers versions pass only input_ids into data_collator.
            if "attention_mask" in f and f["attention_mask"] is not None:
                item["attention_mask"] = f["attention_mask"]
            core_features.append(item)
        batch = self.tokenizer.pad(
            core_features,
            padding=True,
            return_tensors="pt",
        )
        if "attention_mask" not in batch:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                raise ValueError("Tokenizer has no pad_token_id and attention_mask is missing.")
            batch["attention_mask"] = (batch["input_ids"] != pad_id).long()
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for i in range(labels.size(0)):
            seq_len = int(batch["attention_mask"][i].sum().item())
            seq = batch["input_ids"][i, :seq_len].tolist()
            response_start = -1
            for marker in self.marker_token_variants:
                pos = _find_subsequence(seq, marker)
                if pos != -1:
                    response_start = pos + len(marker)
                    break
            if response_start == -1:
                # 异常样本：找不到 marker 时不参与 loss。
                labels[i, :seq_len] = -100
                continue
            labels[i, :response_start] = -100

        batch["labels"] = labels
        return batch


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
    eval_steps = 1 if cfg.eval_steps is None else max(1, cfg.eval_steps)
    save_steps = max(1, cfg.save_steps)
    if save_steps % eval_steps != 0:
        logging.warning(
            "save_steps=%d is not a multiple of eval_steps=%d; override save_steps to eval_steps "
            "to support reliable best-checkpoint tracking.",
            save_steps,
            eval_steps,
        )
        save_steps = eval_steps

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
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "report_to": [],
        "fp16": (get_dtype() == torch.float16),
        "bf16": (get_dtype() == torch.bfloat16),
        "gradient_checkpointing": True,
        "seed": cfg.seed,
        "data_seed": cfg.seed,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
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
    peft_cfg: LoraConfig | None,
    train_args: TrainingArguments,
    max_seq_length: int,
):
    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters
    response_only_collator = ResponseOnlyCollator(tokenizer)

    kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": train_args,
        "callbacks": [EvalLossCallback()],
    }
    if peft_cfg is not None:
        kwargs["peft_config"] = peft_cfg
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
    if "data_collator" in params:
        kwargs["data_collator"] = response_only_collator
    else:
        raise RuntimeError("SFTTrainer has no `data_collator` parameter, cannot enforce response-only loss.")
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
    # 按当前 run 的训练集规模自动计算 eval_steps，保证不同规模数据集评估频率可比。
    if cfg.eval_steps_mode == "manual":
        if cfg.eval_steps is None:
            raise ValueError("eval_steps_mode=manual requires --eval_steps.")
        resolved_eval_steps = max(1, cfg.eval_steps)
    else:
        effective_batch = max(1, cfg.train_batch_size * cfg.gradient_accumulation_steps)
        steps_per_epoch = max(1, (len(train_ds) + effective_batch - 1) // effective_batch)
        total_steps = max(1, int(round(steps_per_epoch * cfg.num_train_epochs)))
        target = max(1, cfg.target_eval_count)
        resolved_eval_steps = max(1, total_steps // target)
        resolved_eval_steps = max(cfg.min_eval_steps, resolved_eval_steps)
        resolved_eval_steps = min(resolved_eval_steps, total_steps)

    logging.info(
        "[%s] train_raw=%d, removed_overlap=%d, train_final=%d, eval_shared=%d, eval_steps=%d",
        job.name,
        len(rows),
        removed,
        len(train_ds),
        len(eval_ds),
        resolved_eval_steps,
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
    trainer_peft_cfg: LoraConfig | None = peft_cfg
    resumed_from_adapter: str | None = None
    if cfg.init_adapter_checkpoint is not None:
        adapter_path = cfg.init_adapter_checkpoint
        if not adapter_path.exists():
            raise FileNotFoundError(f"init_adapter_checkpoint not found: {adapter_path}")
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"init_adapter_checkpoint is not a valid LoRA adapter directory: {adapter_path}"
            )
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        trainer_peft_cfg = None
        resumed_from_adapter = str(adapter_path)
        logging.info("[%s] Resume LoRA adapter from: %s", job.name, adapter_path)

    cfg_for_run = TrainConfig(**{**cfg.__dict__, "eval_steps": resolved_eval_steps})
    train_args = build_training_args(cfg_for_run, job.output_dir)
    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        eval_ds=eval_ds,
        peft_cfg=trainer_peft_cfg,
        train_args=train_args,
        max_seq_length=cfg.max_seq_length,
    )

    trainer.train()
    best_ckpt = trainer.state.best_model_checkpoint
    best_eval_loss = trainer.state.best_metric
    if best_ckpt:
        logging.info("[%s] best checkpoint=%s (best eval_loss=%s)", job.name, best_ckpt, best_eval_loss)
    else:
        logging.warning("[%s] trainer did not record a best checkpoint.", job.name)

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
        "resolved_eval_steps": resolved_eval_steps,
        "best_checkpoint": best_ckpt,
        "best_eval_loss": best_eval_loss,
        "resumed_from_adapter": resumed_from_adapter,
        "final_metrics": final_metrics,
    }


def parse_runs(run_items: list[str]) -> list[MethodJob]:
    jobs = [parse_run_item(x) for x in run_items]
    # if len(jobs) < 2:
    #     raise ValueError("At least 2 runs are required for comparison.")
    seen = set()
    for j in jobs:
        if j.name in seen:
            raise ValueError(f"Duplicate run name: {j.name}")
        seen.add(j.name)
    return jobs


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    runs = parse_runs(args.run)

    cfg = TrainConfig(
        runs=runs,
        shared_eval_jsonl=args.shared_eval_jsonl,
        eval_source_jsonl=args.eval_source_jsonl,
        eval_ratio=max(1e-4, min(0.5, args.eval_ratio)),
        seed=args.seed,
        output_root=args.output_root,
        base_model=args.base_model,
        model_source=args.model_source,
        modelscope_cache_dir=args.modelscope_cache_dir,
        init_adapter_checkpoint=args.init_adapter_checkpoint,
        max_seq_length=max(128, args.max_seq_length),
        learning_rate=args.learning_rate,
        train_batch_size=max(1, args.train_batch_size),
        eval_batch_size=max(1, args.eval_batch_size),
        gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
        num_train_epochs=max(0.1, args.num_train_epochs),
        weight_decay=max(0.0, args.weight_decay),
        warmup_ratio=max(0.0, args.warmup_ratio),
        logging_steps=max(1, args.logging_steps),
        eval_steps=args.eval_steps,
        eval_steps_mode=args.eval_steps_mode,
        target_eval_count=max(1, args.target_eval_count),
        min_eval_steps=max(1, args.min_eval_steps),
        save_steps=max(1, args.save_steps),
        lora_r=max(1, args.lora_r),
        lora_alpha=max(1, args.lora_alpha),
        lora_dropout=max(0.0, args.lora_dropout),
    )

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
    for i, row in enumerate(eval_rows):
        sid = sample_id(row, i)
        if sid is not None:
            eval_id_set.add(sid)
        eval_io_set.add(sample_io_signature(row))

    logging.info("Shared eval path=%s, size=%d", shared_eval_path, len(eval_ds))
    logging.info("Runs=%s", [r.name for r in cfg.runs])
    if cfg.init_adapter_checkpoint is not None:
        logging.info("Init adapter checkpoint=%s", cfg.init_adapter_checkpoint)

    model_path = resolve_model_path(cfg.base_model, cfg.model_source, cfg.modelscope_cache_dir)
    logging.info("Model resolved from %s: %s", cfg.model_source, model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    summary: list[dict[str, Any]] = []
    for job in cfg.runs:
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
                "eval_steps_mode": cfg.eval_steps_mode,
                "manual_eval_steps": cfg.eval_steps,
                "target_eval_count": cfg.target_eval_count,
                "min_eval_steps": cfg.min_eval_steps,
                "runs": [r.name for r in cfg.runs],
                "init_adapter_checkpoint": str(cfg.init_adapter_checkpoint) if cfg.init_adapter_checkpoint else None,
                "jobs": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logging.info("All runs finished. Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
