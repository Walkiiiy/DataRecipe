"""4.1 EXP - 统一变量 LoRA 指令微调脚本（SFTTrainer）。

设计目标：
1) 只允许数据集路径不同，其它训练超参数保持一致，用于严格对比。
2) 使用 transformers + peft + trl 的标准 SFT 流程。
3) 在训练中持续记录 training/eval loss，并导出 JSON/CSV 便于论文画图。
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
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
class TrainConfig:
    dataset_path: Path
    output_dir: Path
    base_model: str
    model_source: str
    modelscope_cache_dir: Path | None
    val_ratio: float
    seed: int
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
    parser = argparse.ArgumentParser(description="LoRA SFT training under controlled hyper-parameters")
    parser.add_argument("--dataset_path", type=Path, required=True, help="输入训练集 JSONL 路径")
    parser.add_argument("--output_dir", type=Path, required=True, help="实验输出目录")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--model_source",
        type=str,
        choices=["modelscope", "hf"],
        default="modelscope",
        help="模型来源：默认从 ModelScope 下载。",
    )
    parser.add_argument(
        "--modelscope_cache_dir",
        type=Path,
        default=None,
        help="ModelScope 本地缓存目录（可选）。",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
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


def build_prompt(row: dict[str, Any]) -> str:
    """统一指令格式模板，保证三组数据训练输入形式完全一致。"""
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


def prepare_dataset(dataset_path: Path, val_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    if len(ds) < 2:
        raise ValueError("Dataset is too small for train/val split.")

    split = ds.train_test_split(test_size=val_ratio, seed=seed, shuffle=True)
    train_ds = split["train"].map(lambda x: {"text": build_prompt(x)})
    eval_ds = split["test"].map(lambda x: {"text": build_prompt(x)})
    return train_ds, eval_ds


def get_dtype() -> torch.dtype:
    # 单卡场景优先 bf16，其次 fp16，最后 fp32
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_model_path(base_model: str, model_source: str, cache_dir: Path | None) -> str:
    """根据来源解析模型路径。modelscope 时先下载到本地再返回本地路径。"""
    if model_source == "hf":
        return base_model
    if snapshot_download is None:
        raise ImportError("ModelScope is not installed. Please install: pip install modelscope")
    local_dir = snapshot_download(
        model_id=base_model,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return str(local_dir)


def build_training_args(cfg: TrainConfig) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(cfg.output_dir),
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
    # 兼容不同 transformers 版本：evaluation_strategy / eval_strategy
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        raise RuntimeError("Neither `evaluation_strategy` nor `eval_strategy` exists in TrainingArguments.")
    return TrainingArguments(**kwargs)


def load_causal_lm(model_path: str, dtype: torch.dtype):
    """兼容不同 transformers 版本：优先 dtype，回退 torch_dtype。"""
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
    """用于在日志里显式强调 eval_loss 记录（SFTTrainer 本身会写 state.log_history）。"""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: D401
        if metrics and "eval_loss" in metrics:
            logging.info("Eval step=%s eval_loss=%.6f", state.global_step, float(metrics["eval_loss"]))


def export_logs(log_history: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_json = out_dir / "log_history.json"
    with raw_json.open("w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    # 结构化导出，便于统一画图
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
    logging.info("Saved log json: %s", raw_json)
    logging.info("Saved log csv: %s", csv_path)


def build_sft_trainer(
    model,
    tokenizer,
    train_ds: Dataset,
    eval_ds: Dataset,
    peft_cfg: LoraConfig,
    train_args: TrainingArguments,
    max_seq_length: int,
):
    """按 trl 版本签名自适配构造 SFTTrainer。"""
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

    # tokenizer 在不同版本中可能叫 tokenizer 或 processing_class
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        kwargs["processing_class"] = tokenizer

    # 文本字段配置在不同版本中差异较大
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in params:
        kwargs["formatting_func"] = lambda x: x["text"]

    if "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq_length
    if "packing" in params:
        kwargs["packing"] = False

    return SFTTrainer(**kwargs)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = TrainConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        model_source=args.model_source,
        modelscope_cache_dir=args.modelscope_cache_dir,
        val_ratio=max(1e-4, min(0.5, args.val_ratio)),
        seed=args.seed,
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

    if not cfg.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.dataset_path}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    train_ds, eval_ds = prepare_dataset(cfg.dataset_path, cfg.val_ratio, cfg.seed)
    logging.info("Train size=%d, Eval size=%d", len(train_ds), len(eval_ds))

    model_path = resolve_model_path(cfg.base_model, cfg.model_source, cfg.modelscope_cache_dir)
    logging.info("Model resolved from %s: %s", cfg.model_source, model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_lm(model_path, get_dtype())

    peft_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_args = build_training_args(cfg)

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
    trainer.save_model(str(cfg.output_dir / "final_checkpoint"))

    with (cfg.output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    export_logs(trainer.state.log_history, cfg.output_dir)
    logging.info("Training done. Final metrics saved to %s", cfg.output_dir / "final_metrics.json")


if __name__ == "__main__":
    main()
