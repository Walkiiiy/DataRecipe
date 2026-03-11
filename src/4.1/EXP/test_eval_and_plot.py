"""4.1 EXP - 统一测试集评估与出图脚本。

功能：
1) 在同一份测试集上，评估三组 LoRA 模型（ours/kmeans/random）的 test loss 与 perplexity。
2) 导出结构化结果（JSON/CSV）。
3) 自动绘制最终测试性能柱状图（DPI=300），可直接用于论文对比。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from modelscope import snapshot_download
except Exception:  # noqa: BLE001
    snapshot_download = None


@dataclass
class EvalConfig:
    test_dataset_path: Path | None
    source_train_jsonl: Path | None
    test_ratio: float
    test_seed: int
    output_dir: Path
    base_model: str
    model_source: str
    modelscope_cache_dir: Path | None
    ours_adapter_dir: Path
    kmeans_adapter_dir: Path
    random_adapter_dir: Path
    batch_size: int
    max_seq_length: int
    dpi: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate three LoRA models on a shared test set and plot")
    parser.add_argument("--test_dataset_path", type=Path, default=None, help="统一测试集 JSONL（可选）")
    parser.add_argument(
        "--source_train_jsonl",
        type=Path,
        default=Path("data/alpaca-gpt4-data-en/train.jsonl"),
        help="当未提供 test_dataset_path 时，从该训练集切分测试集。",
    )
    parser.add_argument("--test_ratio", type=float, default=0.05, help="从训练集切分测试集比例。")
    parser.add_argument("--test_seed", type=int, default=42, help="训练集切分测试集随机种子。")
    parser.add_argument("--output_dir", type=Path, default=Path("data/alpaca-gpt4-data-en/exp/test_eval"))
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--model_source", type=str, choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--modelscope_cache_dir", type=Path, default=None)
    parser.add_argument("--ours_adapter_dir", type=Path, required=True, help="ours 模型 LoRA 目录")
    parser.add_argument("--kmeans_adapter_dir", type=Path, required=True, help="kmeans 模型 LoRA 目录")
    parser.add_argument("--random_adapter_dir", type=Path, required=True, help="random 模型 LoRA 目录")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def resolve_model_path(base_model: str, model_source: str, cache_dir: Path | None) -> str:
    if model_source == "hf":
        return base_model
    if snapshot_download is None:
        raise ImportError("ModelScope is not installed. Please install: pip install modelscope")
    return snapshot_download(
        model_id=base_model,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


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


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_with_lora(base_model_path: str, adapter_dir: Path, device: torch.device):
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True, dtype=get_dtype()
        )
    except TypeError:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True, torch_dtype=get_dtype()
        )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.to(device)
    model.eval()
    return model


def prepare_encoded_dataset(test_dataset_path: Path, tokenizer, max_seq_length: int):
    ds = load_dataset("json", data_files=str(test_dataset_path), split="train")
    if len(ds) == 0:
        raise ValueError("Test dataset is empty.")

    def _map_fn(x: dict[str, Any]) -> dict[str, Any]:
        text = build_prompt(x)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            return_attention_mask=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    ds = ds.map(_map_fn)
    cols = ds.column_names
    keep_cols = {"input_ids", "attention_mask"}
    drop_cols = [c for c in cols if c not in keep_cols]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    return ds


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_test_from_train(
    source_train_jsonl: Path,
    output_dir: Path,
    test_ratio: float,
    test_seed: int,
) -> Path:
    """从训练集切分共享测试集，并保存到输出目录。"""
    if not source_train_jsonl.exists():
        raise FileNotFoundError(f"Source train jsonl not found: {source_train_jsonl}")
    rows = load_jsonl(source_train_jsonl)
    if len(rows) < 2:
        raise ValueError("Source train data is too small to split test set.")

    rng = random.Random(test_seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_test = max(1, int(len(rows) * test_ratio))
    n_test = min(n_test, len(rows))
    test_rows = [rows[i] for i in idx[:n_test]]

    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / "auto_test_from_train.jsonl"
    with test_path.open("w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return test_path


def collate_fn_builder(tokenizer):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        padded = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )
        labels = padded["input_ids"].clone()
        labels[padded["attention_mask"] == 0] = -100
        padded["labels"] = labels
        return padded

    return _collate


@torch.no_grad()
def evaluate_loss(model, dataloader: DataLoader, device: torch.device) -> float:
    total_nll = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()
        target = labels[:, 1:].contiguous()

        # token-level NLL，按有效 token 求平均，确保不同 batch 长度可比
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        valid_tokens = int((target != -100).sum().item())
        total_nll += float(loss.item())
        total_tokens += valid_tokens

    if total_tokens == 0:
        return float("nan")
    return total_nll / total_tokens


def evaluate_one(
    name: str,
    base_model_path: str,
    adapter_dir: Path,
    tokenizer,
    encoded_ds,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model = load_model_with_lora(base_model_path, adapter_dir, device)
    dataloader = DataLoader(
        encoded_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_builder(tokenizer),
    )
    test_loss = evaluate_loss(model, dataloader, device)
    perplexity = float(math.exp(test_loss)) if not math.isnan(test_loss) else float("nan")
    return {
        "method": name,
        "adapter_dir": str(adapter_dir),
        "test_loss": test_loss,
        "perplexity": perplexity,
    }


def save_metrics(metrics: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "test_metrics.json"
    csv_path = out_dir / "test_metrics.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "adapter_dir", "test_loss", "perplexity"])
        writer.writeheader()
        writer.writerows(metrics)
    logging.info("Saved metrics json: %s", json_path)
    logging.info("Saved metrics csv: %s", csv_path)


def plot_bars(metrics: list[dict[str, Any]], out_dir: Path, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [m["method"] for m in metrics]
    loss_vals = [float(m["test_loss"]) for m in metrics]
    ppl_vals = [float(m["perplexity"]) for m in metrics]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # 左图：test loss
    sns.barplot(x=labels, y=loss_vals, ax=axes[0], palette=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0].set_title("Final Test Loss")
    axes[0].set_xlabel("Sampling Strategy")
    axes[0].set_ylabel("Test Loss")
    for i, v in enumerate(loss_vals):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # 右图：perplexity
    sns.barplot(x=labels, y=ppl_vals, ax=axes[1], palette=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].set_title("Final Test Perplexity")
    axes[1].set_xlabel("Sampling Strategy")
    axes[1].set_ylabel("Perplexity")
    for i, v in enumerate(ppl_vals):
        axes[1].text(i, v + 0.05, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig_path = out_dir / "test_performance_bar.png"
    plt.savefig(fig_path, dpi=dpi)
    plt.close()
    logging.info("Saved figure: %s", fig_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = EvalConfig(
        test_dataset_path=args.test_dataset_path,
        source_train_jsonl=args.source_train_jsonl,
        test_ratio=max(1e-4, min(0.5, args.test_ratio)),
        test_seed=args.test_seed,
        output_dir=args.output_dir,
        base_model=args.base_model,
        model_source=args.model_source,
        modelscope_cache_dir=args.modelscope_cache_dir,
        ours_adapter_dir=args.ours_adapter_dir,
        kmeans_adapter_dir=args.kmeans_adapter_dir,
        random_adapter_dir=args.random_adapter_dir,
        batch_size=max(1, args.batch_size),
        max_seq_length=max(128, args.max_seq_length),
        dpi=max(100, args.dpi),
    )

    if cfg.test_dataset_path is None:
        if cfg.source_train_jsonl is None:
            raise ValueError("Either --test_dataset_path or --source_train_jsonl must be provided.")
        cfg.test_dataset_path = build_test_from_train(
            source_train_jsonl=cfg.source_train_jsonl,
            output_dir=cfg.output_dir,
            test_ratio=cfg.test_ratio,
            test_seed=cfg.test_seed,
        )
        logging.info("Auto-split test set from train: %s", cfg.test_dataset_path)
    elif not cfg.test_dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {cfg.test_dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = resolve_model_path(cfg.base_model, cfg.model_source, cfg.modelscope_cache_dir)
    logging.info("Base model resolved from %s: %s", cfg.model_source, base_model_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_ds = prepare_encoded_dataset(cfg.test_dataset_path, tokenizer, cfg.max_seq_length)
    logging.info("Loaded test samples: %d", len(encoded_ds))

    metrics = []
    metrics.append(
        evaluate_one(
            name="Ours",
            base_model_path=base_model_path,
            adapter_dir=cfg.ours_adapter_dir,
            tokenizer=tokenizer,
            encoded_ds=encoded_ds,
            batch_size=cfg.batch_size,
            device=device,
        )
    )
    metrics.append(
        evaluate_one(
            name="KMeans",
            base_model_path=base_model_path,
            adapter_dir=cfg.kmeans_adapter_dir,
            tokenizer=tokenizer,
            encoded_ds=encoded_ds,
            batch_size=cfg.batch_size,
            device=device,
        )
    )
    metrics.append(
        evaluate_one(
            name="Random",
            base_model_path=base_model_path,
            adapter_dir=cfg.random_adapter_dir,
            tokenizer=tokenizer,
            encoded_ds=encoded_ds,
            batch_size=cfg.batch_size,
            device=device,
        )
    )

    save_metrics(metrics, cfg.output_dir)
    plot_bars(metrics, cfg.output_dir, cfg.dpi)
    with (cfg.output_dir / "test_split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "test_dataset_path": str(cfg.test_dataset_path),
                "source_train_jsonl": str(cfg.source_train_jsonl) if cfg.source_train_jsonl else None,
                "test_ratio": cfg.test_ratio,
                "test_seed": cfg.test_seed,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logging.info("Test evaluation finished.")


if __name__ == "__main__":
    main()
