"""Minimal PyTorch SFT script for FLAN-style jsonl data.

Key behavior:
- Read all jsonl files under a train directory.
- Build text as: Instruction + Input + Response.
- Compute loss ONLY on response tokens (prompt tokens are label-masked as -100).

This script intentionally avoids high-level trainer wrappers to keep the training
loop easy to modify.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Response-only SFT on FLAN jsonl")

    parser.add_argument(
        "--train_dir",
        type=str,
        default="/home/walkiiiy/DataRecipe/data/flan/train/niv2_capability_data",
        help="Directory containing train jsonl files.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Base CausalLM model path or HF model id.",
    )
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_jsonl_files(root: str) -> List[str]:
    files = sorted(str(p) for p in Path(root).rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No jsonl files found under: {root}")
    return files


def build_prompt(instruction: str, inp: str) -> str:
    return f"Instruction:\n{instruction}\n\nInput:\n{inp}\n\nResponse:\n"


@dataclass
class PackedSample:
    input_ids: List[int]
    labels: List[int]


class FlanResponseOnlyDataset(IterableDataset):
    """Stream FLAN jsonl and mask labels before response."""

    def __init__(
        self,
        files: List[str],
        tokenizer: AutoTokenizer,
        max_seq_len: int,
        shuffle_files_each_epoch: bool = True,
    ) -> None:
        super().__init__()
        self.files = files
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_files_each_epoch = shuffle_files_each_epoch

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("Tokenizer must have eos_token_id.")

    def _encode_one(self, ex: Dict) -> Optional[PackedSample]:
        instruction = str(ex.get("instruction", ""))
        inp = str(ex.get("input", ""))
        response = str(ex.get("response", ""))

        if not response.strip():
            return None

        prefix_text = build_prompt(instruction, inp)
        full_text = prefix_text + response

        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        # [BOS] full_ids [EOS], but compute loss only on response (plus EOS).
        input_ids: List[int] = []
        labels: List[int] = []

        if self.bos_id is not None:
            input_ids.append(self.bos_id)
            labels.append(-100)

        input_ids.extend(full_ids)

        response_start = (1 if self.bos_id is not None else 0) + len(prefix_ids)
        labels.extend([-100] * len(full_ids))

        if response_start >= len(input_ids):
            return None

        # Unmask response tokens only.
        for i in range(response_start, len(input_ids)):
            labels[i] = input_ids[i]

        # Add EOS and train on EOS as response continuation.
        input_ids.append(self.eos_id)
        labels.append(self.eos_id)

        # Truncate.
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        # If all labels are masked after truncation, skip.
        if all(x == -100 for x in labels):
            return None

        return PackedSample(input_ids=input_ids, labels=labels)

    def __iter__(self) -> Iterator[PackedSample]:
        files = list(self.files)
        if self.shuffle_files_each_epoch:
            random.shuffle(files)

        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    packed = self._encode_one(ex)
                    if packed is not None:
                        yield packed


def collate_fn(batch: List[PackedSample], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x.input_ids) for x in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for x in batch:
        cur_len = len(x.input_ids)
        pad_len = max_len - cur_len

        input_ids.append(x.input_ids + [pad_id] * pad_len)
        labels.append(x.labels + [-100] * pad_len)
        attention_mask.append([1] * cur_len + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class LinearWarmupDecay:
    """Simple LR scheduler: warmup then linear decay to 0."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.step_idx = 0

    def step(self) -> None:
        self.step_idx += 1
        if self.step_idx <= self.warmup_steps:
            scale = self.step_idx / float(self.warmup_steps)
        else:
            remain = self.total_steps - self.step_idx
            decay_steps = max(1, self.total_steps - self.warmup_steps)
            scale = max(0.0, remain / float(decay_steps))

        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * scale


def save_checkpoint(model, tokenizer, output_dir: str, step: int) -> None:
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)


def estimate_steps(files: List[str], batch_size: int, grad_accum_steps: int, num_epochs: int) -> int:
    # Lightweight estimate by line count.
    line_count = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for _ in f:
                line_count += 1
    micro_steps = math.ceil(line_count / max(1, batch_size))
    opt_steps_per_epoch = math.ceil(micro_steps / max(1, grad_accum_steps))
    return max(1, opt_steps_per_epoch * max(1, num_epochs))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)

    files = list_jsonl_files(args.train_dir)
    dataset = FlanResponseOnlyDataset(
        files=files,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        shuffle_files_each_epoch=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = args.learning_rate

    total_steps = estimate_steps(
        files=files,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
    )
    scheduler = LinearWarmupDecay(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    use_amp = torch.cuda.is_available() and (args.fp16 or args.bf16)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(torch.cuda.is_available() and args.fp16))

    model.train()
    global_step = 0
    micro_step = 0
    running_loss = 0.0

    for epoch in range(args.num_epochs):
        for batch in loader:
            micro_step += 1

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(**batch)
                loss = out.loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.grad_accum_steps

            if micro_step % args.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg_loss = running_loss / float(args.log_every)
                    lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"epoch={epoch} step={global_step}/{total_steps} "
                        f"loss={avg_loss:.4f} lr={lr:.8f}",
                        flush=True,
                    )
                    running_loss = 0.0

                if global_step % args.save_every == 0:
                    save_checkpoint(model, tokenizer, args.output_dir, global_step)

                if global_step >= total_steps:
                    break

        if global_step >= total_steps:
            break

    # Final save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training finished. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
