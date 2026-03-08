"""Train on scored FLAN data and record alpha (4-metric) trajectory.

Features:
- Train on `niv2_capability_data_random1000_alphagasus_scored` style jsonl.
- Response-only loss masking (prompt tokens masked to -100).
- Maintain dynamic alpha weights for 4 scored dimensions.
- Record only alpha/loss/lr per optimizer step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root importability when run via absolute path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on scored data with alpha trajectory logging")

    parser.add_argument(
        "--train_dir",
        type=str,
        default="/home/walkiiiy/DataRecipe/data/flan/niv2_capability_data_random1000_alphagasus_scored",
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/walkiiiy/DataRecipe/model/recipe_scored_run",
    )

    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.8, help="MWU update scale for alpha.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Entropy regularization for alpha.")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_jsonl_files(root: str) -> List[str]:
    files = sorted(str(p) for p in Path(root).rglob("*.jsonl") if p.name != "errors.jsonl")
    if not files:
        raise ValueError(f"No jsonl files found under: {root}")
    return files


def build_prompt(instruction: str, inp: str) -> str:
    return f"Instruction:\n{instruction}\n\nInput:\n{inp}\n\nResponse:\n"


@dataclass
class PackedSample:
    input_ids: List[int]
    labels: List[int]
    metrics: List[float]


class ScoredResponseOnlyDataset(IterableDataset):
    """Stream scored jsonl data and build response-only labels."""

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

        for i in range(response_start, len(input_ids)):
            labels[i] = input_ids[i]

        input_ids.append(self.eos_id)
        labels.append(self.eos_id)

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        if all(x == -100 for x in labels):
            return None

        metrics = []
        for k in ("相关度", "准确性", "能力多样性", "难度"):
            try:
                metrics.append(float(ex.get(k, "")))
            except Exception:
                return None

        return PackedSample(input_ids=input_ids, labels=labels, metrics=metrics)

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

    input_ids, labels, attention_mask, metrics = [], [], [], []
    for x in batch:
        cur = len(x.input_ids)
        pad = max_len - cur
        input_ids.append(x.input_ids + [pad_id] * pad)
        labels.append(x.labels + [-100] * pad)
        attention_mask.append([1] * cur + [0] * pad)
        metrics.append(x.metrics)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "metrics": torch.tensor(metrics, dtype=torch.float32),
    }


class LinearWarmupDecay:
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

        for g in self.optimizer.param_groups:
            g["lr"] = g["initial_lr"] * scale


def estimate_steps(files: List[str], batch_size: int, grad_accum: int, epochs: int, max_steps: Optional[int]) -> int:
    line_count = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for _ in f:
                line_count += 1
    micro = math.ceil(line_count / max(1, batch_size))
    opt_per_epoch = math.ceil(micro / max(1, grad_accum))
    total = max(1, opt_per_epoch * max(1, epochs))
    if max_steps is not None:
        total = min(total, max_steps)
    return total


class AlphaTrajectoryRecorder:
    """Persist alpha/loss/lr step metadata."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.traj_dir = self.output_dir / "alpha_trajectory"
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.traj_dir / "trajectory_meta.jsonl"
        self.csv_path = self.traj_dir / "trajectory_meta.csv"
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step", "loss", "lr", "tag", "alpha_0", "alpha_1", "alpha_2", "alpha_3"],
            )
            writer.writeheader()

    def save(self, step: int, loss: float, lr: float, alpha: torch.Tensor, tag: str = "step") -> None:
        alpha_vals = alpha.detach().cpu().tolist()
        row = {
            "step": int(step),
            "loss": float(loss),
            "lr": float(lr),
            "tag": str(tag),
            "alpha_0": float(alpha_vals[0]),
            "alpha_1": float(alpha_vals[1]),
            "alpha_2": float(alpha_vals[2]),
            "alpha_3": float(alpha_vals[3]),
        }
        with self.meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step", "loss", "lr", "tag", "alpha_0", "alpha_1", "alpha_2", "alpha_3"],
            )
            writer.writerow(row)


def per_sample_response_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sample response-only CE loss.

    logits: (B, T, V), labels: (B, T) with -100 masked tokens.
    returns: (B,)
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    tok_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size(0), shift_labels.size(1))
    valid = (shift_labels != -100).to(tok_loss.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (tok_loss * valid).sum(dim=1) / denom


def update_alpha(alpha: torch.Tensor, rewards: torch.Tensor, batch_metrics: torch.Tensor, gamma: float, epsilon: float) -> torch.Tensor:
    """Batch-smoothed MWU alpha update."""
    utilities = (rewards.unsqueeze(1) * batch_metrics).mean(dim=0)
    alpha_tilde = alpha * torch.exp(gamma * utilities)
    alpha_norm = alpha_tilde / alpha_tilde.sum().clamp_min(1e-12)
    return (1.0 - epsilon) * alpha_norm + epsilon / 4.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] device={device}", flush=True)

    files = list_jsonl_files(args.train_dir)
    print(f"[Init] train_files={len(files)} train_dir={args.train_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)

    dataset = ScoredResponseOnlyDataset(
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    for g in optimizer.param_groups:
        g["initial_lr"] = args.learning_rate

    total_steps = estimate_steps(
        files=files,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum_steps,
        epochs=args.num_epochs,
        max_steps=args.max_steps,
    )
    scheduler = LinearWarmupDecay(optimizer=optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps)

    use_amp = torch.cuda.is_available() and (args.fp16 or args.bf16)
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(torch.cuda.is_available() and args.fp16))

    recorder = AlphaTrajectoryRecorder(args.output_dir)
    alpha = torch.full((4,), 0.25, dtype=torch.float32, device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Save initial alpha (step 0)
    recorder.save(step=0, loss=float("nan"), lr=optimizer.param_groups[0]["lr"], alpha=alpha, tag="init")
    print("[Init] saved initial alpha snapshot", flush=True)

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    reward_buffer: List[torch.Tensor] = []
    metric_buffer: List[torch.Tensor] = []

    for epoch in range(args.num_epochs):
        print(f"[Epoch] {epoch+1}/{args.num_epochs} start", flush=True)
        for batch in loader:
            micro_step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch_metrics = batch.pop("metrics")

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            with torch.no_grad():
                sample_loss = per_sample_response_loss(outputs.logits.detach(), batch["labels"])
                rewards = torch.exp(-sample_loss).clamp_min(1e-8)
                reward_buffer.append(rewards)
                metric_buffer.append(batch_metrics.detach())

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

                cur_lr = optimizer.param_groups[0]["lr"]
                cur_loss = running_loss / max(1, args.log_every)

                if reward_buffer:
                    batch_rewards = torch.cat(reward_buffer, dim=0)
                    batch_metrics_all = torch.cat(metric_buffer, dim=0)
                    alpha = update_alpha(
                        alpha=alpha,
                        rewards=batch_rewards,
                        batch_metrics=batch_metrics_all,
                        gamma=args.gamma,
                        epsilon=args.epsilon,
                    )
                    reward_buffer.clear()
                    metric_buffer.clear()

                recorder.save(step=global_step, loss=cur_loss, lr=cur_lr, alpha=alpha, tag="step")

                if global_step % args.log_every == 0:
                    print(
                        f"[Train] epoch={epoch+1} step={global_step}/{total_steps} "
                        f"loss={cur_loss:.6f} lr={cur_lr:.8f} alpha={[round(x, 6) for x in alpha.detach().cpu().tolist()]}",
                        flush=True,
                    )
                    running_loss = 0.0

                if args.max_steps is not None and global_step >= args.max_steps:
                    print(f"[Stop] reached max_steps={args.max_steps}", flush=True)
                    break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[Done] model saved to {args.output_dir}", flush=True)
    print(f"[Done] alpha trajectory dir: {Path(args.output_dir) / 'alpha_trajectory'}", flush=True)


if __name__ == "__main__":
    main()
