"""Train on scored FLAN data and record full weight trajectory.

Features:
- Train on `niv2_capability_data_random1000_alphagasus_scored` style jsonl.
- Response-only loss masking (prompt tokens masked to -100).
- Record full model weights over time by saving state_dict every optimizer step.

Note: saving full state_dict each step is storage-intensive by design.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure project root importability when run via absolute path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on scored data with full weight trajectory logging")

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

    # Trajectory controls
    parser.add_argument(
        "--record_every_step",
        action="store_true",
        help="If set, save full state_dict at every optimizer step.",
    )
    parser.add_argument(
        "--save_weight_every",
        type=int,
        default=1,
        help="Save full state_dict every N optimizer steps (used when record_every_step is not set).",
    )

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

    input_ids, labels, attention_mask = [], [], []
    for x in batch:
        cur = len(x.input_ids)
        pad = max_len - cur
        input_ids.append(x.input_ids + [pad_id] * pad)
        labels.append(x.labels + [-100] * pad)
        attention_mask.append([1] * cur + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
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


class WeightTrajectoryRecorder:
    """Persist full state_dict snapshots and step metadata."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.traj_dir = self.output_dir / "weight_trajectory"
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.traj_dir / "trajectory_meta.jsonl"

    def save(self, model: torch.nn.Module, step: int, loss: float, lr: float, tag: str = "step") -> None:
        ckpt_path = self.traj_dir / f"{tag}_{step:08d}.pt"
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(state, ckpt_path)
        with self.meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "loss": loss, "lr": lr, "file": ckpt_path.name, "tag": tag}) + "\n")


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

    recorder = WeightTrajectoryRecorder(args.output_dir)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Save initial weights (step 0)
    recorder.save(model, step=0, loss=float("nan"), lr=optimizer.param_groups[0]["lr"], tag="init")
    print("[Init] saved initial weight snapshot", flush=True)

    global_step = 0
    micro_step = 0
    running_loss = 0.0

    for epoch in range(args.num_epochs):
        print(f"[Epoch] {epoch+1}/{args.num_epochs} start", flush=True)
        for batch in loader:
            micro_step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum_steps

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

                cur_lr = optimizer.param_groups[0]["lr"]
                cur_loss = running_loss / max(1, args.log_every)

                # Record full weights each step (or every N steps).
                save_now = args.record_every_step or (global_step % max(1, args.save_weight_every) == 0)
                if save_now:
                    recorder.save(model, step=global_step, loss=cur_loss, lr=cur_lr, tag="step")

                if global_step % args.log_every == 0:
                    print(
                        f"[Train] epoch={epoch+1} step={global_step}/{total_steps} "
                        f"loss={cur_loss:.6f} lr={cur_lr:.8f}",
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
    print(f"[Done] weight trajectory dir: {Path(args.output_dir) / 'weight_trajectory'}", flush=True)


if __name__ == "__main__":
    main()
