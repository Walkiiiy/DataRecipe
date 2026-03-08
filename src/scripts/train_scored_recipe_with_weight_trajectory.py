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
from typing import Dict, Iterator, List, Optional, Tuple

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
    parser.add_argument(
        "--alpha_data_mode",
        type=str,
        default="none",
        choices=["none", "loss_weight", "topk", "topk_loss_weight"],
        help="How alpha affects data/training each micro-step.",
    )
    parser.add_argument(
        "--topk_ratio",
        type=float,
        default=1.0,
        help="Keep top-k ratio by alpha-weighted quality when mode includes topk.",
    )
    parser.add_argument(
        "--anchor_dir",
        type=str,
        default="/home/walkiiiy/DataRecipe/data/flan/niv2_capability_data_ramdom1000_preprocessed/capability_anchors",
        help="Directory for capability-anchor data (recommended: .../capability_anchors).",
    )
    parser.add_argument(
        "--anchor_refresh_every",
        type=int,
        default=5,
        help="Refresh capability anchor gradients every N optimizer steps.",
    )
    parser.add_argument(
        "--anchor_samples_per_capability",
        type=int,
        default=2,
        help="Max anchor samples used per capability when building anchor pool.",
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
    metrics: List[float]
    anchor_key: str


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

        metric_keys = [
            ("相关度",),
            ("准确性",),
            ("能力多样性", "多样性"),
            ("难度",),
        ]
        metrics = []
        for key_group in metric_keys:
            found = None
            for k in key_group:
                if k in ex and ex.get(k) not in (None, ""):
                    found = ex.get(k)
                    break
            if found is None:
                return None
            try:
                metrics.append(float(found))
            except Exception:
                return None

        anchor_key = str(ex.get("能力锚点属性") or ex.get("capability_tag") or ex.get("task_name") or "")
        if not anchor_key:
            return None

        return PackedSample(input_ids=input_ids, labels=labels, metrics=metrics, anchor_key=anchor_key)

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

    input_ids, labels, attention_mask, metrics, anchor_keys = [], [], [], [], []
    for x in batch:
        cur = len(x.input_ids)
        pad = max_len - cur
        input_ids.append(x.input_ids + [pad_id] * pad)
        labels.append(x.labels + [-100] * pad)
        attention_mask.append([1] * cur + [0] * pad)
        metrics.append(x.metrics)
        anchor_keys.append(x.anchor_key)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "metrics": torch.tensor(metrics, dtype=torch.float32),
        "anchor_keys": anchor_keys,
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


def build_anchor_pool(
    files: List[str],
    tokenizer: AutoTokenizer,
    max_seq_len: int,
    per_capability_limit: int,
) -> Dict[str, List[PackedSample]]:
    """Build a small capability->samples pool for anchor gradient refresh."""
    pool: Dict[str, List[PackedSample]] = {}
    ds = ScoredResponseOnlyDataset(
        files=files,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        shuffle_files_each_epoch=False,
    )
    for sample in ds:
        bucket = pool.setdefault(sample.anchor_key, [])
        if len(bucket) < per_capability_limit:
            bucket.append(sample)
    return pool


def flatten_last_layer_grad(model: torch.nn.Module) -> torch.Tensor:
    g_w = model.lm_head.weight.grad
    if g_w is None:
        return torch.zeros(model.lm_head.weight.numel(), device=model.lm_head.weight.device)
    flat = g_w.reshape(-1)
    if model.lm_head.bias is not None and model.lm_head.bias.grad is not None:
        flat = torch.cat([flat, model.lm_head.bias.grad.reshape(-1)], dim=0)
    return flat / flat.norm().clamp_min(1e-12)


def compute_capability_anchor_grads(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    anchor_pool: Dict[str, List[PackedSample]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute latest normalized anchor gradients per capability."""
    grads: Dict[str, torch.Tensor] = {}
    pad_id = tokenizer.pad_token_id
    for cap, samples in anchor_pool.items():
        if not samples:
            continue
        batch = collate_fn(samples, pad_id)
        t_batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor) and k != "metrics"}
        model.zero_grad(set_to_none=True)
        out = model(
            input_ids=t_batch["input_ids"],
            attention_mask=t_batch["attention_mask"],
            labels=t_batch["labels"],
        )
        out.loss.backward()
        grads[cap] = flatten_last_layer_grad(model).detach().clone()
        model.zero_grad(set_to_none=True)
    return grads


def per_sample_last_layer_grads(model: torch.nn.Module, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-sample normalized grads for lm_head."""
    sample_losses = per_sample_response_loss(logits, labels)
    grads: List[torch.Tensor] = []
    grad_params = [model.lm_head.weight] + ([model.lm_head.bias] if model.lm_head.bias is not None else [])
    for i in range(sample_losses.size(0)):
        g = torch.autograd.grad(
            sample_losses[i],
            grad_params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        g_w = torch.zeros_like(model.lm_head.weight) if g[0] is None else g[0]
        flat = g_w.reshape(-1)
        if model.lm_head.bias is not None:
            g_b = torch.zeros_like(model.lm_head.bias) if g[1] is None else g[1]
            flat = torch.cat([flat, g_b.reshape(-1)], dim=0)
        grads.append(flat / flat.norm().clamp_min(1e-12))
    return torch.stack(grads, dim=0)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not (0.0 < args.topk_ratio <= 1.0):
        raise ValueError("--topk_ratio must be in (0, 1].")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] device={device}", flush=True)
    print(
        f"[Init] alpha_data_mode={args.alpha_data_mode} topk_ratio={args.topk_ratio}",
        flush=True,
    )

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
    anchor_files: List[str]
    anchor_dir = Path(args.anchor_dir) if args.anchor_dir else None
    if anchor_dir is not None and anchor_dir.exists():
        anchor_files = list_jsonl_files(str(anchor_dir))
    else:
        print(
            f"[Warn] anchor_dir not found ({args.anchor_dir}), fallback to train_dir for anchors.",
            flush=True,
        )
        anchor_files = files
    anchor_pool = build_anchor_pool(
        files=anchor_files,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        per_capability_limit=args.anchor_samples_per_capability,
    )
    print(
        f"[Init] anchor_dir={(str(anchor_dir) if (anchor_dir and anchor_dir.exists()) else args.train_dir)} "
        f"anchor_caps={len(anchor_pool)} "
        f"refresh_every={args.anchor_refresh_every} "
        f"samples_per_cap={args.anchor_samples_per_capability}",
        flush=True,
    )
    anchor_grads = compute_capability_anchor_grads(
        model=model,
        tokenizer=tokenizer,
        anchor_pool=anchor_pool,
        device=device,
    )
    global_anchor = None
    if anchor_grads:
        global_anchor = torch.stack(list(anchor_grads.values()), dim=0).mean(dim=0)
        global_anchor = global_anchor / global_anchor.norm().clamp_min(1e-12)

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
            if (micro_step - 1) % args.grad_accum_steps == 0:
                if (global_step % max(1, args.anchor_refresh_every) == 0) or (not anchor_grads):
                    anchor_grads = compute_capability_anchor_grads(
                        model=model,
                        tokenizer=tokenizer,
                        anchor_pool=anchor_pool,
                        device=device,
                    )
                    if anchor_grads:
                        global_anchor = torch.stack(list(anchor_grads.values()), dim=0).mean(dim=0)
                        global_anchor = global_anchor / global_anchor.norm().clamp_min(1e-12)

            anchor_keys = list(batch.pop("anchor_keys"))
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch_metrics = batch.pop("metrics")

            # Alpha-conditioned quality score for this micro-batch.
            quality_scores = batch_metrics @ alpha
            if args.alpha_data_mode in ("topk", "topk_loss_weight"):
                total = quality_scores.size(0)
                k = max(1, int(math.ceil(total * args.topk_ratio)))
                if k < total:
                    topk_idx = torch.topk(quality_scores, k=k, dim=0, largest=True).indices
                    batch = {name: t.index_select(0, topk_idx) for name, t in batch.items()}
                    batch_metrics = batch_metrics.index_select(0, topk_idx)
                    quality_scores = quality_scores.index_select(0, topk_idx)
                    keep = topk_idx.detach().cpu().tolist()
                    anchor_keys = [anchor_keys[i] for i in keep]

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**batch)
                sample_loss = per_sample_response_loss(outputs.logits, batch["labels"])
                if args.alpha_data_mode in ("loss_weight", "topk_loss_weight"):
                    weights = quality_scores.clamp_min(1e-8)
                    weights = weights / weights.sum().clamp_min(1e-8)
                    train_loss = (sample_loss * weights).sum()
                else:
                    train_loss = outputs.loss
                loss = train_loss / args.grad_accum_steps

            g_samples = per_sample_last_layer_grads(model, outputs.logits, batch["labels"])

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            with torch.no_grad():
                anchor_vecs: List[torch.Tensor] = []
                for key in anchor_keys:
                    g_ref = anchor_grads.get(key, global_anchor)
                    if g_ref is None:
                        g_ref = torch.ones(g_samples.size(1), device=device)
                        g_ref = g_ref / g_ref.norm().clamp_min(1e-12)
                    anchor_vecs.append(g_ref)
                g_ref_batch = torch.stack(anchor_vecs, dim=0)
                rewards = (g_samples * g_ref_batch).sum(dim=1).detach()
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
