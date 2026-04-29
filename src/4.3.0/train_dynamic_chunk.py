#!/usr/bin/env python3
"""Module 1: Dynamic filtering + training on a single chunk.

Key requirements implemented:
- Built on top of SFTTrainer and ResponseOnlyCollator logic.
- Override `compute_loss` to:
  1) capture per-sample gradients,
  2) compute reward R(x) from anchor gradients,
  3) update alpha online,
  4) apply sample mask KEEP/DROP by zeroing dropped sample loss weight.
- Persist selected rows for this chunk and save latest checkpoint.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, TrainingArguments, set_seed
from trl import SFTTrainer

from recipe_common import (
    EPS,
    MapperInput,
    ResponseOnlyCollator,
    build_prompt,
    choose_row_id,
    get_dtype,
    load_causal_lm,
    load_jsonl,
    minmax_l1_normalize,
    parse_mapper_item,
    parse_topk_indices,
    parse_vector_to_dense,
    resolve_model_path,
    to_float,
    write_jsonl,
)

SCALAR_FALLBACK_FIELDS = (
    "score",
    "delta_scalar",
    "alpagasus_scalar",
    "judge_score",
    "pdm_scalar",
)


@dataclass
class DynamicSelectorState:
    method_names: list[str]
    method_scores: torch.Tensor  # (N, K, M), CPU float32
    topk_indices: torch.Tensor  # (N, topk), CPU long
    beta: torch.Tensor  # (M,), CPU float32
    alpha: torch.Tensor  # (K,), CPU float32
    anchor_grads: dict[int, torch.Tensor]  # cap -> (D,), CPU float32 normalized
    grad_dim: int

    alpha_ema: float
    alpha_temperature: float

    data_weight: float
    reward_weight: float

    keep_policy: str
    keep_threshold: float
    keep_ratio: float
    softmax_temperature: float
    score_ema_mean: float
    score_ema_var: float
    score_ema_momentum: float
    min_keep_prob: float
    keep_every_n: int

    # Online alpha dynamics (method-level)
    reward_ema_mean: float
    reward_ema_var: float
    reward_ema_momentum: float
    alpha_signal_ema: float
    alpha_lr: float
    min_alpha: float
    utility_ema: torch.Tensor  # (K,), CPU float32

    kept_local_indices: list[int]
    seen_kept: set[int]


class DynamicChunkTrainer(SFTTrainer):
    """SFTTrainer with dynamic sample-level selection in compute_loss."""

    def __init__(
        self,
        *args,
        selector_state: DynamicSelectorState,
        grad_params: list[tuple[str, torch.nn.Parameter]],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.selector_state = selector_state
        self.grad_params = grad_params
        self._anchor_device_cache: dict[int, torch.Tensor] = {}
        self._dynamic_step = 0

    @staticmethod
    def _per_sample_response_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        tok_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.size(0), shift_labels.size(1))

        valid = (shift_labels != -100).to(tok_loss.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (tok_loss * valid).sum(dim=1) / denom

    def _flatten_single_grad(
        self,
        loss_scalar: torch.Tensor,
        retain_graph: bool,
    ) -> torch.Tensor:
        grads = torch.autograd.grad(
            loss_scalar,
            [p for _, p in self.grad_params],
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True,
        )
        parts: list[torch.Tensor] = []
        remaining = int(self.selector_state.grad_dim)
        for g, (_, p) in zip(grads, self.grad_params):
            if remaining <= 0:
                break
            take = min(int(p.numel()), remaining)
            if take <= 0:
                continue
            if g is None:
                parts.append(torch.zeros((take,), dtype=torch.float32, device=loss_scalar.device))
            else:
                flat = g.detach().to(torch.float32).reshape(-1)
                parts.append(flat[:take])
            remaining -= take
        if not parts:
            return torch.zeros((1,), dtype=torch.float32, device=loss_scalar.device)
        gvec = torch.cat(parts, dim=0)
        return gvec / gvec.norm().clamp_min(EPS)

    def _compute_reward_for_sample(
        self,
        sample_grad: torch.Tensor,
        topk_caps: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        device = sample_grad.device

        weighted_sim = torch.zeros((), dtype=torch.float32, device=device)
        weight_sum = torch.zeros((), dtype=torch.float32, device=device)

        for cap in topk_caps.tolist():
            if cap < 0:
                continue
            g_cap_cpu = self.selector_state.anchor_grads.get(int(cap))
            if g_cap_cpu is None:
                continue

            if cap not in self._anchor_device_cache:
                self._anchor_device_cache[cap] = g_cap_cpu.to(device=device, dtype=torch.float32)
            g_cap = self._anchor_device_cache[cap]
            dim = min(int(sample_grad.numel()), int(g_cap.numel()))
            if dim <= 0:
                continue
            sg = sample_grad[:dim]
            g_cap = g_cap[:dim]
            g_cap = g_cap / g_cap.norm().clamp_min(EPS)

            sim = torch.dot(sg, g_cap)
            w = beta[cap] if cap < beta.numel() else torch.tensor(0.0, device=device)
            if w.item() <= 0:
                continue

            weighted_sim = weighted_sim + w * sim
            weight_sum = weight_sum + w

        if weight_sum.item() <= 0:
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        return weighted_sim / weight_sum.clamp_min(EPS)

    def _decide_keep_mask(
        self,
        final_scores: torch.Tensor,
    ) -> torch.Tensor:
        bsz = int(final_scores.numel())
        if bsz == 0:
            return torch.zeros((0,), dtype=torch.bool, device=final_scores.device)

        # Single-sample micro-batch needs a dedicated policy. If we always keep
        # at B=1, dynamic selection degenerates (selected_ratio -> 1.0).
        if bsz == 1:
            if self.selector_state.keep_policy == "softmax":
                p = max(float(self.selector_state.min_keep_prob), float(self.selector_state.keep_ratio))
                keep_flag = random.random() < p
            else:
                s = float(final_scores[0].detach().item())
                mean = float(self.selector_state.score_ema_mean)
                var = max(float(self.selector_state.score_ema_var), 1e-8)
                std = max(math.sqrt(var), 1e-6)
                z = (s - mean) / std
                p = 1.0 / (1.0 + math.exp(-z))
                p = 0.7 * p + 0.3 * float(self.selector_state.keep_ratio)
                p = max(float(self.selector_state.min_keep_prob), min(1.0, p))
                keep_flag = random.random() < p

            keep_every_n = int(max(0, self.selector_state.keep_every_n))
            if (not keep_flag) and keep_every_n > 0 and (self._dynamic_step % keep_every_n == 0):
                keep_flag = True
            return torch.tensor([keep_flag], dtype=torch.bool, device=final_scores.device)

        if self.selector_state.keep_policy == "softmax":
            k = max(1, int(round(bsz * self.selector_state.keep_ratio)))
            k = min(k, bsz)
            logits = final_scores / max(self.selector_state.softmax_temperature, 1e-6)
            probs = torch.softmax(logits - logits.max(), dim=0)
            chosen = torch.multinomial(probs, num_samples=k, replacement=False)
            keep = torch.zeros((bsz,), dtype=torch.bool, device=final_scores.device)
            keep[chosen] = True
            return keep

        # threshold policy
        centered = final_scores - final_scores.mean()
        scaled = centered / final_scores.std(unbiased=False).clamp_min(1e-6)
        probs = torch.sigmoid(scaled)
        keep = probs >= self.selector_state.keep_threshold

        # Avoid empty batch that may stall training
        if int(keep.sum().item()) == 0:
            keep[torch.argmax(final_scores)] = True
        return keep

    def _update_alpha(
        self,
        mapper_util: torch.Tensor,  # (B, K)
        reward_scaled: torch.Tensor,  # (B,)
    ) -> None:
        # Use method-relative signal (within-sample z-score across methods) to
        # avoid scale collapse after global normalization. Then weight it by
        # reward advantage vs running reward baseline.
        if mapper_util.numel() == 0:
            return

        device = mapper_util.device
        k = int(mapper_util.size(1))
        if k <= 0:
            return

        mu = mapper_util.mean(dim=1, keepdim=True)
        sigma = mapper_util.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
        method_rel = (mapper_util - mu) / sigma  # (B,K), centered per sample
        method_signal = method_rel.mean(dim=0)  # (K,)

        r = float(reward_scaled.detach().mean().item())
        r_mom = float(min(max(self.selector_state.reward_ema_momentum, 1e-4), 1.0))
        r_mean_old = float(self.selector_state.reward_ema_mean)
        r_mean_new = (1.0 - r_mom) * r_mean_old + r_mom * r
        r_var_new = (1.0 - r_mom) * float(self.selector_state.reward_ema_var) + r_mom * ((r - r_mean_new) ** 2)
        r_std_new = max(math.sqrt(max(r_var_new, 1e-8)), 1e-6)
        reward_adv = float((r - r_mean_new) / r_std_new)
        self.selector_state.reward_ema_mean = r_mean_new
        self.selector_state.reward_ema_var = max(r_var_new, 1e-8)

        signal = method_signal * reward_adv  # (K,)
        util_ema = self.selector_state.utility_ema.to(device=device, dtype=torch.float32)
        sig_mom = float(min(max(self.selector_state.alpha_signal_ema, 1e-4), 1.0))
        if util_ema.numel() != k:
            util_ema = torch.zeros((k,), dtype=torch.float32, device=device)
        util_ema = (1.0 - sig_mom) * util_ema + sig_mom * signal
        self.selector_state.utility_ema = util_ema.detach().cpu()

        alpha_old = self.selector_state.alpha.to(device=device, dtype=torch.float32)
        if alpha_old.numel() != k:
            alpha_old = torch.full((k,), 1.0 / float(k), dtype=torch.float32, device=device)
        logits = torch.log(alpha_old.clamp_min(EPS))
        temp = float(max(self.selector_state.alpha_temperature, 1e-6))
        lr = float(max(self.selector_state.alpha_lr, 1e-6))
        target = torch.softmax(logits + lr * (util_ema / temp), dim=0)

        # Optional floor to avoid early collapse
        min_alpha = float(max(0.0, self.selector_state.min_alpha))
        if min_alpha > 0.0:
            max_floor = 0.99 / float(k)
            floor = min(min_alpha, max_floor)
            target = target * (1.0 - floor * float(k)) + floor
            target = target / target.sum().clamp_min(EPS)

        ema = float(min(max(self.selector_state.alpha_ema, 0.0), 1.0))
        alpha = (1.0 - ema) * alpha_old + ema * target
        alpha = alpha.clamp_min(EPS)
        alpha = alpha / alpha.sum().clamp_min(EPS)
        self.selector_state.alpha = alpha.detach().cpu()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: D401
        del num_items_in_batch
        self._dynamic_step += 1

        sample_ids = inputs.pop("sample_ids")
        labels = inputs["labels"]

        # Pass-1: selection graph (used only for sample-gradient probing).
        outputs_select = model(**inputs)
        per_sample_loss_select = self._per_sample_response_loss(outputs_select.logits, labels)

        bsz = int(per_sample_loss_select.size(0))
        device = per_sample_loss_select.device

        # 1) Per-sample gradient interception
        sample_grads: list[torch.Tensor] = []
        for i in range(bsz):
            retain = i < (bsz - 1)
            gvec = self._flatten_single_grad(per_sample_loss_select[i], retain_graph=retain)
            sample_grads.append(gvec)
        grad_mat = torch.stack(sample_grads, dim=0) if sample_grads else torch.zeros((0, 1), device=device)

        # 2) Compute data score from normalized multi-method vectors
        sid_cpu = sample_ids.detach().to("cpu", dtype=torch.long)
        method_batch = self.selector_state.method_scores.index_select(0, sid_cpu).to(device=device)  # (B,K,M)

        beta = self.selector_state.beta.to(device)
        if beta.numel() < method_batch.size(-1):
            beta = F.pad(beta, (0, method_batch.size(-1) - beta.numel()))
        elif beta.numel() > method_batch.size(-1):
            beta = beta[: method_batch.size(-1)]
        beta = beta / beta.sum().clamp_min(EPS)

        mapper_util = torch.einsum("bkm,m->bk", method_batch, beta)  # (B,K)
        alpha = self.selector_state.alpha.to(device)
        if alpha.numel() < mapper_util.size(1):
            alpha = F.pad(alpha, (0, mapper_util.size(1) - alpha.numel()))
        elif alpha.numel() > mapper_util.size(1):
            alpha = alpha[: mapper_util.size(1)]
        alpha = alpha / alpha.sum().clamp_min(EPS)
        data_score = torch.einsum("bk,k->b", mapper_util, alpha)  # (B,)

        # 3) Compute R(x) with anchor gradients + top-k capabilities
        topk_batch = self.selector_state.topk_indices.index_select(0, sid_cpu)  # CPU
        rewards: list[torch.Tensor] = []
        for i in range(bsz):
            reward_i = self._compute_reward_for_sample(
                sample_grad=grad_mat[i],
                topk_caps=topk_batch[i],
                beta=beta,
            )
            rewards.append(reward_i)
        reward = torch.stack(rewards, dim=0) if rewards else torch.zeros((0,), device=device)

        # map cosine range [-1,1] to [0,1] for stable combination
        reward_scaled = (reward + 1.0) * 0.5

        # 4) Final S(x), keep/drop mask, and weighted loss
        final_score = self.selector_state.data_weight * data_score + self.selector_state.reward_weight * reward_scaled

        # Update running score statistics (used by B=1 keep policy).
        s_mean = float(final_score.detach().mean().item())
        mom = float(min(max(self.selector_state.score_ema_momentum, 1e-4), 1.0))
        old_mean = float(self.selector_state.score_ema_mean)
        new_mean = (1.0 - mom) * old_mean + mom * s_mean
        centered = float(((final_score.detach() - new_mean) ** 2).mean().item())
        new_var = (1.0 - mom) * float(self.selector_state.score_ema_var) + mom * centered
        self.selector_state.score_ema_mean = new_mean
        self.selector_state.score_ema_var = max(new_var, 1e-8)

        keep_mask = self._decide_keep_mask(final_score)
        keep_weight = keep_mask.to(torch.float32)

        # Pass-2: training graph (clean graph for actual optimizer backward).
        outputs_train = model(**inputs)
        per_sample_loss_train = self._per_sample_response_loss(outputs_train.logits, labels)
        keep_weight = keep_weight.to(per_sample_loss_train.dtype)
        if float(keep_weight.sum().item()) > 0:
            final_loss = (per_sample_loss_train * keep_weight).sum() / keep_weight.sum().clamp_min(1.0)
        else:
            final_loss = outputs_train.loss * 0.0

        # 5) Online alpha update (EMA)
        self._update_alpha(mapper_util=mapper_util.detach(), reward_scaled=reward_scaled.detach())

        # 6) Record selected sample ids for this chunk
        for sid, keep in zip(sid_cpu.tolist(), keep_mask.detach().to("cpu").tolist()):
            if bool(keep) and sid not in self.selector_state.seen_kept:
                self.selector_state.seen_kept.add(int(sid))
                self.selector_state.kept_local_indices.append(int(sid))

        if return_outputs:
            return final_loss, outputs_train
        return final_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one dynamic chunk with online alpha update")

    parser.add_argument("--chunk-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--model-source", type=str, choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--modelscope-cache-dir", type=Path, default=None)

    parser.add_argument("--topk-jsonl", type=Path, required=True)
    parser.add_argument("--score-path", type=str, action="append", required=True)

    parser.add_argument("--beta-json", type=Path, required=True)
    parser.add_argument("--anchor-grads", type=Path, required=True)
    parser.add_argument("--alpha-json", type=Path, required=True)

    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
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
    parser.add_argument("--alpha-lr", type=float, default=0.8)
    parser.add_argument("--alpha-signal-ema", type=float, default=0.05)
    parser.add_argument("--reward-ema-momentum", type=float, default=0.02)
    parser.add_argument("--min-alpha", type=float, default=0.02)
    parser.add_argument(
        "--min-response-tokens",
        type=int,
        default=4,
        help="Drop sample if truncated sequence keeps fewer than this many response tokens.",
    )

    parser.add_argument(
        "--grad-param-mode",
        type=str,
        choices=["lora", "lm_head", "all_trainable", "name_contains"],
        default="lora",
    )
    parser.add_argument("--grad-param-filter", type=str, default="")
    parser.add_argument(
        "--max-grad-dim",
        type=int,
        default=65536,
        help="Truncate flattened sample gradient vector to first N dims. 0 means full anchor dim.",
    )

    parser.add_argument("--torch-dtype", type=str, choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--grad-ckpt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")

    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_checkpoint_model_for_training(
    checkpoint_path: str,
    base_model: str,
    model_source: str,
    modelscope_cache_dir: Path | None,
    dtype: torch.dtype,
    lora_cfg: LoraConfig,
):
    resolved_base = resolve_model_path(base_model, model_source, modelscope_cache_dir)

    ckpt = Path(checkpoint_path)
    if ckpt.exists() and (ckpt / "adapter_config.json").exists():
        base = load_causal_lm(resolved_base, dtype=dtype)
        model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=True)
        return model, resolved_base

    if ckpt.exists():
        model = load_causal_lm(str(ckpt), dtype=dtype)
        has_lora = any("lora_" in n for n, _ in model.named_parameters())
        if not has_lora:
            model = get_peft_model(model, lora_cfg)
        return model, str(ckpt)

    # Non-local identifier branch.
    # For first chunk in orchestrator, checkpoint_path is usually base_model string.
    # Under modelscope mode we should prefer resolved local snapshot.
    if checkpoint_path == base_model and model_source == "modelscope":
        logging.info(
            "checkpoint_path=%s is non-local base model under modelscope; use resolved local snapshot: %s",
            checkpoint_path,
            resolved_base,
        )
        model = load_causal_lm(resolved_base, dtype=dtype)
        has_lora = any("lora_" in n for n, _ in model.named_parameters())
        if not has_lora:
            model = get_peft_model(model, lora_cfg)
        return model, resolved_base

    # Generic model-id fallback
    model = load_causal_lm(checkpoint_path, dtype=dtype)
    has_lora = any("lora_" in n for n, _ in model.named_parameters())
    if not has_lora:
        model = get_peft_model(model, lora_cfg)
    return model, checkpoint_path


def build_tokenized_chunk_dataset(
    rows: list[dict[str, Any]],
    tokenizer,
    max_seq_length: int,
    min_response_tokens: int,
) -> tuple[Dataset, list[dict[str, Any]]]:
    def _build_prompt_without_output(row: dict[str, Any]) -> str:
        instruction = str(row.get("instruction", "")).strip()
        inp = row.get("input")
        inp = "" if inp is None else str(inp).strip()
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

    features: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    skipped_short = 0
    skipped_empty = 0

    for row in rows:
        text = build_prompt(row)
        prompt_only = _build_prompt_without_output(row)
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )
        prompt_ids = tokenizer(
            prompt_only,
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]

        full_ids = enc["input_ids"]
        resp_tokens = max(0, len(full_ids) - min(len(full_ids), len(prompt_ids)))
        if resp_tokens <= 0:
            skipped_empty += 1
            continue
        if resp_tokens < max(1, int(min_response_tokens)):
            skipped_short += 1
            continue

        local_idx = len(kept_rows)
        kept_rows.append(row)
        features.append(
            {
                "input_ids": full_ids,
                "attention_mask": enc.get("attention_mask"),
                "sample_id": int(local_idx),
            }
        )

    if not features:
        raise ValueError(
            "No valid tokenized samples after response-token filtering. "
            f"max_seq_length={max_seq_length}, min_response_tokens={min_response_tokens}."
        )

    logging.info(
        "Tokenized chunk: kept=%d/%d, skipped_empty_response=%d, skipped_short_response=%d (min_response_tokens=%d)",
        len(features),
        len(rows),
        skipped_empty,
        skipped_short,
        int(min_response_tokens),
    )
    return Dataset.from_list(features), kept_rows


def load_topk_for_ids(topk_jsonl: Path, wanted_ids: set[str]) -> tuple[dict[str, list[int]], int]:
    if not topk_jsonl.exists():
        raise FileNotFoundError(f"topk_jsonl not found: {topk_jsonl}")

    topk_map: dict[str, list[int]] = {}
    max_cap = -1

    with topk_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = choose_row_id(row, line_no - 1)
            if rid not in wanted_ids:
                continue
            vals = parse_topk_indices(row.get("top_k_indices"))
            if not vals:
                continue
            topk_map[rid] = vals
            max_cap = max(max_cap, max(vals))

    return topk_map, max_cap


def extract_scalar_from_row(row: dict[str, Any]) -> float | None:
    for field in SCALAR_FALLBACK_FIELDS:
        if field not in row:
            continue
        fv = to_float(row.get(field))
        if fv is not None:
            return float(fv)
    return None


def load_mapper_vectors_for_ids(
    mapper: MapperInput,
    wanted_ids: set[str],
) -> tuple[dict[str, dict[int, float]], int]:
    if not mapper.path.exists():
        raise FileNotFoundError(f"score file not found: {mapper.path}")

    out: dict[str, dict[int, float]] = {}
    max_cap = -1

    with mapper.path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = choose_row_id(row, line_no - 1)
            if rid not in wanted_ids:
                continue

            vec = parse_vector_to_dense(row.get("mapped_vector"))
            if not vec:
                vec = parse_vector_to_dense(row.get("score"))

            if not vec:
                scalar = extract_scalar_from_row(row)
                if scalar is not None:
                    caps = parse_topk_indices(row.get("top_k_indices"))
                    if caps:
                        share = float(scalar) / float(len(caps))
                        vec = {int(c): share for c in caps}

            if not vec:
                continue

            vec = {int(k): float(v) for k, v in vec.items() if int(k) >= 0 and float(v) > 0}
            if not vec:
                continue

            max_cap = max(max_cap, max(vec.keys()))
            out[rid] = vec

    return out, max_cap


def load_beta(beta_json: Path) -> torch.Tensor:
    if not beta_json.exists():
        raise FileNotFoundError(f"beta_json not found: {beta_json}")
    with beta_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    arr = obj.get("beta")
    if not isinstance(arr, list) or not arr:
        raise ValueError(f"Invalid beta file: {beta_json}")
    vals = []
    for v in arr:
        try:
            vals.append(float(v))
        except Exception:  # noqa: BLE001
            vals.append(0.0)
    t = torch.tensor(vals, dtype=torch.float32)
    t = t.clamp_min(0.0)
    return t / t.sum().clamp_min(EPS)


def load_alpha(alpha_json: Path, k_methods: int) -> torch.Tensor:
    if alpha_json.exists():
        with alpha_json.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        arr = obj.get("alpha")
        if isinstance(arr, list) and arr:
            vals = []
            for v in arr:
                try:
                    vals.append(float(v))
                except Exception:  # noqa: BLE001
                    vals.append(0.0)
            t = torch.tensor(vals, dtype=torch.float32)
            if t.numel() < k_methods:
                t = F.pad(t, (0, k_methods - t.numel()))
            elif t.numel() > k_methods:
                t = t[:k_methods]
            t = t.clamp_min(0.0)
            if float(t.sum().item()) > 0:
                return t / t.sum().clamp_min(EPS)

    return torch.full((k_methods,), 1.0 / float(k_methods), dtype=torch.float32)


def load_anchor_grads(anchor_grads_path: Path) -> tuple[dict[int, torch.Tensor], int, int]:
    if not anchor_grads_path.exists():
        raise FileNotFoundError(f"anchor_grads not found: {anchor_grads_path}")
    obj = torch.load(anchor_grads_path, map_location="cpu")
    grads_obj = obj.get("grads", {})
    m_dim = int(obj.get("m_dim", 0))
    grad_dim = int(obj.get("gradient_dim", 0))

    out: dict[int, torch.Tensor] = {}
    if isinstance(grads_obj, dict):
        for k, v in grads_obj.items():
            try:
                cap = int(k)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(v, torch.Tensor):
                continue
            vec = v.to(torch.float32).reshape(-1)
            n = vec.norm().clamp_min(EPS)
            out[cap] = vec / n
            if grad_dim <= 0:
                grad_dim = int(vec.numel())
    return out, m_dim, grad_dim


def select_grad_parameters(
    model,
    mode: str,
    name_filter: str,
) -> list[tuple[str, torch.nn.Parameter]]:
    all_named = list(model.named_parameters())

    if mode == "all_trainable":
        selected = [(n, p) for n, p in all_named if p.requires_grad]
    elif mode == "lora":
        selected = [(n, p) for n, p in all_named if "lora_" in n and p.requires_grad]
        if not selected:
            selected = [(n, p) for n, p in all_named if p.requires_grad]
    elif mode == "lm_head":
        selected = [(n, p) for n, p in all_named if "lm_head" in n and p.requires_grad]
        if not selected:
            selected = [(n, p) for n, p in all_named if p.requires_grad]
    else:
        selected = [(n, p) for n, p in all_named if name_filter and (name_filter in n) and p.requires_grad]
        if not selected:
            selected = [(n, p) for n, p in all_named if p.requires_grad]

    return selected


def build_score_tensor_bundle(
    chunk_rows: list[dict[str, Any]],
    mapper_inputs: list[MapperInput],
    topk_jsonl: Path,
    beta: torch.Tensor,
    anchor_m_dim: int,
    top_k_default: int = 5,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    row_ids = [choose_row_id(row, i) for i, row in enumerate(chunk_rows)]
    wanted = set(row_ids)

    topk_map, max_cap_topk = load_topk_for_ids(topk_jsonl, wanted)

    mapper_maps: list[dict[str, dict[int, float]]] = []
    mapper_names: list[str] = []
    max_cap_mapper = -1
    for mapper in mapper_inputs:
        m, mmax = load_mapper_vectors_for_ids(mapper, wanted)
        mapper_maps.append(m)
        mapper_names.append(mapper.name)
        max_cap_mapper = max(max_cap_mapper, mmax)

    max_cap = max(max_cap_mapper, max_cap_topk, int(beta.numel()) - 1, int(anchor_m_dim) - 1)
    if max_cap < 0:
        raise ValueError("Cannot infer capability dimension from topk/score/beta inputs.")
    m_dim = max_cap + 1

    n = len(chunk_rows)
    k = len(mapper_maps)

    method_tensors: list[torch.Tensor] = []
    for mapper_map in mapper_maps:
        mat = torch.zeros((n, m_dim), dtype=torch.float32)
        for i, rid in enumerate(row_ids):
            vec = mapper_map.get(rid)
            if not vec:
                continue
            for cap, val in vec.items():
                if 0 <= cap < m_dim and val > 0:
                    mat[i, cap] = float(val)
        method_tensors.append(minmax_l1_normalize(mat))

    method_scores = torch.stack(method_tensors, dim=1)  # (N,K,M)

    kmax = top_k_default
    for rid in row_ids:
        vals = topk_map.get(rid, [])
        if vals:
            kmax = max(kmax, len(vals))

    topk_tensor = torch.full((n, kmax), -1, dtype=torch.long)
    aggregate = method_scores.sum(dim=1)  # (N,M)
    for i, rid in enumerate(row_ids):
        vals = topk_map.get(rid)
        if vals:
            valid = [int(x) for x in vals if 0 <= int(x) < m_dim]
            if valid:
                t = torch.tensor(valid[:kmax], dtype=torch.long)
                topk_tensor[i, : t.numel()] = t
                continue

        # fallback: non-zero top-k from summed mapper vectors
        row = aggregate[i]
        nonzero = torch.nonzero(row > 0, as_tuple=False).squeeze(1)
        if nonzero.numel() > 0:
            take = min(kmax, int(nonzero.numel()))
            vals_idx = torch.topk(row, k=take, dim=0).indices
            topk_tensor[i, : take] = vals_idx

    return mapper_names, method_scores, topk_tensor


def build_training_args(args: argparse.Namespace, output_dir: Path, dtype: torch.dtype) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(args.learning_rate),
        "per_device_train_batch_size": int(max(1, args.train_batch_size)),
        "gradient_accumulation_steps": int(max(1, args.gradient_accumulation_steps)),
        "num_train_epochs": float(max(0.1, args.num_train_epochs)),
        "weight_decay": float(max(0.0, args.weight_decay)),
        "warmup_ratio": float(max(0.0, args.warmup_ratio)),
        "logging_steps": int(max(1, args.logging_steps)),
        "logging_strategy": "steps",
        "report_to": [],
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "seed": int(args.seed),
        "data_seed": int(args.seed),
        "fp16": (dtype == torch.float16),
        "bf16": (dtype == torch.bfloat16),
    }

    save_steps = int(max(0, args.save_steps))
    if save_steps > 0:
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = save_steps
    else:
        kwargs["save_strategy"] = "no"

    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "no"

    return TrainingArguments(**kwargs)


def build_dynamic_trainer(
    model,
    tokenizer,
    train_ds: Dataset,
    train_args: TrainingArguments,
    selector_state: DynamicSelectorState,
    grad_params: list[tuple[str, torch.nn.Parameter]],
):
    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters

    kwargs: dict[str, Any] = {
        "model": model,
        "args": train_args,
        "train_dataset": train_ds,
        "eval_dataset": None,
        "data_collator": ResponseOnlyCollator(tokenizer),
    }

    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if "formatting_func" in params:
        kwargs["formatting_func"] = None

    return DynamicChunkTrainer(
        **kwargs,
        selector_state=selector_state,
        grad_params=grad_params,
    )


def save_alpha(alpha_json: Path, alpha: torch.Tensor, method_names: list[str]) -> None:
    alpha_json.parent.mkdir(parents=True, exist_ok=True)
    with alpha_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "alpha": [float(x) for x in alpha.tolist()],
                "method_names": method_names,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    if not args.chunk_jsonl.exists():
        raise FileNotFoundError(f"chunk_jsonl not found: {args.chunk_jsonl}")

    mapper_inputs = [parse_mapper_item(x) for x in args.score_path]
    if len(mapper_inputs) < 1:
        raise ValueError("At least one --score-path is required.")

    dtype = get_dtype(args.torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    chunk_rows = load_jsonl(args.chunk_jsonl)
    if len(chunk_rows) == 0:
        raise ValueError("chunk_jsonl is empty.")

    beta = load_beta(args.beta_json)
    anchor_grads, anchor_m_dim, anchor_grad_dim = load_anchor_grads(args.anchor_grads)

    mapper_names, method_scores, topk_tensor = build_score_tensor_bundle(
        chunk_rows=chunk_rows,
        mapper_inputs=mapper_inputs,
        topk_jsonl=args.topk_jsonl,
        beta=beta,
        anchor_m_dim=anchor_m_dim,
        top_k_default=5,
    )

    k_methods = len(mapper_names)
    alpha = load_alpha(args.alpha_json, k_methods=k_methods)

    effective_grad_dim = int(anchor_grad_dim)
    if int(args.max_grad_dim) > 0:
        effective_grad_dim = min(effective_grad_dim, int(args.max_grad_dim)) if effective_grad_dim > 0 else int(
            args.max_grad_dim
        )
    if effective_grad_dim <= 0:
        effective_grad_dim = int(args.max_grad_dim) if int(args.max_grad_dim) > 0 else 65536

    # Align anchor vectors to effective dimension.
    if anchor_grads:
        trimmed: dict[int, torch.Tensor] = {}
        for cap, vec in anchor_grads.items():
            v = vec[:effective_grad_dim].to(torch.float32)
            v = v / v.norm().clamp_min(EPS)
            trimmed[int(cap)] = v
        anchor_grads = trimmed

    logging.info(
        "Dynamic gradient dim: anchor_grad_dim=%d, configured_max_grad_dim=%d, effective_grad_dim=%d",
        int(anchor_grad_dim),
        int(args.max_grad_dim),
        int(effective_grad_dim),
    )

    selector_state = DynamicSelectorState(
        method_names=mapper_names,
        method_scores=method_scores,
        topk_indices=topk_tensor,
        beta=beta,
        alpha=alpha,
        anchor_grads=anchor_grads,
        grad_dim=int(effective_grad_dim),
        alpha_ema=float(args.alpha_ema),
        alpha_temperature=float(args.alpha_temperature),
        data_weight=float(args.data_weight),
        reward_weight=float(args.reward_weight),
        keep_policy=args.keep_policy,
        keep_threshold=float(args.keep_threshold),
        keep_ratio=float(args.keep_ratio),
        softmax_temperature=float(args.softmax_temperature),
        score_ema_mean=0.0,
        score_ema_var=1.0,
        score_ema_momentum=float(args.score_ema_momentum),
        min_keep_prob=float(args.min_keep_prob),
        keep_every_n=int(args.keep_every_n),
        reward_ema_mean=0.5,
        reward_ema_var=1.0,
        reward_ema_momentum=float(args.reward_ema_momentum),
        alpha_signal_ema=float(args.alpha_signal_ema),
        alpha_lr=float(args.alpha_lr),
        min_alpha=float(args.min_alpha),
        utility_ema=torch.zeros((k_methods,), dtype=torch.float32),
        kept_local_indices=[],
        seen_kept=set(),
    )

    lora_cfg = LoraConfig(
        r=max(1, args.lora_r),
        lora_alpha=max(1, args.lora_alpha),
        lora_dropout=max(0.0, args.lora_dropout),
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model, tokenizer_source = load_checkpoint_model_for_training(
        checkpoint_path=args.checkpoint_path,
        base_model=args.base_model,
        model_source=args.model_source,
        modelscope_cache_dir=args.modelscope_cache_dir,
        dtype=dtype,
        lora_cfg=lora_cfg,
    )

    if int(args.grad_ckpt) == 1 and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    model.to(device)
    model.train()

    grad_params = select_grad_parameters(
        model=model,
        mode=args.grad_param_mode,
        name_filter=args.grad_param_filter,
    )
    if not grad_params:
        raise RuntimeError("No grad params selected for dynamic gradient interception.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if int(args.max_seq_length) <= 256:
        logging.warning(
            "max_seq_length=%d may truncate response heavily on dolly; consider >=512 for stable loss.",
            int(args.max_seq_length),
        )

    train_ds, train_rows_effective = build_tokenized_chunk_dataset(
        rows=chunk_rows,
        tokenizer=tokenizer,
        max_seq_length=max(64, args.max_seq_length),
        min_response_tokens=max(1, args.min_response_tokens),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = build_training_args(args=args, output_dir=args.output_dir, dtype=dtype)

    trainer = build_dynamic_trainer(
        model=model,
        tokenizer=tokenizer,
        train_ds=train_ds,
        train_args=train_args,
        selector_state=selector_state,
        grad_params=grad_params,
    )

    trainer.train()

    selected_local = selector_state.kept_local_indices
    selected_rows = [train_rows_effective[i] for i in selected_local if 0 <= i < len(train_rows_effective)]

    selected_path = args.output_dir / "chunk_selected.jsonl"
    write_jsonl(selected_rows, selected_path)

    final_ckpt = args.output_dir / "final_checkpoint"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_ckpt))
    tokenizer.save_pretrained(str(final_ckpt))

    save_alpha(args.alpha_json, selector_state.alpha, mapper_names)

    result = {
        "chunk_jsonl": str(args.chunk_jsonl),
        "chunk_size": int(len(chunk_rows)),
        "chunk_effective_size": int(len(train_rows_effective)),
        "selected_size": int(len(selected_rows)),
        "selected_ratio": float(len(selected_rows) / max(1, len(chunk_rows))),
        "selected_ratio_effective": float(len(selected_rows) / max(1, len(train_rows_effective))),
        "selected_jsonl": str(selected_path),
        "next_checkpoint": str(final_ckpt),
        "alpha_json": str(args.alpha_json),
        "alpha": [float(x) for x in selector_state.alpha.tolist()],
        "beta_dim": int(selector_state.beta.numel()),
        "method_names": mapper_names,
        "keep_policy": args.keep_policy,
        "grad_param_mode": args.grad_param_mode,
    }

    result_path = args.output_dir / "chunk_result.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logging.info(
        "Chunk done. selected=%d/%d (effective=%d), result=%s",
        len(selected_rows),
        len(chunk_rows),
        len(train_rows_effective),
        result_path,
    )


if __name__ == "__main__":
    main()
