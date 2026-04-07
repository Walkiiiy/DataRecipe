#!/usr/bin/env python3
"""Module 2: Anchor-gradient evaluation + beta update.

Responsibilities:
- Load current model checkpoint.
- Compute anchor gradients per capability cluster.
- Update beta by gradient norm and write `current_beta.json`.
- Serialize anchor vectors to `anchor_grads.pt`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from recipe_common import (
    ResponseOnlyCollator,
    build_prompt,
    choose_row_id,
    get_dtype,
    load_causal_lm,
    load_jsonl,
    parse_topk_indices,
    resolve_model_path,
)

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update beta with anchor gradients")

    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--model-source", type=str, choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--modelscope-cache-dir", type=Path, default=None)
    parser.add_argument("--torch-dtype", type=str, choices=["auto", "fp32", "fp16", "bf16"], default="auto")

    parser.add_argument("--anchor-jsonl", type=Path, required=True)
    parser.add_argument("--topk-jsonl", type=Path, default=None)
    parser.add_argument("--capability-key", type=str, default="auto")
    parser.add_argument("--anchor-size-per-cap", type=int, default=4)
    parser.add_argument("--anchor-batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-capabilities", type=int, default=0, help="0 means all")

    parser.add_argument("--prev-beta-json", type=Path, default=None)
    parser.add_argument("--beta-ema", type=float, default=0.0, help="EMA on beta. 0 means disabled")
    parser.add_argument("--output-beta-json", type=Path, required=True)
    parser.add_argument("--output-anchor-grads", type=Path, required=True)

    parser.add_argument(
        "--grad-param-mode",
        type=str,
        choices=["lora", "lm_head", "all_trainable", "name_contains"],
        default="lora",
    )
    parser.add_argument("--grad-param-filter", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_model_from_checkpoint(
    checkpoint_path: str,
    base_model: str,
    model_source: str,
    modelscope_cache_dir: Path | None,
    dtype: torch.dtype,
):
    resolved_base = resolve_model_path(base_model, model_source, modelscope_cache_dir)

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
        base = load_causal_lm(resolved_base, dtype=dtype)
        model = PeftModel.from_pretrained(base, str(ckpt_path), is_trainable=True)
        return model, resolved_base

    if ckpt_path.exists():
        model = load_causal_lm(str(ckpt_path), dtype=dtype)
        return model, str(ckpt_path)

    # Non-local identifier branch.
    # For first chunk in orchestrator, checkpoint_path is usually base_model string.
    # Under modelscope mode we should prefer the local snapshot path instead of
    # letting transformers resolve via HF hub.
    if checkpoint_path == base_model and model_source == "modelscope":
        logging.info(
            "checkpoint_path=%s is non-local base model under modelscope; use resolved local snapshot: %s",
            checkpoint_path,
            resolved_base,
        )
        model = load_causal_lm(resolved_base, dtype=dtype)
        return model, resolved_base

    # Fallback: treat checkpoint path as generic model identifier.
    model = load_causal_lm(checkpoint_path, dtype=dtype)
    return model, checkpoint_path


def load_topk_map(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"topk_jsonl not found: {path}")
    rows = load_jsonl(path)
    out: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        rid = choose_row_id(row, i)
        vals = parse_topk_indices(row.get("top_k_indices"))
        if vals:
            out[rid] = vals
    return out


def infer_capability_id(
    row: dict[str, Any],
    rid: str,
    topk_map: dict[str, list[int]],
    capability_key: str,
) -> int | None:
    candidate_keys = []
    if capability_key != "auto":
        candidate_keys.append(capability_key)
    candidate_keys.extend(
        [
            "capability_id",
            "capability",
            "cap_idx",
            "cluster_id",
            "cluster",
            "top_capability",
            "top1_capability",
        ]
    )

    for key in candidate_keys:
        if key not in row:
            continue
        value = row.get(key)
        if isinstance(value, list):
            if value:
                try:
                    cap = int(value[0])
                except Exception:  # noqa: BLE001
                    continue
                if cap >= 0:
                    return cap
            continue
        try:
            cap = int(value)
        except Exception:  # noqa: BLE001
            continue
        if cap >= 0:
            return cap

    row_topk = parse_topk_indices(row.get("top_k_indices"))
    if row_topk:
        return int(row_topk[0])

    fallback_topk = topk_map.get(rid, [])
    if fallback_topk:
        return int(fallback_topk[0])

    return None


def group_anchor_rows(
    rows: list[dict[str, Any]],
    topk_map: dict[str, list[int]],
    capability_key: str,
) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for i, row in enumerate(rows):
        rid = choose_row_id(row, i)
        cap = infer_capability_id(row, rid, topk_map, capability_key)
        if cap is None:
            continue
        groups.setdefault(cap, []).append(row)
    return groups


def select_grad_parameters(
    model,
    mode: str,
    name_filter: str,
) -> list[tuple[str, torch.nn.Parameter]]:
    all_named = list(model.named_parameters())

    if mode == "all_trainable":
        selected = [(n, p) for n, p in all_named if p.requires_grad]
    elif mode == "lora":
        selected = [(n, p) for n, p in all_named if "lora_" in n]
        if not selected:
            selected = [(n, p) for n, p in all_named if p.requires_grad]
    elif mode == "lm_head":
        selected = [(n, p) for n, p in all_named if "lm_head" in n]
        if not selected:
            selected = [(n, p) for n, p in all_named if p.requires_grad]
    else:
        selected = [(n, p) for n, p in all_named if name_filter and name_filter in n]
        if not selected:
            selected = [(n, p) for n, p in all_named if p.requires_grad]

    selected_ids = {id(p) for _, p in selected}
    for _, p in all_named:
        p.requires_grad_(id(p) in selected_ids)

    return [(n, p) for n, p in all_named if id(p) in selected_ids]


def flatten_grads(grad_params: list[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for _, p in grad_params:
        g = p.grad
        if g is None:
            parts.append(torch.zeros((p.numel(),), dtype=torch.float32, device=p.device))
        else:
            parts.append(g.detach().to(torch.float32).reshape(-1))
    if not parts:
        return torch.zeros((1,), dtype=torch.float32)
    return torch.cat(parts, dim=0)


def collate_anchor_batch(
    rows: list[dict[str, Any]],
    tokenizer,
    collator: ResponseOnlyCollator,
    max_seq_length: int,
) -> dict[str, torch.Tensor]:
    features: list[dict[str, Any]] = []
    for row in rows:
        prompt = build_prompt(row)
        enc = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )
        features.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc.get("attention_mask"),
                "sample_id": -1,
            }
        )
    return collator(features)


def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    out: list[list[Any]] = []
    for st in range(0, len(items), chunk_size):
        out.append(items[st : st + chunk_size])
    return out


def load_prev_beta(path: Path | None) -> torch.Tensor | None:
    if path is None or (not path.exists()):
        return None
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    arr = obj.get("beta")
    if not isinstance(arr, list):
        return None
    vals = []
    for v in arr:
        try:
            vals.append(float(v))
        except Exception:  # noqa: BLE001
            vals.append(0.0)
    if not vals:
        return None
    t = torch.tensor(vals, dtype=torch.float32)
    t = t.clamp_min(0.0)
    s = t.sum().clamp_min(EPS)
    return t / s


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.anchor_jsonl.exists():
        raise FileNotFoundError(f"anchor_jsonl not found: {args.anchor_jsonl}")

    dtype = get_dtype(args.torch_dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and dtype != torch.float32:
        dtype = torch.float32

    model, tokenizer_source = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        base_model=args.base_model,
        model_source=args.model_source,
        modelscope_cache_dir=args.modelscope_cache_dir,
        dtype=dtype,
    )
    model.to(device)
    model.train()

    grad_params = select_grad_parameters(
        model=model,
        mode=args.grad_param_mode,
        name_filter=args.grad_param_filter,
    )
    if not grad_params:
        raise RuntimeError("No target gradient parameters found.")

    grad_dim = int(sum(p.numel() for _, p in grad_params))
    logging.info("Selected %d gradient params, grad_dim=%d", len(grad_params), grad_dim)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator = ResponseOnlyCollator(tokenizer)

    topk_map = load_topk_map(args.topk_jsonl)
    anchor_rows = load_jsonl(args.anchor_jsonl)
    grouped = group_anchor_rows(anchor_rows, topk_map=topk_map, capability_key=args.capability_key)
    if not grouped:
        raise ValueError("No valid capability anchors found from anchor_jsonl.")

    caps = sorted(grouped.keys())
    if args.max_capabilities > 0 and len(caps) > args.max_capabilities:
        random.shuffle(caps)
        caps = sorted(caps[: args.max_capabilities])

    anchor_grads: dict[int, torch.Tensor] = {}
    anchor_norms: dict[int, float] = {}

    for cap in caps:
        rows = grouped.get(cap, [])
        if not rows:
            continue
        random.shuffle(rows)
        picked = rows[: max(1, min(args.anchor_size_per_cap, len(rows)))]

        cap_vecs: list[torch.Tensor] = []
        for part in chunk_list(picked, max(1, args.anchor_batch_size)):
            model.zero_grad(set_to_none=True)
            batch = collate_anchor_batch(
                rows=part,
                tokenizer=tokenizer,
                collator=collator,
                max_seq_length=max(64, args.max_seq_length),
            )
            batch = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "attention_mask", "labels"}}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            out.loss.backward()
            cap_vecs.append(flatten_grads(grad_params))

        if not cap_vecs:
            continue

        g_cap = torch.stack(cap_vecs, dim=0).mean(dim=0)
        g_norm = float(g_cap.norm().item())
        anchor_norms[cap] = g_norm
        if g_norm > EPS:
            anchor_grads[cap] = (g_cap / g_norm).detach().to(torch.float16).cpu()
        else:
            anchor_grads[cap] = torch.zeros_like(g_cap, dtype=torch.float16).cpu()

    if not anchor_norms:
        raise RuntimeError("Anchor gradient computation produced empty norms.")

    m_dim = max(anchor_norms.keys()) + 1
    prev_beta = load_prev_beta(args.prev_beta_json)
    if prev_beta is not None:
        m_dim = max(m_dim, int(prev_beta.numel()))

    norm_vec = torch.zeros((m_dim,), dtype=torch.float32)
    for cap, val in anchor_norms.items():
        if 0 <= cap < m_dim:
            norm_vec[cap] = float(val)

    if norm_vec.sum().item() <= EPS:
        if prev_beta is not None and prev_beta.numel() == m_dim:
            beta = prev_beta.clone()
        else:
            beta = torch.full((m_dim,), 1.0 / float(m_dim), dtype=torch.float32)
    else:
        beta_new = norm_vec / norm_vec.sum().clamp_min(EPS)
        if prev_beta is not None:
            if prev_beta.numel() < m_dim:
                pad = torch.zeros((m_dim - prev_beta.numel(),), dtype=torch.float32)
                prev_beta = torch.cat([prev_beta, pad], dim=0)
            elif prev_beta.numel() > m_dim:
                prev_beta = prev_beta[:m_dim]
            ema = min(max(args.beta_ema, 0.0), 0.999)
            beta = ema * prev_beta + (1.0 - ema) * beta_new
            beta = beta.clamp_min(0.0)
            beta = beta / beta.sum().clamp_min(EPS)
        else:
            beta = beta_new

    args.output_beta_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_beta_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_path": args.checkpoint_path,
                "m_dim": int(m_dim),
                "updated_capabilities": [int(c) for c in sorted(anchor_norms.keys())],
                "beta": [float(x) for x in beta.tolist()],
                "beta_ema": float(args.beta_ema),
                "grad_param_mode": args.grad_param_mode,
                "grad_param_filter": args.grad_param_filter,
                "anchor_jsonl": str(args.anchor_jsonl),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    args.output_anchor_grads.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "grads": {int(k): v for k, v in anchor_grads.items()},
            "norms": {int(k): float(v) for k, v in anchor_norms.items()},
            "m_dim": int(m_dim),
            "gradient_dim": int(grad_dim),
            "target_param_names": [name for name, _ in grad_params],
            "checkpoint_path": args.checkpoint_path,
        },
        args.output_anchor_grads,
    )

    logging.info(
        "Done. caps=%d, m_dim=%d, beta_path=%s, anchor_grads_path=%s",
        len(anchor_norms),
        m_dim,
        args.output_beta_json,
        args.output_anchor_grads,
    )


if __name__ == "__main__":
    main()
