#!/usr/bin/env python3
"""Real training runner for Online Recipe Evolution (4.3).

Features:
- dialogsum/lawyer/dolly style JSONL instruction datasets
- streaming score loader for multiple mapper files (memory-safe)
- dynamic alpha/beta update via RecipeEvolver
- HuggingFace model loading + optional LoRA
- checkpointing and JSONL training logs
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from recipe_evolver import RecipeEvolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real training runner for RecipeEvolver (4.3).")

    # Data / score
    parser.add_argument("--train-jsonl", type=Path, required=True, help="Training jsonl path.")
    parser.add_argument(
        "--score-path",
        action="append",
        required=True,
        help="Mapper score file path (repeatable). Example: --score-path data/dialogsum/score/pdm_scored.jsonl",
    )
    parser.add_argument("--top-k-path", type=Path, default=None, help="Optional top-k routing file (jsonl).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")

    # Field names
    parser.add_argument("--id-field", type=str, default="id")
    parser.add_argument("--vector-field", type=str, default="mapped_vector")
    parser.add_argument("--fallback-vector-field", type=str, default="score")
    parser.add_argument("--top-k-field", type=str, default="top_k_indices")

    # Tokenization
    parser.add_argument("--base-model", type=str, required=True, help="HF/modelscope model path or id.")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--tokenize-batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use full dataset.")
    parser.add_argument(
        "--response-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask prompt tokens in labels and train only on response tokens.",
    )

    # RecipeEvolution hyper-params
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--anchor-size-per-capability", type=int, default=16)
    parser.add_argument("--anchor-chunk-size", type=int, default=8)
    parser.add_argument(
        "--anchor-refresh-interval",
        type=int,
        default=10,
        help="Recompute anchor gradients and beta every N steps. 1 means every step.",
    )
    parser.add_argument(
        "--anchor-ema-momentum",
        type=float,
        default=0.8,
        help="EMA momentum for anchor gradients.",
    )
    parser.add_argument("--eta-beta", type=float, default=0.1)
    parser.add_argument("--gamma-alpha", type=float, default=5.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--gamma-T", type=float, default=1.0)
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.1,
        help="Penalty coefficient for repeatedly sampled data points.",
    )
    parser.add_argument(
        "--prune-patience",
        type=int,
        default=3,
        help="Prune when consecutive bad rewards >= prune_patience.",
    )
    parser.add_argument(
        "--prune-reward-threshold",
        type=float,
        default=-0.05,
        help="Reward below this threshold is treated as bad sample.",
    )
    parser.add_argument(
        "--mapper-utility-mode",
        type=str,
        default="beta_weighted",
        choices=["beta_weighted", "mean", "sum"],
    )

    # Capability dimension clamp (for practical memory/speed tradeoff)
    parser.add_argument(
        "--max-capability-dim",
        type=int,
        default=0,
        help="If >0, clip m to this value (dims >= m are dropped).",
    )

    # Optim / model
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches for gradient accumulation in each optimizer step.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--torch-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing on model.",
    )

    # Target layer for feedback gradients
    parser.add_argument(
        "--layer-name",
        type=str,
        default="auto",
        help="Layer prefix for feedback gradients. Use 'auto' to infer a practical target layer.",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=8, help="<=0 to disable LoRA.")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj",
        help="Comma-separated module names.",
    )

    # Logging/checkpoints
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Only build everything and print shapes, no training.")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL object at {path}:{line_no}: expected dict")
            yield line_no, row


def choose_row_id(row: Dict[str, Any], fallback_idx: int, id_field: str) -> str:
    if id_field and id_field in row and row.get(id_field) is not None:
        return str(row.get(id_field))
    for key in ("id", "data_id", "uid", "idx", "index"):
        if key in row and row.get(key) is not None:
            return str(row.get(key))
    return str(fallback_idx)


def parse_vector_entries(value: Any) -> List[Tuple[int, float]]:
    entries: List[Tuple[int, float]] = []
    if isinstance(value, list):
        for i, v in enumerate(value):
            try:
                fv = float(v)
            except Exception:  # noqa: BLE001
                continue
            if fv != 0.0:
                entries.append((int(i), fv))
        return entries
    if isinstance(value, dict):
        for k, v in value.items():
            try:
                idx = int(k)
                fv = float(v)
            except Exception:  # noqa: BLE001
                continue
            if idx >= 0 and fv != 0.0:
                entries.append((idx, fv))
        return entries
    return entries


def get_row_vector_entries(
    row: Dict[str, Any],
    vector_field: str,
    fallback_vector_field: str,
) -> List[Tuple[int, float]]:
    entries = parse_vector_entries(row.get(vector_field))
    if entries:
        return entries
    return parse_vector_entries(row.get(fallback_vector_field))


def extract_topk_indices(row: Dict[str, Any], top_k_field: str) -> List[int]:
    val = row.get(top_k_field)
    if not isinstance(val, list):
        return []
    out: List[int] = []
    for x in val:
        try:
            out.append(int(x))
        except Exception:  # noqa: BLE001
            continue
    return out


def build_prompt_and_text(row: Dict[str, Any]) -> Tuple[str, str]:
    instruction = str(row.get("instruction", "") or "").strip()
    input_text = str(row.get("input", "") or "").strip()
    output = str(row.get("output", "") or "").strip()

    if instruction or input_text or output:
        prompt_parts = ["Below is an instruction that describes a task"]
        if input_text:
            prompt_parts[-1] += ", paired with an input that provides further context."
        prompt_parts[-1] += " Write a response that appropriately completes the request.\n\n"
        prompt_parts.append("### Instruction:\n")
        prompt_parts.append(instruction)
        prompt_parts.append("\n\n")
        if input_text:
            prompt_parts.append("### Input:\n")
            prompt_parts.append(input_text)
            prompt_parts.append("\n\n")
        prompt_parts.append("### Response:\n")
        prompt = "".join(prompt_parts)
        return prompt, prompt + output

    text = str(row.get("text", "") or "").strip()
    if text:
        marker = "### Response:\n"
        pos = text.rfind(marker)
        if pos >= 0:
            split = pos + len(marker)
            return text[:split], text
        return "", text

    fallback = json.dumps(row, ensure_ascii=False)
    return "", fallback


def tokenize_pool_from_jsonl(
    *,
    train_jsonl: Path,
    tokenizer: AutoTokenizer,
    id_field: str,
    max_length: int,
    tokenize_batch_size: int,
    max_samples: int,
    response_only: bool,
) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    row_ids: List[str] = []
    input_chunks: List[torch.Tensor] = []
    attention_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []

    buf_texts: List[str] = []
    buf_prompts: List[str] = []

    def flush_buffer() -> None:
        if not buf_texts:
            return

        enc = tokenizer(
            buf_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        if response_only:
            prompt_enc = tokenizer(
                buf_prompts,
                truncation=True,
                padding=False,
                max_length=max_length,
                add_special_tokens=True,
            )
            prompt_lens = [len(x) for x in prompt_enc["input_ids"]]
            for i, plen in enumerate(prompt_lens):
                if plen > 0:
                    labels[i, : min(plen, max_length)] = -100

        # 防止某些样本在 response-only + 截断后标签全为 -100 导致 loss=nan。
        for i in range(labels.size(0)):
            if torch.all(labels[i] == -100):
                valid_pos = torch.nonzero(attention_mask[i], as_tuple=False).reshape(-1)
                if valid_pos.numel() > 0:
                    last_pos = int(valid_pos[-1].item())
                    labels[i, last_pos] = input_ids[i, last_pos]

        input_chunks.append(input_ids)
        attention_chunks.append(attention_mask)
        label_chunks.append(labels)

        buf_texts.clear()
        buf_prompts.clear()

    count = 0
    for line_no, row in iter_jsonl(train_jsonl):
        rid = choose_row_id(row, line_no - 1, id_field)
        prompt, full_text = build_prompt_and_text(row)

        row_ids.append(rid)
        buf_prompts.append(prompt)
        buf_texts.append(full_text)
        count += 1

        if len(buf_texts) >= tokenize_batch_size:
            flush_buffer()

        if max_samples > 0 and count >= max_samples:
            break

    flush_buffer()

    if not row_ids:
        raise ValueError(f"No rows loaded from {train_jsonl}")

    input_ids = torch.cat(input_chunks, dim=0)
    attention_mask = torch.cat(attention_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)

    if input_ids.size(0) != len(row_ids):
        raise RuntimeError("Tokenization count mismatch.")

    return row_ids, input_ids, attention_mask, labels


def infer_m_and_kmax(
    *,
    score_paths: Sequence[Path],
    top_k_path: Path | None,
    row_pos: Dict[str, int],
    id_field: str,
    vector_field: str,
    fallback_vector_field: str,
    top_k_field: str,
    max_capability_dim: int,
) -> Tuple[int, int]:
    max_dim = -1
    kmax = 0

    def update_from_row(row: Dict[str, Any]) -> None:
        nonlocal max_dim, kmax
        entries = get_row_vector_entries(row, vector_field, fallback_vector_field)
        for d, _v in entries:
            if d > max_dim:
                max_dim = d
        tk = extract_topk_indices(row, top_k_field)
        if tk:
            kmax = max(kmax, len(tk))
            for d in tk:
                if d > max_dim:
                    max_dim = d

    for sp in score_paths:
        for line_no, row in iter_jsonl(sp):
            rid = choose_row_id(row, line_no - 1, id_field)
            if rid not in row_pos:
                continue
            update_from_row(row)

    if top_k_path is not None:
        for line_no, row in iter_jsonl(top_k_path):
            rid = choose_row_id(row, line_no - 1, id_field)
            if rid not in row_pos:
                continue
            update_from_row(row)

    m = max_dim + 1
    if m <= 0:
        raise ValueError("Failed to infer capability dimension m.")
    if max_capability_dim > 0:
        m = min(m, int(max_capability_dim))
    if kmax <= 0:
        raise ValueError("Failed to infer K_max. Ensure top_k indices exist in score/top-k files.")
    return m, kmax


def build_score_tensors_streaming(
    *,
    row_ids: List[str],
    score_paths: Sequence[Path],
    top_k_path: Path | None,
    id_field: str,
    vector_field: str,
    fallback_vector_field: str,
    top_k_field: str,
    max_capability_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    row_pos = {rid: i for i, rid in enumerate(row_ids)}
    n = len(row_ids)
    k = len(score_paths)

    m, kmax = infer_m_and_kmax(
        score_paths=score_paths,
        top_k_path=top_k_path,
        row_pos=row_pos,
        id_field=id_field,
        vector_field=vector_field,
        fallback_vector_field=fallback_vector_field,
        top_k_field=top_k_field,
        max_capability_dim=max_capability_dim,
    )

    e_matrix = torch.zeros((n, k, m), dtype=torch.float32)
    top_k_indices = torch.full((n, kmax), fill_value=-1, dtype=torch.long)

    for mapper_idx, sp in enumerate(score_paths):
        filled = 0
        for line_no, row in iter_jsonl(sp):
            rid = choose_row_id(row, line_no - 1, id_field)
            pos = row_pos.get(rid)
            if pos is None:
                continue
            entries = get_row_vector_entries(row, vector_field, fallback_vector_field)
            if entries:
                filled += 1
            for dim_idx, val in entries:
                if 0 <= dim_idx < m:
                    e_matrix[pos, mapper_idx, dim_idx] = float(val)

            if top_k_path is None and mapper_idx == 0:
                tk = extract_topk_indices(row, top_k_field)
                for t, dim_idx in enumerate(tk[:kmax]):
                    if 0 <= dim_idx < m:
                        top_k_indices[pos, t] = int(dim_idx)
        print(f"[Score] mapper#{mapper_idx} loaded from {sp} | rows_with_vector={filled}", flush=True)

    if top_k_path is not None:
        filled_topk = 0
        for line_no, row in iter_jsonl(top_k_path):
            rid = choose_row_id(row, line_no - 1, id_field)
            pos = row_pos.get(rid)
            if pos is None:
                continue
            tk = extract_topk_indices(row, top_k_field)
            if tk:
                filled_topk += 1
            for t, dim_idx in enumerate(tk[:kmax]):
                if 0 <= dim_idx < m:
                    top_k_indices[pos, t] = int(dim_idx)
        print(f"[Score] top-k loaded from {top_k_path} | rows_with_topk={filled_topk}", flush=True)

    return e_matrix, top_k_indices


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg == "bf16":
        return torch.bfloat16
    if dtype_arg == "fp16":
        return torch.float16

    # auto
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def maybe_apply_lora(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if args.lora_r <= 0:
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    if not targets:
        raise ValueError("LoRA enabled but --lora-target-modules is empty.")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model


def infer_auto_layer_name(model: nn.Module) -> str:
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_names:
        raise ValueError("No trainable parameters found in model.")

    best_idx = -1
    best_prefix = ""
    for name in trainable_names:
        if ".mlp.down_proj." not in name:
            continue
        prefix = name.split(".lora_")[0]
        if prefix.endswith(".weight"):
            prefix = prefix[: -len(".weight")]
        if prefix.endswith(".bias"):
            prefix = prefix[: -len(".bias")]
        m = re.search(r"\.layers\.(\d+)\.", prefix)
        if not m:
            continue
        idx = int(m.group(1))
        if idx > best_idx:
            best_idx = idx
            best_prefix = prefix

    if best_prefix:
        return best_prefix

    if any(name.startswith("lm_head") for name in trainable_names):
        return "lm_head"

    fallback = trainable_names[0]
    if fallback.endswith(".weight"):
        fallback = fallback[: -len(".weight")]
    if fallback.endswith(".bias"):
        fallback = fallback[: -len(".bias")]
    return fallback


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def topk_pairs(t: torch.Tensor, k: int) -> List[List[float]]:
    if t.numel() == 0:
        return []
    kk = min(k, int(t.numel()))
    vals, idx = torch.topk(t, k=kk)
    out: List[List[float]] = []
    for i, v in zip(idx.tolist(), vals.tolist()):
        out.append([int(i), float(v)])
    return out


def args_to_jsonable(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, list):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            out[k] = v
    return out


def save_checkpoint(
    *,
    output_dir: Path,
    tag: str,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    evolver: RecipeEvolver,
    args: argparse.Namespace,
) -> None:
    ckpt_dir = output_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    state = {
        "step": int(evolver.step_id),
        "alpha": evolver.alpha.detach().cpu(),
        "beta": evolver.beta.detach().cpu(),
        "args": args_to_jsonable(args),
    }
    torch.save(state, ckpt_dir / "recipe_state.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    config_path.write_text(json.dumps(args_to_jsonable(args), ensure_ascii=False, indent=2), encoding="utf-8")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.torch_dtype, device)
    print(f"[Init] device={device} dtype={dtype}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[Data] tokenizing train pool...", flush=True)
    row_ids, input_ids, attention_mask, labels = tokenize_pool_from_jsonl(
        train_jsonl=args.train_jsonl,
        tokenizer=tokenizer,
        id_field=args.id_field,
        max_length=args.max_length,
        tokenize_batch_size=args.tokenize_batch_size,
        max_samples=args.max_samples,
        response_only=args.response_only,
    )
    print(
        f"[Data] pool_size={len(row_ids)} token_shape={tuple(input_ids.shape)} response_only={args.response_only}",
        flush=True,
    )

    score_paths = [Path(p) for p in args.score_path]
    print("[Score] building E_matrix/top_k_indices (streaming)...", flush=True)
    e_matrix, top_k_indices = build_score_tensors_streaming(
        row_ids=row_ids,
        score_paths=score_paths,
        top_k_path=args.top_k_path,
        id_field=args.id_field,
        vector_field=args.vector_field,
        fallback_vector_field=args.fallback_vector_field,
        top_k_field=args.top_k_field,
        max_capability_dim=args.max_capability_dim,
    )
    print(
        f"[Score] E_matrix={tuple(e_matrix.shape)} top_k_indices={tuple(top_k_indices.shape)}",
        flush=True,
    )

    print(f"[Model] loading base model: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model = maybe_apply_lora(model, args)
    model.to(device)
    model.train()

    trainable, total = count_trainable_params(model)
    print(f"[Model] trainable_params={trainable} / total_params={total}", flush=True)

    layer_name = args.layer_name
    if layer_name == "auto":
        layer_name = infer_auto_layer_name(model)
    print(f"[Model] feedback layer_name={layer_name}", flush=True)

    pool_batch = {
        "input_ids": input_ids,
        "labels": labels,
    }

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    evolver = RecipeEvolver(
        model=model,
        optimizer=optimizer,
        pool_batch=pool_batch,
        E_matrix=e_matrix,
        top_k_indices=top_k_indices,
        layer_name=layer_name,
        eta_beta=args.eta_beta,
        gamma_alpha=args.gamma_alpha,
        epsilon=args.epsilon,
        gamma_T=args.gamma_T,
        frequency_penalty=args.frequency_penalty,
        anchor_ema_momentum=args.anchor_ema_momentum,
        prune_patience=args.prune_patience,
        prune_reward_threshold=args.prune_reward_threshold,
        mapper_utility_mode=args.mapper_utility_mode,
        score_device="cpu",
    )

    print(
        f"[Recipe] N={evolver.N} k={evolver.k} m={evolver.m} K_max={evolver.K_max} gradient_dim={evolver.gradient_dim}",
        flush=True,
    )
    effective_train_batch = args.batch_size * max(1, int(args.gradient_accumulation_steps))
    print(
        f"[Train] sampled_batch={args.batch_size} grad_accum={max(1, int(args.gradient_accumulation_steps))} "
        f"effective_train_batch={effective_train_batch}",
        flush=True,
    )

    if args.dry_run:
        print("[DryRun] setup complete; exit without training.", flush=True)
        return

    log_path = args.output_dir / "train_log.jsonl"
    alpha_hist_path = args.output_dir / "alpha_history.pt"
    beta_hist_path = args.output_dir / "beta_history.pt"

    anchor_grads: torch.Tensor | None = None

    grad_accum_steps = max(1, int(args.gradient_accumulation_steps))

    t0 = time.time()
    for step in range(1, args.num_steps + 1):
        step_start = time.time()

        refresh_interval = max(1, int(args.anchor_refresh_interval))
        need_refresh = anchor_grads is None or ((step - 1) % refresh_interval == 0)

        running_anchor = evolver.get_running_anchor_gradients()
        if need_refresh or running_anchor is None:
            anchor_indices = evolver.sample_anchor_indices(args.anchor_size_per_capability)
            current_anchor_grads = evolver.compute_anchor_gradients(
                anchor_indices_by_capability=anchor_indices,
                chunk_size=args.anchor_chunk_size,
            )
            anchor_grads = evolver.update_running_anchor_gradients(
                current_anchor_grads,
                momentum=args.anchor_ema_momentum,
            )
            evolver.update_beta(anchor_grads)
        else:
            anchor_grads = running_anchor
            if anchor_grads is None:
                raise RuntimeError("Running anchor gradients are unexpectedly None.")

        evolver.optimizer.zero_grad(set_to_none=True)

        loss_vals: List[float] = []
        rewards_list: List[torch.Tensor] = []
        utilities_list: List[torch.Tensor] = []
        pruned_mask = evolver.pruned_mask

        for _acc_idx in range(grad_accum_steps):
            sampled_idx, _probs, _scores = evolver.score_and_sample_batch(batch_size=args.batch_size)
            batch = evolver._gather_pool_batch(sampled_idx)  # intentionally using internal helper

            per_sample_grads = evolver.compute_per_sample_layer_gradients_vmap(batch)
            rewards = evolver.compute_local_rewards(
                batch_indices=sampled_idx,
                per_sample_grads=per_sample_grads,
                anchor_gradients=anchor_grads,
            )
            _bad_counter, pruned_mask = evolver.update_pruning_state(
                batch_indices=sampled_idx,
                rewards=rewards,
                prune_patience=args.prune_patience,
                prune_reward_threshold=args.prune_reward_threshold,
            )
            loss_i = evolver.train_on_sampled_batch(
                batch,
                gradient_accumulation_steps=1,
                loss_divisor=float(grad_accum_steps),
                zero_grad=False,
                step_optimizer=False,
            )
            utilities_i = evolver.compute_mapper_utilities(batch_indices=sampled_idx, rewards=rewards)

            loss_vals.append(float(loss_i))
            rewards_list.append(rewards.detach())
            utilities_list.append(utilities_i.detach())

        evolver.optimizer.step()

        train_loss = float(sum(loss_vals) / max(1, len(loss_vals)))
        rewards = torch.cat(rewards_list, dim=0) if rewards_list else torch.zeros(0, dtype=torch.float32)
        if utilities_list:
            utilities = torch.stack(utilities_list, dim=0).mean(dim=0)
        else:
            utilities = torch.zeros((evolver.k,), dtype=torch.float32, device=evolver.score_device)
        alpha_new = evolver.update_alpha(utilities)

        evolver.step_id += 1
        evolver.alpha_history.append(alpha_new.detach().clone().cpu())
        evolver.beta_history.append(evolver.beta.detach().clone().cpu())

        elapsed = time.time() - step_start
        log_row = {
            "step": int(step),
            "train_loss": float(train_loss),
            "avg_reward": float(rewards.mean().item()) if rewards.numel() > 0 else 0.0,
            "sampled_total": int(rewards.numel()),
            "grad_accum_steps": int(grad_accum_steps),
            "anchor_refreshed": bool(need_refresh),
            "pruned_count": int(pruned_mask.sum().item()),
            "active_count": int((~pruned_mask).sum().item()),
            "alpha": [float(x) for x in alpha_new.detach().cpu().tolist()],
            "utilities": [float(x) for x in utilities.detach().cpu().tolist()],
            "beta_top10": topk_pairs(evolver.beta.detach().cpu(), 10),
            "step_sec": float(elapsed),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

        if step % max(1, args.log_every) == 0 or step == 1:
            print(
                f"[Step {step}/{args.num_steps}] "
                f"loss={train_loss:.4f} reward={log_row['avg_reward']:.4f} "
                f"anchor_refresh={need_refresh} pruned={log_row['pruned_count']} "
                f"active={log_row['active_count']} sec={elapsed:.2f}",
                flush=True,
            )

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(
                output_dir=args.output_dir,
                tag=f"checkpoint-{step}",
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                evolver=evolver,
                args=args,
            )
            torch.save(evolver.alpha_history, alpha_hist_path)
            torch.save(evolver.beta_history, beta_hist_path)

    # final save
    save_checkpoint(
        output_dir=args.output_dir,
        tag="final",
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        evolver=evolver,
        args=args,
    )
    torch.save(evolver.alpha_history, alpha_hist_path)
    torch.save(evolver.beta_history, beta_hist_path)

    total_sec = time.time() - t0
    print(f"[Done] steps={args.num_steps} total_sec={total_sec:.1f} output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
