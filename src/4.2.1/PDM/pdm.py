#!/usr/bin/env python3
"""Perplexity Deviation Mapping (PDM) scorer for 4.2.

This implementation aligns with Section 4.3 in dataSelectionRecipe_LLM:
- global random contextual baseline loss
- capability-conditioned loss on active clusters
- deviation mapping with soft penalty

It integrates existing SRM outputs by:
1) taking active capability dimensions from SRM
2) building per-cluster representative pools from SRM dimension scores
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn.functional as F
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # noqa: BLE001
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


logger = logging.getLogger(__name__)

PROMPT_HEADER = (
    "You are a helpful assistant. Learn from demonstrations and output the correct answer.\n"
    "Only produce the final output text.\n\n"
)


def choose_row_id(row: Dict[str, Any], fallback_idx: int) -> Any:
    for key in ("id", "data_id", "uid", "idx", "index"):
        if key in row and row[key] is not None:
            return row[key]
    return fallback_idx


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def load_json_or_jsonl(path: str) -> Tuple[List[Dict[str, Any]], str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return [], "jsonl"

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj, "json"
        raise ValueError("Input JSON must be a list of rows.")
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return rows, "jsonl"


def write_json_or_jsonl(path: str, rows: List[Dict[str, Any]], prefer_jsonl: bool) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_is_jsonl = prefer_jsonl or path.lower().endswith(".jsonl")
    with open(path, "w", encoding="utf-8") as f:
        if output_is_jsonl:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            json.dump(rows, f, ensure_ascii=False, indent=2)


def default_qwen_model_path() -> str:
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, ".cache/modelscope/hub/Qwen/Qwen2.5-0.5B-Instruct"),
        os.path.join(home, ".cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct"),
        os.path.join(home, ".cache/modelscope/hub/Qwen/Qwen2___5-0___5B-Instruct"),
        os.path.join(home, ".cache/modelscope/hub/models/Qwen/Qwen2___5-0___5B-Instruct"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "Qwen/Qwen2.5-0.5B-Instruct"


def parse_score_vector(value: Any) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if isinstance(value, list):
        for i, v in enumerate(value):
            fv = to_float(v, 0.0)
            if fv > 0.0:
                out[int(i)] = float(fv)
        return out

    if isinstance(value, dict):
        for k, v in value.items():
            idx = to_int(k, -1)
            if idx < 0:
                continue
            fv = to_float(v, 0.0)
            if fv > 0.0:
                out[int(idx)] = float(fv)
        return out

    return out


def infer_m_from_srm_row(row: Dict[str, Any]) -> int:
    max_idx = -1
    for key in ("top_k_indices",):
        val = row.get(key)
        if isinstance(val, list):
            for x in val:
                idx = to_int(x, -1)
                if idx > max_idx:
                    max_idx = idx
    vec = parse_score_vector(row.get("mapped_vector"))
    if not vec:
        vec = parse_score_vector(row.get("score"))
    for idx in vec.keys():
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    var = sum((float(v) - mu) ** 2 for v in values) / len(values)
    return float(math.sqrt(max(0.0, var)))


def extract_optional_list(row: Dict[str, Any], key: str) -> Optional[List[Any]]:
    v = row.get(key)
    if isinstance(v, list):
        return v
    return None


def extract_target_text(row: Dict[str, Any]) -> str:
    for key in ("output", "response", "label", "category", "answer"):
        val = row.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s:
            return s
    return ""


def format_instruction_input(row: Dict[str, Any]) -> Tuple[str, str]:
    instruction = str(row.get("instruction", "") or "").strip()
    input_text = str(row.get("input", "") or "").strip()
    if not instruction and not input_text:
        text = str(row.get("text", "") or "").strip()
        return text, ""
    return instruction, input_text


def format_demo_tail(instruction: str, input_text: str, answer: str) -> str:
    parts = [f"Instruction: {instruction}\n"]
    if input_text:
        parts.append(f"Input: {input_text}\n")
    parts.append(f"Output: {answer}\n\n")
    return "".join(parts)


def format_query_prefix_text(instruction: str, input_text: str) -> str:
    parts = ["### Query\n", f"Instruction: {instruction}\n"]
    if input_text:
        parts.append(f"Input: {input_text}\n")
    parts.append("Output:")
    return "".join(parts)


def precompute_text_cache(
    rows: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[str]]:
    demo_tails: List[str] = []
    query_prefixes: List[str] = []
    targets: List[str] = []
    for row in rows:
        instruction, input_text = format_instruction_input(row)
        answer = extract_target_text(row)
        demo_tails.append(format_demo_tail(instruction, input_text, answer))
        query_prefixes.append(format_query_prefix_text(instruction, input_text))
        targets.append(answer)
    return demo_tails, query_prefixes, targets


@dataclass(frozen=True)
class SrmItem:
    row_index: int
    sample_id: Any
    active_indices: List[int]
    score_by_dim: Dict[int, float]
    top_k_indices: List[int]
    top_k_scores: List[float]
    top_k_node_ids: List[str]
    top_k_node_names: List[str]
    top_k_node_paths: List[str]


def build_active_indices(
    sparse_scores: Dict[int, float],
    top_k_indices: List[int],
    max_top_k: int,
    mode: str,
) -> List[int]:
    sorted_nonzero = [k for k, _v in sorted(sparse_scores.items(), key=lambda kv: kv[1], reverse=True)]
    if mode == "srm_nonzero":
        out = sorted_nonzero
    elif mode == "topk":
        out = [int(x) for x in top_k_indices if int(x) >= 0]
    else:
        seen: set[int] = set()
        out = []
        for idx in sorted_nonzero:
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        for idx in top_k_indices:
            idx_i = int(idx)
            if idx_i < 0 or idx_i in seen:
                continue
            seen.add(idx_i)
            out.append(idx_i)

    if max_top_k > 0:
        out = out[:max_top_k]
    return out


def load_srm_items(
    srm_path: str,
    max_top_k: int,
    active_set_mode: str,
) -> Tuple[Dict[str, SrmItem], int]:
    rows, _fmt = load_json_or_jsonl(srm_path)
    items: Dict[str, SrmItem] = {}
    inferred_m = 0

    for i, row in enumerate(rows):
        sample_id = row.get("id", i)
        top_k_indices_raw = extract_optional_list(row, "top_k_indices") or []
        top_k_scores_raw = extract_optional_list(row, "top_k_scores") or []

        top_k_indices: List[int] = []
        top_k_scores: List[float] = []
        for rank, raw_idx in enumerate(top_k_indices_raw):
            idx = to_int(raw_idx, -1)
            if idx < 0:
                continue
            score = to_float(top_k_scores_raw[rank], 0.0) if rank < len(top_k_scores_raw) else 0.0
            top_k_indices.append(idx)
            top_k_scores.append(score)
            inferred_m = max(inferred_m, idx + 1)

        sparse_scores = parse_score_vector(row.get("mapped_vector"))
        if not sparse_scores:
            sparse_scores = parse_score_vector(row.get("score"))
        for idx in sparse_scores.keys():
            inferred_m = max(inferred_m, int(idx) + 1)

        active_indices = build_active_indices(
            sparse_scores=sparse_scores,
            top_k_indices=top_k_indices,
            max_top_k=max_top_k,
            mode=active_set_mode,
        )

        top_k_node_ids_raw = extract_optional_list(row, "top_k_node_ids") or []
        top_k_node_names_raw = extract_optional_list(row, "top_k_node_names") or []
        top_k_node_paths_raw = extract_optional_list(row, "top_k_node_paths") or []

        item = SrmItem(
            row_index=i,
            sample_id=sample_id,
            active_indices=[int(x) for x in active_indices],
            score_by_dim=dict(sparse_scores),
            top_k_indices=top_k_indices,
            top_k_scores=top_k_scores,
            top_k_node_ids=[str(x or "").strip() for x in top_k_node_ids_raw],
            top_k_node_names=[str(x or "").strip() for x in top_k_node_names_raw],
            top_k_node_paths=[str(x or "").strip() for x in top_k_node_paths_raw],
        )
        items[str(sample_id)] = item

    return items, inferred_m


def resolve_srm_item(
    srm_map: Dict[str, SrmItem],
    sample: Dict[str, Any],
    fallback_idx: int,
) -> Tuple[Any, Optional[SrmItem]]:
    sample_id = choose_row_id(sample, fallback_idx)
    item = srm_map.get(str(sample_id))
    if item is None and str(sample_id) != str(fallback_idx):
        item = srm_map.get(str(fallback_idx))
    return sample_id, item


def parse_torch_dtype(dtype_name: str) -> Any:
    if torch is None:
        return None
    dn = dtype_name.strip().lower()
    if dn == "auto":
        return None
    if dn == "float16":
        return torch.float16
    if dn == "bfloat16":
        return torch.bfloat16
    if dn == "float32":
        return torch.float32
    raise ValueError("--torch_dtype must be one of: auto, float16, bfloat16, float32")


def load_model_and_tokenizer(
    model_name_or_path: str,
    torch_dtype: str,
    device_map: str,
) -> Tuple[Any, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required. Please install torch.")
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required. Please install transformers.")

    model_path = str(model_name_or_path).strip()
    if not model_path:
        raise ValueError("Missing --model_name_or_path.")

    logger.info("Loading reference model from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    td = parse_torch_dtype(torch_dtype)
    if td is not None:
        model_kwargs["torch_dtype"] = td

    dm = str(device_map).strip().lower()
    if dm and dm != "none":
        model_kwargs["device_map"] = dm

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()

    if "device_map" not in model_kwargs:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)

    return model, tokenizer


def auto_adjust_nll_batch_size(scorer: "PDMScorer", requested_batch_size: int) -> int:
    if torch is None:
        return max(1, int(requested_batch_size))
    req = max(1, int(requested_batch_size))
    try:
        dev = scorer._device()
    except Exception:  # noqa: BLE001
        return req
    if str(getattr(dev, "type", "")).lower() != "cuda":
        return req

    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(dev)
    except Exception:  # noqa: BLE001
        return req

    try:
        vocab_size = int(getattr(scorer.model.config, "vocab_size", 0))
    except Exception:  # noqa: BLE001
        vocab_size = 0
    if vocab_size <= 0:
        vocab_size = int(getattr(scorer.tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        return req

    try:
        param_dtype = next(scorer.model.parameters()).dtype
        dtype_bytes = int(torch.tensor([], dtype=param_dtype).element_size())
    except Exception:  # noqa: BLE001
        dtype_bytes = 2

    # Approximate per-sample logits memory: [T-1, V] * dtype_bytes.
    # Keep conservative headroom because CE and activations need extra memory.
    seq = max(2, int(scorer.max_seq_len))
    per_sample_bytes = max(1, (seq - 1) * vocab_size * max(1, dtype_bytes))
    headroom_ratio = 0.35
    safe = int((float(free_bytes) * headroom_ratio) // float(per_sample_bytes))
    return max(1, min(req, safe if safe > 0 else 1))


class PDMScorer:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_seq_len: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max(128, int(max_seq_len))
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        self.pad_id = int(pad_id)
        self._prompt_header_ids: Optional[List[int]] = None
        self._demo_header_ids: Dict[int, List[int]] = {}

    def _device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except Exception:  # noqa: BLE001
            if torch is None:
                raise RuntimeError("PyTorch is required.")
            return torch.device("cpu")

    def _format_demo_block(self, row: Dict[str, Any], idx: int) -> str:
        instruction, input_text = format_instruction_input(row)
        answer = extract_target_text(row)
        return f"### Demonstration {idx}\n" + format_demo_tail(instruction, input_text, answer)

    def _format_query_prefix(self, row: Dict[str, Any]) -> str:
        instruction, input_text = format_instruction_input(row)
        return format_query_prefix_text(instruction, input_text)

    def _build_prefix(self, query_row: Dict[str, Any], context_rows: List[Dict[str, Any]]) -> str:
        chunks = [PROMPT_HEADER]
        for i, row in enumerate(context_rows, start=1):
            chunks.append(self._format_demo_block(row, i))
        chunks.append(self._format_query_prefix(query_row))
        return "".join(chunks)

    def build_prefix_from_cache(
        self,
        query_prefix: str,
        context_indices: List[int],
        demo_tails: List[str],
    ) -> str:
        chunks = [PROMPT_HEADER]
        for i, row_idx in enumerate(context_indices, start=1):
            chunks.append(f"### Demonstration {i}\n")
            chunks.append(demo_tails[row_idx])
        chunks.append(query_prefix)
        return "".join(chunks)

    def encode_text(self, text: str) -> List[int]:
        return [int(x) for x in self.tokenizer.encode(text, add_special_tokens=False)]

    def prompt_header_ids(self) -> List[int]:
        if self._prompt_header_ids is None:
            self._prompt_header_ids = self.encode_text(PROMPT_HEADER)
        return self._prompt_header_ids

    def demo_header_ids(self, demo_rank: int) -> List[int]:
        out = self._demo_header_ids.get(int(demo_rank))
        if out is not None:
            return out
        out = self.encode_text(f"### Demonstration {int(demo_rank)}\n")
        self._demo_header_ids[int(demo_rank)] = out
        return out

    def build_prefix_ids_from_token_cache(
        self,
        query_prefix_ids: List[int],
        context_indices: List[int],
        demo_tail_ids_cache: Dict[int, List[int]],
        demo_tails: List[str],
    ) -> List[int]:
        out: List[int] = []
        out.extend(self.prompt_header_ids())
        for i, row_idx in enumerate(context_indices, start=1):
            out.extend(self.demo_header_ids(i))
            tail_ids = demo_tail_ids_cache.get(int(row_idx))
            if tail_ids is None:
                tail_ids = self.encode_text(demo_tails[int(row_idx)])
                demo_tail_ids_cache[int(row_idx)] = tail_ids
            out.extend(tail_ids)
        out.extend(query_prefix_ids)
        return out

    def _encode_for_loss_ids(self, prefix_ids: List[int], target_ids: List[int]) -> Tuple[List[int], List[int]]:
        prefix_ids = [int(x) for x in prefix_ids]
        target_ids = [int(x) for x in target_ids]

        if not target_ids:
            eos_id = self.tokenizer.eos_token_id
            target_ids = [int(eos_id)] if eos_id is not None else [self.pad_id]

        if len(target_ids) == 1:
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                target_ids = target_ids + [int(eos_id)]

        if len(target_ids) >= self.max_seq_len:
            target_ids = target_ids[: self.max_seq_len]
            prefix_ids = []
        else:
            remain = self.max_seq_len - len(target_ids)
            if len(prefix_ids) > remain:
                prefix_ids = prefix_ids[-remain:]

        input_ids = prefix_ids + target_ids
        labels = ([-100] * len(prefix_ids)) + target_ids
        return input_ids, labels

    def _encode_for_loss(self, prefix: str, target: str) -> Tuple[List[int], List[int]]:
        prefix_ids = self.encode_text(prefix)
        target_ids = self.encode_text(target)
        return self._encode_for_loss_ids(prefix_ids=prefix_ids, target_ids=target_ids)

    def _batch_nll(self, encoded_batch: List[Tuple[List[int], List[int]]]) -> List[float]:
        if torch is None or F is None:
            raise RuntimeError("PyTorch is required.")
        if not encoded_batch:
            return []

        bsz = len(encoded_batch)
        max_len = max(len(x[0]) for x in encoded_batch)
        if max_len <= 1:
            return [0.0 for _ in encoded_batch]

        dev = self._device()
        input_ids = torch.full((bsz, max_len), int(self.pad_id), dtype=torch.long, device=dev)
        labels = torch.full((bsz, max_len), -100, dtype=torch.long, device=dev)
        attn = torch.zeros((bsz, max_len), dtype=torch.long, device=dev)

        for i, (ids, lbs) in enumerate(encoded_batch):
            n = len(ids)
            if n == 0:
                continue
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long, device=dev)
            labels[i, :n] = torch.tensor(lbs, dtype=torch.long, device=dev)
            attn[i, :n] = 1

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False, return_dict=True)

        logits = outputs.logits  # [B, T, V]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(bsz, -1)

        valid = (shift_labels != -100).float()
        denom = valid.sum(dim=1).clamp_min(1.0)
        nll = (token_losses * valid).sum(dim=1) / denom
        return [float(x.item()) for x in nll]

    def compute_losses(
        self,
        query_row: Dict[str, Any],
        context_rows_batch: List[List[Dict[str, Any]]],
    ) -> List[float]:
        target = extract_target_text(query_row)
        if not target:
            return [0.0 for _ in context_rows_batch]

        encoded: List[Tuple[List[int], List[int]]] = []
        for ctx_rows in context_rows_batch:
            prefix = self._build_prefix(query_row, ctx_rows)
            encoded.append(self._encode_for_loss(prefix=prefix, target=target))
        return self._batch_nll(encoded)


def build_cluster_pools(
    srm_map: Dict[str, SrmItem],
    id_to_data_index: Dict[str, int],
    top_n: int,
) -> Dict[int, List[int]]:
    accum: Dict[int, List[Tuple[float, int]]] = {}
    for item in srm_map.values():
        sample_idx = id_to_data_index.get(str(item.sample_id))
        if sample_idx is None:
            continue
        for dim, sc in item.score_by_dim.items():
            if float(sc) <= 0.0:
                continue
            accum.setdefault(int(dim), []).append((float(sc), int(sample_idx)))

    pools: Dict[int, List[int]] = {}
    for dim, pairs in accum.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
        if top_n > 0:
            pairs_sorted = pairs_sorted[:top_n]
        pools[dim] = [idx for _score, idx in pairs_sorted]
    return pools


def sample_context_indices(
    primary_candidates: List[int],
    global_candidates: List[int],
    exclude_index: int,
    k: int,
    rng: random.Random,
) -> List[int]:
    if k <= 0:
        return []

    primary = [x for x in primary_candidates if x != exclude_index]
    selected: List[int] = []
    selected_set: set[int] = set()

    if len(primary) >= k:
        chosen = rng.sample(primary, k)
        selected.extend(chosen)
        return selected

    for x in primary:
        if x not in selected_set:
            selected_set.add(x)
            selected.append(x)

    needed = k - len(selected)
    fallback = [x for x in global_candidates if x != exclude_index and x not in selected_set]
    if len(fallback) > needed:
        fallback = rng.sample(fallback, needed)
    selected.extend(fallback)
    return selected


def sample_global_context_indices(
    global_candidates: List[int],
    exclude_index: int,
    k: int,
    rng: random.Random,
) -> List[int]:
    if k <= 0:
        return []
    n = len(global_candidates)
    if n <= 1:
        return []
    if k >= n:
        return [x for x in global_candidates if x != exclude_index][:k]

    # Here global_candidates is constructed as list(range(N)).
    # Sample from [0, N-2] then shift around exclude to avoid O(N) filtering.
    picks = rng.sample(range(n - 1), k)
    out: List[int] = []
    ex = int(exclude_index)
    for p in picks:
        idx = int(p if p < ex else p + 1)
        out.append(int(global_candidates[idx]))
    return out


def infer_m_from_srm_map(srm_map: Dict[str, SrmItem]) -> int:
    max_idx = -1
    for item in srm_map.values():
        for idx in item.top_k_indices:
            if idx > max_idx:
                max_idx = idx
        for idx in item.score_by_dim.keys():
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1 if max_idx >= 0 else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDM scorer: contextual perplexity deviation mapping guided by SRM."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Raw data path (JSON/JSONL).")
    parser.add_argument("--srm_path", type=str, required=True, help="SRM output path (JSON/JSONL).")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for scored rows.")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=default_qwen_model_path(),
        help="Reference model path for perplexity loss (default: Qwen2.5-0.5B-Instruct from ModelScope cache).",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Transformers device_map; use 'none' to disable.",
    )
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length for loss computation.")
    parser.add_argument(
        "--nll_batch_size",
        type=int,
        default=32,
        help="Batch size for NLL forward passes across contexts (higher = faster, more VRAM).",
    )

    parser.add_argument("--global_trials", type=int, default=3, help="Number of global random baseline trials.")
    parser.add_argument("--context_size", type=int, default=4, help="Context sample count per trial.")
    parser.add_argument("--cluster_pool_top_n", type=int, default=256, help="Top-N SRM representatives per cluster.")
    parser.add_argument(
        "--active_set_mode",
        type=str,
        choices=["union", "srm_nonzero", "topk"],
        default="union",
        help="How to build active clusters from SRM row.",
    )
    parser.add_argument("--max_top_k", type=int, default=5, help="Max active clusters per sample; <=0 means all.")

    parser.add_argument("--m_dimensions", type=int, default=None, help="Capability dimension m; auto if omitted.")
    parser.add_argument("--min_std", type=float, default=1e-4, help="Minimum std for deviation normalization.")
    parser.add_argument("--loss_threshold", type=float, default=3.0, help="Empirical loss threshold tau.")
    parser.add_argument("--penalty_smoothness", type=float, default=1.0, help="Penalty smoothness omega.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--starting_sample", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--cache_text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Precompute demo/query text cache for faster prompt building.",
    )
    parser.add_argument(
        "--cache_token_ids",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache tokenized query/target/demo pieces to reduce repeated tokenizer.encode calls.",
    )

    parser.add_argument(
        "--attach_routing_meta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Attach SRM top-k metadata fields into output.",
    )
    parser.add_argument(
        "--include_debug_fields",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include detailed intermediate PDM fields in output.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Skip model loss; use SRM scores as proxy signals.")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    rng = random.Random(int(args.seed))

    data_rows, data_fmt = load_json_or_jsonl(args.data_path)
    if not data_rows:
        raise ValueError("No rows loaded from --data_path.")

    srm_map, inferred_m_from_file = load_srm_items(
        srm_path=args.srm_path,
        max_top_k=int(args.max_top_k),
        active_set_mode=str(args.active_set_mode),
    )
    if not srm_map:
        raise ValueError("No valid SRM rows loaded from --srm_path.")

    inferred_m = max(int(inferred_m_from_file), infer_m_from_srm_map(srm_map))
    if args.m_dimensions is None:
        m_dimensions = inferred_m
    else:
        m_dimensions = int(args.m_dimensions)
        if m_dimensions <= 0:
            raise ValueError("--m_dimensions must be > 0")
        if m_dimensions < inferred_m:
            raise ValueError(f"--m_dimensions={m_dimensions} is too small; requires >= {inferred_m}.")
    if m_dimensions <= 0:
        raise ValueError("Cannot infer m_dimensions from SRM. Please set --m_dimensions explicitly.")

    start = max(0, int(args.starting_sample))
    if start >= len(data_rows):
        raise ValueError(f"starting_sample={start} out of range for dataset size {len(data_rows)}.")
    if int(args.max_samples) < 0:
        end = len(data_rows)
    else:
        end = min(len(data_rows), start + int(args.max_samples))
    work_rows = data_rows[start:end]

    logger.info(
        "Loaded data rows=%d, scoring range=[%d,%d) -> %d",
        len(data_rows),
        start,
        end,
        len(work_rows),
    )
    logger.info("Loaded SRM rows=%d, inferred m=%d", len(srm_map), m_dimensions)

    id_to_data_index: Dict[str, int] = {}
    for i, row in enumerate(data_rows):
        rid = choose_row_id(row, i)
        id_to_data_index[str(rid)] = i

    cluster_pools = build_cluster_pools(
        srm_map=srm_map,
        id_to_data_index=id_to_data_index,
        top_n=max(0, int(args.cluster_pool_top_n)),
    )
    global_indices = list(range(len(data_rows)))
    logger.info("Built cluster pools for %d dimensions", len(cluster_pools))

    scorer: Optional[PDMScorer] = None
    if not args.dry_run:
        if torch is not None and torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:  # noqa: BLE001
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:  # noqa: BLE001
                pass
        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        scorer = PDMScorer(
            model=model,
            tokenizer=tokenizer,
            max_seq_len=int(args.max_seq_len),
        )
    else:
        logger.info("Dry run enabled: using SRM scores as proxy signals for PDM.")

    demo_tails: List[str] = []
    query_prefixes: List[str] = []
    targets: List[str] = []
    if bool(args.cache_text):
        logger.info("Precomputing text cache for %d rows", len(data_rows))
        demo_tails, query_prefixes, targets = precompute_text_cache(data_rows)

    intermediates: List[Dict[str, Any]] = []
    deviations_by_dim: Dict[int, List[float]] = {}

    iterator = enumerate(work_rows)
    if tqdm is not None:
        iterator = tqdm(iterator, total=len(work_rows), desc="PDM pass-1")  # type: ignore[assignment]

    pending_encoded: List[Tuple[List[int], List[int]]] = []
    pending_meta: List[Tuple[int, int]] = []
    nll_batch_size = max(1, int(args.nll_batch_size))
    if scorer is not None:
        adjusted_bsz = auto_adjust_nll_batch_size(scorer, nll_batch_size)
        if adjusted_bsz < nll_batch_size:
            logger.warning(
                "Adjusted --nll_batch_size from %d to %d for available CUDA memory.",
                nll_batch_size,
                adjusted_bsz,
            )
        nll_batch_size = adjusted_bsz
    use_token_cache = bool(args.cache_text) and bool(args.cache_token_ids) and (not args.dry_run)
    demo_tail_ids_cache: Dict[int, List[int]] = {}
    query_prefix_ids_cache: Dict[int, List[int]] = {}
    target_ids_cache: Dict[int, List[int]] = {}
    if use_token_cache:
        logger.info("Token-id cache enabled for query/target/demo text pieces.")

    def flush_pending() -> None:
        if not pending_encoded:
            return
        assert scorer is not None
        losses = scorer._batch_nll(pending_encoded)
        for (sample_idx, meta_idx), loss in zip(pending_meta, losses):
            intermediates[sample_idx]["context_losses"][meta_idx] = float(loss)
        pending_encoded.clear()
        pending_meta.clear()

    for local_idx, row in iterator:
        global_idx = start + local_idx
        sample_id, srm_item = resolve_srm_item(srm_map=srm_map, sample=row, fallback_idx=global_idx)

        active_indices: List[int] = []
        top_k_indices: List[int] = []
        top_k_scores: List[float] = []
        top_k_node_ids: List[str] = []
        top_k_node_names: List[str] = []
        top_k_node_paths: List[str] = []
        score_by_dim: Dict[int, float] = {}
        if srm_item is not None:
            active_indices = [int(x) for x in srm_item.active_indices if 0 <= int(x) < m_dimensions]
            top_k_indices = list(srm_item.top_k_indices)
            top_k_scores = list(srm_item.top_k_scores)
            top_k_node_ids = list(srm_item.top_k_node_ids)
            top_k_node_names = list(srm_item.top_k_node_names)
            top_k_node_paths = list(srm_item.top_k_node_paths)
            score_by_dim = dict(srm_item.score_by_dim)

        baseline_loss = 0.0
        cond_losses: Dict[int, float] = {}
        deviations: Dict[int, float] = {}
        context_meta: List[Tuple[str, int]] = []
        context_indices_list: List[List[int]] = []

        if active_indices and not args.dry_run:
            g_trials = max(1, int(args.global_trials))
            ctx_size = max(1, int(args.context_size))

            for _ in range(g_trials):
                idxs = sample_global_context_indices(
                    global_candidates=global_indices,
                    exclude_index=global_idx,
                    k=ctx_size,
                    rng=rng,
                )
                context_indices_list.append(idxs)
                context_meta.append(("global", -1))

            for dim in active_indices:
                pool = cluster_pools.get(int(dim), [])
                idxs = sample_context_indices(
                    primary_candidates=pool,
                    global_candidates=global_indices,
                    exclude_index=global_idx,
                    k=ctx_size,
                    rng=rng,
                )
                context_indices_list.append(idxs)
                context_meta.append(("cluster", int(dim)))

        intermediates.append(
            {
                "global_idx": global_idx,
                "sample_id": sample_id,
                "active_indices": active_indices,
                "baseline_loss": baseline_loss,
                "cond_losses": cond_losses,
                "deviations": deviations,
                "top_k_indices": top_k_indices,
                "top_k_scores": top_k_scores,
                "top_k_node_ids": top_k_node_ids,
                "top_k_node_names": top_k_node_names,
                "top_k_node_paths": top_k_node_paths,
                "context_meta": context_meta,
                "context_losses": [None] * len(context_meta),
            }
        )

        if not active_indices:
            continue

        if args.dry_run:
            baseline_loss = 1.0
            for dim in active_indices:
                proxy = float(score_by_dim.get(dim, 0.0))
                cond = max(0.0, 1.0 - proxy)
                cond_losses[dim] = cond
                dev = baseline_loss - cond
                deviations[dim] = dev
                deviations_by_dim.setdefault(dim, []).append(dev)
            intermediates[local_idx]["baseline_loss"] = baseline_loss
            intermediates[local_idx]["cond_losses"] = cond_losses
            intermediates[local_idx]["deviations"] = deviations
            continue

        if not context_meta:
            continue

        query_prefix_ids: List[int] = []
        target_ids: List[int] = []
        if bool(args.cache_text):
            query_prefix = query_prefixes[global_idx]
            target = targets[global_idx]
            if use_token_cache:
                assert scorer is not None
                qids = query_prefix_ids_cache.get(global_idx)
                if qids is None:
                    qids = scorer.encode_text(query_prefix)
                    query_prefix_ids_cache[global_idx] = qids
                tids = target_ids_cache.get(global_idx)
                if tids is None:
                    tids = scorer.encode_text(target) if target else []
                    target_ids_cache[global_idx] = tids
                query_prefix_ids = qids
                target_ids = tids
        else:
            query_prefix = scorer._format_query_prefix(row) if scorer is not None else ""
            target = extract_target_text(row)

        if not target:
            for i in range(len(context_meta)):
                intermediates[local_idx]["context_losses"][i] = 0.0
            continue

        for meta_idx, idxs in enumerate(context_indices_list):
            if use_token_cache:
                assert scorer is not None
                prefix_ids = scorer.build_prefix_ids_from_token_cache(
                    query_prefix_ids=query_prefix_ids,
                    context_indices=idxs,
                    demo_tail_ids_cache=demo_tail_ids_cache,
                    demo_tails=demo_tails,
                )
                encoded = scorer._encode_for_loss_ids(prefix_ids=prefix_ids, target_ids=target_ids)
            elif bool(args.cache_text):
                prefix = scorer.build_prefix_from_cache(query_prefix, idxs, demo_tails)
                encoded = scorer._encode_for_loss(prefix=prefix, target=target)
            else:
                context_rows = [data_rows[i] for i in idxs]
                prefix = scorer._build_prefix(row, context_rows)
                encoded = scorer._encode_for_loss(prefix=prefix, target=target)
            pending_encoded.append(encoded)
            pending_meta.append((local_idx, meta_idx))
            if len(pending_encoded) >= nll_batch_size:
                flush_pending()

    if not args.dry_run:
        flush_pending()

        for meta in intermediates:
            context_meta = meta.get("context_meta", [])
            context_losses = meta.get("context_losses", [])
            if not meta.get("active_indices"):
                meta.pop("context_meta", None)
                meta.pop("context_losses", None)
                continue
            if context_meta and any(loss is None for loss in context_losses):
                raise RuntimeError("Missing losses in batched NLL computation; try smaller --nll_batch_size.")

            base_losses = [loss for (kind, _d), loss in zip(context_meta, context_losses) if kind == "global"]
            baseline_loss = mean([float(x) for x in base_losses if x is not None])
            cond_losses = {}
            deviations = {}
            for (kind, dim), loss in zip(context_meta, context_losses):
                if kind != "cluster" or dim < 0 or loss is None:
                    continue
                cond_losses[int(dim)] = float(loss)
                dev = float(baseline_loss - float(loss))
                deviations[int(dim)] = dev
                deviations_by_dim.setdefault(int(dim), []).append(dev)

            meta["baseline_loss"] = float(baseline_loss)
            meta["cond_losses"] = cond_losses
            meta["deviations"] = deviations
            meta.pop("context_meta", None)
            meta.pop("context_losses", None)

    dim_mean: Dict[int, float] = {}
    dim_std: Dict[int, float] = {}
    min_std = max(float(args.min_std), 1e-12)
    for dim, vals in deviations_by_dim.items():
        mu = mean(vals)
        sd = std(vals)
        if sd < min_std:
            sd = min_std
        dim_mean[int(dim)] = float(mu)
        dim_std[int(dim)] = float(sd)

    logger.info(
        "PDM pass-1 done: scored=%d, dims_with_stats=%d, avg_active_dims=%.3f",
        len(intermediates),
        len(dim_mean),
        mean([len(x["active_indices"]) for x in intermediates]),
    )

    out_rows: List[Dict[str, Any]] = []
    tau = float(args.loss_threshold)
    omega = max(float(args.penalty_smoothness), 1e-8)

    iterator2 = enumerate(work_rows)
    if tqdm is not None:
        iterator2 = tqdm(iterator2, total=len(work_rows), desc="PDM pass-2")  # type: ignore[assignment]

    for local_idx, row in iterator2:
        meta = intermediates[local_idx]
        mapped_vector = [0.0] * m_dimensions

        for dim in meta["active_indices"]:
            dim_i = int(dim)
            if dim_i < 0 or dim_i >= m_dimensions:
                continue
            dev = float(meta["deviations"].get(dim_i, 0.0))
            cond = float(meta["cond_losses"].get(dim_i, meta["baseline_loss"]))

            mu = float(dim_mean.get(dim_i, 0.0))
            sd = float(dim_std.get(dim_i, min_std))
            gain = sigmoid((dev - mu) / max(sd, min_std))
            penalty = 1.0 - sigmoid((cond - tau) / omega)
            mapped_vector[dim_i] = float(max(0.0, gain * penalty))

        item = dict(row)
        item["id"] = meta["sample_id"]
        item["mapped_vector"] = mapped_vector
        item["score"] = mapped_vector
        item["score_type"] = "pdm_mapped_vector"
        item["pdm_scalar"] = float(sum(mapped_vector))
        item["pdm_active_count"] = int(len(meta["active_indices"]))

        if bool(args.attach_routing_meta):
            item["top_k_indices"] = meta["top_k_indices"]
            item["top_k_scores"] = meta["top_k_scores"]
            if meta["top_k_node_ids"]:
                item["top_k_node_ids"] = meta["top_k_node_ids"]
            if meta["top_k_node_names"]:
                item["top_k_node_names"] = meta["top_k_node_names"]
            if meta["top_k_node_paths"]:
                item["top_k_node_paths"] = meta["top_k_node_paths"]

        if bool(args.include_debug_fields):
            item["pdm_baseline_loss"] = float(meta["baseline_loss"])
            item["pdm_cond_losses"] = {str(k): float(v) for k, v in meta["cond_losses"].items()}
            item["pdm_deviations"] = {str(k): float(v) for k, v in meta["deviations"].items()}
            item["pdm_dim_mean"] = {str(k): float(v) for k, v in dim_mean.items()}
            item["pdm_dim_std"] = {str(k): float(v) for k, v in dim_std.items()}
            item["pdm_loss_threshold"] = tau
            item["pdm_penalty_smoothness"] = omega

        out_rows.append(item)

    write_json_or_jsonl(args.output_path, out_rows, prefer_jsonl=(data_fmt == "jsonl"))
    logger.info("Saved PDM scored rows [%d, %d) to %s", start, end, args.output_path)


if __name__ == "__main__":
    main()
