#!/usr/bin/env python3
"""DataWhisperer ICL scorer aligned with the 4.2 pipeline.

This implementation follows the core method from:
  Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning
  via Few-Shot In-Context Learning

Core behaviors implemented:
1) Few-shot ICL scoring with random demonstration/query sampling
2) Attention-aware weighting from internal model attention matrices
3) Multi-iteration score accumulation and averaging
4) Weak-to-strong support (scorer model can differ from target model)

Pipeline compatibility:
- Input: data_path (JSON/JSONL), routing_path (optional)
- Output rows include: score, score_type
- Supports: --select_ratio / --select_top_k
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # noqa: BLE001
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutedItem:
    row_index: int
    sample_id: Any
    top_k_indices: List[int]
    top_k_scores: List[float]
    top_k_node_ids: List[str]
    top_k_node_names: List[str]
    top_k_node_paths: List[str]


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
        data = json.loads(text)
        if isinstance(data, list):
            return data, "json"
        raise ValueError("Input JSON must be a list of samples.")
    except json.JSONDecodeError:
        lines = [line for line in text.splitlines() if line.strip()]
        return [json.loads(line) for line in lines], "jsonl"


def write_rows(path: str, rows: List[Dict[str, Any]], prefer_jsonl: bool) -> None:
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


def derive_selected_output_path(output_path: str) -> str:
    root, ext = os.path.splitext(output_path)
    if ext.lower() == ".jsonl":
        return f"{root}_selected.jsonl"
    if ext.lower() == ".json":
        return f"{root}_selected.json"
    return f"{output_path}_selected.jsonl"


def select_subset(
    rows: List[Dict[str, Any]],
    select_ratio: Optional[float],
    select_top_k: Optional[int],
) -> Optional[List[Dict[str, Any]]]:
    if select_ratio is not None and select_top_k is not None:
        raise ValueError("Use only one of --select_ratio or --select_top_k.")

    if select_ratio is None and select_top_k is None:
        return None

    sorted_rows = sorted(
        rows,
        key=lambda x: float(x.get("datawhisperer_scalar", 0.0)),
        reverse=True,
    )

    if select_ratio is not None:
        ratio = float(select_ratio)
        if not (0 < ratio <= 1):
            raise ValueError("--select_ratio must be in (0, 1].")
        k = max(1, int(len(sorted_rows) * ratio))
        return sorted_rows[:k]

    assert select_top_k is not None
    if int(select_top_k) <= 0:
        raise ValueError("--select_top_k must be > 0.")
    return sorted_rows[: min(int(select_top_k), len(sorted_rows))]


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


def normalize_for_match(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    n = len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, len(a) + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev, curr = curr, prev
    return prev[n]


def rouge_l_f1(reference: str, prediction: str) -> float:
    ref_tokens = str(reference or "").split()
    pred_tokens = str(prediction or "").split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    ll = lcs_len(ref_tokens, pred_tokens)
    p = ll / max(1, len(pred_tokens))
    r = ll / max(1, len(ref_tokens))
    if p + r <= 1e-12:
        return 0.0
    return 2 * p * r / (p + r)


def extract_target(sample: Dict[str, Any]) -> str:
    out = str(sample.get("output", sample.get("response", "")) or "").strip()
    if out:
        return out
    # fallback for classification-like data
    if "label" in sample:
        return str(sample.get("label", "") or "").strip()
    if "category" in sample:
        return str(sample.get("category", "") or "").strip()
    return ""


def format_instruction_input(sample: Dict[str, Any]) -> Tuple[str, str]:
    instruction = str(sample.get("instruction", "") or "").strip()
    input_text = str(sample.get("input", "") or "").strip()
    if not instruction and not input_text:
        text = str(sample.get("text", "") or "").strip()
        return text, ""
    return instruction, input_text


def normalize_weights(values: List[float]) -> List[float]:
    if not values:
        return []
    clipped = [max(0.0, float(v)) for v in values]
    s = sum(clipped)
    if s <= 0:
        u = 1.0 / len(clipped)
        return [u for _ in clipped]
    return [v / s for v in clipped]


def extract_optional_list(row: Dict[str, Any], key: str) -> Optional[List[Any]]:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    return value


def load_routing_jsonl(routing_path: str, max_top_k: int) -> Tuple[Dict[str, RoutedItem], int]:
    items: Dict[str, RoutedItem] = {}
    inferred_m = 0

    with open(routing_path, "r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)

            sample_id = row.get("id", row_index)
            top_k_indices_raw = row.get("top_k_indices")
            top_k_scores_raw = row.get("top_k_scores")
            if not isinstance(top_k_indices_raw, list) or not isinstance(top_k_scores_raw, list):
                raise ValueError(
                    f"Invalid routing row at line={row_index + 1}: missing list top_k_indices/top_k_scores"
                )
            if len(top_k_indices_raw) != len(top_k_scores_raw):
                raise ValueError(
                    f"Routing row length mismatch at line={row_index + 1}: "
                    f"top_k_indices={len(top_k_indices_raw)} vs top_k_scores={len(top_k_scores_raw)}"
                )

            limit = len(top_k_indices_raw)
            if max_top_k > 0:
                limit = min(limit, max_top_k)

            top_k_node_ids_raw = extract_optional_list(row, "top_k_node_ids")
            top_k_node_names_raw = extract_optional_list(row, "top_k_node_names")
            top_k_node_paths_raw = extract_optional_list(row, "top_k_node_paths")

            top_k_indices: List[int] = []
            top_k_scores: List[float] = []
            top_k_node_ids: List[str] = []
            top_k_node_names: List[str] = []
            top_k_node_paths: List[str] = []

            for rank in range(limit):
                idx = to_int(top_k_indices_raw[rank], -1)
                if idx < 0:
                    continue
                score = to_float(top_k_scores_raw[rank], 0.0)

                node_id = ""
                if top_k_node_ids_raw is not None and rank < len(top_k_node_ids_raw):
                    node_id = str(top_k_node_ids_raw[rank] or "").strip()

                node_name = ""
                if top_k_node_names_raw is not None and rank < len(top_k_node_names_raw):
                    node_name = str(top_k_node_names_raw[rank] or "").strip()

                node_path = ""
                if top_k_node_paths_raw is not None and rank < len(top_k_node_paths_raw):
                    node_path = str(top_k_node_paths_raw[rank] or "").strip()

                top_k_indices.append(idx)
                top_k_scores.append(score)
                top_k_node_ids.append(node_id)
                top_k_node_names.append(node_name)
                top_k_node_paths.append(node_path)
                inferred_m = max(inferred_m, idx + 1)

            if not top_k_indices:
                logger.warning("Routing row id=%s has no valid candidates and will be ignored.", sample_id)
                continue

            key = str(sample_id)
            if key in items:
                logger.warning("Duplicate routing id=%s found; keeping the last row.", key)

            items[key] = RoutedItem(
                row_index=row_index,
                sample_id=sample_id,
                top_k_indices=top_k_indices,
                top_k_scores=top_k_scores,
                top_k_node_ids=top_k_node_ids,
                top_k_node_names=top_k_node_names,
                top_k_node_paths=top_k_node_paths,
            )

    return items, inferred_m


def resolve_sample_routing(
    sample: Dict[str, Any],
    fallback_idx: int,
    routing_map: Dict[str, RoutedItem],
) -> Tuple[Any, Optional[RoutedItem]]:
    sample_id = choose_row_id(sample, fallback_idx)
    routed = routing_map.get(str(sample_id))
    if routed is None and str(sample_id) != str(fallback_idx):
        routed = routing_map.get(str(fallback_idx))
    return sample_id, routed


def map_scalar_to_vector(
    scalar: float,
    routed: RoutedItem,
    m_dimensions: int,
    routing_weight_mode: str,
) -> List[float]:
    mapped = [0.0] * m_dimensions
    if routing_weight_mode == "uniform":
        weights = normalize_weights([1.0 for _ in routed.top_k_scores])
    else:
        weights = normalize_weights(routed.top_k_scores)
    for idx, w in zip(routed.top_k_indices, weights):
        if 0 <= idx < m_dimensions:
            mapped[idx] += float(scalar * w)
    return mapped


class DataWhispererScorer:
    """Data Whisperer scorer: ICL + attention-aware weighting + multi-iteration."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        attention_layer: int = 13,
        n_demonstrations: int = 10,
        n_queries: int = 5,
        n_iterations: int = 10,
        metric: str = "accuracy",
        max_new_tokens: int = 32,
        max_input_tokens: int = 2048,
        query_batch_size: int = 1,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.attention_layer = int(attention_layer)
        self.n_demonstrations = int(n_demonstrations)
        self.n_queries = int(n_queries)
        self.n_iterations = int(n_iterations)
        self.metric = metric.strip().lower()
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_tokens = int(max_input_tokens)
        self.query_batch_size = max(1, int(query_batch_size))
        self.rng = random.Random(int(seed))

        if self.metric not in {"accuracy", "exact_match", "rouge-l"}:
            raise ValueError("--icl_metric must be one of: accuracy, exact_match, rouge-L")

    def _score_one_prediction(self, prediction: str, ground_truth: str) -> float:
        if self.metric in {"accuracy", "exact_match"}:
            return 1.0 if normalize_for_match(prediction) == normalize_for_match(ground_truth) else 0.0
        return float(rouge_l_f1(ground_truth, prediction))

    def _compute_prediction_scores(self, predictions: List[str], ground_truths: List[str]) -> float:
        if not predictions:
            return 0.0
        values = [self._score_one_prediction(p, g) for p, g in zip(predictions, ground_truths)]
        return float(sum(values) / max(1, len(values)))

    def _safe_attention_layer(self, model_output: Any) -> int:
        attentions = getattr(model_output, "attentions", None)
        if not attentions:
            return -1
        n_layers = len(attentions)
        if n_layers <= 0:
            return -1
        idx = self.attention_layer
        if idx < 0:
            idx = n_layers + idx
        idx = max(0, min(idx, n_layers - 1))
        return idx

    def _extract_attention_weights(
        self,
        model_output: Any,
        batch_index: int,
        demo_spans: List[Tuple[int, int]],
        query_span: Tuple[int, int],
    ) -> List[float]:
        if not demo_spans:
            return []

        layer_idx = self._safe_attention_layer(model_output)
        if layer_idx < 0:
            return normalize_weights([1.0 for _ in demo_spans])

        attn_layers = model_output.attentions
        layer = attn_layers[layer_idx]  # [B, H, T, T]
        if layer is None:
            return normalize_weights([1.0 for _ in demo_spans])

        # average over attention heads
        attn = layer[batch_index].float().mean(dim=0)  # [T, T]
        t = int(attn.shape[-1])

        q0, q1 = query_span
        q0 = max(0, min(q0, t))
        q1 = max(q0, min(q1, t))
        if q1 <= q0:
            return normalize_weights([1.0 for _ in demo_spans])

        values: List[float] = []
        for d0, d1 in demo_spans:
            d0 = max(0, min(d0, t))
            d1 = max(d0, min(d1, t))
            if d1 <= d0:
                values.append(0.0)
                continue
            block = attn[q0:q1, d0:d1]
            if block.numel() == 0:
                values.append(0.0)
            else:
                values.append(float(block.mean().item()))
        return normalize_weights(values)

    def _format_demo_block(self, idx: int, sample: Dict[str, Any]) -> str:
        instruction, input_text = format_instruction_input(sample)
        answer = extract_target(sample)
        out = [f"### Demonstration {idx}\n"]
        out.append(f"Instruction: {instruction}\n")
        if input_text:
            out.append(f"Input: {input_text}\n")
        out.append(f"Answer: {answer}\n\n")
        return "".join(out)

    def _format_query_block(self, sample: Dict[str, Any]) -> str:
        instruction, input_text = format_instruction_input(sample)
        out = ["### Query\n"]
        out.append(f"Instruction: {instruction}\n")
        if input_text:
            out.append(f"Input: {input_text}\n")
        out.append("Answer:")
        return "".join(out)

    def _encode_no_special(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_icl_prompt(
        self,
        demo_samples: List[Dict[str, Any]],
        query_sample: Dict[str, Any],
    ) -> Tuple[str, List[Tuple[int, int]], Tuple[int, int]]:
        header = (
            "You are an expert assistant. Learn the task pattern from demonstrations.\n"
            "Follow the same style and answer the query.\n"
            "Return only the final answer text.\n\n"
        )

        # If too long, progressively reduce number of demos.
        active_demos = list(demo_samples)
        while True:
            segments: List[str] = [header]
            demo_spans: List[Tuple[int, int]] = []

            token_cursor = len(self._encode_no_special(header))
            for idx, sample in enumerate(active_demos, start=1):
                seg = self._format_demo_block(idx, sample)
                seg_len = len(self._encode_no_special(seg))
                start = token_cursor
                end = token_cursor + seg_len
                demo_spans.append((start, end))
                token_cursor = end
                segments.append(seg)

            query_seg = self._format_query_block(query_sample)
            q_len = len(self._encode_no_special(query_seg))
            q_start = token_cursor
            q_end = token_cursor + q_len
            segments.append(query_seg)

            total_len = q_end
            if total_len <= self.max_input_tokens or len(active_demos) <= 1:
                prompt = "".join(segments)
                return prompt, demo_spans, (q_start, q_end)

            # drop earliest demo and retry
            active_demos = active_demos[1:]

    def _decode_new_tokens(self, full_ids: Any, input_len: int) -> str:
        new_ids = full_ids[input_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        text = str(text or "").strip()
        # keep first answer line for classification-style tasks
        if "\n" in text:
            text = text.split("\n", 1)[0].strip()
        text = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        return text

    def _model_input_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except Exception:  # noqa: BLE001
            dev = getattr(self.model, "device", None)
            if dev is not None:
                return dev
            if torch is None:
                raise RuntimeError("PyTorch is required.")
            return torch.device("cpu")

    def _run_icl_iteration(
        self,
        dataset: List[Dict[str, Any]],
        demo_indices: List[int],
        query_indices: List[int],
    ) -> Tuple[List[float], List[float], float]:
        demo_samples = [dataset[i] for i in demo_indices]
        n_demo = len(demo_samples)
        if n_demo == 0 or not query_indices:
            return [0.0] * n_demo, [0.0] * n_demo, 0.0

        demo_score_updates = [0.0] * n_demo
        demo_weight_sums = [0.0] * n_demo

        all_predictions: List[str] = []
        all_ground_truths: List[str] = []

        query_batches: List[List[int]] = []
        for i in range(0, len(query_indices), self.query_batch_size):
            query_batches.append(query_indices[i : i + self.query_batch_size])

        iterator = tqdm(query_batches, desc="DataWhisperer ICL iter", leave=False) if tqdm else query_batches
        for batch in iterator:
            prompts: List[str] = []
            demo_spans_batch: List[List[Tuple[int, int]]] = []
            query_spans_batch: List[Tuple[int, int]] = []
            refs: List[str] = []

            for q_idx in batch:
                sample = dataset[q_idx]
                prompt, demo_spans, query_span = self._build_icl_prompt(demo_samples, sample)
                prompts.append(prompt)
                demo_spans_batch.append(demo_spans)
                query_spans_batch.append(query_span)
                refs.append(extract_target(sample))

            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            )
            input_device = self._model_input_device()
            enc = {k: v.to(input_device) for k, v in enc.items()}
            attention_mask = enc["attention_mask"]

            with torch.no_grad():  # type: ignore[union-attr]
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                model_out = self.model(
                    **enc,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )

            for b in range(len(batch)):
                input_len = int(attention_mask[b].sum().item())
                pred = self._decode_new_tokens(generated[b], input_len)
                gt = refs[b]
                score = self._score_one_prediction(pred, gt)

                weights = self._extract_attention_weights(
                    model_output=model_out,
                    batch_index=b,
                    demo_spans=demo_spans_batch[b],
                    query_span=query_spans_batch[b],
                )
                if len(weights) != n_demo:
                    weights = normalize_weights([1.0 for _ in range(n_demo)])

                for i_demo, w in enumerate(weights):
                    demo_score_updates[i_demo] += float(score * w)
                    demo_weight_sums[i_demo] += float(w)

                all_predictions.append(pred)
                all_ground_truths.append(gt)

        iteration_avg = self._compute_prediction_scores(all_predictions, all_ground_truths)
        return demo_score_updates, demo_weight_sums, iteration_avg

    def compute_sample_scores(self, dataset: List[Dict[str, Any]]) -> List[float]:
        n = len(dataset)
        if n == 0:
            return []
        if n == 1:
            return [0.0]

        score_sums = [0.0] * n
        weight_sums = [0.0] * n
        seen_counts = [0] * n

        # Number of demonstrations/queries per iteration with disjoint sets.
        n_demo = min(max(1, self.n_demonstrations), n - 1)
        n_query = min(max(1, self.n_queries), n - n_demo)

        logger.info(
            "DataWhisperer start | n=%d n_demo=%d n_query=%d n_iter=%d metric=%s",
            n,
            n_demo,
            n_query,
            self.n_iterations,
            self.metric,
        )

        iter_indices = list(range(n))
        iterator = range(self.n_iterations)
        iterator = tqdm(iterator, total=self.n_iterations, desc="DW iterations") if tqdm else iterator

        for it in iterator:
            self.rng.shuffle(iter_indices)
            demo_indices = iter_indices[:n_demo]
            query_indices = iter_indices[n_demo : n_demo + n_query]
            if not query_indices:
                continue

            demo_updates, demo_weights, iter_avg = self._run_icl_iteration(
                dataset=dataset,
                demo_indices=demo_indices,
                query_indices=query_indices,
            )
            for local_i, data_i in enumerate(demo_indices):
                score_sums[data_i] += float(demo_updates[local_i])
                weight_sums[data_i] += float(demo_weights[local_i])
                seen_counts[data_i] += 1

            logger.info(
                "Iteration %d/%d done | avg_query_score=%.4f",
                it + 1,
                self.n_iterations,
                iter_avg,
            )

        # Final average over attention-weighted accumulations.
        final_scores = [0.0] * n
        seen_values: List[float] = []
        for i in range(n):
            if weight_sums[i] > 1e-12:
                final_scores[i] = score_sums[i] / weight_sums[i]
                seen_values.append(final_scores[i])

        fallback = float(sum(seen_values) / len(seen_values)) if seen_values else 0.0
        unseen = 0
        for i in range(n):
            if weight_sums[i] <= 1e-12:
                final_scores[i] = fallback
                unseen += 1

        if unseen > 0:
            logger.warning(
                "DataWhisperer: %d samples not selected as demonstrations; assigned fallback score %.4f",
                unseen,
                fallback,
            )
        return final_scores


def load_model_and_tokenizer(
    model_name_or_path: str,
    weak_model_path: str,
    torch_dtype: str,
    device_map: str,
) -> Tuple[Any, Any, str]:
    if torch is None:
        raise RuntimeError("PyTorch is required. Please install torch.")
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers is required. Please install transformers.")

    scorer_path = weak_model_path.strip() or model_name_or_path.strip()
    if not scorer_path:
        raise ValueError("Missing model path. Set --model_name_or_path or --weak_model_path.")

    dtype = None
    dtype_name = torch_dtype.strip().lower()
    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_name == "float32":
        dtype = torch.float32
    elif dtype_name == "auto":
        dtype = None
    else:
        raise ValueError("--torch_dtype must be one of: auto, float16, bfloat16, float32")

    logger.info("Loading scorer model from %s", scorer_path)
    tokenizer = AutoTokenizer.from_pretrained(scorer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    map_arg = device_map.strip().lower()
    if map_arg and map_arg != "none":
        model_kwargs["device_map"] = map_arg
    model = AutoModelForCausalLM.from_pretrained(scorer_path, **model_kwargs)
    model.eval()

    if "device_map" not in model_kwargs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    return model, tokenizer, scorer_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DataWhisperer ICL scorer aligned with pipeline outputs."
    )
    parser.add_argument("--data_path", type=str, required=True, help="Input dataset path (JSON/JSONL).")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output score file path (JSON/JSONL).",
    )
    parser.add_argument(
        "--routing_path",
        type=str,
        default="",
        help="Optional coarse routing jsonl (e.g., train_coarse_topk5.jsonl).",
    )
    parser.add_argument(
        "--selected_output_path",
        type=str,
        default="",
        help="Optional selected subset output path.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=default_qwen_model_path(),
        help="Target model path (default: ModelScope cache Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--weak_model_path",
        type=str,
        default="",
        help="Optional weak scorer model path. If set, use this model for DataWhisperer scoring.",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="",
        help="Optional target model identifier for metadata only.",
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
        help="Transformers device_map value. Use 'none' to disable device_map.",
    )

    parser.add_argument("--n_demonstrations", type=int, default=10)
    parser.add_argument("--n_queries", type=int, default=5)
    parser.add_argument("--n_iterations", type=int, default=10)
    parser.add_argument("--attention_layer", type=int, default=13)
    parser.add_argument(
        "--icl_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "exact_match", "rouge-L"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument(
        "--concurrency",
        "--concurrancy",
        dest="concurrency",
        type=int,
        default=1,
        help="Query batch size per ICL iteration.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--starting_sample", type=int, default=0)

    parser.add_argument("--m_dimensions", type=int, default=None)
    parser.add_argument("--max_top_k", type=int, default=5)
    parser.add_argument(
        "--routing_weight_mode",
        type=str,
        choices=["coarse", "uniform"],
        default="coarse",
    )
    parser.add_argument(
        "--attach_routing_meta",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--select_ratio", type=float, default=None)
    parser.add_argument("--select_top_k", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--include_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reserved for compatibility; prompt is not stored in this implementation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric = "rouge-l" if args.icl_metric.lower() == "rouge-l" else args.icl_metric.lower()

    data, input_fmt = load_json_or_jsonl(args.data_path)
    if not data:
        raise ValueError("No data rows loaded from --data_path.")

    start = max(0, int(args.starting_sample))
    if start >= len(data):
        raise ValueError(f"starting_sample={start} out of range for dataset size {len(data)}.")
    if int(args.max_samples) < 0:
        end = len(data)
    else:
        end = min(len(data), start + int(args.max_samples))
    selected_data = data[start:end]
    logger.info("Loaded %d rows, scoring rows [%d, %d) -> %d", len(data), start, end, len(selected_data))

    if args.dry_run:
        scalar_scores = [3.0 for _ in selected_data]
        scorer_model_path = args.weak_model_path.strip() or args.model_name_or_path.strip()
        logger.info("Dry run enabled. Use constant scalar score=3.0.")
    else:
        model, tokenizer, scorer_model_path = load_model_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            weak_model_path=args.weak_model_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        scorer = DataWhispererScorer(
            model=model,
            tokenizer=tokenizer,
            attention_layer=int(args.attention_layer),
            n_demonstrations=max(1, int(args.n_demonstrations)),
            n_queries=max(1, int(args.n_queries)),
            n_iterations=max(1, int(args.n_iterations)),
            metric=metric,
            max_new_tokens=max(1, int(args.max_new_tokens)),
            max_input_tokens=max(128, int(args.max_input_tokens)),
            query_batch_size=max(1, int(args.concurrency)),
            seed=int(args.seed),
        )
        scalar_scores = scorer.compute_sample_scores(selected_data)

    # Optional routing for mapped vector output
    routing_map: Dict[str, RoutedItem] = {}
    inferred_m = 0
    if args.routing_path:
        routing_map, inferred_m = load_routing_jsonl(args.routing_path, max_top_k=int(args.max_top_k))
        logger.info("Loaded %d routing rows from %s", len(routing_map), args.routing_path)

    if args.m_dimensions is None:
        m_dimensions = inferred_m
    else:
        m_dimensions = int(args.m_dimensions)
        if m_dimensions <= 0:
            raise ValueError("--m_dimensions must be > 0 when set.")
        if inferred_m > 0 and m_dimensions < inferred_m:
            raise ValueError(
                f"--m_dimensions={m_dimensions} is too small; routing requires >= {inferred_m}."
            )

    out_rows: List[Dict[str, Any]] = []
    for local_idx, (row, scalar) in enumerate(zip(selected_data, scalar_scores)):
        global_idx = start + local_idx
        sample_id, routed = resolve_sample_routing(row, global_idx, routing_map)

        item = dict(row)
        item["id"] = sample_id
        item["datawhisperer_scalar"] = float(scalar)
        item["datawhisperer_metric"] = metric
        item["datawhisperer_scorer_model"] = scorer_model_path
        item["datawhisperer_target_model"] = args.target_model or args.model_name_or_path
        item["datawhisperer_n_demonstrations"] = int(args.n_demonstrations)
        item["datawhisperer_n_queries"] = int(args.n_queries)
        item["datawhisperer_n_iterations"] = int(args.n_iterations)
        item["datawhisperer_attention_layer"] = int(args.attention_layer)

        if routed is not None and m_dimensions > 0:
            mapped_vector = map_scalar_to_vector(
                scalar=float(scalar),
                routed=routed,
                m_dimensions=m_dimensions,
                routing_weight_mode=args.routing_weight_mode,
            )
            item["mapped_vector"] = mapped_vector
            item["score"] = mapped_vector
            item["score_type"] = "datawhisperer_icl_mapped_vector"
            item["routing_weight_mode"] = args.routing_weight_mode
            if bool(args.attach_routing_meta):
                item["top_k_indices"] = routed.top_k_indices
                item["top_k_scores"] = routed.top_k_scores
                item["top_k_node_ids"] = routed.top_k_node_ids
                item["top_k_node_names"] = routed.top_k_node_names
                item["top_k_node_paths"] = routed.top_k_node_paths
        else:
            item["score"] = float(scalar)
            item["score_type"] = "datawhisperer_icl_scalar"

        out_rows.append(item)

    write_rows(args.output_path, out_rows, prefer_jsonl=(input_fmt == "jsonl"))
    logger.info("Saved scored rows [%d, %d) to %s", start, end, args.output_path)

    selected = select_subset(out_rows, args.select_ratio, args.select_top_k)
    if selected is not None:
        selected_path = args.selected_output_path or derive_selected_output_path(args.output_path)
        write_rows(selected_path, selected, prefer_jsonl=(input_fmt == "jsonl"))
        logger.info("Saved selected rows=%d to %s", len(selected), selected_path)


if __name__ == "__main__":
    main()
