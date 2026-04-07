#!/usr/bin/env python3
"""Shared helpers for decoupled Recipe Evolution scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

try:
    from modelscope import snapshot_download
except Exception:  # noqa: BLE001
    snapshot_download = None

RESPONSE_MARKER = "### Response:\n"
EPS = 1e-12


@dataclass
class MapperInput:
    name: str
    path: Path


def parse_mapper_item(text: str) -> MapperInput:
    raw = text.strip()
    if not raw:
        raise ValueError("Empty --score-path item.")
    if "::" in raw:
        name, path = raw.split("::", 1)
        name = name.strip()
        p = Path(path.strip())
        if not name:
            raise ValueError(f"Invalid mapper name in --score-path: {text}")
    else:
        p = Path(raw)
        name = p.stem
    return MapperInput(name=name, path=p)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object JSON at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def build_prompt(row: dict[str, Any]) -> str:
    instruction = safe_text(row.get("instruction", "")).strip()
    inp = safe_text(row.get("input", "")).strip()
    output = safe_text(row.get("output", "")).strip()
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


def choose_row_id(row: dict[str, Any], fallback_idx: int) -> str:
    for key in ("id", "data_id", "uid", "idx", "index"):
        if key in row and row.get(key) is not None:
            return str(row[key])
    return str(fallback_idx)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:  # noqa: BLE001
        return None


def parse_topk_indices(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            idx = int(item)
        except Exception:  # noqa: BLE001
            continue
        if idx >= 0:
            out.append(idx)
    return out


def parse_vector_to_dense(value: Any) -> dict[int, float]:
    out: dict[int, float] = {}
    if isinstance(value, list):
        for i, v in enumerate(value):
            fv = to_float(v)
            if fv is None:
                continue
            out[int(i)] = float(fv)
    elif isinstance(value, dict):
        for k, v in value.items():
            try:
                idx = int(k)
            except Exception:  # noqa: BLE001
                continue
            fv = to_float(v)
            if fv is None:
                continue
            out[idx] = float(fv)
    return out


def minmax_l1_normalize(mat: torch.Tensor) -> torch.Tensor:
    out = torch.nan_to_num(mat.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_v = float(out.min().item())
    max_v = float(out.max().item())
    if max_v > min_v + EPS:
        out = (out - min_v) / (max_v - min_v)
    out = out.clamp_min(0.0)
    row_sum = out.sum(dim=1, keepdim=True)
    out = torch.where(row_sum > 0, out / row_sum.clamp_min(EPS), out)
    return out


def get_dtype(torch_dtype: str = "auto") -> torch.dtype:
    if torch_dtype == "fp32":
        return torch.float32
    if torch_dtype == "fp16":
        return torch.float16
    if torch_dtype == "bf16":
        return torch.bfloat16
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_model_path(base_model: str, model_source: str, cache_dir: Path | None) -> str:
    if model_source == "hf":
        return base_model
    if snapshot_download is None:
        raise ImportError("ModelScope is not installed. Please install: pip install modelscope")
    return snapshot_download(
        model_id=base_model,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )


def load_causal_lm(model_path: str, dtype: torch.dtype):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )


class ResponseOnlyCollator:
    """Prompt 部分 labels=-100，仅 Response 参与 loss。"""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        variants = [RESPONSE_MARKER, f"\n{RESPONSE_MARKER}", f"\n\n{RESPONSE_MARKER}"]
        self.marker_token_variants: list[list[int]] = []
        for text in variants:
            marker_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if marker_ids and marker_ids not in self.marker_token_variants:
                self.marker_token_variants.append(marker_ids)
        if not self.marker_token_variants:
            raise ValueError("Failed to build response marker token ids.")

    @staticmethod
    def _find_subsequence(seq: list[int], pattern: list[int]) -> int:
        if not pattern or len(pattern) > len(seq):
            return -1
        end = len(seq) - len(pattern) + 1
        for i in range(end):
            if seq[i : i + len(pattern)] == pattern:
                return i
        return -1

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sample_ids: list[int] = []
        core_features = []
        for f in features:
            item: dict[str, Any] = {"input_ids": f["input_ids"]}
            if "attention_mask" in f and f["attention_mask"] is not None:
                item["attention_mask"] = f["attention_mask"]
            core_features.append(item)
            sample_ids.append(int(f.get("sample_id", -1)))

        batch = self.tokenizer.pad(core_features, padding=True, return_tensors="pt")
        if "attention_mask" not in batch:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                raise ValueError("Tokenizer has no pad_token_id and attention_mask is missing.")
            batch["attention_mask"] = (batch["input_ids"] != pad_id).long()

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for i in range(labels.size(0)):
            seq_len = int(batch["attention_mask"][i].sum().item())
            seq = batch["input_ids"][i, :seq_len].tolist()
            response_start = -1
            for marker in self.marker_token_variants:
                pos = self._find_subsequence(seq, marker)
                if pos != -1:
                    response_start = pos + len(marker)
                    break
            if response_start == -1:
                labels[i, :seq_len] = -100
                continue
            labels[i, :response_start] = -100

        batch["labels"] = labels
        batch["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
        return batch
