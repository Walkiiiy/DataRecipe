#!/usr/bin/env python3
"""4.3 Online Recipe Evolution 训练脚本。

核心目标：
1. 基于多评分文件构建静态特征张量 E_matrix，形状 (N, k, m)。
2. 使用 RecipeEvolver 执行在线配方演化：动态更新 alpha/beta，并按玻尔兹曼分布采样。
3. 第一轮(epoch=1)在线选择 data_size 条样本并训练；后续 epoch 仅复训该子集。
4. 阶段4单样本梯度严格使用 torch.func.vmap + torch.func.grad。

实现约束：
- 只提取指定层(默认 lm_head)梯度，不对全模型做全量梯度提取。
- alpha/beta 更新均在 no_grad 下执行，避免干扰大模型计算图。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.func import functional_call, grad, vmap
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

try:
    from modelscope import snapshot_download
except Exception:  # noqa: BLE001
    snapshot_download = None


RESPONSE_MARKER = "### Response:\n"
EPS = 1e-12

VECTOR_FALLBACK_FIELDS = (
    "mapped_vector",
    "score",
)
SCALAR_FALLBACK_FIELDS = (
    "score",
    "delta_scalar",
    "alpagasus_scalar",
    "judge_score",
    "pdm_scalar",
)
TOPK_FALLBACK_FIELDS = (
    "top_k_indices",
    "raw_top_k_indices",
)


@dataclass
class MapperInput:
    name: str
    path: Path


@dataclass
class MapperRowInfo:
    vector: dict[int, float]
    scalar: float | None
    top_k_indices: list[int]


@dataclass
class TokenizedPool:
    rows: list[dict[str, Any]]
    original_indices: list[int]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


@dataclass
class ScoreTensorBundle:
    mapper_names: list[str]
    e_matrix: torch.Tensor  # (N, k, m), CPU float32
    top_k_indices: torch.Tensor  # (N, K_max), CPU long


def parse_mapper_item(text: str) -> MapperInput:
    """解析评分文件参数：支持 name::path 或 path。"""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4.3 Online Recipe Evolution training")

    # 数据与评分输入
    parser.add_argument("--train-jsonl", type=Path, required=True, help="训练数据 jsonl")
    parser.add_argument(
        "--score-path",
        type=str,
        action="append",
        required=True,
        help="可重复：name::score_jsonl 或 score_jsonl",
    )
    parser.add_argument(
        "--top-k-path",
        type=Path,
        default=None,
        help="可选 top_k 索引文件（如 train_coarse_topk5.jsonl）",
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    # 与 4.1 对齐的模型/训练参数
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--model_source", type=str, choices=["modelscope", "hf"], default="modelscope")
    parser.add_argument("--modelscope_cache_dir", type=Path, default=None)

    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=0, help="按优化器步数保存检查点，0 表示不保存中间点")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # 用户特别强调参数
    parser.add_argument("--data_size", "--train_size", dest="data_size", type=int, default=130)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=None,
        help="兼容 4.1 命名；若传入则覆盖 --epoch（按四舍五入到整数）",
    )

    # Recipe Evolution 参数
    parser.add_argument("--gamma_t", type=float, default=0.8, help="玻尔兹曼温度")
    parser.add_argument("--gamma_alpha", type=float, default=1.0, help="alpha 指数更新系数")
    parser.add_argument("--alpha_epsilon", type=float, default=0.05, help="alpha 正则项 epsilon")
    parser.add_argument("--eta_beta", type=float, default=0.2, help="beta 更新步长")

    parser.add_argument("--anchor_size", type=int, default=4, help="每个能力簇用于锚点梯度的样本数")
    parser.add_argument("--anchor_refresh_steps", type=int, default=20, help="每多少个 recipe batch 刷新锚点")
    parser.add_argument("--anchor_batch_size", type=int, default=2, help="计算锚点梯度时的微批大小")
    parser.add_argument(
        "--max_anchor_capabilities",
        type=int,
        default=0,
        help="每次刷新最多计算多少个能力簇的锚点梯度；0 表示全量 m",
    )

    parser.add_argument("--target_layer", type=str, default="lm_head", help="用于梯度反馈的目标层名")
    parser.add_argument("--vmap_chunk_size", type=int, default=4, help="单样本梯度 vmap 分块大小")
    parser.add_argument("--top_k_max", type=int, default=5, help="top_k_indices 目标宽度下限")

    # 工程选项
    parser.add_argument("--torch_dtype", type=str, choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--grad_ckpt", type=int, default=1, help="1 启用 gradient checkpointing，0 关闭")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", type=int, default=0, help="1 时只跑第一轮的一个 recipe batch")
    parser.add_argument("--log_level", type=str, default="INFO")

    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


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
                raise ValueError(f"Expected dict JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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


def build_prompt_without_output(row: dict[str, Any]) -> str:
    instruction = safe_text(row.get("instruction", "")).strip()
    inp = safe_text(row.get("input", "")).strip()
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


def get_dtype(torch_dtype: str) -> torch.dtype:
    if torch_dtype == "fp32":
        return torch.float32
    if torch_dtype == "fp16":
        return torch.float16
    if torch_dtype == "bf16":
        return torch.bfloat16

    # auto
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


def choose_row_id(row: dict[str, Any], fallback_idx: int) -> str:
    for key in ("id", "data_id", "uid", "idx", "index"):
        if key in row and row.get(key) is not None:
            return str(row.get(key))
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
    for x in value:
        if isinstance(x, bool):
            continue
        try:
            idx = int(x)
        except Exception:  # noqa: BLE001
            continue
        if idx >= 0:
            out.append(idx)
    return out


def parse_vector_to_sparse(value: Any) -> dict[int, float]:
    """把 dense/sparse 表示统一为稀疏字典 {cap_idx: score}。"""
    out: dict[int, float] = {}
    if isinstance(value, list):
        for i, v in enumerate(value):
            fv = to_float(v)
            if fv is None:
                continue
            if fv > 0:
                out[int(i)] = float(fv)
        return out
    if isinstance(value, dict):
        for k, v in value.items():
            try:
                idx = int(k)
            except Exception:  # noqa: BLE001
                continue
            if idx < 0:
                continue
            fv = to_float(v)
            if fv is None:
                continue
            if fv > 0:
                out[int(idx)] = float(fv)
        return out
    return out


def extract_vector_from_row(row: dict[str, Any]) -> dict[int, float]:
    """优先提取映射后的能力向量；兼容 fused top-k 结构。"""
    for field in VECTOR_FALLBACK_FIELDS:
        if field not in row:
            continue
        vec = parse_vector_to_sparse(row.get(field))
        if vec:
            return vec

    # 某些文件只有 top_k_indices + top_k_scores / fused_top_k
    top_k = []
    for field in TOPK_FALLBACK_FIELDS:
        top_k = parse_topk_indices(row.get(field))
        if top_k:
            break
    if top_k:
        score_list = row.get("fused_top_k")
        if not isinstance(score_list, list):
            score_list = row.get("top_k_scores")
        if isinstance(score_list, list):
            out: dict[int, float] = {}
            for i, cap in enumerate(top_k):
                if i >= len(score_list):
                    break
                fv = to_float(score_list[i])
                if fv is not None and fv > 0:
                    out[cap] = float(fv)
            if out:
                return out
    return {}


def extract_scalar_from_row(row: dict[str, Any]) -> float | None:
    for field in SCALAR_FALLBACK_FIELDS:
        if field not in row:
            continue
        fv = to_float(row.get(field))
        if fv is not None:
            return fv
    return None


def extract_topk_from_row(row: dict[str, Any]) -> list[int]:
    for field in TOPK_FALLBACK_FIELDS:
        vals = parse_topk_indices(row.get(field))
        if vals:
            return vals
    return []


def load_mapper_rows(path: Path) -> tuple[dict[str, MapperRowInfo], int, int]:
    """加载一个评分文件，返回 id->信息、观测到的最大能力索引、观测到的最大 topk 长度。"""
    if not path.exists():
        raise FileNotFoundError(f"Score file not found: {path}")

    out: dict[str, MapperRowInfo] = {}
    max_cap_idx = -1
    max_topk_len = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                continue

            rid = choose_row_id(row, line_no - 1)
            vector = extract_vector_from_row(row)
            scalar = extract_scalar_from_row(row)
            top_k_indices = extract_topk_from_row(row)

            if vector:
                max_cap_idx = max(max_cap_idx, max(vector.keys()))
            if top_k_indices:
                max_cap_idx = max(max_cap_idx, max(top_k_indices))
                max_topk_len = max(max_topk_len, len(top_k_indices))

            out[rid] = MapperRowInfo(vector=vector, scalar=scalar, top_k_indices=top_k_indices)

    return out, max_cap_idx, max_topk_len


def load_topk_file(path: Path) -> tuple[dict[str, list[int]], int, int]:
    if not path.exists():
        raise FileNotFoundError(f"top-k file not found: {path}")

    topk_map: dict[str, list[int]] = {}
    max_cap_idx = -1
    max_topk_len = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            rid = choose_row_id(row, line_no - 1)
            topk = extract_topk_from_row(row)
            if not topk:
                continue
            topk_map[rid] = topk
            max_topk_len = max(max_topk_len, len(topk))
            max_cap_idx = max(max_cap_idx, max(topk))
    return topk_map, max_cap_idx, max_topk_len


def topk_from_sparse_vector(vec: dict[int, float], k: int) -> list[int]:
    if not vec:
        return []
    ranked = sorted(vec.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:k]]


def normalize_mapper_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """评分方式归一化：
    1) 先做 mapper 级 min-max 到 [0,1]
    2) 再做样本级 L1 归一化，保证不同 mapper 量纲一致
    """
    out = torch.nan_to_num(matrix.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_v = float(out.min().item())
    max_v = float(out.max().item())
    if max_v > min_v + EPS:
        out = (out - min_v) / (max_v - min_v)
    out = out.clamp_min(0.0)
    row_sum = out.sum(dim=1, keepdim=True)
    out = torch.where(row_sum > 0, out / row_sum.clamp_min(EPS), out)
    return out


def build_score_tensors(
    mapper_inputs: list[MapperInput],
    row_ids: list[str],
    top_k_path: Path | None,
    top_k_max_floor: int,
) -> ScoreTensorBundle:
    """构建 E_matrix(N,k,m) 与 top_k_indices(N,K_max)。"""
    mapper_rows_list: list[dict[str, MapperRowInfo]] = []
    mapper_names: list[str] = []

    global_max_cap_idx = -1
    observed_topk_max = 0

    auto_topk_map: dict[str, list[int]] = {}

    for mapper in mapper_inputs:
        rows_map, max_cap_idx, max_topk_len = load_mapper_rows(mapper.path)
        mapper_rows_list.append(rows_map)
        mapper_names.append(mapper.name)
        global_max_cap_idx = max(global_max_cap_idx, max_cap_idx)
        observed_topk_max = max(observed_topk_max, max_topk_len)

        for rid, info in rows_map.items():
            if info.top_k_indices and rid not in auto_topk_map:
                auto_topk_map[rid] = info.top_k_indices

    explicit_topk_map: dict[str, list[int]] = {}
    if top_k_path is not None:
        explicit_topk_map, max_cap_idx, max_topk_len = load_topk_file(top_k_path)
        global_max_cap_idx = max(global_max_cap_idx, max_cap_idx)
        observed_topk_max = max(observed_topk_max, max_topk_len)

    if global_max_cap_idx < 0:
        raise ValueError("Unable to infer capability dimension m from score/top-k files.")

    n_samples = len(row_ids)
    m_dim = global_max_cap_idx + 1

    k_max = max(1, top_k_max_floor, observed_topk_max)
    top_k_indices = torch.full((n_samples, k_max), -1, dtype=torch.long)

    # 先建立 top_k_indices
    for i, rid in enumerate(row_ids):
        cur_topk = explicit_topk_map.get(rid) or auto_topk_map.get(rid) or []
        if not cur_topk:
            # 兜底：从第一个 mapper 向量取 top-k
            first_map = mapper_rows_list[0].get(rid)
            if first_map is not None:
                cur_topk = topk_from_sparse_vector(first_map.vector, k_max)
        if not cur_topk:
            continue

        valid = [x for x in cur_topk if 0 <= x < m_dim]
        if not valid:
            continue
        t = torch.tensor(valid[:k_max], dtype=torch.long)
        top_k_indices[i, : t.numel()] = t

    mapper_tensors: list[torch.Tensor] = []

    # 再构建每个 mapper 的 (N,m) 分数矩阵
    for mapper_idx, rows_map in enumerate(mapper_rows_list):
        mat = torch.zeros((n_samples, m_dim), dtype=torch.float32)

        for i, rid in enumerate(row_ids):
            info = rows_map.get(rid)
            if info is None:
                continue

            if info.vector:
                for cap, val in info.vector.items():
                    if 0 <= cap < m_dim and val > 0:
                        mat[i, cap] = float(val)
                continue

            # 标量评分 fallback：均匀分配到该样本的 top-k 能力上
            if info.scalar is not None:
                topk_list = explicit_topk_map.get(rid) or info.top_k_indices
                if not topk_list:
                    # 使用已构造好的 top_k_indices 兜底
                    topk_list = [int(x) for x in top_k_indices[i].tolist() if x >= 0]
                valid_caps = [c for c in topk_list if 0 <= c < m_dim]
                if valid_caps:
                    v = float(info.scalar) / float(len(valid_caps))
                    for cap in valid_caps:
                        mat[i, cap] = max(mat[i, cap], v)
                else:
                    mat[i, 0] = max(mat[i, 0], float(info.scalar))

        mat = normalize_mapper_matrix(mat)
        mapper_tensors.append(mat)
        logging.info(
            "Mapper[%d]=%s loaded: nonzero=%d / %d",
            mapper_idx,
            mapper_names[mapper_idx],
            int((mat > 0).sum().item()),
            mat.numel(),
        )

    e_matrix = torch.stack(mapper_tensors, dim=1)  # (N,k,m)
    return ScoreTensorBundle(
        mapper_names=mapper_names,
        e_matrix=e_matrix,
        top_k_indices=top_k_indices,
    )


def encode_response_only(
    rows: list[dict[str, Any]],
    tokenizer,
    max_seq_len: int,
) -> TokenizedPool:
    """把训练数据编码为固定长度张量，并构造 response-only labels。"""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must have eos_token_id.")

    input_tensor_list: list[torch.Tensor] = []
    mask_tensor_list: list[torch.Tensor] = []
    label_tensor_list: list[torch.Tensor] = []
    kept_rows: list[dict[str, Any]] = []
    kept_original_indices: list[int] = []

    for idx, row in enumerate(rows):
        prompt = build_prompt_without_output(row)
        response = safe_text(row.get("output", ""))
        if not response.strip():
            continue

        full_text = prompt + response
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        input_ids: list[int] = []
        labels: list[int] = []

        if bos_id is not None:
            input_ids.append(bos_id)
            labels.append(-100)

        input_ids.extend(full_ids)
        labels.extend([-100] * len(full_ids))

        response_start = (1 if bos_id is not None else 0) + len(prompt_ids)
        if response_start >= len(input_ids):
            continue

        for i in range(response_start, len(input_ids)):
            labels[i] = input_ids[i]

        input_ids.append(eos_id)
        labels.append(eos_id)

        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]

        if all(v == -100 for v in labels):
            continue

        cur_len = len(input_ids)
        attn = [1] * cur_len

        if cur_len < max_seq_len:
            pad_len = max_seq_len - cur_len
            input_ids = input_ids + [pad_id] * pad_len
            labels = labels + [-100] * pad_len
            attn = attn + [0] * pad_len

        input_tensor_list.append(torch.tensor(input_ids, dtype=torch.long))
        mask_tensor_list.append(torch.tensor(attn, dtype=torch.long))
        label_tensor_list.append(torch.tensor(labels, dtype=torch.long))
        kept_rows.append(row)
        kept_original_indices.append(idx)

    if not input_tensor_list:
        raise ValueError("No valid training samples after tokenization.")

    input_ids_t = torch.stack(input_tensor_list, dim=0)
    attention_mask_t = torch.stack(mask_tensor_list, dim=0)
    labels_t = torch.stack(label_tensor_list, dim=0)

    return TokenizedPool(
        rows=kept_rows,
        original_indices=kept_original_indices,
        input_ids=input_ids_t,
        attention_mask=attention_mask_t,
        labels=labels_t,
    )


def select_by_indices(pool: TokenizedPool, indices: list[int]) -> dict[str, torch.Tensor]:
    idx_t = torch.tensor(indices, dtype=torch.long)
    return {
        "input_ids": pool.input_ids.index_select(0, idx_t),
        "attention_mask": pool.attention_mask.index_select(0, idx_t),
        "labels": pool.labels.index_select(0, idx_t),
    }


class RecipeEvolver:
    """Online Recipe Evolution 主类。

    按照用户指定的四阶段流程实现：
    - 阶段1：计算能力簇锚点梯度 g_{C_j}
    - 阶段2：用锚点梯度范数更新 beta
    - 阶段3：按 S(x)=alpha^T E_x beta 打分，并按 Boltzmann 分布采样
    - 阶段4：用 vmap+grad 计算单样本梯度和奖励，更新 alpha

    关键工程点：
    - 只提取指定层参数的梯度，避免全模型梯度提取开销
    - alpha/beta 更新使用 no_grad，不进入主模型训练图
    - 单样本梯度严格使用 torch.func.vmap + torch.func.grad
    """

    def __init__(
        self,
        model,
        tokenizer,
        pool: TokenizedPool,
        e_matrix_cpu: torch.Tensor,
        top_k_indices_cpu: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler,
        output_dir: Path,
        device: torch.device,
        train_batch_size: int,
        gradient_accumulation_steps: int,
        gamma_t: float,
        gamma_alpha: float,
        alpha_epsilon: float,
        eta_beta: float,
        anchor_size: int,
        anchor_refresh_steps: int,
        anchor_batch_size: int,
        max_anchor_capabilities: int,
        target_layer: str,
        vmap_chunk_size: int,
        max_grad_norm: float,
        logging_steps: int,
        save_steps: int,
        dry_run: bool,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.pool = pool
        self.e_matrix_cpu = e_matrix_cpu
        self.top_k_indices_cpu = top_k_indices_cpu
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.output_dir = output_dir
        self.device = device

        self.train_batch_size = max(1, train_batch_size)
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.gamma_t = max(gamma_t, 1e-6)
        self.gamma_alpha = gamma_alpha
        self.alpha_epsilon = min(max(alpha_epsilon, 0.0), 1.0)
        self.eta_beta = max(0.0, eta_beta)

        self.anchor_size = max(1, anchor_size)
        self.anchor_refresh_steps = max(1, anchor_refresh_steps)
        self.anchor_batch_size = max(1, anchor_batch_size)
        self.max_anchor_capabilities = max(0, max_anchor_capabilities)

        self.target_layer = target_layer
        self.vmap_chunk_size = max(1, vmap_chunk_size)
        self.max_grad_norm = max_grad_norm
        self.logging_steps = max(1, logging_steps)
        self.save_steps = max(0, save_steps)
        self.dry_run = dry_run

        self.n_samples = int(e_matrix_cpu.size(0))
        self.k_mappers = int(e_matrix_cpu.size(1))
        self.m_caps = int(e_matrix_cpu.size(2))

        # alpha/beta 初始化为均匀分布；不作为主模型参数参与反向图
        self.alpha = torch.full((self.k_mappers,), 1.0 / self.k_mappers, dtype=torch.float32, device=self.device)
        self.beta = torch.full((self.m_caps,), 1.0 / self.m_caps, dtype=torch.float32, device=self.device)

        self.target_param_names, self.target_param_refs = self._resolve_target_layer_params(self.target_layer)
        self.gradient_dim = int(sum(p.numel() for p in self.target_param_refs))
        logging.info(
            "Target layer=%s, grad_param_count=%d, gradient_dim=%d",
            self.target_layer,
            len(self.target_param_names),
            self.gradient_dim,
        )

        # g_{C_j} 缓存：形状 (m, gradient_dim)
        self.anchor_grads = torch.zeros((self.m_caps, self.gradient_dim), dtype=torch.float32, device=self.device)

        # capability -> 样本索引列表，用于锚点采样
        self.capability_to_samples = self._build_capability_index()

        # 训练状态
        self.recipe_step = 0
        self.micro_step = 0
        self.update_step = 0
        self.history: list[dict[str, Any]] = []

        self.optimizer.zero_grad(set_to_none=True)

    def _resolve_target_layer_params(self, target_layer: str) -> tuple[list[str], list[torch.nn.Parameter]]:
        modules = dict(self.model.named_modules())
        chosen_prefix = None

        if target_layer in modules:
            chosen_prefix = target_layer
        else:
            candidates = [name for name in modules if name.endswith(f".{target_layer}") or name.endswith(target_layer)]
            if candidates:
                chosen_prefix = min(candidates, key=len)

        if chosen_prefix is None:
            raise ValueError(f"Unable to resolve target layer: {target_layer}")

        all_named_params_list = list(self.model.named_parameters())
        all_named_params = dict(all_named_params_list)
        id_to_global_name: dict[int, str] = {}
        for n, p in all_named_params_list:
            if id(p) not in id_to_global_name:
                id_to_global_name[id(p)] = n

        target_module = modules[chosen_prefix]
        try:
            local_named = list(target_module.named_parameters(recurse=True, remove_duplicate=False))
        except TypeError:
            local_named = list(target_module.named_parameters(recurse=True))

        # 某些模型 lm_head 与 embedding 共享权重时，前缀名下可能“看起来无参数”。
        # 兜底：尝试输出层模块（get_output_embeddings）。
        if not local_named and hasattr(self.model, "get_output_embeddings"):
            out_head = self.model.get_output_embeddings()
            if out_head is not None:
                try:
                    local_named = list(out_head.named_parameters(recurse=True, remove_duplicate=False))
                except TypeError:
                    local_named = list(out_head.named_parameters(recurse=True))
                if local_named:
                    logging.warning(
                        "Target layer %s has no direct params; fallback to get_output_embeddings().",
                        chosen_prefix,
                    )

        # 仍为空时，最后兜底为全局可训练参数（避免直接中断训练）。
        if not local_named:
            fallback_trainable = [(n, p) for n, p in all_named_params_list if p.requires_grad]
            if fallback_trainable:
                logging.warning(
                    "No params found under target layer %s; fallback to all trainable params.",
                    chosen_prefix,
                )
                names = [n for n, _ in fallback_trainable]
                refs = [p for _, p in fallback_trainable]
                return names, refs
            raise ValueError(f"No parameters found under target layer: {chosen_prefix}")

        # 按参数对象去重（保持顺序）。
        unique_refs: list[torch.nn.Parameter] = []
        seen_ids: set[int] = set()
        for _, p in local_named:
            pid = id(p)
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            unique_refs.append(p)

        selected_refs = [p for p in unique_refs if p.requires_grad]
        if not selected_refs:
            # 若目标层参数都冻结，仅为梯度反馈临时打开 requires_grad。
            selected_refs = unique_refs
            if selected_refs:
                logging.warning(
                    "No trainable params under target layer %s; enable requires_grad for feedback only.",
                    chosen_prefix,
                )
                for p in selected_refs:
                    p.requires_grad_(True)

        selected_names: list[str] = []
        selected_refs_canonical: list[torch.nn.Parameter] = []
        for p in selected_refs:
            gname = id_to_global_name.get(id(p))
            if gname is None:
                continue
            selected_names.append(gname)
            selected_refs_canonical.append(all_named_params[gname])

        if not selected_names:
            raise ValueError(
                f"Resolved target layer {chosen_prefix}, but cannot map params to global parameter names."
            )

        return selected_names, selected_refs_canonical

    def _build_capability_index(self) -> list[list[int]]:
        cap_to_samples: list[list[int]] = [[] for _ in range(self.m_caps)]
        topk = self.top_k_indices_cpu.tolist()
        for sample_idx, row in enumerate(topk):
            for c in row:
                if c >= 0 and c < self.m_caps:
                    cap_to_samples[c].append(sample_idx)
        return cap_to_samples

    @contextmanager
    def _temporary_eval(self):
        """临时切到 eval，避免 dropout 随机性干扰梯度对齐计算。"""
        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            if was_training:
                self.model.train()

    def _to_device_batch(self, cpu_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": cpu_batch["input_ids"].to(self.device, non_blocking=True),
            "attention_mask": cpu_batch["attention_mask"].to(self.device, non_blocking=True),
            "labels": cpu_batch["labels"].to(self.device, non_blocking=True),
        }

    @staticmethod
    def _per_sample_response_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """response-only 单样本 loss，返回形状 (B,)。"""
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

    def _functional_forward(
        self,
        layer_params: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        use_attention_mask: bool = True,
    ):
        """兼容不同 PyTorch 版本的 functional_call 调用。"""
        kwargs = {
            "input_ids": batch["input_ids"],
            "labels": batch["labels"],
        }
        if use_attention_mask and "attention_mask" in batch and batch["attention_mask"] is not None:
            kwargs["attention_mask"] = batch["attention_mask"]
        try:
            return functional_call(self.model, layer_params, (), kwargs, strict=False)
        except TypeError:
            return functional_call(self.model, layer_params, (), kwargs)

    def extract_layer_gradients(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """只提取目标层梯度向量。

        这是阶段1锚点梯度与阶段4局部梯度的基础操作。
        注意：这里显式只对目标层参数做 autograd.grad，不做全模型梯度读取。
        """
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss

        grads = torch.autograd.grad(
            loss,
            self.target_param_refs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        flat: list[torch.Tensor] = []
        for g, p in zip(grads, self.target_param_refs):
            if g is None:
                flat.append(torch.zeros_like(p, dtype=torch.float32).reshape(-1))
            else:
                flat.append(g.detach().to(torch.float32).reshape(-1))
        vec = torch.cat(flat, dim=0)
        return vec

    def compute_anchor_gradients(self) -> torch.Tensor:
        """阶段1：计算能力簇锚点梯度 g_{C_j}。

        实现逻辑：
        - 对每个能力簇 j，从该簇关联样本中采样 anchor_size 条。
        - 通过 extract_layer_gradients 只提取目标层梯度，并求均值作为 g_{C_j}。
        - 结果缓存到 self.anchor_grads，形状 (m, gradient_dim)。
        """
        # 选取这次要刷新的能力簇（可选子采样，降低开销）
        caps = [j for j, ids in enumerate(self.capability_to_samples) if ids]
        if self.max_anchor_capabilities > 0 and len(caps) > self.max_anchor_capabilities:
            caps = random.sample(caps, self.max_anchor_capabilities)

        if not caps:
            return self.anchor_grads

        with self._temporary_eval():
            for cap in caps:
                members = self.capability_to_samples[cap]
                if not members:
                    continue
                sample_count = min(self.anchor_size, len(members))
                chosen = random.sample(members, sample_count)

                chunk_vecs: list[torch.Tensor] = []
                for st in range(0, len(chosen), self.anchor_batch_size):
                    part = chosen[st : st + self.anchor_batch_size]
                    cpu_batch = select_by_indices(self.pool, part)
                    batch = self._to_device_batch(cpu_batch)
                    g = self.extract_layer_gradients(batch)
                    chunk_vecs.append(g)

                if not chunk_vecs:
                    continue

                cap_grad = torch.stack(chunk_vecs, dim=0).mean(dim=0)
                cap_norm = cap_grad.norm().clamp_min(EPS)
                self.anchor_grads[cap] = cap_grad / cap_norm

        return self.anchor_grads

    def update_beta(self) -> torch.Tensor:
        """阶段2：基于锚点梯度范数更新 beta，并做 L1 归一化。"""
        with torch.no_grad():
            norms = self.anchor_grads.norm(dim=1)  # (m,)
            denom = norms.sum().clamp_min(EPS)
            delta = self.eta_beta * (norms / denom)
            self.beta = self.beta + delta
            self.beta = self.beta.clamp_min(0.0)
            self.beta = self.beta / self.beta.sum().clamp_min(EPS)
        return self.beta

    def compute_global_scores(self) -> torch.Tensor:
        """阶段3：S(x)=alpha^T E_x beta，返回 CPU 向量 (N,)。"""
        alpha_cpu = self.alpha.detach().cpu().to(torch.float32)
        beta_cpu = self.beta.detach().cpu().to(torch.float32)
        scores = torch.einsum("k,nkm,m->n", alpha_cpu, self.e_matrix_cpu, beta_cpu)
        return scores

    def sample_batch_indices(self, available_mask: torch.Tensor, batch_size: int) -> list[int]:
        """阶段3：根据玻尔兹曼分布采样 batch 索引。"""
        if available_mask.dtype != torch.bool:
            available_mask = available_mask.to(torch.bool)

        avail_idx = torch.nonzero(available_mask, as_tuple=False).squeeze(1)
        if avail_idx.numel() == 0:
            return []

        scores = self.compute_global_scores()
        avail_scores = scores.index_select(0, avail_idx)
        logits = avail_scores / self.gamma_t
        logits = logits - logits.max()
        probs = torch.softmax(logits, dim=0)

        take = min(max(1, batch_size), int(avail_idx.numel()))
        chosen_rel = torch.multinomial(probs, num_samples=take, replacement=False)
        chosen = avail_idx.index_select(0, chosen_rel)
        return [int(x) for x in chosen.tolist()]

    def per_sample_layer_gradients_vmap(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """阶段4：使用 vmap + grad 计算单样本目标层梯度。

        返回形状：(B, gradient_dim)
        """
        all_named_params = dict(self.model.named_parameters())
        layer_params: dict[str, torch.Tensor] = {}
        for name in self.target_param_names:
            if name in all_named_params:
                layer_params[name] = all_named_params[name]

        if not layer_params:
            raise ValueError(
                "No target params found in current model.named_parameters() during vmap gradient computation."
            )

        def single_loss(
            params: dict[str, torch.Tensor],
            input_ids_i: torch.Tensor,
            attention_mask_i: torch.Tensor,
            labels_i: torch.Tensor,
        ) -> torch.Tensor:
            single_batch = {
                "input_ids": input_ids_i.unsqueeze(0),
                "attention_mask": attention_mask_i.unsqueeze(0),
                "labels": labels_i.unsqueeze(0),
            }
            # vmap 场景下部分模型的 attention_mask 分支包含数据依赖控制流，
            # 会触发 functorch 限制。这里关闭 attention_mask，仅用于奖励梯度估计。
            out = self._functional_forward(params, single_batch, use_attention_mask=False)
            per_sample_loss = self._per_sample_response_loss_from_logits(out.logits, single_batch["labels"])
            return per_sample_loss.squeeze(0)

        grad_fn = grad(single_loss)

        batch_size = int(batch["input_ids"].size(0))
        outputs: list[torch.Tensor] = []

        # 分块 vmap，避免一次性内存峰值过高
        for st in range(0, batch_size, self.vmap_chunk_size):
            ed = min(batch_size, st + self.vmap_chunk_size)
            chunk_input_ids = batch["input_ids"][st:ed]
            chunk_attention = batch["attention_mask"][st:ed]
            chunk_labels = batch["labels"][st:ed]

            grad_tree = vmap(grad_fn, in_dims=(None, 0, 0, 0))(
                layer_params,
                chunk_input_ids,
                chunk_attention,
                chunk_labels,
            )

            flat_parts: list[torch.Tensor] = []
            chunk_bsz = ed - st
            for name, ref in zip(self.target_param_names, self.target_param_refs):
                g = grad_tree.get(name)
                if g is None:
                    flat_parts.append(
                        torch.zeros((chunk_bsz, ref.numel()), dtype=torch.float32, device=self.device)
                    )
                else:
                    flat_parts.append(g.reshape(chunk_bsz, -1).to(torch.float32))

            chunk_flat = torch.cat(flat_parts, dim=1)
            chunk_flat = chunk_flat / chunk_flat.norm(dim=1, keepdim=True).clamp_min(EPS)
            outputs.append(chunk_flat)

        return torch.cat(outputs, dim=0)

    def compute_rewards(self, batch_indices: list[int], per_sample_grads: torch.Tensor) -> torch.Tensor:
        """阶段4：按 top-k 能力簇计算局部奖励 R(x)。"""
        if not batch_indices:
            return torch.empty((0,), dtype=torch.float32, device=self.device)

        idx_cpu = torch.tensor(batch_indices, dtype=torch.long)
        topk_cpu = self.top_k_indices_cpu.index_select(0, idx_cpu)  # (B, K)
        topk = topk_cpu.to(self.device)

        safe_topk = topk.clamp_min(0)
        anchor = self.anchor_grads.index_select(0, safe_topk.view(-1)).view(topk.size(0), topk.size(1), -1)

        # cosine(sim) = dot(g_x, g_Cj) / (||g_x|| * ||g_Cj||)
        gx = per_sample_grads / per_sample_grads.norm(dim=1, keepdim=True).clamp_min(EPS)
        anchor_norm = anchor.norm(dim=2)
        dot = torch.einsum("bkd,bd->bk", anchor, gx)
        denom = anchor_norm * gx.norm(dim=1, keepdim=True).clamp_min(EPS)
        sim = dot / denom.clamp_min(EPS)

        valid_mask = (topk >= 0) & (anchor_norm > 0)
        reward = (sim * valid_mask.to(sim.dtype)).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1).to(sim.dtype)
        return reward

    def update_alpha(self, batch_indices: list[int], rewards: torch.Tensor) -> torch.Tensor:
        """阶段4：计算 U_i 并指数更新 alpha。"""
        if len(batch_indices) == 0:
            return self.alpha

        idx_cpu = torch.tensor(batch_indices, dtype=torch.long)
        batch_e = self.e_matrix_cpu.index_select(0, idx_cpu).to(self.device)  # (B,k,m)

        # 把 v_{x,i}(m维) 用当前 beta 投影成标量，得到每个 mapper 在每个样本上的贡献
        mapper_values = torch.einsum("bkm,m->bk", batch_e, self.beta)  # (B,k)
        utilities = (rewards.unsqueeze(1) * mapper_values).mean(dim=0)  # (k,)

        with torch.no_grad():
            alpha_tilde = self.alpha * torch.exp(self.gamma_alpha * utilities)
            alpha_norm = alpha_tilde / alpha_tilde.sum().clamp_min(EPS)
            self.alpha = (1.0 - self.alpha_epsilon) * alpha_norm + self.alpha_epsilon / float(self.k_mappers)
        return self.alpha

    def _optimize_batch(self, batch: dict[str, torch.Tensor]) -> float:
        """标准参数更新（用于 epoch1 在线训练与后续复训）。"""
        self.model.train()
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss

        scaled = loss / float(self.gradient_accumulation_steps)
        scaled.backward()

        self.micro_step += 1
        if self.micro_step % self.gradient_accumulation_steps == 0:
            if self.max_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.update_step += 1

            if self.save_steps > 0 and self.update_step % self.save_steps == 0:
                ckpt = self.output_dir / f"checkpoint-step{self.update_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(ckpt))

        return float(loss.item())

    def flush_optimizer(self) -> None:
        """处理最后不足 grad_accum 的尾 batch。"""
        if self.micro_step == 0:
            return
        if self.micro_step % self.gradient_accumulation_steps == 0:
            return

        if self.max_grad_norm > 0:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.update_step += 1

    def train_recipe_batch(self, epoch_idx: int, batch_indices: list[int]) -> dict[str, Any]:
        """一个 recipe batch：阶段4奖励+alpha更新 + 常规训练反向。"""
        cpu_batch = select_by_indices(self.pool, batch_indices)
        batch = self._to_device_batch(cpu_batch)

        with self._temporary_eval():
            per_sample_grads = self.per_sample_layer_gradients_vmap(batch)
            rewards = self.compute_rewards(batch_indices, per_sample_grads)

        self.update_alpha(batch_indices, rewards)
        loss = self._optimize_batch(batch)

        stat = {
            "epoch": epoch_idx,
            "recipe_step": self.recipe_step,
            "update_step": self.update_step,
            "loss": loss,
            "avg_reward": float(rewards.mean().item()) if rewards.numel() > 0 else 0.0,
            "alpha": [float(x) for x in self.alpha.detach().cpu().tolist()],
            "beta": [float(x) for x in self.beta.detach().cpu().tolist()],
            "batch_size": len(batch_indices),
        }
        self.history.append(stat)
        return stat

    def train_plain_batch(self, epoch_idx: int, batch_indices: list[int]) -> dict[str, Any]:
        """后续 epoch 仅复训固定子集，不再做采样与策略更新。"""
        cpu_batch = select_by_indices(self.pool, batch_indices)
        batch = self._to_device_batch(cpu_batch)
        loss = self._optimize_batch(batch)

        stat = {
            "epoch": epoch_idx,
            "recipe_step": None,
            "update_step": self.update_step,
            "loss": loss,
            "avg_reward": None,
            "alpha": [float(x) for x in self.alpha.detach().cpu().tolist()],
            "beta": [float(x) for x in self.beta.detach().cpu().tolist()],
            "batch_size": len(batch_indices),
        }
        self.history.append(stat)
        return stat

    def run_first_epoch(self, data_size: int) -> list[int]:
        """第一轮：在线采样并训练，最终得到 data_size 个被选样本。"""
        target = min(max(1, data_size), self.n_samples)
        available = torch.ones((self.n_samples,), dtype=torch.bool)

        selected: list[int] = []

        while len(selected) < target:
            if self.recipe_step % self.anchor_refresh_steps == 0:
                self.compute_anchor_gradients()

            self.update_beta()

            current_batch = min(self.train_batch_size, target - len(selected))
            batch_indices = self.sample_batch_indices(available, current_batch)
            if not batch_indices:
                break

            self.recipe_step += 1
            stat = self.train_recipe_batch(epoch_idx=1, batch_indices=batch_indices)

            for idx in batch_indices:
                available[idx] = False
            selected.extend(batch_indices)

            if self.recipe_step % self.logging_steps == 0:
                logging.info(
                    "[epoch1][recipe_step=%d] loss=%.6f reward=%.6f update_step=%d selected=%d/%d",
                    self.recipe_step,
                    stat["loss"],
                    stat["avg_reward"],
                    self.update_step,
                    len(selected),
                    target,
                )

            if self.dry_run:
                logging.info("dry_run=1: stop after first recipe batch")
                break

        return selected

    def run_later_epochs(self, selected_indices: list[int], total_epochs: int) -> None:
        if total_epochs <= 1 or not selected_indices:
            return

        for epoch_idx in range(2, total_epochs + 1):
            shuffled = list(selected_indices)
            random.shuffle(shuffled)

            for st in range(0, len(shuffled), self.train_batch_size):
                part = shuffled[st : st + self.train_batch_size]
                stat = self.train_plain_batch(epoch_idx=epoch_idx, batch_indices=part)

                local_step = (st // self.train_batch_size) + 1
                if local_step % self.logging_steps == 0:
                    logging.info(
                        "[epoch=%d][step=%d] loss=%.6f update_step=%d",
                        epoch_idx,
                        local_step,
                        stat["loss"],
                        self.update_step,
                    )

            if self.dry_run:
                break

    def dump_history(self) -> tuple[Path, Path]:
        out_jsonl = self.output_dir / "recipe_history.jsonl"
        out_csv = self.output_dir / "recipe_history.csv"

        with out_jsonl.open("w", encoding="utf-8") as f:
            for row in self.history:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "recipe_step", "update_step", "loss", "avg_reward", "batch_size"],
            )
            writer.writeheader()
            for row in self.history:
                writer.writerow(
                    {
                        "epoch": row.get("epoch"),
                        "recipe_step": row.get("recipe_step"),
                        "update_step": row.get("update_step"),
                        "loss": row.get("loss"),
                        "avg_reward": row.get("avg_reward"),
                        "batch_size": row.get("batch_size"),
                    }
                )

        return out_jsonl, out_csv


def export_selected_dataset(pool: TokenizedPool, selected_indices: list[int], path: Path) -> None:
    rows = [pool.rows[i] for i in selected_indices if 0 <= i < len(pool.rows)]
    write_jsonl(rows, path)


def resolve_total_epochs(epoch: int, num_train_epochs: float | None) -> int:
    if num_train_epochs is not None:
        return max(1, int(round(num_train_epochs)))
    return max(1, int(epoch))


def estimate_total_update_steps(
    data_size: int,
    batch_size: int,
    grad_accum: int,
    total_epochs: int,
) -> int:
    first_micro = math.ceil(data_size / max(1, batch_size))
    later_micro = max(0, total_epochs - 1) * math.ceil(data_size / max(1, batch_size))
    total_micro = first_micro + later_micro
    return max(1, math.ceil(total_micro / max(1, grad_accum)))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if not hasattr(torch, "func"):
        raise RuntimeError("This script requires PyTorch 2.0+ (torch.func).")

    total_epochs = resolve_total_epochs(args.epoch, args.num_train_epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    mapper_inputs = [parse_mapper_item(x) for x in args.score_path]
    if len(mapper_inputs) < 1:
        raise ValueError("At least one --score-path is required.")

    if not args.train_jsonl.exists():
        raise FileNotFoundError(f"train_jsonl not found: {args.train_jsonl}")

    logging.info("Loading train data: %s", args.train_jsonl)
    train_rows = load_jsonl(args.train_jsonl)
    if len(train_rows) < 2:
        raise ValueError("train_jsonl is too small.")

    model_path = resolve_model_path(args.base_model, args.model_source, args.modelscope_cache_dir)
    logging.info("Model resolved from %s: %s", args.model_source, model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Tokenizing train pool...")
    pool = encode_response_only(train_rows, tokenizer, max_seq_len=max(64, args.max_seq_length))

    # 分数字典的 ID 一般与原始行号对齐；tokenization 过滤后需要同步子集索引
    row_ids = [str(i) for i in pool.original_indices]

    logging.info("Building E_matrix and top_k_indices from score files...")
    score_bundle = build_score_tensors(
        mapper_inputs=mapper_inputs,
        row_ids=row_ids,
        top_k_path=args.top_k_path,
        top_k_max_floor=max(1, args.top_k_max),
    )

    e_matrix = score_bundle.e_matrix
    top_k_indices = score_bundle.top_k_indices

    n, k, m = e_matrix.shape
    logging.info("E_matrix shape: N=%d, k=%d, m=%d", n, k, m)
    logging.info("top_k_indices shape: %s", tuple(top_k_indices.shape))

    data_size = min(max(1, args.data_size), n)
    if args.data_size > n:
        logging.warning("data_size=%d > N=%d, clamped to %d", args.data_size, n, data_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(args.torch_dtype)
    if device.type == "cpu" and dtype != torch.float32:
        logging.warning("CPU device detected, force dtype to float32.")
        dtype = torch.float32

    logging.info("Loading model and attaching LoRA...")
    model = load_causal_lm(model_path, dtype=dtype)
    if hasattr(model, "config"):
        # 提升 torch.func.vmap 兼容性，避免部分模型走 sdpa 分支时触发数据依赖控制流报错。
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
    if int(args.grad_ckpt) == 1 and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=max(1, args.lora_r),
        lora_alpha=max(1, args.lora_alpha),
        lora_dropout=max(0.0, args.lora_dropout),
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=max(0.0, args.weight_decay),
    )

    total_update_steps = estimate_total_update_steps(
        data_size=data_size,
        batch_size=max(1, args.train_batch_size),
        grad_accum=max(1, args.gradient_accumulation_steps),
        total_epochs=total_epochs,
    )
    warmup_steps = int(max(0.0, args.warmup_ratio) * total_update_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    evolver = RecipeEvolver(
        model=model,
        tokenizer=tokenizer,
        pool=pool,
        e_matrix_cpu=e_matrix,
        top_k_indices_cpu=top_k_indices,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=args.output_dir,
        device=device,
        train_batch_size=max(1, args.train_batch_size),
        gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
        gamma_t=args.gamma_t,
        gamma_alpha=args.gamma_alpha,
        alpha_epsilon=args.alpha_epsilon,
        eta_beta=args.eta_beta,
        anchor_size=max(1, args.anchor_size),
        anchor_refresh_steps=max(1, args.anchor_refresh_steps),
        anchor_batch_size=max(1, args.anchor_batch_size),
        max_anchor_capabilities=max(0, args.max_anchor_capabilities),
        target_layer=args.target_layer,
        vmap_chunk_size=max(1, args.vmap_chunk_size),
        max_grad_norm=args.max_grad_norm,
        logging_steps=max(1, args.logging_steps),
        save_steps=max(0, args.save_steps),
        dry_run=bool(args.dry_run),
    )

    t0 = time.time()

    logging.info("Epoch1: online recipe evolution starts (target data_size=%d)", data_size)
    selected_indices = evolver.run_first_epoch(data_size=data_size)
    selected_indices = selected_indices[:data_size]

    logging.info("Epoch2..%d: retrain fixed subset (size=%d)", total_epochs, len(selected_indices))
    evolver.run_later_epochs(selected_indices=selected_indices, total_epochs=total_epochs)

    evolver.flush_optimizer()

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    selected_path = args.output_dir / "selected_epoch1.jsonl"
    export_selected_dataset(pool=pool, selected_indices=selected_indices, path=selected_path)

    hist_jsonl, hist_csv = evolver.dump_history()

    summary = {
        "train_jsonl": str(args.train_jsonl),
        "score_paths": [str(x.path) for x in mapper_inputs],
        "mapper_names": score_bundle.mapper_names,
        "top_k_path": str(args.top_k_path) if args.top_k_path is not None else None,
        "output_dir": str(args.output_dir),
        "model_path": model_path,
        "device": str(device),
        "dtype": str(dtype),
        "seed": args.seed,
        "N": int(n),
        "k": int(k),
        "m": int(m),
        "K_max": int(top_k_indices.size(1)),
        "data_size": int(data_size),
        "epoch": int(total_epochs),
        "train_batch_size": int(args.train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "learning_rate": float(args.learning_rate),
        "warmup_ratio": float(args.warmup_ratio),
        "gamma_t": float(args.gamma_t),
        "gamma_alpha": float(args.gamma_alpha),
        "eta_beta": float(args.eta_beta),
        "alpha_epsilon": float(args.alpha_epsilon),
        "anchor_size": int(args.anchor_size),
        "anchor_refresh_steps": int(args.anchor_refresh_steps),
        "anchor_batch_size": int(args.anchor_batch_size),
        "max_anchor_capabilities": int(args.max_anchor_capabilities),
        "target_layer": args.target_layer,
        "vmap_chunk_size": int(args.vmap_chunk_size),
        "selected_count": int(len(selected_indices)),
        "history_jsonl": str(hist_jsonl),
        "history_csv": str(hist_csv),
        "selected_epoch1_jsonl": str(selected_path),
        "final_dir": str(final_dir),
        "alpha_final": [float(x) for x in evolver.alpha.detach().cpu().tolist()],
        "beta_final": [float(x) for x in evolver.beta.detach().cpu().tolist()],
        "total_update_steps": int(evolver.update_step),
        "elapsed_seconds": float(time.time() - t0),
    }

    summary_path = args.output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Done. Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
