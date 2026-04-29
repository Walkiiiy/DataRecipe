#!/usr/bin/env python3
"""
Stage 2: Coarse Dense Routing via Embeddings

For each data point x and m capability centroids {mu_1 ... mu_m}:
1) S_k = cos(E(x), mu_k)
2) K* = Top-K indices of S
3) S_emb^(k) = exp(S_k / tau) / sum_{j in K*} exp(S_j / tau)
4) Non-Top-K dimensions are treated as 0 and not saved.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc

try:
    from datasets import Dataset, load_dataset
    DATASETS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    Dataset = Any  # type: ignore[misc,assignment]
    load_dataset = None  # type: ignore[assignment]
    DATASETS_IMPORT_ERROR = exc

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    TRANSFORMERS_IMPORT_ERROR = exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # noqa: BLE001
    pa = None
    pq = None

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


ID_CANDIDATES = ("id", "data_id", "uid", "idx", "index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coarse dense routing with embedding centroids.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Local JSONL/Parquet path or HF dataset name.")
    parser.add_argument("--dataset_split", type=str, default="train", help="HF split when dataset_path is dataset name.")
    parser.add_argument("--dataset_config", type=str, default=None, help="Optional HF dataset config name.")
    parser.add_argument("--centroids_path", type=Path, required=True, help="Path to centroids (.npy/.pt/.jsonl).")
    parser.add_argument("--output_path", type=Path, default=None, help="Output file path (.jsonl/.parquet).")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--query_prefix", type=str, default="", help="Optional prefix prepended to each text before encoding.")
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser.parse_args()


def ensure_dependencies() -> None:
    missing: list[str] = []
    if torch is None:
        missing.append(f"torch ({TORCH_IMPORT_ERROR})")
    if load_dataset is None:
        missing.append(f"datasets ({DATASETS_IMPORT_ERROR})")
    if AutoModel is None or AutoTokenizer is None:
        missing.append(f"transformers ({TRANSFORMERS_IMPORT_ERROR})")
    if missing:
        raise RuntimeError("Missing required dependencies: " + "; ".join(missing))


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device)


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def build_text_from_row(row: dict[str, Any]) -> str:
    instruction = clean_text(row.get("instruction"))
    inp = clean_text(row.get("input"))
    out = clean_text(row.get("output") or row.get("response"))
    chunks = [x for x in [instruction, inp, out] if x]
    return "\n".join(chunks)


def batch_size_from_columns(batch: dict[str, list[Any]]) -> int:
    if not batch:
        return 0
    first_col = next(iter(batch.values()))
    return len(first_col)


def build_texts_from_batch(batch: dict[str, list[Any]]) -> list[str]:
    n = batch_size_from_columns(batch)
    instruction_col = batch.get("instruction")
    input_col = batch.get("input")
    output_col = batch.get("output")
    response_col = batch.get("response")

    texts: list[str] = []
    for i in range(n):
        row = {
            "instruction": instruction_col[i] if instruction_col is not None else "",
            "input": input_col[i] if input_col is not None else "",
            "output": output_col[i] if output_col is not None else "",
            "response": response_col[i] if response_col is not None else "",
        }
        texts.append(build_text_from_row(row))
    return texts


def resolve_id_column(dataset: Dataset) -> str | None:
    names = set(dataset.column_names)
    for cand in ID_CANDIDATES:
        if cand in names:
            return cand
    return None


def build_ids_from_batch(
    batch: dict[str, list[Any]],
    id_col: str | None,
    global_start_idx: int,
) -> list[Any]:
    n = batch_size_from_columns(batch)
    if id_col is None:
        return [global_start_idx + i for i in range(n)]
    values = batch.get(id_col)
    if values is None:
        return [global_start_idx + i for i in range(n)]
    out: list[Any] = []
    for i in range(n):
        value = values[i]
        out.append(global_start_idx + i if value is None else value)
    return out


def load_input_dataset(dataset_path: str, split: str, config_name: str | None) -> Dataset:
    p = Path(dataset_path)
    if p.exists():
        if p.is_file():
            suffix = p.suffix.lower()
            if suffix in {".jsonl", ".json"}:
                return load_dataset("json", data_files=str(p), split="train")
            if suffix == ".parquet":
                return load_dataset("parquet", data_files=str(p), split="train")
            raise ValueError(f"Unsupported local file extension: {suffix}")
        if p.is_dir():
            train_jsonl = p / "train.jsonl"
            train_parquet = p / "train.parquet"
            if train_jsonl.exists():
                return load_dataset("json", data_files=str(train_jsonl), split="train")
            if train_parquet.exists():
                return load_dataset("parquet", data_files=str(train_parquet), split="train")
            raise ValueError(f"Directory does not contain train.jsonl/train.parquet: {p}")
    return load_dataset(dataset_path, name=config_name, split=split)


def _ensure_2d_float32(arr: Any) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        out = out[None, :]
    if out.ndim != 2:
        raise ValueError(f"Centroids must be 2D; got shape={out.shape}")
    return out


def _extract_from_pt(obj: Any) -> tuple[np.ndarray, list[str] | None]:
    node_ids: list[str] | None = None
    tensor_like: Any = obj
    if isinstance(obj, dict):
        for key in ("centroids", "center_vectors", "vectors", "embeddings"):
            if key in obj:
                tensor_like = obj[key]
                break
        nid = obj.get("node_ids")
        if isinstance(nid, list):
            node_ids = [str(x) for x in nid]

    if isinstance(tensor_like, torch.Tensor):
        arr = tensor_like.detach().cpu().float().numpy()
    else:
        arr = _ensure_2d_float32(tensor_like)
    arr = _ensure_2d_float32(arr)
    return arr, node_ids


def _extract_from_centroid_jsonl(path: Path) -> tuple[np.ndarray, list[str]]:
    vectors: list[list[float]] = []
    node_ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            vec = obj.get("center_vector")
            if not isinstance(vec, list) or not vec:
                continue
            vectors.append([float(x) for x in vec])
            node_ids.append(str(obj.get("node_id", f"cluster_{lineno - 1}")))

    if not vectors:
        raise ValueError(f"No non-empty center_vector found in {path}")
    arr = _ensure_2d_float32(vectors)
    if len(node_ids) != arr.shape[0]:
        raise ValueError("node_ids count does not match centroid count.")
    return arr, node_ids


def load_centroids(path: Path) -> tuple[torch.Tensor, list[str] | None]:
    if not path.exists():
        raise FileNotFoundError(f"Centroids path not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = _ensure_2d_float32(np.load(path))
        node_ids = None
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        arr, node_ids = _extract_from_pt(obj)
    elif suffix == ".jsonl":
        arr, node_ids = _extract_from_centroid_jsonl(path)
    else:
        raise ValueError(f"Unsupported centroids extension: {suffix}. Use .npy/.pt/.jsonl")

    centroids = torch.tensor(arr, dtype=torch.float32)
    centroids = F.normalize(centroids, p=2, dim=1)
    return centroids, node_ids


def read_centroid_summary_embedding_model(centroids_path: Path) -> str | None:
    if centroids_path.suffix.lower() != ".jsonl":
        return None
    summary_path = centroids_path.with_name(f"{centroids_path.stem}_summary.json")
    if not summary_path.exists():
        return None
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    embedding = obj.get("embedding", {})
    model_name = embedding.get("embedding_model")
    if isinstance(model_name, str) and model_name.strip():
        return model_name.strip()
    return None


class TransformerEncoder:
    def __init__(self, model_name: str, device: torch.device, max_length: int):
        self.device = device
        self.max_length = max(16, int(max_length))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    def encode(self, texts: list[str]) -> torch.Tensor:
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with torch.inference_mode():
            outputs = self.model(**tokenized)
            sentence_emb = self.mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
            # Force a stable dtype for downstream matmul with centroid matrix.
            return F.normalize(sentence_emb, p=2, dim=1).float()


def infer_output_path(dataset_path: str, top_k: int) -> Path:
    p = Path(dataset_path)
    if p.exists() and p.is_file():
        return p.with_name(f"{p.stem}_coarse_topk{top_k}.jsonl")
    if p.exists() and p.is_dir():
        return p / f"coarse_topk{top_k}.jsonl"
    safe_name = dataset_path.replace("/", "_")
    return Path(f"{safe_name}_coarse_topk{top_k}.jsonl")


def build_top_k_node_ids(top_k_indices: list[list[int]], node_ids: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    for row in top_k_indices:
        out.append([node_ids[idx] for idx in row])
    return out


def open_jsonl_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def iter_batch_starts(total: int, batch_size: int) -> Iterable[int]:
    return range(0, total, max(1, batch_size))


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ensure_dependencies()

    if args.top_k <= 0:
        raise ValueError("--top_k must be > 0")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")

    dataset = load_input_dataset(
        dataset_path=args.dataset_path,
        split=args.dataset_split,
        config_name=args.dataset_config,
    )
    if len(dataset) == 0:
        raise ValueError("Input dataset is empty.")

    output_path = args.output_path or infer_output_path(args.dataset_path, args.top_k)
    output_suffix = output_path.suffix.lower()
    if output_suffix not in {".jsonl", ".parquet"}:
        raise ValueError("--output_path must end with .jsonl or .parquet")

    centroids_cpu, node_ids = load_centroids(args.centroids_path)
    centroid_model_hint = read_centroid_summary_embedding_model(args.centroids_path)
    m = int(centroids_cpu.shape[0])
    d = int(centroids_cpu.shape[1])
    if args.top_k > m:
        raise ValueError(f"--top_k ({args.top_k}) cannot exceed centroid count ({m})")

    device = resolve_device(args.device)
    encoder = TransformerEncoder(args.model_name, device=device, max_length=args.max_length)
    centroids = centroids_cpu.to(device)

    id_col = resolve_id_column(dataset)
    logging.info(
        "Routing start: rows=%d, centroids=%d, dim=%d, top_k=%d, tau=%.4f, device=%s, id_col=%s",
        len(dataset),
        m,
        d,
        args.top_k,
        args.temperature,
        device,
        id_col or "<row_index>",
    )

    jsonl_writer = None
    parquet_writer = None
    try:
        if output_suffix == ".jsonl":
            jsonl_writer = open_jsonl_writer(output_path)
        else:
            if pa is None or pq is None:
                raise RuntimeError("pyarrow is required to write parquet output.")
            output_path.parent.mkdir(parents=True, exist_ok=True)

        starts = iter_batch_starts(len(dataset), args.batch_size)
        if tqdm is not None:
            starts = tqdm(starts, total=(len(dataset) + args.batch_size - 1) // args.batch_size, desc="Dense routing")

        total_written = 0
        for start in starts:
            end = min(start + args.batch_size, len(dataset))
            batch = dataset[start:end]

            texts = build_texts_from_batch(batch)
            if args.query_prefix:
                texts = [f"{args.query_prefix}{t}" for t in texts]
            ids = build_ids_from_batch(batch, id_col=id_col, global_start_idx=start)

            emb = encoder.encode(texts)  # [B, D], normalized
            if emb.shape[1] != centroids.shape[1]:
                hint = (
                    f"Centroid model from summary: {centroid_model_hint}. "
                    if centroid_model_hint
                    else ""
                )
                raise ValueError(
                    "Embedding dim mismatch. "
                    f"data emb dim={emb.shape[1]}, centroid dim={centroids.shape[1]}, "
                    f"routing model={args.model_name}. "
                    f"{hint}"
                    "Use the same embedding model for Stage-1 centroid building and Stage-2 routing. "
                    "Recommended fix: rerun stage_1_capability_extraction.py with "
                    f"--embedding-model {args.model_name} to regenerate centroids."
                )

            # S = cos(E(x), mu) because both are L2-normalized.
            sim = torch.matmul(emb, centroids.T)  # [B, M]
            topk_vals, topk_idx = torch.topk(sim, k=args.top_k, dim=1, largest=True, sorted=True)  # [B, K]
            topk_scores = torch.softmax(topk_vals / args.temperature, dim=1)  # [B, K]

            topk_idx_list: list[list[int]] = topk_idx.detach().cpu().tolist()
            topk_scores_list: list[list[float]] = topk_scores.detach().cpu().tolist()

            if output_suffix == ".jsonl":
                topk_node_ids_list = build_top_k_node_ids(topk_idx_list, node_ids) if node_ids is not None else None
                for i in range(len(topk_idx_list)):
                    row = {
                        "id": ids[i],
                        "top_k_indices": [int(x) for x in topk_idx_list[i]],
                        "top_k_scores": [float(x) for x in topk_scores_list[i]],
                    }
                    if topk_node_ids_list is not None:
                        row["top_k_node_ids"] = topk_node_ids_list[i]
                    jsonl_writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                ids_str = [str(x) for x in ids]
                block: dict[str, Any] = {
                    "id": ids_str,
                    "top_k_indices": topk_idx_list,
                    "top_k_scores": topk_scores_list,
                }
                if node_ids is not None:
                    block["top_k_node_ids"] = build_top_k_node_ids(topk_idx_list, node_ids)
                table = pa.table(block)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(str(output_path), table.schema)
                parquet_writer.write_table(table)

            total_written += (end - start)

        logging.info("Routing complete: wrote %d rows -> %s", total_written, output_path)
    finally:
        if jsonl_writer is not None:
            jsonl_writer.close()
        if parquet_writer is not None:
            parquet_writer.close()


if __name__ == "__main__":
    main()
