"""Analyze weight trajectory checkpoints and plot 4-dimension curves.

Given full-state-dict checkpoints (*.pt), this script evaluates model NLL on
four dimension-specific subsets built from scored data fields:
- 相关度
- 准确性
- 能力多样性
- 难度

Output:
- CSV with per-checkpoint per-dimension metrics
- PNG line chart for the 4 dimensions
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make project import-safe for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIMENSIONS = ["相关度", "准确性", "能力多样性", "难度"]


@dataclass
class Sample:
    input_ids: List[int]
    labels: List[int]
    dim_scores: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing model config/tokenizer (e.g., recipe_scored_run_quick).",
    )
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        required=True,
        help="Directory containing weight checkpoints (*.pt) and trajectory_meta.jsonl.",
    )
    parser.add_argument(
        "--scored_data_dir",
        type=str,
        default="/home/walkiiiy/DataRecipe/data/flan/niv2_capability_data_random1000_alphagasus_scored",
    )
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_samples_total",
        type=int,
        default=2000,
        help="Cap total scored samples loaded for analysis speed.",
    )
    parser.add_argument(
        "--topk_per_dimension",
        type=int,
        default=400,
        help="For each dimension, evaluate on top-k scored samples.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="/home/walkiiiy/DataRecipe/model/dimension_curve.csv",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="/home/walkiiiy/DataRecipe/model/dimension_curve.png",
    )
    return parser.parse_args()


def build_prompt(instruction: str, inp: str) -> str:
    return f"Instruction:\n{instruction}\n\nInput:\n{inp}\n\nResponse:\n"


def iter_jsonl_files(root: Path) -> Iterable[Path]:
    for fp in sorted(root.rglob("*.jsonl")):
        if fp.name == "errors.jsonl":
            continue
        yield fp


def parse_float(x) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def encode_sample(ex: Dict, tokenizer: AutoTokenizer, max_seq_len: int) -> Sample | None:
    instruction = str(ex.get("instruction", ""))
    inp = str(ex.get("input", ""))
    response = str(ex.get("response", ""))
    if not response.strip():
        return None

    dim_scores: Dict[str, float] = {}
    for d in DIMENSIONS:
        v = parse_float(ex.get(d))
        if v is None:
            return None
        dim_scores[d] = v

    prefix = build_prompt(instruction, inp)
    full_text = prefix + response

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        return None

    input_ids: List[int] = []
    labels: List[int] = []

    if bos_id is not None:
        input_ids.append(bos_id)
        labels.append(-100)

    input_ids.extend(full_ids)
    labels.extend([-100] * len(full_ids))

    response_start = (1 if bos_id is not None else 0) + len(prefix_ids)
    if response_start >= len(input_ids):
        return None

    for i in range(response_start, len(input_ids)):
        labels[i] = input_ids[i]

    input_ids.append(eos_id)
    labels.append(eos_id)

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

    if all(v == -100 for v in labels):
        return None

    return Sample(input_ids=input_ids, labels=labels, dim_scores=dim_scores)


def collate(samples: List[Sample], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(s.input_ids) for s in samples)
    input_ids, labels, attn = [], [], []
    for s in samples:
        cur = len(s.input_ids)
        pad = max_len - cur
        input_ids.append(s.input_ids + [pad_id] * pad)
        labels.append(s.labels + [-100] * pad)
        attn.append([1] * cur + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


def load_samples(scored_dir: Path, tokenizer: AutoTokenizer, max_seq_len: int, max_total: int) -> List[Sample]:
    samples: List[Sample] = []
    for fp in iter_jsonl_files(scored_dir):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                s = encode_sample(ex, tokenizer, max_seq_len)
                if s is not None:
                    samples.append(s)
                    if len(samples) >= max_total:
                        return samples
    return samples


def build_dimension_subsets(samples: List[Sample], topk: int) -> Dict[str, List[Sample]]:
    out: Dict[str, List[Sample]] = {}
    for d in DIMENSIONS:
        ranked = sorted(samples, key=lambda s: s.dim_scores[d], reverse=True)
        out[d] = ranked[: min(topk, len(ranked))]
    return out


def evaluate_subset(
    model: AutoModelForCausalLM,
    subset: List[Sample],
    batch_size: int,
    pad_id: int,
    device: torch.device,
) -> float:
    if not subset:
        return float("nan")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for i in range(0, len(subset), batch_size):
            batch = subset[i : i + batch_size]
            x, y, a = collate(batch, pad_id)
            x = x.to(device)
            y = y.to(device)
            a = a.to(device)

            logits = model(input_ids=x, attention_mask=a).logits
            # token-level CE with ignore_index=-100
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                y[:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid = (y[:, 1:] != -100).sum().item()
            total_loss += float(loss.item())
            total_tokens += int(valid)

    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


def load_checkpoints(trajectory_dir: Path) -> List[Tuple[int, Path]]:
    meta = trajectory_dir / "trajectory_meta.jsonl"
    ckpts: List[Tuple[int, Path]] = []

    if meta.exists():
        with meta.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                step = int(row["step"])
                path = trajectory_dir / row["file"]
                if path.exists():
                    ckpts.append((step, path))
    else:
        for p in sorted(trajectory_dir.glob("*.pt")):
            # fallback if no meta file
            name = p.stem
            step = 0
            if "_" in name:
                try:
                    step = int(name.split("_")[-1])
                except Exception:
                    step = 0
            ckpts.append((step, p))

    ckpts = sorted(ckpts, key=lambda x: x[0])
    return ckpts


def maybe_plot(csv_rows: List[Dict[str, str]], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] matplotlib unavailable, skip png plot: {exc}")
        return

    steps = [int(r["step"]) for r in csv_rows]
    plt.figure(figsize=(10, 6))
    for d in DIMENSIONS:
        vals = [float(r[d]) for r in csv_rows]
        plt.plot(steps, vals, marker="o", label=d)

    plt.xlabel("Checkpoint Step")
    plt.ylabel("Response-only NLL (lower better)")
    plt.title("Dimension Curves over Weight Trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir)
    traj_dir = Path(args.trajectory_dir)
    scored_dir = Path(args.scored_data_dir)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

    print("[Data] loading scored samples...")
    samples = load_samples(scored_dir, tokenizer, args.max_seq_len, args.max_samples_total)
    print(f"[Data] loaded_samples={len(samples)}")
    if not samples:
        raise ValueError("No valid samples loaded from scored data.")

    subsets = build_dimension_subsets(samples, args.topk_per_dimension)
    for d in DIMENSIONS:
        print(f"[Data] subset[{d}]={len(subsets[d])}")

    ckpts = load_checkpoints(traj_dir)
    print(f"[Traj] checkpoints={len(ckpts)}")
    if not ckpts:
        raise ValueError(f"No checkpoints found in {traj_dir}")

    rows: List[Dict[str, str]] = []

    for idx, (step, ckpt_path) in enumerate(ckpts, start=1):
        print(f"[Eval] ({idx}/{len(ckpts)}) step={step} ckpt={ckpt_path.name}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=True)

        row: Dict[str, str] = {"step": str(step), "checkpoint": ckpt_path.name}
        for d in DIMENSIONS:
            nll = evaluate_subset(
                model=model,
                subset=subsets[d],
                batch_size=args.batch_size,
                pad_id=tokenizer.pad_token_id,
                device=device,
            )
            row[d] = f"{nll:.8f}"
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "checkpoint", *DIMENSIONS])
        writer.writeheader()
        writer.writerows(rows)

    maybe_plot(rows, out_png)

    print(f"[Done] csv: {out_csv}")
    print(f"[Done] png: {out_png}")


if __name__ == "__main__":
    main()
