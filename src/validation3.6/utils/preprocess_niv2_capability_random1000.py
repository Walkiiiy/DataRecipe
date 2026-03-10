"""Preprocess niv2_capability_data_random1000 for capability-anchor experiments.

Rules implemented:
- Different task == different capability.
- Select 20 tasks as effective capabilities.
- Sample 40 items per selected capability as anchors.
- Anchors are excluded from training data.
- Add fields to output records:
  - 相关度
  - 准确性
  - 多样性
  - 难度
  - 能力锚点属性

If a record is an anchor:
- 相关度 = 1
- 能力锚点属性 = capability name
Else:
- 相关度 = ""
- 能力锚点属性 = ""
Other added attributes are kept empty strings by default.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/walkiiiy/DataRecipe/data/flan/niv2_capability_data_random1000",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/walkiiiy/DataRecipe/data/flan/niv2_capability_data_ramdom1000_preprocessed",
    )
    parser.add_argument("--num_capabilities", type=int, default=20)
    parser.add_argument("--anchors_per_capability", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def add_fields(ex: Dict, is_anchor: bool, capability_name: str) -> Dict:
    out = dict(ex)
    out["相关度"] = 1 if is_anchor else ""
    out["准确性"] = ""
    out["多样性"] = ""
    out["难度"] = ""
    out["能力锚点属性"] = capability_name if is_anchor else ""
    return out


def read_jsonl(path: Path) -> List[Dict]:
    data: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No jsonl files found in {input_dir}")

    # Count lines and keep tasks that can provide enough anchors.
    task_rows: List[Tuple[Path, int]] = []
    for fp in files:
        n = sum(1 for _ in fp.open("r", encoding="utf-8"))
        task_rows.append((fp, n))

    eligible = [fp for fp, n in task_rows if n >= args.anchors_per_capability]
    if len(eligible) < args.num_capabilities:
        raise ValueError(
            f"Not enough eligible tasks with >= {args.anchors_per_capability} rows: "
            f"need {args.num_capabilities}, got {len(eligible)}"
        )

    # Random selection with fixed seed for reproducibility.
    selected_caps = rng.sample(eligible, args.num_capabilities)
    selected_caps = sorted(selected_caps)
    selected_names = {fp.stem for fp in selected_caps}

    output_root = Path(args.output_root)
    eff_dir = output_root / "effective_capabilities"
    anchor_dir = output_root / "capability_anchors"
    train_dir = output_root / "train_data"
    meta_dir = output_root / "meta"
    for d in (eff_dir, anchor_dir, train_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Store chosen capability list.
    with (meta_dir / "selected_capabilities.txt").open("w", encoding="utf-8") as f:
        for fp in selected_caps:
            f.write(fp.stem + "\n")

    total_train = 0
    total_anchor = 0

    for fp in files:
        records = read_jsonl(fp)
        capability_name = fp.stem

        anchor_idx: Set[int] = set()
        if capability_name in selected_names:
            anchor_idx = set(rng.sample(range(len(records)), args.anchors_per_capability))

        # Effective capability storage (only selected tasks, keep all records with added fields).
        if capability_name in selected_names:
            eff_records = [
                add_fields(ex, i in anchor_idx, capability_name) for i, ex in enumerate(records)
            ]
            write_jsonl(eff_dir / fp.name, eff_records)

        # Anchor storage (only sampled anchors from selected tasks).
        if capability_name in selected_names:
            anchor_records = [
                add_fields(records[i], True, capability_name)
                for i in sorted(anchor_idx)
            ]
            write_jsonl(anchor_dir / fp.name, anchor_records)
            total_anchor += len(anchor_records)

        # Train storage (all non-anchor records from all tasks).
        train_records = [
            add_fields(ex, False, capability_name)
            for i, ex in enumerate(records)
            if i not in anchor_idx
        ]
        write_jsonl(train_dir / fp.name, train_records)
        total_train += len(train_records)

    # Summary
    summary = {
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "num_all_tasks": len(files),
        "num_selected_capabilities": len(selected_caps),
        "anchors_per_capability": args.anchors_per_capability,
        "total_anchor_records": total_anchor,
        "total_train_records": total_train,
        "seed": args.seed,
    }
    with (meta_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
