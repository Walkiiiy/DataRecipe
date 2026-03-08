"""Convert trajectory_meta.jsonl to CSV and plot curves.

Input:
- weight_trajectory/trajectory_meta.jsonl

Output:
- CSV with columns: step, loss, lr, tag, file
- PNG with two subplots: loss-step and lr-step
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta_jsonl",
        type=str,
        required=True,
        help="Path to trajectory_meta.jsonl",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        required=True,
        help="Output PNG path",
    )
    return parser.parse_args()


def load_rows(meta_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    "step": int(row.get("step", 0)),
                    "loss": float(row["loss"]) if row.get("loss") is not None else float("nan"),
                    "lr": float(row["lr"]) if row.get("lr") is not None else float("nan"),
                    "tag": str(row.get("tag", "")),
                    "file": str(row.get("file", "")),
                }
            )
    rows.sort(key=lambda x: x["step"])
    return rows


def write_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "lr", "tag", "file"])
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: List[Dict], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Warn] matplotlib unavailable, skip png plot: {exc}")
        return

    steps = [int(r["step"]) for r in rows]
    losses = [float(r["loss"]) for r in rows]
    lrs = [float(r["lr"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(steps, losses, marker="o")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss over Steps")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, lrs, marker="o")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_xlabel("Step")
    axes[1].set_title("Learning Rate over Steps")
    axes[1].grid(True, alpha=0.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    meta_path = Path(args.meta_jsonl)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)

    rows = load_rows(meta_path)
    if not rows:
        raise ValueError(f"No rows found in {meta_path}")

    write_csv(rows, out_csv)
    plot(rows, out_png)
    print(f"[Done] csv: {out_csv}")
    print(f"[Done] png: {out_png}")


if __name__ == "__main__":
    main()
