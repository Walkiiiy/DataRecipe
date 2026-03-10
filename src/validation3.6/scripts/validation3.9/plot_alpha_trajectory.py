"""Plot alpha trajectory (4 metric weights) from jsonl/csv.

Supported inputs:
- alpha_trajectory.jsonl (step, alpha_0..alpha_3)
- alpha_trajectory.csv (same schema)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to alpha_trajectory.jsonl or .csv")
    parser.add_argument("--out_png", type=str, required=True, help="Output line plot path")
    return parser.parse_args()


def load_rows(input_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(
                    {
                        "step": float(row["step"]),
                        "alpha_0": float(row["alpha_0"]),
                        "alpha_1": float(row["alpha_1"]),
                        "alpha_2": float(row["alpha_2"]),
                        "alpha_3": float(row["alpha_3"]),
                    }
                )
    elif input_path.suffix.lower() == ".csv":
        with input_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "step": float(row["step"]),
                        "alpha_0": float(row["alpha_0"]),
                        "alpha_1": float(row["alpha_1"]),
                        "alpha_2": float(row["alpha_2"]),
                        "alpha_3": float(row["alpha_3"]),
                    }
                )
    else:
        raise ValueError("input_path must be .jsonl or .csv")

    rows.sort(key=lambda x: x["step"])
    return rows


def plot(rows: List[Dict[str, float]], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib unavailable: {exc}") from exc

    steps = [r["step"] for r in rows]
    labels = ["alpha_0", "alpha_1", "alpha_2", "alpha_3"]

    plt.figure(figsize=(10, 6))
    for key in labels:
        vals = [r[key] for r in rows]
        plt.plot(steps, vals, marker="o", label=key)

    plt.xlabel("Step")
    plt.ylabel("Alpha Weight")
    plt.title("Alpha Trajectory (4 Metrics)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    out_png = Path(args.out_png)
    rows = load_rows(input_path)
    if not rows:
        raise ValueError(f"No rows in {input_path}")
    plot(rows, out_png)
    print(f"[Done] plot: {out_png}")


if __name__ == "__main__":
    main()
