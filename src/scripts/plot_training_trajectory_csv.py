"""Plot training trajectory line charts from a CSV file.

Default behavior:
- x axis: step
- y axes: all numeric columns except x axis
- one subplot per y column
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Input trajectory CSV path")
    parser.add_argument("--out_png", type=str, required=True, help="Output PNG path")
    parser.add_argument("--x_col", type=str, default="step", help="X-axis column name")
    parser.add_argument(
        "--y_cols",
        type=str,
        default="",
        help="Comma-separated Y-axis columns. If empty, auto-detect numeric columns except x_col.",
    )
    parser.add_argument(
        "--single_axis",
        action="store_true",
        help="Plot all y columns on a single axis instead of one subplot per column.",
    )
    return parser.parse_args()


def _to_float(value: str) -> float:
    text = value.strip()
    if not text:
        return float("nan")
    return float(text)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def infer_y_cols(rows: List[Dict[str, str]], x_col: str) -> List[str]:
    keys = list(rows[0].keys())
    y_cols: List[str] = []
    for key in keys:
        if key == x_col:
            continue
        numeric = True
        for row in rows:
            try:
                _to_float(row[key])
            except Exception:
                numeric = False
                break
        if numeric:
            y_cols.append(key)
    return y_cols


def build_series(rows: List[Dict[str, str]], x_col: str, y_cols: List[str]) -> tuple[List[float], Dict[str, List[float]]]:
    x_vals = [_to_float(r[x_col]) for r in rows]
    series: Dict[str, List[float]] = {k: [] for k in y_cols}
    for row in rows:
        for key in y_cols:
            series[key].append(_to_float(row[key]))
    return x_vals, series


def sort_by_x(x_vals: List[float], series: Dict[str, List[float]]) -> tuple[List[float], Dict[str, List[float]]]:
    order = sorted(range(len(x_vals)), key=lambda i: x_vals[i])
    sorted_x = [x_vals[i] for i in order]
    sorted_series: Dict[str, List[float]] = {}
    for key, vals in series.items():
        sorted_series[key] = [vals[i] for i in order]
    return sorted_x, sorted_series


def plot_lines(
    x_vals: List[float],
    series: Dict[str, List[float]],
    out_png: Path,
    x_col: str,
    single_axis: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib unavailable: {exc}") from exc

    y_cols = list(series.keys())
    if single_axis:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for key in y_cols:
            y_vals = series[key]
            if all(math.isnan(v) for v in y_vals):
                continue
            ax.plot(x_vals, y_vals, marker="o", linewidth=1.6, markersize=3.5, label=key)
        ax.set_title("Training Trajectory")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        n = len(y_cols)
        fig, axes = plt.subplots(n, 1, figsize=(10, max(3 * n, 4)), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, key in zip(axes, y_cols):
            y_vals = series[key]
            if all(math.isnan(v) for v in y_vals):
                continue
            ax.plot(x_vals, y_vals, marker="o", linewidth=1.6, markersize=3.5)
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.3)

        axes[0].set_title("Training Trajectory")
        axes[-1].set_xlabel(x_col)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    out_png = Path(args.out_png)
    rows = load_csv_rows(input_csv)

    if args.x_col not in rows[0]:
        raise ValueError(f"x_col '{args.x_col}' not found in CSV columns: {list(rows[0].keys())}")

    if args.y_cols.strip():
        y_cols = [c.strip() for c in args.y_cols.split(",") if c.strip()]
    else:
        y_cols = infer_y_cols(rows, args.x_col)

    if not y_cols:
        raise ValueError("No numeric y columns found to plot.")

    for c in y_cols:
        if c not in rows[0]:
            raise ValueError(f"y_col '{c}' not found in CSV columns: {list(rows[0].keys())}")

    x_vals, series = build_series(rows, args.x_col, y_cols)
    x_vals, series = sort_by_x(x_vals, series)
    plot_lines(x_vals, series, out_png, args.x_col, args.single_axis)
    print(f"[Done] plot: {out_png}")
    print(f"[Info] x_col: {args.x_col}")
    print(f"[Info] y_cols: {', '.join(y_cols)}")


if __name__ == "__main__":
    main()
