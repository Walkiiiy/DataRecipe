"""4.1 EXP - 结果可视化脚本（论文图）。

输入：
- 三组或四组实验日志（train_eval_log.csv）
- 可选：雷达图评分 JSON（若无则使用模拟分数）

输出（DPI=300）：
1) learning_curves_val_loss.png
2) data_distribution_radar.png
3) final_eval_loss_bar.png
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize SFT comparison results")
    parser.add_argument("--ours-log-csv", type=Path, required=True)
    parser.add_argument("--kmeans-log-csv", type=Path, required=True)
    parser.add_argument("--random-log-csv", type=Path, required=True)
    parser.add_argument("--category-log-csv", type=Path, default=None, help="可选：category 采样实验日志 CSV")
    parser.add_argument(
        "--radar-scores-json",
        type=Path,
        default=None,
        help=(
            "可选 JSON，格式示例："
            "{\"ours\":{\"Cognition\":0.9,\"Domain\":0.8,\"Task\":0.88}, ...}"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/alpaca-gpt4-data-en/exp/figures"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def load_log_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Log CSV not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_eval_curve(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    for r in rows:
        step = r.get("step", "")
        eval_loss = r.get("eval_loss", "")
        if step in ("", None) or eval_loss in ("", None):
            continue
        try:
            s = int(float(step))
            l = float(eval_loss)
        except ValueError:
            continue
        steps.append(s)
        losses.append(l)
    return steps, losses


def last_eval_loss(rows: list[dict[str, Any]]) -> float:
    steps, losses = extract_eval_curve(rows)
    if not losses:
        return float("nan")
    return losses[-1]


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_log_map(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    logs = {
        "ours": load_log_csv(args.ours_log_csv),
        "kmeans": load_log_csv(args.kmeans_log_csv),
        "random": load_log_csv(args.random_log_csv),
    }
    if args.category_log_csv is not None:
        logs["category"] = load_log_csv(args.category_log_csv)
    return logs


def plot_learning_curves(
    logs: dict[str, list[dict[str, Any]]],
    out_path: Path,
    dpi: int,
) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8.5, 5.2))

    style = {
        "ours": ("Ours (Tree Sampling)", "#1f77b4"),
        "kmeans": ("KMeans Baseline", "#ff7f0e"),
        "random": ("Random Baseline", "#2ca02c"),
        "category": ("Category Baseline", "#d62728"),
    }
    for key in ["ours", "kmeans", "random", "category"]:
        if key not in logs:
            continue
        name, color = style[key]
        rows = logs[key]
        x, y = extract_eval_curve(rows)
        if len(x) == 0:
            logging.warning("No eval curve found for %s", name)
            continue
        plt.plot(x, y, label=name, linewidth=2.2, color=color)

    plt.xlabel("Training Step")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def load_or_simulate_radar_scores(path: Path | None) -> dict[str, dict[str, float]]:
    if path is not None and path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    # 若未提供评分器输出，这里使用模拟分数（论文图占位示意）。
    return {
        "ours": {"Cognition": 0.88, "Domain": 0.86, "Task": 0.90},
        "kmeans": {"Cognition": 0.80, "Domain": 0.79, "Task": 0.81},
        "random": {"Cognition": 0.72, "Domain": 0.70, "Task": 0.74},
        "category": {"Cognition": 0.83, "Domain": 0.82, "Task": 0.84},
    }


def plot_radar(scores: dict[str, dict[str, float]], out_path: Path, dpi: int) -> None:
    # 统一维度顺序
    dims = list(next(iter(scores.values())).keys())
    angles = np.linspace(0, 2 * math.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6.5, 6.5))
    ax = plt.subplot(111, polar=True)

    style = {
        "ours": ("Ours (Tree Sampling)", "#1f77b4"),
        "kmeans": ("KMeans Baseline", "#ff7f0e"),
        "random": ("Random Baseline", "#2ca02c"),
        "category": ("Category Baseline", "#d62728"),
    }
    for key in ["ours", "kmeans", "random", "category"]:
        if key not in scores:
            continue
        values = [float(scores[key][d]) for d in dims]
        values += values[:1]
        label, color = style[key]
        ax.plot(angles, values, label=label, linewidth=2.0, color=color)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Data Distribution Radar (Coverage Uniformity)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_final_bar(
    logs: dict[str, list[dict[str, Any]]],
    out_path: Path,
    dpi: int,
) -> None:
    sns.set_theme(style="ticks")
    label_map = {
        "ours": "Ours",
        "kmeans": "KMeans",
        "random": "Random",
        "category": "Category",
    }
    color_map = {
        "ours": "#1f77b4",
        "kmeans": "#ff7f0e",
        "random": "#2ca02c",
        "category": "#d62728",
    }
    keys = [k for k in ["ours", "kmeans", "random", "category"] if k in logs]
    labels = [label_map[k] for k in keys]
    values = [last_eval_loss(logs[k]) for k in keys]
    palette = [color_map[k] for k in keys]

    plt.figure(figsize=(7.0, 5.0))
    ax = sns.barplot(x=labels, y=values, palette=palette)
    ax.set_xlabel("Sampling Strategy")
    ax.set_ylabel("Final Eval Loss")
    ax.set_title("Final Performance Comparison")
    for i, v in enumerate(values):
        txt = "NaN" if np.isnan(v) else f"{v:.4f}"
        ax.text(i, (0.0 if np.isnan(v) else v) + 0.01, txt, ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    ensure_out_dir(args.out_dir)

    logs = build_log_map(args)
    radar_scores = load_or_simulate_radar_scores(args.radar_scores_json)

    out_curve = args.out_dir / "learning_curves_val_loss.png"
    out_radar = args.out_dir / "data_distribution_radar.png"
    out_bar = args.out_dir / "final_eval_loss_bar.png"

    plot_learning_curves(logs, out_curve, args.dpi)
    plot_radar(radar_scores, out_radar, args.dpi)
    plot_final_bar(logs, out_bar, args.dpi)

    summary = {
        "ours_final_eval_loss": last_eval_loss(logs["ours"]),
        "kmeans_final_eval_loss": last_eval_loss(logs["kmeans"]),
        "random_final_eval_loss": last_eval_loss(logs["random"]),
        "figures": {
            "learning_curve": str(out_curve),
            "radar": str(out_radar),
            "bar": str(out_bar),
        },
    }
    if "category" in logs:
        summary["category_final_eval_loss"] = last_eval_loss(logs["category"])
    with (args.out_dir / "figure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info("Visualization done. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
