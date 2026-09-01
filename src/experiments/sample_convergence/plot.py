"""Figures for the sample-count convergence check.

uv run -m src.experiments.sample_convergence.plot --results_dir results/sample_convergence --output_dir results/figures
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import simple_parsing

from src.utils.io import read_json

LOGGER = logging.getLogger(__name__)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
K_COLORS = {20: BLUE, 50: ORANGE, 100: AQUA}


@dataclass
class Config:
    results_dir: str = ""
    output_dir: str = "results/figures"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    config = simple_parsing.parse(Config)
    results_dir = Path(config.results_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(results_dir / "cell_table_by_k.csv")
    per_k = read_json(results_dir / "convergence.json")["per_k"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ref = long[long.k == 100].set_index(["model_id", "dataset_name"]).faithfulness_proxy
    for k in sorted(long.k.unique()):
        if k == 100:
            continue
        sub = long[long.k == k].set_index(["model_id", "dataset_name"])
        ax.scatter(ref.reindex(sub.index), sub.faithfulness_proxy, s=38, alpha=0.85,
                   color=K_COLORS[k], edgecolor="white", linewidth=0.8, label=f"k={k}", zorder=3)
    lims = [long.faithfulness_proxy.min() - 0.03, long.faithfulness_proxy.max() + 0.03]
    ax.plot(lims, lims, color=MUTED, linestyle="--", linewidth=1.2, zorder=2, label="identity")
    ax.set_xlabel("Proxy at k=100", fontsize=8.5, color=MUTED)
    ax.set_ylabel("Proxy at fewer samples", fontsize=8.5, color=MUTED)
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)
    _style(ax)

    ax = axes[1]
    ks = [r["k"] for r in per_k]
    ax.plot(ks, [r["mean_ci_width"] for r in per_k], "-o", color=BLUE, linewidth=2.2, markersize=8,
            markeredgecolor="white", markeredgewidth=0.9)
    ax.set_xticks(ks)
    ax.set_xlabel("CoT samples per question", fontsize=8.5, color=MUTED)
    ax.set_ylabel("Mean bootstrap CI width", fontsize=8.5, color=MUTED)
    _style(ax)

    ax = axes[2]
    reductions = [r["decomposition"]["pct_reduction"] for r in per_k]
    lowers = [r["decomposition"]["pct_reduction_ci"][0] for r in per_k]
    uppers = [r["decomposition"]["pct_reduction_ci"][1] for r in per_k]
    ax.axhline(50, color=ORANGE, linestyle="--", linewidth=1.6, zorder=2,
               label="Pre-registered 50% threshold")
    ax.vlines(ks, lowers, uppers, color=BLUE, linewidth=2.4, alpha=0.55, zorder=3)
    ax.plot(ks, reductions, "o", color=BLUE, markersize=9, zorder=4,
            markeredgecolor="white", markeredgewidth=1.0)
    for k, v in zip(ks, reductions):
        ax.annotate(f"{v:.1f}%", (k, v), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(ks)
    ax.set_ylim(0, 105)
    ax.set_xlabel("CoT samples per question", fontsize=8.5, color=MUTED)
    ax.set_ylabel("Coefficient reduction (%)", fontsize=8.5, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=MUTED)
    _style(ax)

    fig.tight_layout()
    path = output_dir / "sample_convergence.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()
