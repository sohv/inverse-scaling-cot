"""Figure for a paired two-run comparison (position randomisation, prompts, AWQ vs BF16).

uv run -m src.experiments.comparison.plot --cells_csv results/quantization/awq_vs_bf16_cells.csv --label awq_vs_bf16 --output_dir results/figures
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

LOGGER = logging.getLogger(__name__)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

TASK_LABELS = {
    "aqua": "AQuA",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "logiqa": "LogiQA",
    "openbookqa": "OpenBookQA",
}


@dataclass
class Config:
    cells_csv: str = ""
    label: str = "comparison"
    baseline_name: str = "Baseline"
    variant_name: str = "Variant"
    output_dir: str = "results/figures"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _dumbbell(ax, df, metric, baseline_name, variant_name, title):
    df = df.sort_values(["dataset_name", "size_b"]).reset_index(drop=True)
    y = np.arange(len(df))
    base = df[f"{metric}_baseline"]
    var = df[f"{metric}_variant"]

    ax.hlines(y, base, var, color=GRID, linewidth=2.4, zorder=2)
    ax.plot(base, y, "o", markersize=7, color=BLUE, zorder=3, label=baseline_name,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(var, y, "s", markersize=6.5, color=ORANGE, zorder=4, label=variant_name,
            markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{TASK_LABELS.get(r.dataset_name, r.dataset_name)} {r.size_b:g}B" for r in df.itertuples()],
        fontsize=7.5, color=INK,
    )
    ax.invert_yaxis()
    _style(ax)


def main():
    config = simple_parsing.parse(Config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.cells_csv)

    height = max(3.4, 0.28 * len(df) + 1.6)
    fig, axes = plt.subplots(1, 2, figsize=(12, height))

    _dumbbell(axes[0], df, "faithfulness_proxy", config.baseline_name, config.variant_name,
              "Faithfulness proxy")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right", labelcolor=MUTED)
    axes[0].set_xlabel("Match fraction", fontsize=8.5, color=MUTED)

    _dumbbell(axes[1], df, "dependence", config.baseline_name, config.variant_name,
              "CoT dependence (real minus shuffled)")
    axes[1].set_xlabel("Dependence", fontsize=8.5, color=MUTED)

    fig.tight_layout()
    path = output_dir / f"{config.label}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()
