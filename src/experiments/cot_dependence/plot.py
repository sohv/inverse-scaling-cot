"""Figures for the empirical null and CoT-dependence analysis.

uv run -m src.experiments.cot_dependence.plot --results_dir results/cot_dependence --output_dir results/figures
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
from src.utils.plotting import legend_below

LOGGER = logging.getLogger(__name__)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"
NULL_FILL = "#c9c8c2"

TASK_LABELS = {
    "aqua": "AQuA",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "logiqa": "LogiQA",
    "openbookqa": "OpenBookQA",
}
FAMILY_LABELS = {"qwen": "Qwen2.5-Instruct", "llama": "Llama-3-Instruct", "olmo": "OLMo-2-Instruct"}
FAMILY_COLORS = {"qwen": BLUE, "llama": ORANGE, "olmo": AQUA}


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


def plot_real_vs_null(df: pd.DataFrame, output_dir: Path) -> Path:
    """Real match rate against each cell's own permutation null, one panel per task."""
    tasks = sorted(df.dataset_name.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(16, 3.6), sharey=True)

    families = [f for f in ["qwen", "llama", "olmo"] if f in set(df.family)]
    for ax, task in zip(axes, tasks):
        sub = df[df.dataset_name == task]
        for family in families:
            fam = sub[sub.family == family].sort_values("size_b")
            if fam.empty:
                continue
            color = FAMILY_COLORS[family]
            ax.plot(
                fam.size_b,
                fam.real_match_rate,
                "-o",
                color=color,
                linewidth=2,
                markersize=6,
                label=f"{FAMILY_LABELS[family]} (real)",
                zorder=4,
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
            ax.plot(
                fam.size_b,
                fam.shuffled_mean,
                "--",
                color=color,
                linewidth=1.6,
                alpha=0.85,
                label=f"{FAMILY_LABELS[family]} (shuffled null)",
                zorder=3,
            )
            ax.fill_between(
                fam.size_b, fam.null_ci_lower, fam.null_ci_upper, color=NULL_FILL, alpha=0.75, zorder=2, linewidth=0
            )
        ax.set_xscale("log")
        ax.set_xlabel("Model size (B params)", fontsize=8, color=MUTED)
        _style(ax)

    axes[0].set_ylabel("Match fraction\n(with-CoT vs without-CoT)", fontsize=8.5, color=MUTED)
    axes[0].legend(fontsize=6.6, frameon=False, loc="upper left", labelcolor=MUTED)
    fig.tight_layout()
    legend_below(fig)
    path = output_dir / "real_vs_empirical_null.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dependence_scaling(df: pd.DataFrame, fits: list[dict], output_dir: Path) -> Path:
    """CoT dependence (real - shuffled) against model size, per family."""
    fit_by_label = {f["label"]: f for f in fits}
    families = [f for f in ["qwen", "llama", "olmo"] if f in set(df.family)]
    fig, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, family in zip(axes, families):
        sub = df[df.family == family]
        fit = fit_by_label[f"family={family}"]
        ends = []
        for task in sorted(sub.dataset_name.unique()):
            t = sub[sub.dataset_name == task].sort_values("size_b")
            ax.plot(t.size_b, t.dependence, "-o", linewidth=1.4, markersize=5, alpha=0.55, color=MUTED, zorder=3)
            ends.append([t.size_b.iloc[-1], t.dependence.iloc[-1], TASK_LABELS[task]])
        # nudge end labels apart so curves converging at the largest size stay legible
        min_gap = 0.035 * (sub.dependence.max() - sub.dependence.min())
        ends.sort(key=lambda e: e[1])
        for i in range(1, len(ends)):
            if ends[i][1] - ends[i - 1][1] < min_gap:
                ends[i][1] = ends[i - 1][1] + min_gap
        for x, y_label, name in ends:
            ax.annotate(
                name, (x, y_label), textcoords="offset points", xytext=(7, 0), fontsize=7, color=MUTED, va="center"
            )
        xs = np.linspace(np.log10(sub.size_b.min()), np.log10(sub.size_b.max()), 50)
        ax.plot(
            10**xs,
            fit["intercept"] + fit["slope"] * (xs + 9),
            color=FAMILY_COLORS[family],
            linewidth=2.6,
            zorder=5,
            label=f"slope={fit['slope']:+.3f} [{fit['slope_ci'][0]:+.3f}, {fit['slope_ci'][1]:+.3f}]",
        )
        ax.axhline(0, color=MUTED, linewidth=1.0, zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(right=sub.size_b.max() * 3.2)
        ax.set_xlabel("Model size (billions of parameters)", fontsize=8.5, color=MUTED)
        ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=MUTED)
        _style(ax)

    axes[0].set_ylabel("CoT dependence\n(real minus shuffled match rate)", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    path = output_dir / "cot_dependence_scaling.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    config = simple_parsing.parse(Config)
    results_dir = Path(config.results_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "null_table.csv")
    fits = read_json(results_dir / "dependence_fits.json")["fits"]

    for path in [plot_real_vs_null(df, output_dir), plot_dependence_scaling(df, fits, output_dir)]:
        print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()
