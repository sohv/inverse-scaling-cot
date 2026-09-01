"""Figures for the capability confound: reconstruction and regime structure.

uv run -m src.experiments.capability.plot --results_dir results/capability --output_dir results/figures
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

from src.metrics.capability import BIN_EDGES
from src.utils.io import read_json

LOGGER = logging.getLogger(__name__)

# first three categorical slots: the documented subset that validates on all pairs
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d8d7d2"

BIN_COLORS = {"near_random": BLUE, "intermediate": ORANGE, "ceiling": AQUA}
BIN_LABELS = {
    "near_random": "Near-random (acc < 0.35)",
    "intermediate": "Intermediate (0.35-0.75)",
    "ceiling": "Ceiling (acc >= 0.75)",
}


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


def plot_reconstruction(df: pd.DataFrame, recon: dict, output_dir: Path) -> Path:
    """Observed scaling curve vs the curve predicted from no-CoT accuracy alone."""
    work = df.copy()
    work["proxy_predicted"] = recon["accuracy_intercept"] + recon["accuracy_slope"] * work["accuracy_no_cot"]

    families = [f for f in ["qwen", "llama", "olmo"] if f in set(work.family)]
    fig, axes = plt.subplots(1, len(families), figsize=(5.5 * len(families), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, family in zip(axes, families):
        sub = work[work.family == family]
        observed = sub.groupby("size_b").faithfulness_proxy.mean()
        predicted = sub.groupby("size_b").proxy_predicted.mean()
        ax.plot(
            observed.index,
            observed.values,
            "-o",
            color=BLUE,
            linewidth=2.4,
            markersize=7,
            label="Observed proxy",
            zorder=4,
            markeredgecolor="white",
            markeredgewidth=0.9,
        )
        ax.plot(
            predicted.index,
            predicted.values,
            "--s",
            color=ORANGE,
            linewidth=2.4,
            markersize=6,
            label="Predicted from no-CoT accuracy alone",
            zorder=3,
            markeredgecolor="white",
            markeredgewidth=0.9,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Model size (billions of parameters)", fontsize=8.5, color=MUTED)
        ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=MUTED)
        _style(ax)

    axes[0].set_ylabel("Faithfulness proxy\n(mean over five tasks)", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    path = output_dir / "reconstruction.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_regimes(df: pd.DataFrame, bins: list[dict], output_dir: Path) -> Path:
    """Proxy against no-CoT accuracy, coloured by pre-registered capability regime."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for edge in BIN_EDGES[1:-1]:
        ax.axvline(edge, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    for name, color in BIN_COLORS.items():
        sub = df[df.capability_bin == name]
        ax.scatter(
            sub.accuracy_no_cot,
            sub.faithfulness_proxy,
            s=42,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
            label=BIN_LABELS[name],
        )
    lims = [0.15, 1.0]
    ax.plot(lims, lims, color=MUTED, linewidth=1.2, linestyle="--", zorder=2, label="Proxy = accuracy")
    ax.set_xlabel("Accuracy without CoT", fontsize=8.5, color=MUTED)
    ax.set_ylabel("Faithfulness proxy", fontsize=8.5, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=MUTED)
    _style(ax)

    ax = axes[1]
    valid = [b for b in bins if not b.get("insufficient_data")]
    x = np.arange(len(valid))
    width = 0.36
    ax.bar(
        x - width / 2,
        [b["slope_vs_log_params"] for b in valid],
        width,
        color=BLUE,
        label="Slope vs log model size",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        [b["slope_vs_accuracy"] for b in valid],
        width,
        color=ORANGE,
        label="Slope vs no-CoT accuracy",
        zorder=3,
    )
    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{BIN_LABELS[b['bin']].split(' (')[0]}\n(n={b['n_obs']})" for b in valid], fontsize=8, color=INK
    )
    ax.set_ylabel("Regression slope", fontsize=8.5, color=MUTED)
    ax.legend(fontsize=8, frameon=False, labelcolor=MUTED)
    _style(ax)

    fig.tight_layout()
    path = output_dir / "capability_regimes.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    config = simple_parsing.parse(Config)
    results_dir = Path(config.results_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "binned_cells.csv")
    recon = read_json(results_dir / "reconstruction.json")
    bins = read_json(results_dir / "capability_bins.json")["bins"]

    for path in [plot_reconstruction(df, recon, output_dir), plot_regimes(df, bins, output_dir)]:
        print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()
