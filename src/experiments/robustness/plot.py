"""Figures for the Phase 1a regression battery.

uv run -m src.experiments.robustness.plot --results_dir results/robustness --output_dir results/figures
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

TASK_LABELS = {
    "aqua": "AQuA",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "logiqa": "LogiQA",
    "openbookqa": "OpenBookQA",
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


def plot_residualized(results_dir: Path, output_dir: Path) -> Path:
    """Small multiples: accuracy-residualised proxy vs model size, pooled and per task.

    Faceted rather than colour-coded because five categorical series cannot clear the
    all-pairs colour-separation floor on a scatter.
    """
    df = pd.read_csv(results_dir / "residual_table.csv")
    fits = {f["label"]: f for f in read_json(results_dir / "residualized.json")["fits"]}

    panels = [("pooled", df)] + [(f"task={t}", df[df.dataset_name == t]) for t in sorted(df.dataset_name.unique())]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey=True)

    for ax, (label, sub) in zip(axes.flat, panels):
        fit = fits[label]
        ax.axhline(0, color=MUTED, linewidth=1.0, zorder=1)
        ax.scatter(
            sub.size_b, sub.proxy_residual, s=34, color=BLUE, alpha=0.85, zorder=3, edgecolor="white", linewidth=0.8
        )
        xs = np.linspace(np.log10(sub.size_b.min()), np.log10(sub.size_b.max()), 50)
        ax.plot(10**xs, fit["intercept"] + fit["slope"] * (xs + 9), color=ORANGE, linewidth=2, zorder=4)
        ax.set_xscale("log")
        title = "All 55 cells" if label == "pooled" else TASK_LABELS[label.split("=")[1]]
        ci = fit.get("slope_ci") or [float("nan"), float("nan")]
        ax.set_title(f"{title}\nslope={fit['slope']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", fontsize=9, color=INK)
        _style(ax)

    for ax in axes[1, :]:
        ax.set_xlabel("Model size (billions of parameters)", fontsize=8, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("Faithfulness residual\n(after controlling for no-CoT accuracy)", fontsize=8, color=MUTED)

    fig.suptitle(
        "Residualised faithfulness proxy vs model size\nQwen2.5-Instruct and Llama-3-Instruct, 55 model-task cells",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    path = output_dir / "residualized_scaling.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _forest(ax, labels, values, lowers, uppers, threshold=50.0):
    y = np.arange(len(labels))
    ax.axvline(threshold, color=ORANGE, linewidth=1.6, linestyle="--", zorder=2)
    ax.hlines(y, lowers, uppers, color=BLUE, linewidth=2.4, alpha=0.55, zorder=3)
    ax.plot(values, y, "o", markersize=8, color=BLUE, zorder=4, markeredgecolor="white", markeredgewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.invert_yaxis()
    for yi, v, u in zip(y, values, uppers):
        ax.annotate(
            f"{v:.1f}%",
            (u, yi),
            textcoords="offset points",
            xytext=(6, 0),
            ha="left",
            va="center",
            fontsize=7.5,
            color=MUTED,
        )
    ax.set_xlim(min(lowers) - 4, max(uppers) + 12)
    _style(ax)


def plot_per_task(results_dir: Path, output_dir: Path) -> Path:
    """Forest plot of the per-task coefficient reduction with bootstrap CIs."""
    fits = read_json(results_dir / "per_task.json")["fits"]
    labels = [TASK_LABELS[f["label"].split("=")[1]] for f in fits]
    values = [f["pct_reduction"] for f in fits]
    lowers = [f["pct_reduction_ci"][0] for f in fits]
    uppers = [f["pct_reduction_ci"][1] for f in fits]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _forest(ax, labels, values, lowers, uppers)
    ax.set_xlabel(
        "Reduction in log-parameter coefficient after controlling for no-CoT accuracy (%)", fontsize=8.5, color=MUTED
    )
    ax.set_title(
        "Capability control reduces the size coefficient in every task\n"
        "Dashed line: pre-registered 50% threshold; bars are 95% bootstrap CIs",
        fontsize=9.5,
        color=INK,
    )
    fig.tight_layout()
    path = output_dir / "per_task_reduction.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_leave_one_out(results_dir: Path, output_dir: Path) -> Path:
    """Leave-one-task-out and leave-one-model-out coefficient reductions."""
    loto = read_json(results_dir / "leave_one_task_out.json")["fits"]
    lomo = read_json(results_dir / "leave_one_model_out.json")["fits"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    labels = [TASK_LABELS[f["label"].split("=")[1]] for f in loto]
    _forest(
        axes[0],
        labels,
        [f["pct_reduction"] for f in loto],
        [f["pct_reduction_ci"][0] for f in loto],
        [f["pct_reduction_ci"][1] for f in loto],
    )
    axes[0].set_title("Leave-one-task-out", fontsize=10, color=INK)

    order = sorted(lomo, key=lambda f: f["pct_reduction"])
    labels = [f["label"].split("=", 1)[1].split("/")[-1] for f in order]
    _forest(
        axes[1],
        labels,
        [f["pct_reduction"] for f in order],
        [f["pct_reduction_ci"][0] for f in order],
        [f["pct_reduction_ci"][1] for f in order],
    )
    axes[1].set_title("Leave-one-model-out", fontsize=10, color=INK)

    for ax in axes:
        ax.set_xlabel("Coefficient reduction (%)", fontsize=8.5, color=MUTED)
    fig.suptitle("The capability confound survives dropping any single task or checkpoint", fontsize=11, color=INK)
    fig.tight_layout()
    path = output_dir / "loo_reduction.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    config = simple_parsing.parse(Config)
    results_dir = Path(config.results_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [
        plot_residualized(results_dir, output_dir),
        plot_per_task(results_dir, output_dir),
        plot_leave_one_out(results_dir, output_dir),
    ]:
        print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()
