# regenerates every existing figure under the thesis chapter-4 visual style, for a side-by-side style comparison.
# uv run -m experiments.figure_style.260901_thesis_style_v1.1_regenerate_figures --output_dir results/fig_new --seed 42

import importlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import simple_parsing
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.utils.plotting import reserve_legend_space

LOGGER = logging.getLogger(__name__)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
BLUE, ORANGE, AQUA, RED, PURPLE = PALETTE
INK = "#222222"
MUTED = "#555555"
GRID = "#cccccc"

RCPARAMS = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
}

# each entry: module path, the argv it needs, and the module constants to repaint.
PLOTS = [
    ("src.experiments.core_sweep.plot", ["--results_dir", "results/core_sweep"]),
    (
        "src.experiments.shuffled_cot.plot",
        ["--core_sweep_results_dir", "results/core_sweep", "--shuffled_cot_results_dir", "results/shuffled_cot"],
    ),
    ("src.experiments.regression.plot", ["--regression_table", "results/regression/regression_table.csv"]),
    ("src.experiments.robustness.plot", ["--results_dir", "results/final"]),
    ("src.experiments.capability.plot", ["--results_dir", "results/final_capability"]),
    ("src.experiments.cot_dependence.plot", ["--results_dir", "results/final_dependence"]),
    ("src.experiments.sample_convergence.plot", ["--results_dir", "results/sample_convergence"]),
    (
        "src.experiments.comparison.plot",
        ["--cells_csv", "results/quantization/awq_vs_bf16_cells.csv", "--label", "awq_vs_bf16",
         "--baseline_name", "AWQ 4-bit", "--variant_name", "BF16"],
    ),
    (
        "src.experiments.comparison.plot",
        ["--cells_csv", "results/aqua_position_analysis/aqua_position_cells.csv", "--label", "aqua_position",
         "--baseline_name", "Original order", "--variant_name", "Positions randomised"],
    ),
    (
        "src.experiments.comparison.plot",
        ["--cells_csv", "results/prompt_analysis/prompt_v1_cells.csv", "--label", "prompt_v1",
         "--baseline_name", "Prompt v0", "--variant_name", "Prompt v1"],
    ),
    (
        "src.experiments.comparison.plot",
        ["--cells_csv", "results/prompt_analysis/prompt_v2_cells.csv", "--label", "prompt_v2",
         "--baseline_name", "Prompt v0", "--variant_name", "Prompt v2"],
    ),
]

# repaint by value, not by name, so dicts keyed on families or sample counts survive intact
OLD_TO_NEW = {
    "#2a78d6": BLUE,
    "#eb6834": ORANGE,
    "#1baf7a": AQUA,
    "#0b0b0b": INK,
    "#52514e": MUTED,
    "#d8d7d2": GRID,
    "#c9c8c2": GRID,
}



def style_axes(ax) -> None:
    """The chapter-4 axis look: open frame, dashed y-grid behind the marks."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(False)
    ax.yaxis.grid(True, ls="--", alpha=0.3, color=GRID)
    ax.set_axisbelow(True)
    for text in [ax.xaxis.label, ax.yaxis.label]:
        text.set_color(INK)


def share_ylabel(fig) -> None:
    """One y-axis label for the whole figure when several panels repeat the same one."""
    labels = [ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()]
    if len(labels) < 2 or len(set(labels)) != 1: return
    for ax in fig.axes:
        ax.set_ylabel("")
    fig.supylabel(labels[0], color=INK)


def patch_matplotlib() -> None:
    """Strip per-call font sizes and dpi so the global rcParams actually win."""
    for name in ["set_xlabel", "set_ylabel", "set_xticklabels", "set_yticklabels", "legend"]:
        original = getattr(Axes, name)

        def wrapper(self, *args, _original=original, **kwargs):
            kwargs.pop("fontsize", None)
            return _original(self, *args, **kwargs)

        setattr(Axes, name, wrapper)

    original_tick_params = Axes.tick_params

    def tick_params(self, *args, **kwargs):
        kwargs.pop("labelsize", None)
        return original_tick_params(self, *args, **kwargs)

    Axes.tick_params = tick_params

    original_savefig = Figure.savefig

    def savefig(self, *args, **kwargs):
        kwargs.pop("dpi", None)
        for ax in self.axes:
            style_axes(ax)
        share_ylabel(self)
        # the larger serif type needs more room, or long axis labels run off the canvas
        self.tight_layout()
        # a label centred on an inset axes can still run off the canvas, so widen until it fits
        for _ in range(4):
            renderer = self.canvas.get_renderer()
            width_px = self.get_figwidth() * self.dpi
            overflow = 0.0
            for ax in self.axes:
                bb = ax.xaxis.label.get_window_extent(renderer)
                overflow = max(overflow, bb.x1 - width_px, -bb.x0)
            if overflow <= 1: break
            grow = 2 * overflow / self.dpi + 0.4
            self.set_size_inches(self.get_figwidth() + grow, self.get_figheight())
            self.tight_layout()
        # this wrapper's tight_layout reclaims the strip a source-side legend reserved
        for legend in self.legends:
            reserve_legend_space(self, legend)
        return original_savefig(self, *args, **kwargs)

    Figure.savefig = savefig

    # these modules call sns.set_theme inside main(), which would reset everything
    sns.set_theme = lambda *a, **k: plt.rcParams.update(RCPARAMS)
    sns.color_palette = lambda *a, **k: [PALETTE[i % len(PALETTE)] for i in range(a[1] if len(a) > 1 else 5)]


@dataclass
class Config:
    output_dir: str = "results/fig_new"
    seed: int = 42


def main():
    config = simple_parsing.parse(Config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "run.log")],
    )

    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8]
    metadata = {"git_hash": git_hash, "seed": config.seed, "rcparams": RCPARAMS, "palette": PALETTE}
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2))

    plt.rcParams.update(RCPARAMS)
    patch_matplotlib()

    for module_path, args in PLOTS:
        module = importlib.import_module(module_path)
        for name, value in vars(module).copy().items():
            if isinstance(value, str) and value in OLD_TO_NEW:
                setattr(module, name, OLD_TO_NEW[value])
            elif isinstance(value, dict) and any(v in OLD_TO_NEW for v in value.values() if isinstance(v, str)):
                setattr(module, name, {k: OLD_TO_NEW.get(v, v) for k, v in value.items()})

        sys.argv = [module_path, *args, "--output_dir", str(output_dir)]
        LOGGER.info(f"regenerating {module_path}")
        module.main()

    print(f"Restyled figures written to {output_dir}")


if __name__ == "__main__":
    main()
