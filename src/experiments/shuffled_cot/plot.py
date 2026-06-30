"""Plot Experiment 2: Shuffled-CoT baseline overlaid on Experiment 1.

Shows real CoT match fraction vs shuffled CoT match fraction.

Usage:
    uv run -m src.experiments.shuffled_cot.plot \
        --core_sweep_results_dir results/core_sweep \
        --shuffled_cot_results_dir results/shuffled_cot \
        --output_dir results/figures
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import simple_parsing

from src.utils.io import read_json
from src.utils.models import get_model_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    core_sweep_results_dir: str = ""
    shuffled_cot_results_dir: str = ""
    output_dir: str = "results/figures"


def load_results(results_dir: str, filename: str) -> list[dict]:
    """Load all JSON results matching filename from a results directory."""
    results = []
    for path in Path(results_dir).rglob(filename):
        results.append(read_json(path))
    return results


def main():
    config = simple_parsing.parse(Config)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    real_results = load_results(config.core_sweep_results_dir, "faithfulness.json")
    shuffled_results = load_results(config.shuffled_cot_results_dir, "faithfulness_shuffled.json")

    if not real_results or not shuffled_results:
        LOGGER.warning("Missing results for plotting")
        return

    # Organize by family and dataset
    all_datasets = sorted({r["dataset_name"] for r in real_results})
    families = ["qwen", "llama"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    dataset_colors = sns.color_palette("husl", len(all_datasets))
    color_map = {ds: dataset_colors[i] for i, ds in enumerate(all_datasets)}

    for ax, family in zip(axes, families):
        for dataset in all_datasets:
            # Real results
            real_pts = [
                r
                for r in real_results
                if r["dataset_name"] == dataset and get_model_info(r["model_id"]).family == family
            ]
            real_pts.sort(key=lambda x: get_model_info(x["model_id"]).size_b)

            if not real_pts:
                continue

            sizes = [get_model_info(r["model_id"]).size_b for r in real_pts]
            means = [r["mean_match_fraction"] for r in real_pts]
            ax.plot(sizes, means, "o-", color=color_map[dataset], label=f"{dataset} (real)", markersize=6)

            # Shuffled results
            shuf_pts = [
                r
                for r in shuffled_results
                if r["dataset_name"] == dataset and get_model_info(r["model_id"]).family == family
            ]
            shuf_pts.sort(key=lambda x: get_model_info(x["model_id"]).size_b)

            if shuf_pts:
                shuf_sizes = [get_model_info(r["model_id"]).size_b for r in shuf_pts]
                shuf_means = [r["mean_match_fraction"] for r in shuf_pts]
                ax.plot(
                    shuf_sizes,
                    shuf_means,
                    "x--",
                    color=color_map[dataset],
                    label=f"{dataset} (shuffled)",
                    markersize=6,
                    alpha=0.6,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Model size (billions of parameters)")
        ax.set_title(f"{family.capitalize()} family")
        ax.legend(fontsize=7, ncol=2)

    axes[0].set_ylabel("Match fraction\n(with-CoT vs without-CoT)")
    fig.suptitle(
        "Experiment 2: Real vs shuffled CoT faithfulness\n(Gap between solid and dashed = signal from CoT content)",
        fontsize=12,
    )
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        fig_path = Path(config.output_dir) / f"real_vs_shuffled.{ext}"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {fig_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
