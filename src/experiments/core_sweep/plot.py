"""Plot Experiment 1: Faithfulness vs model size.

Produces the main figure -- faithfulness proxy (% same answer) vs log(params),
one line per dataset, with 95% CI bands. Separate panels for Qwen and Llama.

Usage:
    uv run -m src.experiments.core_sweep.plot \
        --results_dir results/core_sweep \
        --output_dir results/figures
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simple_parsing

from src.utils.io import read_json
from src.utils.models import get_model_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    results_dir: str = ""
    output_dir: str = "results/figures"


def load_all_faithfulness(results_dir: str) -> list[dict]:
    """Load all faithfulness.json files from results directory."""
    results = []
    for faith_path in Path(results_dir).rglob("faithfulness.json"):
        data = read_json(faith_path)
        results.append(data)
    LOGGER.info(f"Loaded {len(results)} faithfulness results from {results_dir}")
    return results


def main():
    config = simple_parsing.parse(Config)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    results = load_all_faithfulness(config.results_dir)
    if not results:
        LOGGER.warning("No faithfulness results found")
        return

    # Organize by family and dataset
    families = {"qwen": {}, "llama": {}}
    for r in results:
        info = get_model_info(r["model_id"])
        family = info.family
        dataset = r["dataset_name"]
        if dataset not in families[family]:
            families[family][dataset] = []
        families[family][dataset].append(
            {
                "size_b": info.size_b,
                "mean": r["mean_match_fraction"],
                "ci_lower": r["bootstrap_ci_lower"],
                "ci_upper": r["bootstrap_ci_upper"],
            }
        )

    # Sort each dataset by model size
    for family in families:
        for dataset in families[family]:
            families[family][dataset].sort(key=lambda x: x["size_b"])

    # Plot: 2-panel figure (Qwen left, Llama right)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    dataset_colors = sns.color_palette("husl", 5)
    all_datasets = sorted({r["dataset_name"] for r in results})
    color_map = {ds: dataset_colors[i] for i, ds in enumerate(all_datasets)}

    for ax, (family_name, family_data) in zip(axes, families.items()):
        for dataset, points in family_data.items():
            sizes = [p["size_b"] for p in points]
            means = [p["mean"] for p in points]
            ci_lowers = [p["ci_lower"] for p in points]
            ci_uppers = [p["ci_upper"] for p in points]

            ax.plot(sizes, means, "o-", color=color_map[dataset], label=dataset, markersize=6)
            ax.fill_between(sizes, ci_lowers, ci_uppers, alpha=0.15, color=color_map[dataset])

        ax.set_xscale("log")
        ax.set_xlabel("Model size (billions of parameters)")
        ax.set_title(f"{family_name.capitalize()} family")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Faithfulness proxy\n(% same answer with vs without CoT)")
    fig.suptitle("Experiment 1: CoT faithfulness vs model scale", fontsize=14)
    plt.tight_layout()

    # Save
    for ext in ["pdf", "png"]:
        fig_path = Path(config.output_dir) / f"faithfulness_vs_size.{ext}"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {fig_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
