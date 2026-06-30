"""Plot Experiment 4: Regression scatter plots.

Scatter of faithfulness vs accuracy, colored by model family, with regression lines.

Usage:
    uv run -m src.experiments.regression.plot \
        --regression_table results/regression/regression_table.csv \
        --output_dir results/figures
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simple_parsing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    regression_table: str = ""
    output_dir: str = "results/figures"


def main():
    config = simple_parsing.parse(Config)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.regression_table)

    sns.set_theme(style="whitegrid")

    # Figure 1: Faithfulness vs log(params), colored by dataset
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for dataset in df["dataset_name"].unique():
        subset = df[df["dataset_name"] == dataset]
        ax1.scatter(subset["log_params"], subset["faithfulness_proxy"], label=dataset, s=60, alpha=0.7)
    # Add regression line
    z = np.polyfit(df["log_params"], df["faithfulness_proxy"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df["log_params"].min(), df["log_params"].max(), 100)
    ax1.plot(x_range, p(x_range), "k--", alpha=0.5, label=f"OLS fit (slope={z[0]:.4f})")
    ax1.set_xlabel("log10(parameters)")
    ax1.set_ylabel("Faithfulness proxy (% same answer)")
    ax1.set_title("Faithfulness vs model scale (unadjusted)")
    ax1.legend(fontsize=8)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = Path(config.output_dir) / f"faith_vs_params.{ext}"
        fig1.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig1)

    # Figure 2: Faithfulness vs accuracy, colored by family
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    markers = {"qwen": "o", "llama": "s"}
    families = df["family_is_llama"].map({0: "qwen", 1: "llama"})
    for family in ["qwen", "llama"]:
        mask = families == family
        subset = df[mask]
        ax2.scatter(
            subset["accuracy_no_cot"],
            subset["faithfulness_proxy"],
            label=f"{family.capitalize()}",
            marker=markers[family],
            s=60,
            alpha=0.7,
        )
    ax2.set_xlabel("Accuracy without CoT")
    ax2.set_ylabel("Faithfulness proxy (% same answer)")
    ax2.set_title("Faithfulness vs accuracy (the confound)")
    ax2.legend()
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = Path(config.output_dir) / f"faith_vs_accuracy.{ext}"
        fig2.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig2)

    # Figure 3: Residual plot (faithfulness after controlling for accuracy) vs log(params)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    # Residualize faithfulness against accuracy
    from numpy.polynomial.polynomial import polyfit

    acc_fit = np.polyfit(df["accuracy_no_cot"], df["faithfulness_proxy"], 1)
    residuals = df["faithfulness_proxy"] - np.polyval(acc_fit, df["accuracy_no_cot"])
    for dataset in df["dataset_name"].unique():
        mask = df["dataset_name"] == dataset
        ax3.scatter(df.loc[mask, "log_params"], residuals[mask], label=dataset, s=60, alpha=0.7)
    z_res = np.polyfit(df["log_params"], residuals, 1)
    ax3.plot(x_range, np.polyval(z_res, x_range), "k--", alpha=0.5, label=f"OLS fit (slope={z_res[0]:.4f})")
    ax3.set_xlabel("log10(parameters)")
    ax3.set_ylabel("Faithfulness residual\n(after controlling for accuracy)")
    ax3.set_title("Does inverse scaling survive after controlling for accuracy?")
    ax3.legend(fontsize=8)
    ax3.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = Path(config.output_dir) / f"residual_vs_params.{ext}"
        fig3.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig3)


if __name__ == "__main__":
    main()
