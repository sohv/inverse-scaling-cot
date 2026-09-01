"""Revision Phase 1b: capability-matched pairs, regime bins and capability reconstruction. No GPU.

uv run -m src.experiments.capability.run --core_sweep_results_dir results/core_sweep --cell_table results/robustness/cell_table.csv --output_dir results/capability --seed 42
"""

import collections
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import simple_parsing

from src.generation.runner import QuestionResult
from src.metrics.capability import (
    ACC_TOLERANCE,
    MIN_SIZE_RATIO,
    answer_distribution_stats,
    assign_bins,
    capability_bins,
    matched_pairs,
    reconstruct,
    saturation,
    tolerance_sensitivity,
)
from src.metrics.cells import find_cell_dirs
from src.metrics.robustness import fit_decomposition
from src.utils.config import save_run_config
from src.utils.io import read_jsonl, write_json
from src.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    core_sweep_results_dir: str = ""
    cell_table: str = ""
    output_dir: str = "results/capability"
    seed: int = 42


def no_cot_answer_stats(core_sweep_dir: str) -> list[dict]:
    """Per-cell no-CoT answer-letter distribution, for the near-random regime diagnostic."""
    rows = []
    for cell_dir in find_cell_dirs(core_sweep_dir):
        results = [QuestionResult(**r) for r in read_jsonl(cell_dir / "generation_results.jsonl")]
        counts = collections.Counter(
            r.no_cot_extracted_answer for r in results if r.no_cot_extracted_answer is not None
        )
        rows.append(
            {
                "model_id": results[0].model_id,
                "dataset_name": results[0].dataset_name,
                **answer_distribution_stats(dict(counts)),
            }
        )
    return rows


def main():
    config = simple_parsing.parse(Config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "run.log")],
    )
    seed_everything(config.seed)

    df = pd.read_csv(config.cell_table)
    raw_size_coef = fit_decomposition(df, "pooled", n_bootstrap=0).raw_coef

    within = matched_pairs(df, raw_size_coef, label="within_family")
    across = matched_pairs(df, raw_size_coef, within_family=False, label="across_families")
    write_json(
        output_dir / "matched_pairs.json",
        {
            "within_family": within.model_dump(),
            "across_families": across.model_dump(),
            "raw_size_coefficient": raw_size_coef,
        },
    )

    sensitivity = tolerance_sensitivity(df, raw_size_coef)
    write_json(output_dir / "matched_pairs_sensitivity.json", {"sweep": sensitivity})

    bins = capability_bins(df)
    sat = saturation(df)
    write_json(output_dir / "capability_bins.json", {"bins": bins, "saturation": sat})

    recon = reconstruct(df)
    write_json(output_dir / "reconstruction.json", recon)

    answer_stats = no_cot_answer_stats(config.core_sweep_results_dir)
    binned = assign_bins(df)[["model_id", "dataset_name", "capability_bin", "accuracy_no_cot", "faithfulness_proxy"]]
    stats_df = pd.DataFrame(answer_stats).drop(columns=["counts"]).merge(binned, on=["model_id", "dataset_name"])
    stats_df.round(4).to_csv(output_dir / "answer_distribution.csv", index=False)
    write_json(
        output_dir / "answer_distribution_by_bin.json",
        {
            "by_bin": stats_df.groupby("capability_bin", observed=False)[["entropy", "normalized_entropy", "max_share"]]
            .mean()
            .round(4)
            .reset_index()
            .to_dict("records")
        },
    )

    assign_bins(df).round(4).to_csv(output_dir / "binned_cells.csv", index=False)
    save_run_config(output_dir, config, extra_metadata={"n_cells": len(df)})

    print(f"\nCapability-matched pairs (|delta accuracy| <= {ACC_TOLERANCE}, size ratio >= {MIN_SIZE_RATIO}x)")
    for m in [within, across]:
        if m.n_pairs == 0:
            print(f"  {m.label:<18} no qualifying pairs")
            continue
        print(
            f"  {m.label:<18} n={m.n_pairs:<3} mean |delta proxy|={m.mean_abs_proxy_diff:.4f} "
            f"vs {m.predicted_proxy_diff_from_size:.4f} predicted from the size coefficient alone "
            f"(mean size ratio {m.mean_size_ratio:.1f}x, paired p={m.paired_p_value:.3f})"
        )

    print("\nMatched-pair threshold sensitivity (post-hoc; pre-registered point marked *)")
    print(f"  {'scope':<8} {'tol':>6} {'ratio':>6} {'n':>4} {'mean |d proxy|':>15} {'predicted':>10} {'p':>8}")
    for r in sensitivity:
        if r["n_pairs"] == 0:
            continue
        star = "*" if r["is_preregistered_point"] else " "
        scope = "within" if r["within_family"] else "across"
        print(f"{star} {scope:<8} {r['acc_tolerance']:>6.3f} {r['min_size_ratio']:>6.1f} {r['n_pairs']:>4} "
              f"{r['mean_abs_proxy_diff']:>15.4f} {r['predicted_proxy_diff_from_size']:>10.4f} {r['paired_p_value']:>8.3f}")

    print("\nCapability regimes")
    for b in bins:
        if b.get("insufficient_data"):
            print(f"  {b['bin']:<14} n={b['n_obs']} insufficient data")
            continue
        print(
            f"  {b['bin']:<14} n={b['n_obs']:<3} accuracy {b['accuracy_range'][0]:.2f}-{b['accuracy_range'][1]:.2f} "
            f"mean proxy={b['mean_proxy']:.3f} slope vs size={b['slope_vs_log_params']:+.4f} "
            f"slope vs accuracy={b['slope_vs_accuracy']:+.4f}"
        )
    if "slope_ratio_above_over_below" in sat:
        print(
            f"  ceiling saturation: proxy-accuracy slope {sat['below']['slope']:.3f} below "
            f"{sat['threshold']} vs {sat['above']['slope']:.3f} above"
        )

    print("\nAnswer-position concentration by regime (no-CoT answers)")
    for row in (
        stats_df.groupby("capability_bin", observed=False)[["normalized_entropy", "max_share"]]
        .mean()
        .reset_index()
        .to_dict("records")
    ):
        print(
            f"  {row['capability_bin']:<14} normalized entropy={row['normalized_entropy']:.3f} "
            f"modal-answer share={row['max_share']:.3f}"
        )

    print(
        f"\nReconstruction from no-CoT accuracy alone: predicted size slope {recon['predicted_size_slope']:.4f} "
        f"vs observed {recon['observed_size_slope']:.4f} "
        f"({recon['fraction_of_slope_reproduced'] * 100:.1f}% of the observed scaling slope), MAE={recon['mean_absolute_error']:.4f}"
    )
    print(f"\nResults saved to {output_dir}")
    print(
        f"Plot with: uv run -m src.experiments.capability.plot --results_dir {output_dir} --output_dir results/figures"
    )


if __name__ == "__main__":
    main()
