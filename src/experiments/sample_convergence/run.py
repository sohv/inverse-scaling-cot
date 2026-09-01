"""Revision Phase 2a: does the result depend on 20 vs 50 vs 100 CoT samples? No GPU.

Subsamples one pool of 100 generations nestedly, so k=20 is the first 20 of the same pool
that k=50 and k=100 are drawn from. This isolates sample count from sampling noise.

uv run -m src.experiments.sample_convergence.run --samples_100_dir results/samples_100 --output_dir results/sample_convergence --seed 42
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from scipy.stats import spearmanr

from src.metrics.cells import Variant, cells_to_table, load_cells
from src.metrics.robustness import fit_decomposition, per_family, per_task, summarize_fits
from src.utils.config import save_run_config
from src.utils.io import write_json
from src.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)

SAMPLE_COUNTS = [20, 50, 100]


@dataclass
class Config:
    samples_100_dir: str = ""
    output_dir: str = "results/sample_convergence"
    n_bootstrap: int = 1000
    seed: int = 42


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

    variants = [Variant(n_samples=k, parser="permissive", failure_mode="non_match") for k in SAMPLE_COUNTS]
    cells = load_cells(config.samples_100_dir, variants=variants, n_bootstrap=config.n_bootstrap, seed=config.seed)

    tables = {k: cells_to_table(cells[v.key]) for k, v in zip(SAMPLE_COUNTS, variants)}
    reference = tables[max(SAMPLE_COUNTS)]

    long_rows = []
    for k, df in tables.items():
        work = df.copy()
        work["k"] = k
        work["ci_width"] = work.proxy_ci_upper - work.proxy_ci_lower
        long_rows.append(work)
    long = pd.concat(long_rows, ignore_index=True)
    long.round(4).to_csv(output_dir / "cell_table_by_k.csv", index=False)

    per_k = []
    for k, df in tables.items():
        merged = df.merge(reference, on=["model_id", "dataset_name"], suffixes=("", "_ref"))
        rho, _ = spearmanr(merged.faithfulness_proxy, merged.faithfulness_proxy_ref)
        fit = fit_decomposition(df, f"k={k}", config.n_bootstrap, config.seed)
        task_fits = per_task(df, config.n_bootstrap, config.seed)
        fam_fits = per_family(df, config.n_bootstrap, config.seed)
        per_k.append(
            {
                "k": k,
                "n_cells": len(df),
                "mean_proxy": float(df.faithfulness_proxy.mean()),
                "mean_ci_width": float((df.proxy_ci_upper - df.proxy_ci_lower).mean()),
                "max_abs_diff_vs_k100": float((merged.faithfulness_proxy - merged.faithfulness_proxy_ref).abs().max()),
                "mean_abs_diff_vs_k100": float((merged.faithfulness_proxy - merged.faithfulness_proxy_ref).abs().mean()),
                "spearman_vs_k100": float(rho),
                "decomposition": fit.model_dump(),
                "per_task_summary": summarize_fits(task_fits),
                "per_family": [f.model_dump() for f in fam_fits],
            }
        )

    write_json(output_dir / "convergence.json", {"sample_counts": SAMPLE_COUNTS, "per_k": per_k})
    save_run_config(output_dir, config, extra_metadata={"n_cells": len(reference)})

    print(f"\nSample-count convergence on {len(reference)} cells "
          f"({reference.model_id.nunique()} models x {reference.dataset_name.nunique()} tasks)")
    print(f"{'k':>5} {'mean proxy':>11} {'mean CI width':>14} {'max |diff| vs k=100':>21} {'rank rho':>9} "
          f"{'raw beta':>9} {'ctl beta':>9} {'reduction':>10}")
    for r in per_k:
        d = r["decomposition"]
        print(f"{r['k']:>5} {r['mean_proxy']:>11.4f} {r['mean_ci_width']:>14.4f} "
              f"{r['max_abs_diff_vs_k100']:>21.4f} {r['spearman_vs_k100']:>9.4f} "
              f"{d['raw_coef']:>9.4f} {d['controlled_coef']:>9.4f} {d['pct_reduction']:>9.1f}%")

    reductions = [r["decomposition"]["pct_reduction"] for r in per_k]
    print(f"\nCoefficient reduction spans {min(reductions):.1f}-{max(reductions):.1f}% across sample counts "
          f"(range {max(reductions) - min(reductions):.1f} points)")
    print(f"\nResults saved to {output_dir}")
    print(f"Plot with: uv run -m src.experiments.sample_convergence.plot --results_dir {output_dir} --output_dir results/figures")


if __name__ == "__main__":
    main()
