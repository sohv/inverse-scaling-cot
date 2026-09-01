"""Revision Phase 1a: full regression battery on existing generations. No GPU needed.

uv run -m src.experiments.robustness.run --core_sweep_results_dir results/core_sweep --output_dir results/robustness --seed 42
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.metrics.cells import cells_to_table, default_variant, load_cells
from src.metrics.robustness import (
    fit_decomposition,
    fit_quadratic,
    leave_one_model_out,
    leave_one_task_out,
    mixed_effects,
    partial_correlation,
    per_family,
    per_task,
    residual_frame,
    residualize,
    summarize_fits,
)
from src.utils.config import save_run_config
from src.utils.io import write_json
from src.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    core_sweep_results_dir: str = ""
    output_dir: str = "results/robustness"
    n_bootstrap: int = 1000
    seed: int = 42
    exclude_quantized: bool = False


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

    cells = load_cells(config.core_sweep_results_dir, n_bootstrap=config.n_bootstrap, seed=config.seed)
    df = cells_to_table(cells[default_variant().key], exclude_quantized=config.exclude_quantized)
    df.round(4).to_csv(output_dir / "cell_table.csv", index=False)

    nb, seed = config.n_bootstrap, config.seed

    pooled = fit_decomposition(df, "pooled", nb, seed)
    write_json(output_dir / "pooled.json", pooled.model_dump())

    task_fits = per_task(df, nb, seed)
    write_json(
        output_dir / "per_task.json",
        {"fits": [f.model_dump() for f in task_fits], "summary": summarize_fits(task_fits)},
    )

    loto = leave_one_task_out(df, nb, seed)
    write_json(
        output_dir / "leave_one_task_out.json",
        {"fits": [f.model_dump() for f in loto], "summary": summarize_fits(loto)},
    )

    lomo = leave_one_model_out(df, nb, seed)
    write_json(
        output_dir / "leave_one_model_out.json",
        {"fits": [f.model_dump() for f in lomo], "summary": summarize_fits(lomo)},
    )

    fam_fits = per_family(df, nb, seed)
    write_json(
        output_dir / "per_family.json",
        {"fits": [f.model_dump() for f in fam_fits], "summary": summarize_fits(fam_fits)},
    )

    resid = [residualize(df, "pooled", nb, seed)]
    resid += [residualize(df[df.dataset_name == t], f"task={t}", nb, seed) for t in sorted(df.dataset_name.unique())]
    resid += [residualize(df[df.family == f], f"family={f}", nb, seed) for f in sorted(df.family.unique())]
    write_json(output_dir / "residualized.json", {"fits": [r.model_dump() for r in resid]})
    residual_frame(df).round(4).to_csv(output_dir / "residual_table.csv", index=False)

    partials = [partial_correlation(df, "pooled", nb, seed)]
    partials += [
        partial_correlation(df[df.dataset_name == t], f"task={t}", nb, seed) for t in sorted(df.dataset_name.unique())
    ]
    partials += [partial_correlation(df[df.family == f], f"family={f}", nb, seed) for f in sorted(df.family.unique())]
    write_json(output_dir / "partial_correlation.json", {"fits": [p.model_dump() for p in partials]})

    write_json(output_dir / "mixed_effects.json", {"fits": [m.model_dump() for m in mixed_effects(df)]})
    write_json(output_dir / "nonlinear.json", fit_quadratic(df, nb, seed).model_dump())

    save_run_config(output_dir, config, extra_metadata={"n_cells": len(df)})

    print(
        f"\nPooled: raw={pooled.raw_coef:.4f} controlled={pooled.controlled_coef:.4f} "
        f"reduction={pooled.pct_reduction:.1f}% CI={[round(c, 1) for c in pooled.pct_reduction_ci]}"
    )
    print("\nPer-task coefficient reduction")
    for f in task_fits:
        print(
            f"  {f.label:<28} raw={f.raw_coef:.4f} controlled={f.controlled_coef:.4f} "
            f"reduction={f.pct_reduction:.1f}% CI={[round(c, 1) for c in f.pct_reduction_ci]}"
        )
    print("\nPer-family coefficient reduction")
    for f in fam_fits:
        print(
            f"  {f.label:<28} raw={f.raw_coef:.4f} controlled={f.controlled_coef:.4f} reduction={f.pct_reduction:.1f}%"
        )
    for name, fits in [("Leave-one-task-out", loto), ("Leave-one-model-out", lomo)]:
        s = summarize_fits(fits)
        print(
            f"\n{name}: min={s['min_pct_reduction']:.1f}% ({s['min_label']}) "
            f"median={s['median_pct_reduction']:.1f}% max={s['max_pct_reduction']:.1f}% "
            f"all_above_50={s['all_above_50']}"
        )
    print(f"\nResults saved to {output_dir}")
    print(
        f"Plot with: uv run -m src.experiments.robustness.plot --results_dir {output_dir} --output_dir results/figures"
    )


if __name__ == "__main__":
    main()
