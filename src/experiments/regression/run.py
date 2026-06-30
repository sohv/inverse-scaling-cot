"""Experiment 4: Confound decomposition regression.

Merges Experiment 1 faithfulness + Experiment 3 accuracy into a 55-cell table.
Runs 3 OLS regressions with cluster-robust standard errors.
Applies pre-registered decision procedure.

Usage:
    uv run -m src.experiments.regression.run \
        --core_sweep_results_dir results/core_sweep \
        --accuracy_results_dir results/accuracy \
        --output_dir results/regression \
        --seed 42
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.metrics.accuracy import AccuracyResult
from src.metrics.faithfulness import FaithfulnessResult
from src.metrics.regression import build_regression_table, run_decomposition
from src.utils.config import save_run_config
from src.utils.io import read_json, write_json
from src.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    core_sweep_results_dir: str = ""
    accuracy_results_dir: str = ""
    output_dir: str = "results/regression"
    drop_threshold: float = 0.50
    retain_threshold: float = 0.30
    seed: int = 42


def main():
    config = simple_parsing.parse(Config)
    seed_everything(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load faithfulness results from core sweep
    faith_results = []
    for faith_path in Path(config.core_sweep_results_dir).rglob("faithfulness.json"):
        data = read_json(faith_path)
        faith_results.append(FaithfulnessResult(**data))
    LOGGER.info(f"Loaded {len(faith_results)} faithfulness results")

    # Load accuracy results
    acc_results = []
    for acc_path in Path(config.accuracy_results_dir).rglob("accuracy.json"):
        data = read_json(acc_path)
        acc_results.append(AccuracyResult(**data))
    LOGGER.info(f"Loaded {len(acc_results)} accuracy results")

    # Build regression table
    df = build_regression_table(faith_results, acc_results)
    df.to_csv(output_dir / "regression_table.csv", index=False)
    print(f"Saved regression table ({len(df)} rows) to {output_dir / 'regression_table.csv'}")

    # Run decomposition
    decomposition = run_decomposition(df, config.drop_threshold, config.retain_threshold)
    write_json(output_dir / "decomposition.json", decomposition.model_dump())
    save_run_config(output_dir, config)

    # Print results
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: CONFOUND DECOMPOSITION RESULTS")
    print("=" * 72)

    for reg in decomposition.regressions:
        print(f"\n--- {reg.model_name} ---")
        print(f"  Formula: {reg.formula}")
        print(f"  R² = {reg.r_squared:.4f}, adj R² = {reg.adj_r_squared:.4f}")
        print(f"  N = {reg.n_obs}, clusters = {reg.n_clusters}")
        for var in reg.coefficients:
            coef = reg.coefficients[var]
            se = reg.std_errors[var]
            p = reg.p_values[var]
            ci = reg.conf_intervals[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {var:<25} {coef:>10.4f} (SE={se:.4f}, p={p:.4f}) {sig}  [{ci[0]:.4f}, {ci[1]:.4f}]")

    print("\n--- DECISION ---")
    print(f"  Baseline log_params coef:    {decomposition.baseline_coef_log_params:.4f}")
    print(f"  Controlled log_params coef:  {decomposition.controlled_coef_log_params:.4f}")
    print(f"  % drop in magnitude:         {decomposition.pct_drop:.1f}%")
    print(f"  Pooled verdict:              {decomposition.decision}")

    print("\n  Per-task verdicts:")
    for task, verdict in decomposition.per_task_decisions.items():
        print(f"    {task:<20} {verdict}")

    print(
        f"\nPlot with: uv run -m src.experiments.regression.plot "
        f"--regression_table {output_dir / 'regression_table.csv'} --output_dir results/figures"
    )


if __name__ == "__main__":
    main()
