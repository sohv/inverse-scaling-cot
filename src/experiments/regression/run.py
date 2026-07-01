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
            boot_ci = reg.bootstrap_ci.get(var)
            boot_str = f"  bootstrap 95% CI [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]" if boot_ci else ""
            print(f"  {var:<25} {coef:>10.4f} (SE={se:.4f}, p={p:.4f}) {sig}  [{ci[0]:.4f}, {ci[1]:.4f}]{boot_str}")

    print("\n--- DECISION ---")
    print(f"  Baseline log_params coef:    {decomposition.baseline_coef_log_params:.4f}")
    print(f"  Controlled log_params coef:  {decomposition.controlled_coef_log_params:.4f}")
    print(f"  % drop in magnitude:         {decomposition.pct_drop:.1f}%")
    print(f"  Pooled verdict:              {decomposition.decision}")

    print("\n  Per-task verdicts:")
    for task, verdict in decomposition.per_task_decisions.items():
        print(f"    {task:<20} {verdict}")

    if decomposition.bootstrap_log_params_model1_ci:
        ci1 = decomposition.bootstrap_log_params_model1_ci
        ci2 = decomposition.bootstrap_log_params_model2_ci
        print("\n--- BOOTSTRAP 95% CI (1000 iterations, observation-level resample) ---")
        print(f"  log_params (Model 1, baseline):   [{ci1[0]:.4f}, {ci1[1]:.4f}]")
        print(f"  log_params (Model 2, controlled):  [{ci2[0]:.4f}, {ci2[1]:.4f}]")
        contains_zero = ci2[0] <= 0.0 <= ci2[1]
        print(f"  Model 2 CI contains zero: {contains_zero} → {'coefficient indistinguishable from zero' if contains_zero else 'coefficient remains non-zero'}")

    if decomposition.no_aqua_baseline_coef is not None:
        na_b = decomposition.no_aqua_baseline_coef
        na_c = decomposition.no_aqua_controlled_coef
        na_drop = decomposition.no_aqua_pct_drop
        na_ci = decomposition.no_aqua_model2_bootstrap_ci
        print("\n--- ROBUSTNESS: AQuA DROPPED ---")
        print(f"  Baseline log_params coef:    {na_b:.4f}")
        print(f"  Controlled log_params coef:  {na_c:.4f}")
        if na_drop is not None:
            print(f"  % drop in magnitude:         {na_drop:.1f}%")
        if na_ci:
            contains_zero = na_ci[0] <= 0.0 <= na_ci[1]
            print(f"  Bootstrap 95% CI (Model 2):  [{na_ci[0]:.4f}, {na_ci[1]:.4f}]")
            print(f"  CI contains zero: {contains_zero} → {'AQuA was driving the residual' if contains_zero else 'residual persists without AQuA'}")

    if decomposition.quadratic_log_params_coef is not None:
        q_coef = decomposition.quadratic_log_params_coef
        q_ci = decomposition.quadratic_log_params_bootstrap_ci
        q_r2 = decomposition.quadratic_r_squared
        print("\n--- ROBUSTNESS: QUADRATIC ACCURACY TERM (Model 2b) ---")
        print(f"  log_params coef:   {q_coef:.4f}")
        if q_ci:
            contains_zero = q_ci[0] <= 0.0 <= q_ci[1]
            print(f"  Bootstrap 95% CI:  [{q_ci[0]:.4f}, {q_ci[1]:.4f}]")
            print(f"  CI contains zero: {contains_zero} → {'quadratic term absorbs residual' if contains_zero else 'residual persists with quadratic term'}")
        if q_r2 is not None:
            print(f"  R² = {q_r2:.4f}  (vs Model 2 linear R² = {decomposition.regressions[1].r_squared:.4f})")

    print(
        f"\nPlot with: uv run -m src.experiments.regression.plot "
        f"--regression_table {output_dir / 'regression_table.csv'} --output_dir results/figures"
    )


if __name__ == "__main__":
    main()
