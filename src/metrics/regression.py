"""OLS regression with cluster-robust standard errors for Experiment 4.

Three models:
1. Baseline: faithfulness_proxy ~ log(params)
2. Accuracy control: faithfulness_proxy ~ log(params) + accuracy_without_cot
3. Full: faithfulness_proxy ~ log(params) + accuracy_without_cot + family
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel

from src.metrics.accuracy import AccuracyResult
from src.metrics.faithfulness import FaithfulnessResult
from src.utils.models import get_model_info

LOGGER = logging.getLogger(__name__)


class RegressionResult(BaseModel):
    """Results from one OLS regression."""

    model_name: str
    formula: str
    n_obs: int
    n_clusters: int
    r_squared: float
    adj_r_squared: float
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    p_values: dict[str, float]
    conf_intervals: dict[str, list[float]]
    bootstrap_ci: dict[str, list[float]] = {}  # 95% bootstrap CI per coefficient
    bootstrap_n: int = 0


class DecompositionResult(BaseModel):
    """Decision from the confound decomposition (Exp 4)."""

    baseline_coef_log_params: float
    controlled_coef_log_params: float
    pct_drop: float
    decision: str  # "confound_explained" | "effect_survives" | "ambiguous"
    regressions: list[RegressionResult]
    per_task_decisions: dict[str, str]
    bootstrap_log_params_model1_ci: list[float] = []  # 95% CI from bootstrap
    bootstrap_log_params_model2_ci: list[float] = []
    # Robustness: same regressions with AQuA dropped
    no_aqua_baseline_coef: float | None = None
    no_aqua_controlled_coef: float | None = None
    no_aqua_pct_drop: float | None = None
    no_aqua_model2_bootstrap_ci: list[float] = []
    # Robustness: quadratic accuracy term (Model 2b)
    quadratic_log_params_coef: float | None = None
    quadratic_log_params_bootstrap_ci: list[float] = []
    quadratic_r_squared: float | None = None


def build_regression_table(
    faithfulness_results: list[FaithfulnessResult],
    accuracy_results: list[AccuracyResult],
) -> pd.DataFrame:
    """Merge faithfulness and accuracy results into a DataFrame.

    Columns: model_id, dataset_name, faithfulness_proxy (mean_match_fraction),
             accuracy_no_cot, log_params, family_is_llama (0/1), task_cluster
    """
    faith_map = {(f.model_id, f.dataset_name): f for f in faithfulness_results}
    acc_map = {(a.model_id, a.dataset_name): a for a in accuracy_results}

    rows = []
    for key in faith_map:
        f = faith_map[key]
        a = acc_map.get(key)
        if a is None:
            LOGGER.warning(f"Missing accuracy result for {key}, skipping")
            continue

        info = get_model_info(f.model_id)
        rows.append(
            {
                "model_id": f.model_id,
                "dataset_name": f.dataset_name,
                "faithfulness_proxy": f.mean_match_fraction,
                "accuracy_no_cot": a.accuracy,
                "log_params": np.log10(info.size_b * 1e9),
                "family_is_llama": 1 if info.family == "llama" else 0,
                "task_cluster": f.dataset_name,
                "size_b": info.size_b,
            }
        )

    df = pd.DataFrame(rows)
    df["accuracy_no_cot_sq"] = df["accuracy_no_cot"] ** 2
    LOGGER.info(f"Built regression table with {len(df)} rows")
    return df


def run_regression(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    model_name: str,
    cluster_col: str = "task_cluster",
) -> RegressionResult:
    """Run OLS with cluster-robust SEs.

    Uses statsmodels with cov_type='cluster'.
    """
    y = df[y_col].values
    X = sm.add_constant(df[x_cols].values)
    col_names = ["const"] + x_cols

    model = sm.OLS(y, X)
    cluster_groups = df[cluster_col].values
    result = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})

    n_clusters = len(df[cluster_col].unique())
    if n_clusters < 10:
        LOGGER.warning(
            f"Only {n_clusters} clusters for {model_name}. "
            "Cluster-robust SEs are unreliable with fewer than ~30 clusters. "
            "Interpret p-values with caution."
        )

    coefficients = {name: float(result.params[i]) for i, name in enumerate(col_names)}
    std_errors = {name: float(result.bse[i]) for i, name in enumerate(col_names)}
    p_values = {name: float(result.pvalues[i]) for i, name in enumerate(col_names)}
    conf_intervals = {
        name: [float(result.conf_int()[i, 0]), float(result.conf_int()[i, 1])] for i, name in enumerate(col_names)
    }

    formula = f"{y_col} ~ {' + '.join(x_cols)}"
    return RegressionResult(
        model_name=model_name,
        formula=formula,
        n_obs=int(result.nobs),
        n_clusters=n_clusters,
        r_squared=float(result.rsquared),
        adj_r_squared=float(result.rsquared_adj),
        coefficients=coefficients,
        std_errors=std_errors,
        p_values=p_values,
        conf_intervals=conf_intervals,
    )


def _bootstrap_coefs(
    df: pd.DataFrame,
    x_cols: list[str],
    y_col: str = "faithfulness_proxy",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Bootstrap regression coefficients and return per-variable arrays."""
    rng = np.random.default_rng(seed)
    col_names = ["const"] + x_cols
    boot_coefs: dict[str, list[float]] = {name: [] for name in col_names}

    for _ in range(n_boot):
        sample = df.sample(n=len(df), replace=True, random_state=int(rng.integers(1 << 31)))
        y = sample[y_col].values
        X = sm.add_constant(sample[x_cols].values)
        result = sm.OLS(y, X).fit()
        for i, name in enumerate(col_names):
            boot_coefs[name].append(float(result.params[i]))

    return boot_coefs


def run_decomposition(
    df: pd.DataFrame,
    drop_threshold: float = 0.50,
    retain_threshold: float = 0.30,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> DecompositionResult:
    """Run all 3 regressions and apply the pre-registered decision procedure.

    Decision:
    - If coefficient on log_params drops >50% in magnitude from Model 1 to Model 2: "confound_explained"
    - If coefficient retains >70% of magnitude (drop <30%): "effect_survives"
    - Otherwise: "ambiguous"
    """
    # Model 1: baseline
    reg1 = run_regression(df, "faithfulness_proxy", ["log_params"], "baseline")

    # Model 2: accuracy control
    reg2 = run_regression(df, "faithfulness_proxy", ["log_params", "accuracy_no_cot"], "accuracy_control")

    # Model 3: full with family
    reg3 = run_regression(df, "faithfulness_proxy", ["log_params", "accuracy_no_cot", "family_is_llama"], "full")

    # Bootstrap CIs for Model 1 and Model 2
    LOGGER.info(f"Running {n_bootstrap}-iteration bootstrap for Model 1...")
    boot1 = _bootstrap_coefs(df, ["log_params"], n_boot=n_bootstrap, seed=bootstrap_seed)
    reg1.bootstrap_ci = {
        name: [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        for name, vals in boot1.items()
    }
    reg1.bootstrap_n = n_bootstrap

    LOGGER.info(f"Running {n_bootstrap}-iteration bootstrap for Model 2...")
    boot2 = _bootstrap_coefs(df, ["log_params", "accuracy_no_cot"], n_boot=n_bootstrap, seed=bootstrap_seed)
    reg2.bootstrap_ci = {
        name: [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        for name, vals in boot2.items()
    }
    reg2.bootstrap_n = n_bootstrap

    # Decision on pooled data
    baseline_coef = reg1.coefficients["log_params"]
    controlled_coef = reg2.coefficients["log_params"]

    if abs(baseline_coef) < 1e-10:
        pct_drop = 0.0
        decision = "ambiguous"
    else:
        pct_drop = (1 - abs(controlled_coef) / abs(baseline_coef)) * 100
        if pct_drop >= drop_threshold * 100:
            decision = "confound_explained"
        elif pct_drop <= retain_threshold * 100:
            decision = "effect_survives"
        else:
            decision = "ambiguous"

    # Per-task decisions
    per_task_decisions = {}
    for task in df["dataset_name"].unique():
        task_df = df[df["dataset_name"] == task]
        if len(task_df) < 3:
            per_task_decisions[task] = "insufficient_data"
            continue

        # For per-task, use HC3 robust SEs since we can't cluster with 1 task
        y = task_df["faithfulness_proxy"].values
        X1 = sm.add_constant(task_df[["log_params"]].values)
        X2 = sm.add_constant(task_df[["log_params", "accuracy_no_cot"]].values)

        res1 = sm.OLS(y, X1).fit(cov_type="HC3")
        res2 = sm.OLS(y, X2).fit(cov_type="HC3")

        b1 = float(res1.params[1])  # log_params coefficient
        b2 = float(res2.params[1])

        if abs(b1) < 1e-10:
            per_task_decisions[task] = "ambiguous"
        else:
            task_pct_drop = (1 - abs(b2) / abs(b1)) * 100
            if task_pct_drop >= drop_threshold * 100:
                per_task_decisions[task] = "confound_explained"
            elif task_pct_drop <= retain_threshold * 100:
                per_task_decisions[task] = "effect_survives"
            else:
                per_task_decisions[task] = "ambiguous"

    LOGGER.info(f"Pooled decision: {decision} (pct_drop={pct_drop:.1f}%)")
    for task, dec in per_task_decisions.items():
        LOGGER.info(f"  {task}: {dec}")

    boot_m1_ci = reg1.bootstrap_ci.get("log_params", [])
    boot_m2_ci = reg2.bootstrap_ci.get("log_params", [])

    # Model 2b: quadratic accuracy control
    LOGGER.info("Running Model 2b: quadratic accuracy term...")
    reg2b = run_regression(
        df, "faithfulness_proxy", ["log_params", "accuracy_no_cot", "accuracy_no_cot_sq"], "quadratic_accuracy"
    )
    boot2b = _bootstrap_coefs(
        df, ["log_params", "accuracy_no_cot", "accuracy_no_cot_sq"], n_boot=n_bootstrap, seed=bootstrap_seed
    )
    reg2b.bootstrap_ci = {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot2b.items()
    }
    reg2b.bootstrap_n = n_bootstrap

    quadratic_log_params_coef = reg2b.coefficients["log_params"]
    quadratic_log_params_boot_ci = reg2b.bootstrap_ci.get("log_params", [])
    LOGGER.info(
        "Model 2b: log_params coef=%.4f, bootstrap CI=[%.4f, %.4f]",
        quadratic_log_params_coef,
        quadratic_log_params_boot_ci[0] if quadratic_log_params_boot_ci else float("nan"),
        quadratic_log_params_boot_ci[1] if quadratic_log_params_boot_ci else float("nan"),
    )

    # Robustness check: drop AQuA and re-run Models 1 & 2 with bootstrap
    no_aqua_baseline_coef = None
    no_aqua_controlled_coef = None
    no_aqua_pct_drop = None
    no_aqua_boot_m2_ci: list[float] = []

    df_no_aqua = df[df["dataset_name"] != "aqua"].copy()
    if len(df_no_aqua) >= 4:
        LOGGER.info("Running robustness check: AQuA dropped (%d rows)", len(df_no_aqua))
        na_reg1 = run_regression(df_no_aqua, "faithfulness_proxy", ["log_params"], "no_aqua_baseline")
        na_reg2 = run_regression(
            df_no_aqua, "faithfulness_proxy", ["log_params", "accuracy_no_cot"], "no_aqua_accuracy_control"
        )

        na_boot1 = _bootstrap_coefs(df_no_aqua, ["log_params"], n_boot=n_bootstrap, seed=bootstrap_seed)
        na_boot2 = _bootstrap_coefs(
            df_no_aqua, ["log_params", "accuracy_no_cot"], n_boot=n_bootstrap, seed=bootstrap_seed
        )
        na_reg1.bootstrap_ci = {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in na_boot1.items()
        }
        na_reg2.bootstrap_ci = {
            k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in na_boot2.items()
        }
        na_reg1.bootstrap_n = na_reg2.bootstrap_n = n_bootstrap

        no_aqua_baseline_coef = na_reg1.coefficients["log_params"]
        no_aqua_controlled_coef = na_reg2.coefficients["log_params"]
        if abs(no_aqua_baseline_coef) > 1e-10:
            no_aqua_pct_drop = (1 - abs(no_aqua_controlled_coef) / abs(no_aqua_baseline_coef)) * 100
        no_aqua_boot_m2_ci = na_reg2.bootstrap_ci.get("log_params", [])
        LOGGER.info(
            "No-AQuA: baseline coef=%.4f, controlled=%.4f, pct_drop=%.1f%%",
            no_aqua_baseline_coef,
            no_aqua_controlled_coef,
            no_aqua_pct_drop or 0.0,
        )

    return DecompositionResult(
        baseline_coef_log_params=baseline_coef,
        controlled_coef_log_params=controlled_coef,
        pct_drop=pct_drop,
        decision=decision,
        regressions=[reg1, reg2, reg3],
        per_task_decisions=per_task_decisions,
        bootstrap_log_params_model1_ci=boot_m1_ci,
        bootstrap_log_params_model2_ci=boot_m2_ci,
        no_aqua_baseline_coef=no_aqua_baseline_coef,
        no_aqua_controlled_coef=no_aqua_controlled_coef,
        no_aqua_pct_drop=no_aqua_pct_drop,
        no_aqua_model2_bootstrap_ci=no_aqua_boot_m2_ci,
        quadratic_log_params_coef=quadratic_log_params_coef,
        quadratic_log_params_bootstrap_ci=quadratic_log_params_boot_ci,
        quadratic_r_squared=reg2b.r_squared,
    )
