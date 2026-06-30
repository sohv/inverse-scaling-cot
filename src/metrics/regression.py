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


class DecompositionResult(BaseModel):
    """Decision from the confound decomposition (Exp 4)."""

    baseline_coef_log_params: float
    controlled_coef_log_params: float
    pct_drop: float
    decision: str  # "confound_explained" | "effect_survives" | "ambiguous"
    regressions: list[RegressionResult]
    per_task_decisions: dict[str, str]


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


def run_decomposition(
    df: pd.DataFrame,
    drop_threshold: float = 0.50,
    retain_threshold: float = 0.30,
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

    return DecompositionResult(
        baseline_coef_log_params=baseline_coef,
        controlled_coef_log_params=controlled_coef,
        pct_drop=pct_drop,
        decision=decision,
        regressions=[reg1, reg2, reg3],
        per_task_decisions=per_task_decisions,
    )
