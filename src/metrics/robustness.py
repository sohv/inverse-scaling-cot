"""Regression battery for the revision: per-task, leave-one-out, per-family, residualised.

Every analysis reports the same three quantities so results are directly comparable:
the raw log_params coefficient (Model 1), the accuracy-controlled coefficient (Model 2)
and the percent reduction between them. Bootstrap CIs resample cells and refit BOTH
models on each resample, so the CI on the percent reduction accounts for the correlation
between the two coefficients.
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)

RAW_COLS = ["log_params"]
CONTROLLED_COLS = ["log_params", "accuracy_no_cot"]


class DecompositionFit(BaseModel):
    """Model 1 vs Model 2 comparison for one subset of cells."""

    label: str
    n_obs: int
    n_models: int
    n_tasks: int
    raw_coef: float
    controlled_coef: float
    pct_reduction: float
    accuracy_coef: float
    r2_raw: float
    r2_controlled: float
    raw_coef_ci: list[float] = []
    controlled_coef_ci: list[float] = []
    pct_reduction_ci: list[float] = []
    n_bootstrap: int = 0


def _ols(df: pd.DataFrame, x_cols: list[str], y_col: str = "faithfulness_proxy"):
    X = sm.add_constant(df[x_cols].to_numpy(), has_constant="add")
    return sm.OLS(df[y_col].to_numpy(), X).fit()


def _pct_reduction(raw: float, controlled: float) -> float:
    if abs(raw) < 1e-10:
        return float("nan")
    return (1 - abs(controlled) / abs(raw)) * 100


def fit_decomposition(
    df: pd.DataFrame,
    label: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> DecompositionFit:
    """Fit Model 1 and Model 2 on one subset and bootstrap the coefficient reduction."""
    res_raw = _ols(df, RAW_COLS)
    res_ctl = _ols(df, CONTROLLED_COLS)

    raw_coef = float(res_raw.params[1])
    controlled_coef = float(res_ctl.params[1])

    fit = DecompositionFit(
        label=label,
        n_obs=len(df),
        n_models=int(df.model_id.nunique()),
        n_tasks=int(df.dataset_name.nunique()),
        raw_coef=raw_coef,
        controlled_coef=controlled_coef,
        pct_reduction=_pct_reduction(raw_coef, controlled_coef),
        accuracy_coef=float(res_ctl.params[2]),
        r2_raw=float(res_raw.rsquared),
        r2_controlled=float(res_ctl.rsquared),
    )

    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        raws, ctls, reductions = [], [], []
        n = len(df)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            sample = df.iloc[idx]
            if sample["log_params"].nunique() < 2:
                continue
            b_raw = float(_ols(sample, RAW_COLS).params[1])
            b_ctl = float(_ols(sample, CONTROLLED_COLS).params[1])
            raws.append(b_raw)
            ctls.append(b_ctl)
            red = _pct_reduction(b_raw, b_ctl)
            if np.isfinite(red):
                reductions.append(red)

        fit.raw_coef_ci = [float(np.percentile(raws, 2.5)), float(np.percentile(raws, 97.5))]
        fit.controlled_coef_ci = [float(np.percentile(ctls, 2.5)), float(np.percentile(ctls, 97.5))]
        fit.pct_reduction_ci = [float(np.percentile(reductions, 2.5)), float(np.percentile(reductions, 97.5))]
        fit.n_bootstrap = n_bootstrap

    return fit


def per_task(df: pd.DataFrame, n_bootstrap: int = 1000, seed: int = 42) -> list[DecompositionFit]:
    """Run the full decomposition separately for each dataset (revision item 3)."""
    return [
        fit_decomposition(df[df.dataset_name == task], f"task={task}", n_bootstrap, seed)
        for task in sorted(df.dataset_name.unique())
    ]


def leave_one_task_out(df: pd.DataFrame, n_bootstrap: int = 1000, seed: int = 42) -> list[DecompositionFit]:
    """Refit with each dataset removed in turn (revision item 4)."""
    return [
        fit_decomposition(df[df.dataset_name != task], f"drop_task={task}", n_bootstrap, seed)
        for task in sorted(df.dataset_name.unique())
    ]


def leave_one_model_out(df: pd.DataFrame, n_bootstrap: int = 1000, seed: int = 42) -> list[DecompositionFit]:
    """Refit with each model checkpoint removed in turn (revision item 5)."""
    return [
        fit_decomposition(df[df.model_id != model], f"drop_model={model}", n_bootstrap, seed)
        for model in sorted(df.model_id.unique())
    ]


def per_family(df: pd.DataFrame, n_bootstrap: int = 1000, seed: int = 42) -> list[DecompositionFit]:
    """Run the full decomposition separately within each model family (revision item 6)."""
    return [
        fit_decomposition(df[df.family == fam], f"family={fam}", n_bootstrap, seed)
        for fam in sorted(df.family.unique())
    ]


class ResidualFit(BaseModel):
    """Residualised proxy regressed on model size (revision item 8)."""

    label: str
    n_obs: int
    slope: float
    intercept: float
    r2: float
    slope_ci: list[float] = []
    n_bootstrap: int = 0


def residualize(df: pd.DataFrame, label: str = "pooled", n_bootstrap: int = 1000, seed: int = 42) -> ResidualFit:
    """Regress proxy on accuracy, then regress the residual on log_params."""
    stage1 = _ols(df, ["accuracy_no_cot"])
    residual = df["faithfulness_proxy"].to_numpy() - stage1.fittedvalues

    work = df.copy()
    work["residual"] = residual
    stage2 = _ols(work, ["log_params"], y_col="residual")

    fit = ResidualFit(
        label=label,
        n_obs=len(df),
        slope=float(stage2.params[1]),
        intercept=float(stage2.params[0]),
        r2=float(stage2.rsquared),
    )

    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        slopes = []
        n = len(work)
        for _ in range(n_bootstrap):
            sample = work.iloc[rng.integers(0, n, size=n)]
            if sample["log_params"].nunique() < 2:
                continue
            s1 = _ols(sample, ["accuracy_no_cot"])
            tmp = sample.copy()
            tmp["residual"] = sample["faithfulness_proxy"].to_numpy() - s1.fittedvalues
            slopes.append(float(_ols(tmp, ["log_params"], y_col="residual").params[1]))
        fit.slope_ci = [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]
        fit.n_bootstrap = n_bootstrap

    return fit


def residual_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the accuracy-residualised proxy to the cell table, for plotting."""
    stage1 = _ols(df, ["accuracy_no_cot"])
    out = df.copy()
    out["proxy_residual"] = df["faithfulness_proxy"].to_numpy() - stage1.fittedvalues
    out["proxy_predicted_from_accuracy"] = stage1.fittedvalues
    return out


class PartialCorrelation(BaseModel):
    """Partial correlation between log_params and the proxy, controlling for accuracy."""

    label: str
    n_obs: int
    raw_correlation: float
    partial_correlation: float
    partial_ci: list[float] = []
    n_bootstrap: int = 0


def _partial_r(df: pd.DataFrame) -> float:
    """Correlation of the two residuals after regressing each on accuracy."""
    res_y = df["faithfulness_proxy"].to_numpy() - _ols(df, ["accuracy_no_cot"]).fittedvalues
    res_x = df["log_params"].to_numpy() - _ols(df, ["accuracy_no_cot"], y_col="log_params").fittedvalues
    if np.std(res_x) < 1e-12 or np.std(res_y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(res_x, res_y)[0, 1])


def partial_correlation(
    df: pd.DataFrame, label: str = "pooled", n_bootstrap: int = 1000, seed: int = 42
) -> PartialCorrelation:
    """Partial correlation r(log_params, proxy | accuracy_no_cot) (revision item 9)."""
    out = PartialCorrelation(
        label=label,
        n_obs=len(df),
        raw_correlation=float(np.corrcoef(df["log_params"], df["faithfulness_proxy"])[0, 1]),
        partial_correlation=_partial_r(df),
    )

    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        vals = []
        n = len(df)
        for _ in range(n_bootstrap):
            sample = df.iloc[rng.integers(0, n, size=n)]
            if sample["log_params"].nunique() < 2:
                continue
            r = _partial_r(sample)
            if np.isfinite(r):
                vals.append(r)
        out.partial_ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        out.n_bootstrap = n_bootstrap

    return out


class MixedEffectsFit(BaseModel):
    """Task-aware sensitivity analysis with a random effect on task."""

    label: str
    formula: str
    n_obs: int
    n_groups: int
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    p_values: dict[str, float]
    group_variance: float
    converged: bool


def _mixed(df: pd.DataFrame, formula: str, label: str, re_formula: str | None) -> MixedEffectsFit:
    model = smf.mixedlm(formula, df, groups=df["task_cluster"], re_formula=re_formula)
    # lbfgs hits a singular gradient once the task variance approaches its boundary;
    # powell is derivative-free and agrees with nm/cg/bfgs to 4 dp on this data.
    res = model.fit(reml=True, method="powell")
    return MixedEffectsFit(
        label=label,
        formula=formula + (f" + ({re_formula[1:]}|task)" if re_formula else " + (1|task)"),
        n_obs=int(res.nobs),
        n_groups=int(df["task_cluster"].nunique()),
        coefficients={k: float(v) for k, v in res.params.items() if not k.startswith("Group")},
        std_errors={k: float(v) for k, v in res.bse.items() if not k.startswith("Group")},
        p_values={k: float(v) for k, v in res.pvalues.items() if not k.startswith("Group")},
        group_variance=float(res.cov_re.iloc[0, 0]),
        converged=bool(res.converged),
    )


def mixed_effects(df: pd.DataFrame) -> list[MixedEffectsFit]:
    """Random-intercept and random-slope task models, raw and accuracy-controlled.

    Predictors are mean-centred: on the raw scale log_params (~9-11) and accuracy (0-1)
    differ enough in magnitude that the REML gradient hits a singular matrix. Centring
    shifts only the intercept, so the slopes stay directly comparable to the OLS fits.
    With only 5 task groups these are a sensitivity check, not primary inference.
    """
    work = df.copy()
    work["log_params"] = df["log_params"] - df["log_params"].mean()
    work["accuracy_no_cot"] = df["accuracy_no_cot"] - df["accuracy_no_cot"].mean()
    return [
        _mixed(work, "faithfulness_proxy ~ log_params", "raw_random_intercept", None),
        _mixed(work, "faithfulness_proxy ~ log_params + accuracy_no_cot", "controlled_random_intercept", None),
        _mixed(work, "faithfulness_proxy ~ log_params + accuracy_no_cot", "controlled_random_slope", "~log_params"),
    ]


def summarize_fits(fits: list[DecompositionFit]) -> dict:
    """Min / median / max percent reduction across a set of fits, with the extremes named."""
    reductions = [(f.label, f.pct_reduction) for f in fits if np.isfinite(f.pct_reduction)]
    values = [r for _, r in reductions]
    return {
        "n_fits": len(reductions),
        "min_pct_reduction": float(np.min(values)),
        "median_pct_reduction": float(np.median(values)),
        "max_pct_reduction": float(np.max(values)),
        "min_label": min(reductions, key=lambda t: t[1])[0],
        "max_label": max(reductions, key=lambda t: t[1])[0],
        "all_above_50": bool(np.all(np.array(values) >= 50.0)),
    }


def fit_quadratic(df: pd.DataFrame, n_bootstrap: int = 1000, seed: int = 42) -> DecompositionFit:
    """Model 2b: accuracy control with a quadratic term (revision item 19)."""
    cols = ["log_params", "accuracy_no_cot", "accuracy_no_cot_sq"]
    res_raw = _ols(df, RAW_COLS)
    res_ctl = _ols(df, cols)
    raw_coef = float(res_raw.params[1])
    controlled_coef = float(res_ctl.params[1])

    fit = DecompositionFit(
        label="quadratic_accuracy",
        n_obs=len(df),
        n_models=int(df.model_id.nunique()),
        n_tasks=int(df.dataset_name.nunique()),
        raw_coef=raw_coef,
        controlled_coef=controlled_coef,
        pct_reduction=_pct_reduction(raw_coef, controlled_coef),
        accuracy_coef=float(res_ctl.params[2]),
        r2_raw=float(res_raw.rsquared),
        r2_controlled=float(res_ctl.rsquared),
    )

    if n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        ctls = []
        n = len(df)
        for _ in range(n_bootstrap):
            sample = df.iloc[rng.integers(0, n, size=n)]
            if sample["log_params"].nunique() < 2:
                continue
            ctls.append(float(_ols(sample, cols).params[1]))
        fit.controlled_coef_ci = [float(np.percentile(ctls, 2.5)), float(np.percentile(ctls, 97.5))]
        fit.n_bootstrap = n_bootstrap

    return fit
