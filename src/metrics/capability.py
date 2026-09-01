"""Capability-matched comparisons, regime bins and the capability reconstruction.

These make the confound argument legible without a regression: if two checkpoints of very
different size reach the same no-CoT accuracy, the proxy should also match.
Thresholds are pre-registered in docs/decisions.md (2026-09-01).
"""

import itertools
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel
from scipy import stats

LOGGER = logging.getLogger(__name__)

ACC_TOLERANCE = 0.03
MIN_SIZE_RATIO = 4.0
BIN_EDGES = [0.0, 0.35, 0.75, 1.0]
BIN_NAMES = ["near_random", "intermediate", "ceiling"]
CEILING_THRESHOLD = 0.75


class MatchedPairs(BaseModel):
    """Capability-matched comparison (revision item 10)."""

    label: str
    acc_tolerance: float
    min_size_ratio: float
    within_family: bool
    n_pairs: int
    mean_abs_proxy_diff: float
    median_abs_proxy_diff: float
    mean_abs_acc_diff: float
    mean_log_params_diff: float
    mean_size_ratio: float
    predicted_proxy_diff_from_size: float  # raw size coefficient x mean log-param gap
    paired_t_statistic: float
    paired_p_value: float
    pairs: list[dict]


def matched_pairs(
    df: pd.DataFrame,
    raw_size_coef: float,
    acc_tolerance: float = ACC_TOLERANCE,
    min_size_ratio: float = MIN_SIZE_RATIO,
    within_family: bool = True,
    label: str = "within_family",
) -> MatchedPairs:
    """Find cell pairs matched on no-CoT accuracy but far apart in parameter count."""
    group_cols = ["dataset_name", "family"] if within_family else ["dataset_name"]
    pairs = []

    for _, group in df.groupby(group_cols):
        for a, b in itertools.combinations(group.to_dict("records"), 2):
            acc_diff = abs(a["accuracy_no_cot"] - b["accuracy_no_cot"])
            ratio = max(a["size_b"], b["size_b"]) / min(a["size_b"], b["size_b"])
            if acc_diff > acc_tolerance or ratio < min_size_ratio:
                continue
            small, large = (a, b) if a["size_b"] < b["size_b"] else (b, a)
            pairs.append(
                {
                    "dataset_name": a["dataset_name"],
                    "small_model": small["model_id"],
                    "large_model": large["model_id"],
                    "small_size_b": small["size_b"],
                    "large_size_b": large["size_b"],
                    "size_ratio": ratio,
                    "small_accuracy": small["accuracy_no_cot"],
                    "large_accuracy": large["accuracy_no_cot"],
                    "acc_diff": acc_diff,
                    "small_proxy": small["faithfulness_proxy"],
                    "large_proxy": large["faithfulness_proxy"],
                    "proxy_diff": large["faithfulness_proxy"] - small["faithfulness_proxy"],
                    "log_params_diff": large["log_params"] - small["log_params"],
                }
            )

    if not pairs:
        LOGGER.warning(f"no matched pairs for {label} at tolerance {acc_tolerance}")
        return MatchedPairs(
            label=label,
            acc_tolerance=acc_tolerance,
            min_size_ratio=min_size_ratio,
            within_family=within_family,
            n_pairs=0,
            mean_abs_proxy_diff=float("nan"),
            median_abs_proxy_diff=float("nan"),
            mean_abs_acc_diff=float("nan"),
            mean_log_params_diff=float("nan"),
            mean_size_ratio=float("nan"),
            predicted_proxy_diff_from_size=float("nan"),
            paired_t_statistic=float("nan"),
            paired_p_value=float("nan"),
            pairs=[],
        )

    frame = pd.DataFrame(pairs)
    t_stat, p_value = stats.ttest_rel(frame.large_proxy, frame.small_proxy)
    mean_log_gap = float(frame.log_params_diff.mean())

    return MatchedPairs(
        label=label,
        acc_tolerance=acc_tolerance,
        min_size_ratio=min_size_ratio,
        within_family=within_family,
        n_pairs=len(frame),
        mean_abs_proxy_diff=float(frame.proxy_diff.abs().mean()),
        median_abs_proxy_diff=float(frame.proxy_diff.abs().median()),
        mean_abs_acc_diff=float(frame.acc_diff.mean()),
        mean_log_params_diff=mean_log_gap,
        mean_size_ratio=float(frame.size_ratio.mean()),
        predicted_proxy_diff_from_size=float(raw_size_coef * mean_log_gap),
        paired_t_statistic=float(t_stat),
        paired_p_value=float(p_value),
        pairs=frame.round(4).to_dict("records"),
    )


def assign_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Label each cell with its pre-registered capability regime."""
    out = df.copy()
    out["capability_bin"] = pd.cut(out["accuracy_no_cot"], bins=BIN_EDGES, labels=BIN_NAMES, include_lowest=True)
    return out


def capability_bins(df: pd.DataFrame) -> list[dict]:
    """Within-bin behaviour of the proxy (revision item 11)."""
    binned = assign_bins(df)
    out = []
    for name in BIN_NAMES:
        sub = binned[binned.capability_bin == name]
        if len(sub) < 3:
            out.append({"bin": name, "n_obs": len(sub), "insufficient_data": True})
            continue
        X_size = sm.add_constant(sub[["log_params"]].to_numpy())
        X_acc = sm.add_constant(sub[["accuracy_no_cot"]].to_numpy())
        y = sub["faithfulness_proxy"].to_numpy()
        out.append(
            {
                "bin": name,
                "n_obs": len(sub),
                "insufficient_data": False,
                "accuracy_range": [float(sub.accuracy_no_cot.min()), float(sub.accuracy_no_cot.max())],
                "mean_proxy": float(sub.faithfulness_proxy.mean()),
                "mean_accuracy": float(sub.accuracy_no_cot.mean()),
                "slope_vs_log_params": float(sm.OLS(y, X_size).fit().params[1]),
                "slope_vs_accuracy": float(sm.OLS(y, X_acc).fit().params[1]),
                "r2_vs_accuracy": float(sm.OLS(y, X_acc).fit().rsquared),
                "tasks": sorted(sub.dataset_name.unique().tolist()),
            }
        )
    return out


def saturation(df: pd.DataFrame, threshold: float = CEILING_THRESHOLD) -> dict:
    """Compare the proxy-accuracy slope below and above the ceiling threshold."""
    out = {"threshold": threshold}
    for name, sub in [("below", df[df.accuracy_no_cot < threshold]), ("above", df[df.accuracy_no_cot >= threshold])]:
        if len(sub) < 3:
            out[name] = {"n_obs": len(sub), "insufficient_data": True}
            continue
        res = sm.OLS(sub.faithfulness_proxy.to_numpy(), sm.add_constant(sub[["accuracy_no_cot"]].to_numpy())).fit()
        out[name] = {
            "n_obs": len(sub),
            "insufficient_data": False,
            "slope": float(res.params[1]),
            "intercept": float(res.params[0]),
            "r_squared": float(res.rsquared),
        }
    if not out["below"].get("insufficient_data") and not out["above"].get("insufficient_data"):
        out["slope_ratio_above_over_below"] = out["above"]["slope"] / out["below"]["slope"]
    return out


def reconstruct(df: pd.DataFrame) -> dict:
    """Predict the proxy from no-CoT accuracy alone and compare scaling curves.

    Explanatory reconstruction, not a causal decomposition: it asks whether capability
    differences are sufficient to reproduce the observed scaling pattern.
    """
    res = sm.OLS(df.faithfulness_proxy.to_numpy(), sm.add_constant(df[["accuracy_no_cot"]].to_numpy())).fit()
    work = df.copy()
    work["proxy_predicted"] = res.fittedvalues

    observed_by_size = work.groupby("log_params").faithfulness_proxy.mean()
    predicted_by_size = work.groupby("log_params").proxy_predicted.mean()

    observed_range = float(observed_by_size.max() - observed_by_size.min())
    predicted_range = float(predicted_by_size.max() - predicted_by_size.min())

    slope_obs = float(
        sm.OLS(work.faithfulness_proxy.to_numpy(), sm.add_constant(work[["log_params"]].to_numpy())).fit().params[1]
    )
    slope_pred = float(
        sm.OLS(work.proxy_predicted.to_numpy(), sm.add_constant(work[["log_params"]].to_numpy())).fit().params[1]
    )

    return {
        "accuracy_intercept": float(res.params[0]),
        "accuracy_slope": float(res.params[1]),
        "r_squared": float(res.rsquared),
        "observed_scaling_range": observed_range,
        "predicted_scaling_range": predicted_range,
        "fraction_of_range_reproduced": predicted_range / observed_range,
        "observed_size_slope": slope_obs,
        "predicted_size_slope": slope_pred,
        "fraction_of_slope_reproduced": slope_pred / slope_obs,
        "mean_absolute_error": float(np.abs(work.faithfulness_proxy - work.proxy_predicted).mean()),
    }


def answer_distribution_stats(counts: dict[str, int]) -> dict:
    """Entropy and max-share of a cell's no-CoT answer-letter distribution."""
    total = sum(counts.values())
    if total == 0:
        return {"entropy": float("nan"), "max_share": float("nan"), "n_distinct": 0, "counts": counts}
    probs = np.array([c / total for c in counts.values() if c > 0])
    n_options = max(len(counts), 2)
    return {
        "entropy": float(-(probs * np.log2(probs)).sum()),
        "normalized_entropy": float(-(probs * np.log2(probs)).sum() / np.log2(n_options)),
        "max_share": float(max(counts.values()) / total),
        "modal_answer": max(counts, key=counts.get),
        "n_distinct": int(sum(1 for c in counts.values() if c > 0)),
        "counts": counts,
    }
