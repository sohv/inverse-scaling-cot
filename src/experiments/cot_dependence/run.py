"""Revision Phase 1c: empirical shuffled-CoT null and the CoT-dependence scaling fit. No GPU.

uv run -m src.experiments.cot_dependence.run --core_sweep_results_dir results/core_sweep --output_dir results/cot_dependence --n_permutations 1000 --seed 42
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from scipy.stats import spearmanr

from src.data.sampler import choice_labels_by_id
from src.generation.runner import QuestionResult
from src.metrics.cells import find_cell_dirs, is_quantized
from src.metrics.null_baseline import empirical_null
from src.metrics.recompute import answer_matrix
from src.metrics.robustness import _ols
from src.utils.config import save_run_config
from src.utils.io import read_jsonl, write_json
from src.utils.models import get_model_info
from src.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    core_sweep_results_dir: str = ""
    output_dir: str = "results/cot_dependence"
    n_permutations: int = 1000
    n_bootstrap: int = 1000
    seed: int = 42


def dependence_fit(df: pd.DataFrame, label: str, n_bootstrap: int, seed: int) -> dict:
    """Regress CoT dependence (real - shuffled) on log model size."""
    res = _ols(df, ["log_params"], y_col="dependence")
    slope = float(res.params[1])

    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_bootstrap):
        sample = df.iloc[rng.integers(0, len(df), size=len(df))]
        if sample["log_params"].nunique() < 2:
            continue
        slopes.append(float(_ols(sample, ["log_params"], y_col="dependence").params[1]))

    return {
        "label": label,
        "n_obs": len(df),
        "slope": slope,
        "intercept": float(res.params[0]),
        "r_squared": float(res.rsquared),
        "slope_ci": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
        "n_bootstrap": n_bootstrap,
        "mean_dependence": float(df["dependence"].mean()),
    }


def monotonicity(df: pd.DataFrame) -> dict:
    """Spearman rho per family-task curve, with explicit violations of monotone increase."""
    curves = []
    for (family, task), sub in df.groupby(["family", "dataset_name"]):
        sub = sub.sort_values("size_b")
        rho, p = spearmanr(sub["size_b"], sub["real_match_rate"])
        proxies = sub["real_match_rate"].to_numpy()
        sizes = sub["size_b"].to_numpy()
        violations = [
            {
                "from_size_b": float(sizes[i]),
                "to_size_b": float(sizes[i + 1]),
                "from_proxy": float(proxies[i]),
                "to_proxy": float(proxies[i + 1]),
            }
            for i in range(len(proxies) - 1)
            if proxies[i + 1] < proxies[i]
        ]
        curves.append(
            {
                "family": family,
                "dataset_name": task,
                "n_points": len(sub),
                "spearman_rho": float(rho),
                "p_value": float(p),
                "strictly_monotone": len(violations) == 0,
                "violations": violations,
            }
        )
    return {
        "curves": curves,
        "n_curves": len(curves),
        "n_strictly_monotone": sum(c["strictly_monotone"] for c in curves),
        "n_with_violations": sum(not c["strictly_monotone"] for c in curves),
    }


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

    labels_cache: dict[str, dict[str, list[str]]] = {}
    rows = []
    for cell_dir in find_cell_dirs(config.core_sweep_results_dir):
        results = [QuestionResult(**r) for r in read_jsonl(cell_dir / "generation_results.jsonl")]
        dataset_name = results[0].dataset_name
        if dataset_name not in labels_cache:
            labels_cache[dataset_name] = choice_labels_by_id(dataset_name)
        labels = labels_cache[dataset_name]

        cot, no_cot, valid = answer_matrix(results, labels)
        n_choices = len(labels[results[0].id])
        null = empirical_null(
            cot,
            no_cot,
            valid,
            results[0].model_id,
            dataset_name,
            n_choices=n_choices,
            n_permutations=config.n_permutations,
            seed=config.seed,
        )
        info = get_model_info(null.model_id)
        rows.append(
            {
                **null.model_dump(),
                "log_params": float(np.log10(info.size_b * 1e9)),
                "size_b": info.size_b,
                "family": info.family,
                "is_quantized": int(is_quantized(null.model_id)),
            }
        )
        LOGGER.info(
            f"{cell_dir.name}: real={null.real_match_rate:.4f} null={null.shuffled_mean:.4f} "
            f"dependence={null.dependence:.4f}"
        )

    df = pd.DataFrame(rows).sort_values(["dataset_name", "family", "size_b"]).reset_index(drop=True)
    df.round(4).to_csv(output_dir / "null_table.csv", index=False)

    fits = [dependence_fit(df, "pooled", config.n_bootstrap, config.seed)]
    fits += [
        dependence_fit(df[df.dataset_name == t], f"task={t}", config.n_bootstrap, config.seed)
        for t in sorted(df.dataset_name.unique())
    ]
    fits += [
        dependence_fit(df[df.family == f], f"family={f}", config.n_bootstrap, config.seed)
        for f in sorted(df.family.unique())
    ]
    write_json(output_dir / "dependence_fits.json", {"fits": fits})

    write_json(output_dir / "monotonicity.json", monotonicity(df))
    write_json(
        output_dir / "null_summary.json",
        {
            "n_cells": len(df),
            "n_cells_real_above_null": int(df.real_above_null.sum()),
            "mean_shuffled_rate": float(df.shuffled_mean.mean()),
            "min_shuffled_rate": float(df.shuffled_mean.min()),
            "max_shuffled_rate": float(df.shuffled_mean.max()),
            "mean_uniform_chance": float(df.uniform_chance.mean()),
            "mean_dependence": float(df.dependence.mean()),
        },
    )
    save_run_config(output_dir, config, extra_metadata={"n_cells": len(df)})

    print(
        f"\nEmpirical null: {int(df.real_above_null.sum())}/{len(df)} cells have a real match rate "
        f"above the 97.5th percentile of their own permutation null"
    )
    print(
        f"Shuffled rates span {df.shuffled_mean.min():.4f}-{df.shuffled_mean.max():.4f} "
        f"(uniform-chance assumption was {df.uniform_chance.mean():.4f})"
    )
    print("\nCoT dependence (real - shuffled) vs log model size")
    for f in fits:
        print(
            f"  {f['label']:<28} slope={f['slope']:+.4f} CI=[{f['slope_ci'][0]:+.4f}, {f['slope_ci'][1]:+.4f}] "
            f"mean_dependence={f['mean_dependence']:.4f}"
        )
    mono = monotonicity(df)
    print(
        f"\nMonotonicity: {mono['n_strictly_monotone']}/{mono['n_curves']} family-task curves are "
        f"strictly increasing in model size"
    )
    for c in mono["curves"]:
        if not c["strictly_monotone"]:
            for v in c["violations"]:
                print(
                    f"  violation {c['family']}/{c['dataset_name']}: "
                    f"{v['from_size_b']:g}B {v['from_proxy']:.3f} -> {v['to_size_b']:g}B {v['to_proxy']:.3f}"
                )
    print(f"\nResults saved to {output_dir}")
    print(
        f"Plot with: uv run -m src.experiments.cot_dependence.plot --results_dir {output_dir} --output_dir results/figures"
    )


if __name__ == "__main__":
    main()
