"""Paired comparison of two generation runs that differ in exactly one variable. No GPU.

Serves three revision items with one code path, because each is the same question --
hold the cells fixed, change one thing, compare the proxy, no-CoT accuracy, shuffled-CoT
match rate and the real-minus-shuffled gap:
  item 14  AQuA answer-position randomisation  (baseline = core sweep)
  item 15  prompt-template robustness          (baseline = core sweep)
  item 2   AWQ vs BF16 quantization            (baseline = the AWQ checkpoints)

Cells are joined on (family, size_b, dataset_name), so an AWQ checkpoint pairs with its
BF16 counterpart even though the model_id differs.

uv run -m src.experiments.comparison.run --baseline_dir results/core_sweep --variant_dir results/bf16_large --label awq_vs_bf16 --output_dir results/quantization --seed 42
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing

from src.data.sampler import choice_labels_by_id
from src.generation.runner import QuestionResult
from src.metrics.cells import cells_to_table, default_variant, find_cell_dirs, load_cells
from src.metrics.null_baseline import empirical_null
from src.metrics.recompute import answer_matrix
from src.metrics.robustness import fit_decomposition
from src.utils.config import save_run_config
from src.utils.io import read_jsonl, write_json
from src.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)

JOIN_KEYS = ["family", "size_b", "dataset_name"]


@dataclass
class Config:
    baseline_dir: str = ""
    variant_dir: str = ""
    label: str = "comparison"
    output_dir: str = "results/comparison"
    n_permutations: int = 1000
    n_bootstrap: int = 1000
    seed: int = 42


def null_table(results_dir: str, n_permutations: int, seed: int) -> pd.DataFrame:
    """Empirical shuffled-CoT null for every cell in a results directory."""
    labels_cache: dict[str, dict[str, list[str]]] = {}
    rows = []
    for cell_dir in find_cell_dirs(results_dir):
        results = [QuestionResult(**r) for r in read_jsonl(cell_dir / "generation_results.jsonl")]
        dataset_name = results[0].dataset_name
        if dataset_name not in labels_cache:
            labels_cache[dataset_name] = choice_labels_by_id(dataset_name)
        labels = labels_cache[dataset_name]
        cot, no_cot, valid = answer_matrix(results, labels)
        null = empirical_null(
            cot, no_cot, valid, results[0].model_id, dataset_name,
            n_choices=len(labels[results[0].id]), n_permutations=n_permutations, seed=seed,
        )
        rows.append(
            {
                "model_id": null.model_id,
                "dataset_name": dataset_name,
                "shuffled_mean": null.shuffled_mean,
                "dependence": null.dependence,
            }
        )
    return pd.DataFrame(rows)


def side_table(results_dir: str, n_permutations: int, n_bootstrap: int, seed: int) -> pd.DataFrame:
    cells = load_cells(results_dir, n_bootstrap=n_bootstrap, seed=seed)
    df = cells_to_table(cells[default_variant().key])
    return df.merge(null_table(results_dir, n_permutations, seed), on=["model_id", "dataset_name"])


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

    base = side_table(config.baseline_dir, config.n_permutations, config.n_bootstrap, config.seed)
    var = side_table(config.variant_dir, config.n_permutations, config.n_bootstrap, config.seed)

    merged = base.merge(var, on=JOIN_KEYS, suffixes=("_baseline", "_variant"))
    if merged.empty:
        raise RuntimeError(f"no cells joined on {JOIN_KEYS}; baseline has {len(base)} rows, variant {len(var)}")

    for metric in ["faithfulness_proxy", "accuracy_no_cot", "shuffled_mean", "dependence"]:
        merged[f"delta_{metric}"] = merged[f"{metric}_variant"] - merged[f"{metric}_baseline"]
    merged.round(4).to_csv(output_dir / f"{config.label}_cells.csv", index=False)

    summary = {
        "label": config.label,
        "baseline_dir": config.baseline_dir,
        "variant_dir": config.variant_dir,
        "n_paired_cells": len(merged),
        "models_baseline": sorted(merged.model_id_baseline.unique().tolist()),
        "models_variant": sorted(merged.model_id_variant.unique().tolist()),
        "deltas": {
            metric: {
                "mean": float(merged[f"delta_{metric}"].mean()),
                "mean_abs": float(merged[f"delta_{metric}"].abs().mean()),
                "max_abs": float(merged[f"delta_{metric}"].abs().max()),
                "std": float(merged[f"delta_{metric}"].std()),
            }
            for metric in ["faithfulness_proxy", "accuracy_no_cot", "shuffled_mean", "dependence"]
        },
    }

    # the decomposition only means something with several distinct model sizes on each side
    for name, df in [("baseline", base), ("variant", var)]:
        if df.log_params.nunique() >= 2 and len(df) >= 4:
            summary[f"decomposition_{name}"] = fit_decomposition(df, name, config.n_bootstrap, config.seed).model_dump()
        else:
            summary[f"decomposition_{name}"] = {
                "skipped": True,
                "reason": f"{df.log_params.nunique()} distinct sizes over {len(df)} cells",
            }

    write_json(output_dir / f"{config.label}_summary.json", summary)
    save_run_config(output_dir, config, extra_metadata={"n_paired_cells": len(merged)})

    print(f"\n{config.label}: {len(merged)} paired cells")
    print(f"{'metric':<22} {'mean delta':>11} {'mean |delta|':>13} {'max |delta|':>12}")
    for metric in ["faithfulness_proxy", "accuracy_no_cot", "shuffled_mean", "dependence"]:
        d = summary["deltas"][metric]
        print(f"{metric:<22} {d['mean']:>+11.4f} {d['mean_abs']:>13.4f} {d['max_abs']:>12.4f}")

    print("\nPer-cell detail")
    print(f"{'dataset':<16} {'size':>6} {'proxy base':>11} {'proxy var':>10} {'delta':>8} "
          f"{'acc base':>9} {'acc var':>8} {'depend base':>12} {'depend var':>11}")
    for r in merged.sort_values(["dataset_name", "size_b"]).to_dict("records"):
        print(f"{r['dataset_name']:<16} {r['size_b']:>6.1f} {r['faithfulness_proxy_baseline']:>11.4f} "
              f"{r['faithfulness_proxy_variant']:>10.4f} {r['delta_faithfulness_proxy']:>+8.4f} "
              f"{r['accuracy_no_cot_baseline']:>9.4f} {r['accuracy_no_cot_variant']:>8.4f} "
              f"{r['dependence_baseline']:>12.4f} {r['dependence_variant']:>11.4f}")

    for name in ["baseline", "variant"]:
        d = summary[f"decomposition_{name}"]
        if d.get("skipped"):
            print(f"\n{name} decomposition skipped: {d['reason']}")
        else:
            print(f"\n{name} decomposition: raw={d['raw_coef']:.4f} controlled={d['controlled_coef']:.4f} "
                  f"reduction={d['pct_reduction']:.1f}%")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
