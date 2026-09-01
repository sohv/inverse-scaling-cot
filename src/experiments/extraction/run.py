"""Revision Phase 1d: answer-extraction robustness and failure-treatment sensitivity. No GPU.

uv run -m src.experiments.extraction.run --core_sweep_results_dir results/core_sweep --output_dir results/extraction --n_manual_sample 200 --seed 42
"""

import collections
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import simple_parsing

from src.data.sampler import choice_labels_by_id
from src.generation.extraction_alt import extract_answer_strict
from src.generation.runner import QuestionResult, extract_answer_no_cot
from src.metrics.cells import Variant, cells_to_table, find_cell_dirs, load_cells
from src.metrics.robustness import fit_decomposition
from src.utils.config import save_run_config
from src.utils.io import write_json, write_jsonl
from src.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)

AGREEMENT_WARN_THRESHOLD = 0.05


@dataclass
class Config:
    core_sweep_results_dir: str = ""
    output_dir: str = "results/extraction"
    n_manual_sample: int = 200
    n_bootstrap: int = 1000
    seed: int = 42


def compare_parsers(core_sweep_dir: str, seed: int, n_manual_sample: int) -> tuple[list[dict], list[dict]]:
    """Per-cell agreement between the permissive and strict parsers, plus an audit sample."""
    labels_cache: dict[str, dict[str, list[str]]] = {}
    rows = []
    audit_pool = []

    for cell_dir in find_cell_dirs(core_sweep_dir):
        results = [
            QuestionResult(**r)
            for r in pd.read_json(cell_dir / "generation_results.jsonl", lines=True).to_dict("records")
        ]
        dataset_name = results[0].dataset_name
        if dataset_name not in labels_cache:
            labels_cache[dataset_name] = choice_labels_by_id(dataset_name)
        labels = labels_cache[dataset_name]

        counts = collections.Counter()
        for r in results:
            q_labels = labels[r.id]
            items = [("no_cot", r.no_cot_raw_text)] + [("cot", s.final_answer_raw) for s in r.cot_samples]
            for kind, raw in items:
                permissive = extract_answer_no_cot(raw, q_labels)
                strict = extract_answer_strict(raw, q_labels)
                counts["total"] += 1
                if permissive == strict:
                    counts["agree"] += 1
                else:
                    counts["disagree"] += 1
                    if permissive is not None and strict is None:
                        counts["permissive_only"] += 1
                    elif permissive is None and strict is not None:
                        counts["strict_only"] += 1
                    else:
                        counts["different_letter"] += 1
                    audit_pool.append(
                        {
                            "model_id": r.model_id,
                            "dataset_name": dataset_name,
                            "question_id": r.id,
                            "kind": kind,
                            "raw_text": raw[:300],
                            "permissive_answer": permissive,
                            "strict_answer": strict,
                            "human_label": "",
                        }
                    )
                if permissive is None:
                    counts["permissive_failures"] += 1
                if strict is None:
                    counts["strict_failures"] += 1

        rows.append(
            {
                "model_id": results[0].model_id,
                "dataset_name": dataset_name,
                "n_extractions": counts["total"],
                "agreement_rate": counts["agree"] / counts["total"],
                "disagreement_rate": counts["disagree"] / counts["total"],
                "n_permissive_only": counts["permissive_only"],
                "n_strict_only": counts["strict_only"],
                "n_different_letter": counts["different_letter"],
                "permissive_failure_rate": counts["permissive_failures"] / counts["total"],
                "strict_failure_rate": counts["strict_failures"] / counts["total"],
            }
        )
        LOGGER.info(f"{cell_dir.name}: agreement={rows[-1]['agreement_rate']:.4f}")

    rng = random.Random(seed)
    audit = rng.sample(audit_pool, min(n_manual_sample, len(audit_pool))) if audit_pool else []
    return rows, audit


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

    agreement_rows, audit = compare_parsers(config.core_sweep_results_dir, config.seed, config.n_manual_sample)
    agreement = pd.DataFrame(agreement_rows)
    agreement.round(4).to_csv(output_dir / "parser_agreement.csv", index=False)
    if audit:
        write_jsonl(output_dir / "manual_audit_sample.jsonl", audit)

    worst = agreement.loc[agreement.disagreement_rate.idxmax()]
    write_json(
        output_dir / "parser_agreement_summary.json",
        {
            "n_cells": len(agreement),
            "total_extractions": int(agreement.n_extractions.sum()),
            "mean_agreement_rate": float(agreement.agreement_rate.mean()),
            "min_agreement_rate": float(agreement.agreement_rate.min()),
            "max_disagreement_rate": float(agreement.disagreement_rate.max()),
            "worst_cell": f"{worst.model_id}__{worst.dataset_name}",
            "n_cells_above_warn_threshold": int((agreement.disagreement_rate > AGREEMENT_WARN_THRESHOLD).sum()),
            "warn_threshold": AGREEMENT_WARN_THRESHOLD,
            "mean_permissive_failure_rate": float(agreement.permissive_failure_rate.mean()),
            "mean_strict_failure_rate": float(agreement.strict_failure_rate.mean()),
            "n_audit_samples": len(audit),
        },
    )

    variants = [
        Variant(n_samples=None, parser=p, failure_mode=f)
        for p in ["permissive", "strict"]
        for f in ["non_match", "exclude"]
    ]
    cells = load_cells(config.core_sweep_results_dir, variants=variants, n_bootstrap=0, seed=config.seed)

    grid = []
    for v in variants:
        df = cells_to_table(cells[v.key])
        fit = fit_decomposition(df, v.key, config.n_bootstrap, config.seed)
        grid.append(
            {
                "parser": v["parser"],
                "failure_mode": v["failure_mode"],
                "mean_proxy": float(df.faithfulness_proxy.mean()),
                "mean_accuracy": float(df.accuracy_no_cot.mean()),
                **fit.model_dump(),
            }
        )
    write_json(output_dir / "treatment_grid.json", {"grid": grid})
    save_run_config(output_dir, config, extra_metadata={"n_cells": len(agreement)})

    print(
        f"\nParser agreement across {int(agreement.n_extractions.sum()):,} extractions: "
        f"mean={agreement.agreement_rate.mean():.4f}, worst cell={worst.disagreement_rate:.4f} "
        f"({worst.model_id}__{worst.dataset_name})"
    )
    print(
        f"Cells above the pre-registered {AGREEMENT_WARN_THRESHOLD:.0%} disagreement threshold: "
        f"{int((agreement.disagreement_rate > AGREEMENT_WARN_THRESHOLD).sum())}/{len(agreement)}"
    )
    print(
        f"Extraction failure rate: permissive={agreement.permissive_failure_rate.mean():.4f} "
        f"strict={agreement.strict_failure_rate.mean():.4f}"
    )

    print("\nExtraction treatment grid")
    print(f"{'parser':<12} {'failures':<12} {'mean proxy':>11} {'raw beta':>10} {'ctl beta':>10} {'reduction':>11}")
    for g in grid:
        print(
            f"{g['parser']:<12} {g['failure_mode']:<12} {g['mean_proxy']:>11.4f} {g['raw_coef']:>10.4f} "
            f"{g['controlled_coef']:>10.4f} {g['pct_reduction']:>10.1f}%"
        )
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
