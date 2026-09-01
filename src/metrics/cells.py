"""Build (model, dataset) cell tables from stored generations under measurement variants.

Every revision regression -- per-task, leave-one-out, per-family, sample-count,
extraction-treatment -- operates on a table produced here, so the merge logic lives in
one place instead of being duplicated per analysis.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.sampler import choice_labels_by_id
from src.generation.extraction_alt import extract_answer_strict
from src.generation.runner import QuestionResult
from src.metrics.recompute import CellMetrics, recompute_cell
from src.utils.io import read_jsonl
from src.utils.models import get_model_info

LOGGER = logging.getLogger(__name__)

PARSERS: dict[str, Callable[[str, list[str]], str | None] | None] = {
    "permissive": None,  # reuse answers extracted at generation time
    "strict": extract_answer_strict,
}


class Variant(dict):
    """One measurement variant: n_samples, parser name, failure mode."""

    @property
    def key(self) -> str:
        n = self.get("n_samples") or "all"
        return f"k{n}__{self.get('parser', 'permissive')}__{self.get('failure_mode', 'non_match')}"


def default_variant() -> Variant:
    return Variant(n_samples=None, parser="permissive", failure_mode="non_match")


def find_cell_dirs(results_dir: str | Path) -> list[Path]:
    results_dir = Path(results_dir)
    return sorted(d for d in results_dir.iterdir() if d.is_dir() and (d / "generation_results.jsonl").exists())


def load_cells(
    results_dir: str | Path,
    variants: list[Variant] | None = None,
    n_bootstrap: int = 0,
    seed: int = 42,
) -> dict[str, list[CellMetrics]]:
    """Compute metrics for every cell under every variant, reading each cell once.

    Returns {variant.key: [CellMetrics, ...]}.
    """
    if variants is None:
        variants = [default_variant()]

    labels_cache: dict[str, dict[str, list[str]]] = {}
    out: dict[str, list[CellMetrics]] = {v.key: [] for v in variants}

    for cell_dir in find_cell_dirs(results_dir):
        records = read_jsonl(cell_dir / "generation_results.jsonl")
        results = [QuestionResult(**r) for r in records]
        dataset_name = results[0].dataset_name
        if dataset_name not in labels_cache:
            labels_cache[dataset_name] = choice_labels_by_id(dataset_name)
        labels = labels_cache[dataset_name]

        for v in variants:
            parser_name = v.get("parser", "permissive")
            metrics = recompute_cell(
                results,
                labels_by_id=labels,
                n_samples=v.get("n_samples"),
                parser=PARSERS[parser_name],
                parser_name=parser_name,
                failure_mode=v.get("failure_mode", "non_match"),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            out[v.key].append(metrics)
        LOGGER.info(f"processed {cell_dir.name}")

    return out


def is_quantized(model_id: str) -> bool:
    """True for AWQ / INT4 / GPTQ checkpoints."""
    upper = model_id.upper()
    return any(tag in upper for tag in ("AWQ", "INT4", "GPTQ", "-FP8"))


def cells_to_table(
    cells: list[CellMetrics],
    exclude_models: list[str] | None = None,
    exclude_tasks: list[str] | None = None,
    exclude_quantized: bool = False,
) -> pd.DataFrame:
    """Assemble CellMetrics into the regression table, applying exclusions."""
    exclude_models = set(exclude_models or [])
    exclude_tasks = set(exclude_tasks or [])

    rows = []
    for c in cells:
        if c.model_id in exclude_models or c.dataset_name in exclude_tasks:
            continue
        if exclude_quantized and is_quantized(c.model_id):
            continue
        info = get_model_info(c.model_id)
        rows.append(
            {
                "model_id": c.model_id,
                "dataset_name": c.dataset_name,
                "faithfulness_proxy": c.proxy,
                "proxy_ci_lower": c.proxy_ci_lower,
                "proxy_ci_upper": c.proxy_ci_upper,
                "accuracy_no_cot": c.accuracy_no_cot,
                "log_params": float(np.log10(info.size_b * 1e9)),
                "size_b": info.size_b,
                "family": info.family,
                "family_is_llama": 1 if info.family == "llama" else 0,
                "task_cluster": c.dataset_name,
                "is_quantized": int(is_quantized(c.model_id)),
                "n_cot_samples_used": c.n_cot_samples_used,
                "n_cot_extraction_failures": c.n_cot_extraction_failures,
                "n_no_cot_extraction_failures": c.n_no_cot_extraction_failures,
            }
        )

    df = pd.DataFrame(rows)
    df["accuracy_no_cot_sq"] = df["accuracy_no_cot"] ** 2
    df = df.sort_values(["dataset_name", "family", "size_b"]).reset_index(drop=True)
    LOGGER.info(
        f"built cell table with {len(df)} rows, {df.model_id.nunique()} models, {df.dataset_name.nunique()} tasks"
    )
    return df
