"""Experiment 3: Accuracy-without-CoT logging.

Scores no-CoT answers from Experiment 1 against ground truth.
No new generation needed -- reads Experiment 1 outputs.

Usage:
    uv run -m src.experiments.accuracy.run \
        --core_sweep_results_dir results/core_sweep \
        --output_dir results/accuracy \
        --seed 42
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.generation.runner import QuestionResult
from src.metrics.accuracy import compute_accuracy
from src.utils.config import save_run_config
from src.utils.io import read_jsonl, write_json
from src.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    core_sweep_results_dir: str = ""
    output_dir: str = "results/accuracy"
    seed: int = 42


def main():
    config = simple_parsing.parse(Config)
    seed_everything(config.seed)

    core_sweep_dir = Path(config.core_sweep_results_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all cell directories
    cell_dirs = sorted(
        [d for d in core_sweep_dir.iterdir() if d.is_dir() and (d / "generation_results.jsonl").exists()]
    )
    LOGGER.info(f"Found {len(cell_dirs)} cells in {core_sweep_dir}")

    all_results = []
    for cell_dir in cell_dirs:
        LOGGER.info(f"Processing {cell_dir.name}...")
        records = read_jsonl(cell_dir / "generation_results.jsonl")
        results = [QuestionResult(**r) for r in records]

        accuracy = compute_accuracy(results)
        all_results.append(accuracy)

        # Save per-cell accuracy
        cell_output = output_dir / cell_dir.name
        cell_output.mkdir(parents=True, exist_ok=True)
        write_json(cell_output / "accuracy.json", accuracy.model_dump())

    # Save summary
    save_run_config(output_dir, config, extra_metadata={"n_cells_processed": len(all_results)})

    # Print summary table
    print(f"\nProcessed {len(all_results)} cells. Results saved to {output_dir}")
    print("\nAccuracy summary:")
    print(f"{'Model':<45} {'Dataset':<15} {'Accuracy':>10}")
    print("-" * 72)
    for acc in sorted(all_results, key=lambda a: (a.dataset_name, a.model_id)):
        print(f"{acc.model_id:<45} {acc.dataset_name:<15} {acc.accuracy:>10.4f}")


if __name__ == "__main__":
    main()
