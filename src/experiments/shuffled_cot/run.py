"""Experiment 2: Shuffled-CoT null baseline.

For each question, replace its CoT with a CoT from a different random question
(same task, same model). Compute the same match-fraction metric on the shuffled CoTs.
No new generation needed -- reads Experiment 1 outputs.

Usage:
    uv run -m src.experiments.shuffled_cot.run \
        --core_sweep_results_dir results/core_sweep \
        --output_dir results/shuffled_cot \
        --seed 42
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.generation.runner import QuestionResult
from src.metrics.faithfulness import FaithfulnessResult, bootstrap_ci, compute_match_fraction_per_question
from src.utils.config import save_run_config
from src.utils.io import read_json, read_jsonl, write_json, write_jsonl
from src.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    core_sweep_results_dir: str = ""
    output_dir: str = "results/shuffled_cot"
    seed: int = 42
    n_bootstrap: int = 1000


def make_derangement(n: int, rng: random.Random) -> list[int]:
    """Generate a random derangement (permutation with no fixed points).

    Uses rejection sampling. For n >= 2, expected ~2.7 attempts.
    """
    indices = list(range(n))
    while True:
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if all(shuffled[i] != i for i in range(n)):
            return shuffled


def process_cell(
    cell_dir: Path,
    output_dir: Path,
    seed: int,
    n_bootstrap: int,
) -> FaithfulnessResult | None:
    """Process one (model, dataset) cell."""
    results_path = cell_dir / "generation_results.jsonl"
    if not results_path.exists():
        LOGGER.warning(f"No generation results at {results_path}, skipping")
        return None

    records = read_jsonl(results_path)
    results = [QuestionResult(**r) for r in records]

    if len(results) < 2:
        LOGGER.warning(f"Need at least 2 questions for shuffling, got {len(results)}")
        return None

    model_id = results[0].model_id
    dataset_name = results[0].dataset_name
    n_cot_per_q = len(results[0].cot_samples)

    # Create derangement mapping
    rng = random.Random(seed)
    n = len(results)
    derangement = make_derangement(n, rng)

    # Save mapping for audit
    cell_output = output_dir / cell_dir.name
    cell_output.mkdir(parents=True, exist_ok=True)
    mapping_records = [{"id": results[i].id, "donor_id": results[derangement[i]].id} for i in range(n)]
    write_jsonl(cell_output / "shuffle_mapping.jsonl", mapping_records)

    # Compute shuffled match fractions
    per_question_fractions = []
    n_no_cot_failures = 0
    n_cot_failures = 0

    for i in range(n):
        no_cot_answer = results[i].no_cot_extracted_answer
        # Use CoT answers from the DONOR question
        donor = results[derangement[i]]
        shuffled_cot_answers = [s.extracted_answer for s in donor.cot_samples]
        n_cot_failures += sum(1 for a in shuffled_cot_answers if a is None)

        fraction = compute_match_fraction_per_question(shuffled_cot_answers, no_cot_answer)
        if fraction is None:
            n_no_cot_failures += 1
            continue
        per_question_fractions.append(fraction)

    if not per_question_fractions:
        LOGGER.warning(f"No valid fractions for {model_id}/{dataset_name}")
        return None

    import numpy as np

    mean_frac = float(np.mean(per_question_fractions))
    std_frac = float(np.std(per_question_fractions))
    ci_lower, ci_upper = bootstrap_ci(per_question_fractions, n_bootstrap=n_bootstrap, seed=seed)

    result = FaithfulnessResult(
        model_id=model_id,
        dataset_name=dataset_name,
        n_questions=n,
        n_cot_samples_per_question=n_cot_per_q,
        mean_match_fraction=mean_frac,
        std_match_fraction=std_frac,
        bootstrap_ci_lower=ci_lower,
        bootstrap_ci_upper=ci_upper,
        n_extraction_failures_cot=n_cot_failures,
        n_extraction_failures_no_cot=n_no_cot_failures,
        per_question_fractions=per_question_fractions,
    )

    write_json(cell_output / "faithfulness_shuffled.json", result.model_dump())
    LOGGER.info(
        f"{model_id}/{dataset_name}: shuffled match fraction = {mean_frac:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
    )
    return result


def main():
    config = simple_parsing.parse(Config)
    seed_everything(config.seed)

    core_sweep_dir = Path(config.core_sweep_results_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all cell directories in core_sweep results
    cell_dirs = sorted(
        [d for d in core_sweep_dir.iterdir() if d.is_dir() and (d / "generation_results.jsonl").exists()]
    )
    LOGGER.info(f"Found {len(cell_dirs)} cells in {core_sweep_dir}")

    all_results = []
    for cell_dir in cell_dirs:
        LOGGER.info(f"Processing {cell_dir.name}...")
        result = process_cell(cell_dir, output_dir, config.seed, config.n_bootstrap)
        if result:
            all_results.append(result)

    # Save summary
    save_run_config(output_dir, config, extra_metadata={"n_cells_processed": len(all_results)})
    print(f"\nProcessed {len(all_results)} cells. Results saved to {output_dir}")
    print(
        f"Plot with: uv run -m src.experiments.shuffled_cot.plot "
        f"--core_sweep_results_dir {config.core_sweep_results_dir} "
        f"--shuffled_cot_results_dir {config.output_dir} --output_dir results/figures"
    )


if __name__ == "__main__":
    main()
