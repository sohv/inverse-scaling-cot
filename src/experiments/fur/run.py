"""Experiment 5: FUR cross-method check.

Runs the FUR (Faithfulness via Unlearning for Reasoning) method on 2 smallest models
and AQuA only, to check if a structurally different faithfulness measure agrees
with Experiment 1's ranking.

Requires the FUR codebase: https://github.com/technion-cs-nlp/parametric-faithfulness
Clone it and pass --fur_repo_path.

Usage:
    uv run -m src.experiments.fur.run \
        --fur_repo_path /path/to/parametric-faithfulness \
        --core_sweep_results_dir results/core_sweep \
        --output_dir results/fur \
        --seed 42

Fallback: If NPO+KL hyperparameter tuning consumes disproportionate time,
this experiment is dropped. State in Limitations with reason.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.utils.config import save_run_config
from src.utils.io import read_json, write_json
from src.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

# Two smallest models to compare
TARGET_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]
TARGET_DATASET = "aqua"


@dataclass
class Config:
    fur_repo_path: str = ""  # path to cloned FUR repo
    core_sweep_results_dir: str = ""
    output_dir: str = "results/fur"
    seed: int = 42


def main():
    config = simple_parsing.parse(Config)
    seed_everything(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config.fur_repo_path or not Path(config.fur_repo_path).exists():
        LOGGER.error(
            "FUR repo path not provided or does not exist. "
            "Clone https://github.com/technion-cs-nlp/parametric-faithfulness "
            "and pass --fur_repo_path."
        )
        return

    # Load core sweep faithfulness rankings for comparison
    core_sweep_rankings = {}
    for model_id in TARGET_MODELS:
        from src.utils.config import normalize_name

        cell_name = f"{normalize_name(model_id)}__{TARGET_DATASET}"
        faith_path = Path(config.core_sweep_results_dir) / cell_name / "faithfulness.json"
        if faith_path.exists():
            data = read_json(faith_path)
            core_sweep_rankings[model_id] = data["mean_match_fraction"]
            LOGGER.info(f"Core sweep faithfulness for {model_id}: {data['mean_match_fraction']:.4f}")
        else:
            LOGGER.warning(f"No core sweep faithfulness found at {faith_path}")

    # FUR integration placeholder
    # The actual FUR code uses a different API and requires model-specific configuration.
    # This section should be adapted to the FUR codebase's actual interface.
    LOGGER.warning(
        "FUR integration is a placeholder. Implement the actual FUR pipeline "
        "using the parametric-faithfulness repo's API. The key steps are:\n"
        "  1. Load model and tokenizer\n"
        "  2. Generate CoT for the 100 AQuA questions\n"
        "  3. Segment CoT into reasoning steps\n"
        "  4. For each step, run NPO+KL unlearning\n"
        "  5. Measure effect on model's prediction (efficacy metric)\n"
        "  6. Compute specificity metric\n"
        "  7. Compare ranking with core sweep"
    )

    fur_results = {
        "status": "placeholder",
        "core_sweep_rankings": core_sweep_rankings,
        "target_models": TARGET_MODELS,
        "target_dataset": TARGET_DATASET,
        "note": "Implement FUR pipeline using parametric-faithfulness repo",
    }

    write_json(output_dir / "fur_results.json", fur_results)
    save_run_config(output_dir, config)

    print(f"\nResults saved to {output_dir}")
    print("NOTE: FUR integration is a placeholder. See the log for implementation steps.")


if __name__ == "__main__":
    main()
