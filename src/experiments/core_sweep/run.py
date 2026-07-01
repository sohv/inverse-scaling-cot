"""Experiment 1: Core metric sweep.

Generates 20 CoT samples + 1 no-CoT answer per question per model.
Runs one (model, dataset) cell per invocation. Shell loop handles all 55 cells.

Usage:
    uv run -m src.experiments.core_sweep.run \
        --model_id Qwen/Qwen2.5-0.5B-Instruct \
        --dataset_name aqua \
        --output_dir results/core_sweep \
        --n_questions 100 \
        --n_cot_samples 20 \
        --seed 42

For testing (10 questions):
    uv run -m src.experiments.core_sweep.run \
        --model_id Qwen/Qwen2.5-0.5B-Instruct \
        --dataset_name aqua \
        --output_dir results/core_sweep_test \
        --n_questions 10 \
        --seed 42
"""

import logging
from dataclasses import dataclass

import simple_parsing

from src.data.sampler import load_or_sample_questions
from src.generation.engine import GenerationConfig, HFEngine, VLLMEngine
from src.generation.runner import run_generation_for_model, save_generation_results
from src.metrics.faithfulness import compute_faithfulness
from src.utils.config import make_output_dir, save_run_config
from src.utils.io import write_json
from src.utils.seed import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    model_id: str = ""
    dataset_name: str = ""
    output_dir: str = "results/core_sweep"
    splits_dir: str = "data/splits"
    n_questions: int = 100
    n_cot_samples: int = 20
    seed: int = 42
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens_cot: int = 1024
    max_tokens_no_cot: int = 20
    engine: str = "vllm"  # "vllm" or "hf"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int | None = None
    enforce_eager: bool = False
    quantization: str | None = None


def main():
    config = simple_parsing.parse(Config)
    seed_everything(config.seed)

    LOGGER.info(f"Experiment 1: {config.model_id} / {config.dataset_name}")

    # 1. Load or sample questions
    questions = load_or_sample_questions(
        config.dataset_name,
        splits_dir=config.splits_dir,
        n=config.n_questions,
        seed=config.seed,
    )
    LOGGER.info(f"Loaded {len(questions)} questions for {config.dataset_name}")

    # 2. Initialize engine
    if config.engine == "hf":
        engine = HFEngine(model_id=config.model_id)
    else:
        engine = VLLMEngine(
            model_id=config.model_id,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enforce_eager=config.enforce_eager,
            quantization=config.quantization,
        )

    # 3. Configure generation
    cot_gen_config = GenerationConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens_cot,
        n=config.n_cot_samples,
        seed=config.seed,
    )
    no_cot_gen_config = GenerationConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens_no_cot,
        n=1,
        seed=config.seed,
    )

    # 4. Run generation
    results = run_generation_for_model(
        engine=engine,
        questions=questions,
        model_id=config.model_id,
        n_cot_samples=config.n_cot_samples,
        cot_gen_config=cot_gen_config,
        no_cot_gen_config=no_cot_gen_config,
    )

    # 5. Save generation results
    cell_dir = make_output_dir(config.output_dir, config.model_id, config.dataset_name)
    save_generation_results(results, cell_dir)
    save_run_config(cell_dir, config)

    # 6. Compute and save faithfulness metric
    faithfulness = compute_faithfulness(results, n_bootstrap=1000, seed=config.seed)
    faith_path = cell_dir / "faithfulness.json"
    write_json(faith_path, faithfulness.model_dump())

    print(f"\nResults saved to {cell_dir}")
    print(
        f"Faithfulness: {faithfulness.mean_match_fraction:.4f} "
        f"[{faithfulness.bootstrap_ci_lower:.4f}, {faithfulness.bootstrap_ci_upper:.4f}]"
    )
    print(
        f"\nPlot with: uv run -m src.experiments.core_sweep.plot "
        f"--results_dir {config.output_dir} --output_dir results/figures"
    )


if __name__ == "__main__":
    main()
