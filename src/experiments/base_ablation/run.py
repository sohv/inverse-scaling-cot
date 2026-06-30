"""Experiment 6: Base-model ablation.

Same as Experiment 1 but on base (non-instruct) checkpoints with few-shot prompts.
Tests whether the inverse scaling pattern requires instruction tuning.

Uses Qwen2.5-base checkpoints at all 7 sizes where available.

Usage:
    uv run -m src.experiments.base_ablation.run \
        --model_id Qwen/Qwen2.5-0.5B \
        --dataset_name aqua \
        --output_dir results/base_ablation \
        --n_questions 100 \
        --seed 42

Fallback: If base checkpoints aren't available for enough sizes, or if few-shot
prompting is too unreliable (>20% extraction failures), this experiment is dropped.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.data.sampler import load_or_sample_questions
from src.generation.engine import GenerationConfig, VLLMEngine
from src.generation.runner import (
    CoTSample,
    QuestionResult,
    extract_answer,
    extract_answer_no_cot,
    save_generation_results,
)
from src.generation.templates import build_few_shot_cot_prompt, build_few_shot_no_cot_prompt
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
    output_dir: str = "results/base_ablation"
    splits_dir: str = "data/splits"
    n_questions: int = 100
    n_cot_samples: int = 20
    n_few_shot: int = 2
    seed: int = 42
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens_cot: int = 1024
    max_tokens_no_cot: int = 20
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int | None = None


def main():
    config = simple_parsing.parse(Config)
    seed_everything(config.seed)

    LOGGER.info(f"Experiment 6 (base model): {config.model_id} / {config.dataset_name}")

    # 1. Load questions (reuse same splits as Exp 1)
    questions = load_or_sample_questions(
        config.dataset_name,
        splits_dir=config.splits_dir,
        n=config.n_questions,
        seed=config.seed,
    )

    # 2. Initialize vLLM engine
    engine = VLLMEngine(
        model_id=config.model_id,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
    )

    # 3. Build few-shot prompts (plain text, no chat template)
    cot_prompts = [build_few_shot_cot_prompt(q, n_shots=config.n_few_shot) for q in questions]
    no_cot_prompts = [build_few_shot_no_cot_prompt(q, n_shots=config.n_few_shot) for q in questions]

    # 4. Generate CoT samples
    LOGGER.info("Generating CoT samples with few-shot prompts...")
    cot_gen_config = GenerationConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens_cot,
        n=config.n_cot_samples,
        seed=config.seed,
    )
    cot_outputs = engine.generate_text(cot_prompts, cot_gen_config)

    # 5. Generate no-CoT answers
    LOGGER.info("Generating no-CoT answers with few-shot prompts...")
    no_cot_gen_config = GenerationConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens_no_cot,
        n=1,
        seed=config.seed,
    )
    no_cot_outputs = engine.generate_text(no_cot_prompts, no_cot_gen_config)

    # 6. Extract answers and assemble results
    results = []
    n_cot_failures = 0
    n_no_cot_failures = 0

    for q_idx, question in enumerate(questions):
        no_cot_raw = no_cot_outputs[q_idx][0]
        no_cot_answer = extract_answer_no_cot(no_cot_raw, question.choice_labels)
        if no_cot_answer is None:
            n_no_cot_failures += 1

        cot_samples = []
        for s_idx, cot_text in enumerate(cot_outputs[q_idx]):
            extracted = extract_answer(cot_text, question.choice_labels)
            if extracted is None:
                n_cot_failures += 1
            cot_samples.append(
                CoTSample(
                    id=question.id,
                    sample_idx=s_idx,
                    cot_text=cot_text,
                    final_answer_raw=cot_text,
                    extracted_answer=extracted,
                )
            )

        results.append(
            QuestionResult(
                id=question.id,
                dataset_name=question.dataset_name,
                model_id=config.model_id,
                correct_label=question.correct_label,
                no_cot_raw_text=no_cot_raw,
                no_cot_extracted_answer=no_cot_answer,
                cot_samples=cot_samples,
            )
        )

    total_cot = len(questions) * config.n_cot_samples
    cot_failure_rate = n_cot_failures / total_cot * 100 if total_cot > 0 else 0
    no_cot_failure_rate = n_no_cot_failures / len(questions) * 100 if questions else 0

    LOGGER.info(
        f"Extraction failures: CoT={n_cot_failures}/{total_cot} ({cot_failure_rate:.1f}%), "
        f"no-CoT={n_no_cot_failures}/{len(questions)} ({no_cot_failure_rate:.1f}%)"
    )

    if cot_failure_rate > 20 or no_cot_failure_rate > 20:
        LOGGER.warning(
            f"High extraction failure rate ({cot_failure_rate:.1f}% CoT, {no_cot_failure_rate:.1f}% no-CoT). "
            "Few-shot prompting on base models may be unreliable. Consider dropping this experiment."
        )

    # 7. Save results
    cell_dir = make_output_dir(config.output_dir, config.model_id, config.dataset_name)
    save_generation_results(results, cell_dir)
    save_run_config(
        cell_dir,
        config,
        extra_metadata={
            "prompt_type": "few_shot",
            "n_few_shot": config.n_few_shot,
            "cot_extraction_failure_rate": round(cot_failure_rate, 4),
            "no_cot_extraction_failure_rate": round(no_cot_failure_rate, 4),
        },
    )

    faithfulness = compute_faithfulness(results, n_bootstrap=1000, seed=config.seed)
    write_json(cell_dir / "faithfulness.json", faithfulness.model_dump())

    print(f"\nResults saved to {cell_dir}")
    print(
        f"Faithfulness: {faithfulness.mean_match_fraction:.4f} "
        f"[{faithfulness.bootstrap_ci_lower:.4f}, {faithfulness.bootstrap_ci_upper:.4f}]"
    )


if __name__ == "__main__":
    main()
