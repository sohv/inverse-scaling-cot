# Inverse Scaling of CoT Faithfulness

Replicates and extends the chain-of-thought faithfulness analysis from Lanham et al. (2023) and Bentham et al. (2024). Tests whether CoT faithfulness (measured as the fraction of with-CoT answers matching without-CoT answers) decreases with model scale, using a full scale sweep across 11 open-weight models in two families (Qwen2.5-Instruct and Llama-3-Instruct).

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

For GPU inference (required for Experiment 1 and 6):
```bash
uv pip install -e ".[dev,inference]"
```

## Repo structure

```
src/
├── data/           Dataset loading (5 HF datasets) and fixed-seed sampling
├── generation/     Prompt templates, vLLM engine wrapper, answer extraction
├── metrics/        Faithfulness metric, accuracy, OLS regression
├── utils/          I/O, seeding, model registry, config helpers
└── experiments/    One subpackage per experiment
data/splits/        Fixed question IDs per dataset (generated on first run)
results/            Experiment outputs (generation results, metrics, figures)
tests/              Unit tests for metrics, data loading, templates, extraction
docs/               Experimental design, pre-registered decisions, prompt templates
```

## Running experiments

### Experiment 1: Core metric sweep

Tests whether CoT faithfulness decreases with model scale. Generates 20 CoT samples + 1 no-CoT answer per question per model.

**Input:** 100 questions from each of AQuA, LogiQA, ARC-Challenge, OpenBookQA, HellaSwag (downloaded from HuggingFace on first run).
**Output:** `results/core_sweep/{model}__{dataset}/generation_results.jsonl` -- fields: `id`, `model_id`, `correct_label`, `no_cot_extracted_answer`, `cot_samples[].extracted_answer`
**Output:** `results/core_sweep/{model}__{dataset}/faithfulness.json` -- fields: `mean_match_fraction`, `bootstrap_ci_lower`, `bootstrap_ci_upper`

```bash
# Single cell (test with 10 questions first):
uv run -m src.experiments.core_sweep.run \
    --model_id Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name aqua \
    --output_dir results/core_sweep \
    --n_questions 10 \
    --seed 42

# Full run for one model across all datasets:
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do
    uv run -m src.experiments.core_sweep.run \
        --model_id Qwen/Qwen2.5-0.5B-Instruct \
        --dataset_name "$dataset" \
        --output_dir results/core_sweep \
        --n_questions 100 \
        --seed 42
done

# Plot results:
uv run -m src.experiments.core_sweep.plot \
    --results_dir results/core_sweep \
    --output_dir results/figures
```

### Experiment 2: Shuffled-CoT null baseline

Tests how much of Experiment 1's signal reflects actual CoT conditioning vs answer-distribution skew. No new generation.

**Input:** `results/core_sweep/` (Experiment 1 outputs)
**Output:** `results/shuffled_cot/{model}__{dataset}/faithfulness_shuffled.json`

```bash
uv run -m src.experiments.shuffled_cot.run \
    --core_sweep_results_dir results/core_sweep \
    --output_dir results/shuffled_cot \
    --seed 42

uv run -m src.experiments.shuffled_cot.plot \
    --core_sweep_results_dir results/core_sweep \
    --shuffled_cot_results_dir results/shuffled_cot \
    --output_dir results/figures
```

### Experiment 3: Accuracy-without-CoT logging

Scores no-CoT answers against ground truth. No new generation.

**Input:** `results/core_sweep/` (Experiment 1 outputs)
**Output:** `results/accuracy/{model}__{dataset}/accuracy.json` -- fields: `accuracy`, `n_correct`, `n_questions`

```bash
uv run -m src.experiments.accuracy.run \
    --core_sweep_results_dir results/core_sweep \
    --output_dir results/accuracy \
    --seed 42
```

### Experiment 4: Confound decomposition regression

Tests whether inverse scaling in faithfulness is explained by models already knowing the answer at larger scale.

**Input:** `results/core_sweep/` + `results/accuracy/`
**Output:** `results/regression/decomposition.json`, `results/regression/regression_table.csv`

```bash
uv run -m src.experiments.regression.run \
    --core_sweep_results_dir results/core_sweep \
    --accuracy_results_dir results/accuracy \
    --output_dir results/regression \
    --seed 42

uv run -m src.experiments.regression.plot \
    --regression_table results/regression/regression_table.csv \
    --output_dir results/figures
```

### Experiment 5: FUR cross-method check

```bash
uv run -m src.experiments.fur.run \
    --fur_repo_path /path/to/parametric-faithfulness \
    --core_sweep_results_dir results/core_sweep \
    --output_dir results/fur \
    --seed 42
```

### Experiment 6: Base-model ablation

```bash
uv run -m src.experiments.base_ablation.run \
    --model_id Qwen/Qwen2.5-0.5B \
    --dataset_name aqua \
    --output_dir results/base_ablation \
    --n_questions 100 \
    --seed 42
```

## Testing

```bash
uv run -m pytest tests/ -v -s
```

## Conventions

- `results/raw/` is append-only.
- All thresholds and decisions are pre-registered in `docs/decisions.md`.
- `data/raw/` is read-only.
- Prompt templates are pinned in `src/generation/templates.py` and documented in `docs/prompt_templates.md`.

## References

- Lanham, T., et al. (2023). "Measuring Faithfulness in Chain-of-Thought Reasoning."
- Bentham, J., Stringham, N., & Marasovic, A. (2024). "Chain-of-Thought Unfaithfulness as Disguised Accuracy." arXiv:2402.14897.
