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

## Revision experiments (BlackboxNLP reproducibility response)

Execution plan: `docs/revision_plan.md`. Pre-registered thresholds: `docs/decisions.md`.
Phase 1 runs entirely off the committed generations and needs no GPU.

### Phase 1a: Regression battery

Runs the Model 1 vs Model 2 decomposition on every subset of the 55 cells: per task,
leave-one-task-out, leave-one-model-out, per family, plus residualised, partial-correlation,
quadratic and task-random-effect fits.

**Input:** `results/core_sweep/*/generation_results.jsonl` -- fields: `id`, `model_id`, `correct_label`, `no_cot_extracted_answer`, `cot_samples[].extracted_answer`
**Output:** `results/robustness/cell_table.csv` -- fields: `model_id`, `dataset_name`, `faithfulness_proxy`, `accuracy_no_cot`, `log_params`, `family`, `size_b`, `is_quantized`
**Output:** `results/robustness/{pooled,per_task,leave_one_task_out,leave_one_model_out,per_family,residualized,partial_correlation,mixed_effects,nonlinear}.json` -- each fit reports `raw_coef`, `controlled_coef`, `pct_reduction` and bootstrap CIs

```bash
uv run -m src.experiments.robustness.run \
    --core_sweep_results_dir results/core_sweep \
    --output_dir results/robustness --n_bootstrap 1000 --seed 42

uv run -m src.experiments.robustness.plot \
    --results_dir results/robustness --output_dir results/figures
```

### Phase 1b: Capability matching, regimes and reconstruction

Finds cell pairs matched on no-CoT accuracy but far apart in size, quantifies the three
measurement regimes, and reconstructs the scaling curve from accuracy alone.

**Input:** `results/robustness/cell_table.csv` and `results/core_sweep/`
**Output:** `results/capability/matched_pairs.json` -- fields: `n_pairs`, `mean_abs_proxy_diff`, `predicted_proxy_diff_from_size`, `paired_p_value`, `pairs[]`
**Output:** `results/capability/{capability_bins,reconstruction,answer_distribution_by_bin}.json`, `binned_cells.csv`, `answer_distribution.csv`

```bash
uv run -m src.experiments.capability.run \
    --core_sweep_results_dir results/core_sweep \
    --cell_table results/robustness/cell_table.csv \
    --output_dir results/capability --seed 42

uv run -m src.experiments.capability.plot \
    --results_dir results/capability --output_dir results/figures
```

### Phase 1c: Empirical shuffled-CoT null and CoT dependence

Replaces the assumed 0.25 chance level with a per-cell permutation null, then regresses
CoT dependence (real minus shuffled) on model size.

**Input:** `results/core_sweep/*/generation_results.jsonl`
**Output:** `results/cot_dependence/null_table.csv` -- fields: `model_id`, `dataset_name`, `real_match_rate`, `shuffled_mean`, `null_ci_lower`, `null_ci_upper`, `dependence`, `real_above_null`
**Output:** `results/cot_dependence/{dependence_fits,monotonicity,null_summary}.json`

```bash
uv run -m src.experiments.cot_dependence.run \
    --core_sweep_results_dir results/core_sweep \
    --output_dir results/cot_dependence --n_permutations 1000 --seed 42

uv run -m src.experiments.cot_dependence.plot \
    --results_dir results/cot_dependence --output_dir results/figures
```

### Phase 1d: Extraction robustness

Compares an independent strict parser against the permissive one and recomputes the main
result under the 2x2 parser x failure-treatment grid.

**Input:** `results/core_sweep/*/generation_results.jsonl` -- uses `no_cot_raw_text` and `cot_samples[].final_answer_raw`
**Output:** `results/extraction/parser_agreement.csv` -- fields: `model_id`, `dataset_name`, `agreement_rate`, `n_permissive_only`, `n_strict_only`, `n_different_letter`
**Output:** `results/extraction/treatment_grid.json`, `manual_audit_sample.jsonl` (200 stratified disagreements with a blank `human_label` field for manual audit)

```bash
uv run -m src.experiments.extraction.run \
    --core_sweep_results_dir results/core_sweep \
    --output_dir results/extraction --n_manual_sample 200 --seed 42
```

### Phase 2a: Sample-count convergence (100 CoT samples)

Regenerates a fresh pool of 100 CoT samples per question, then subsamples it nestedly to
compare k=20, 50 and 100. Requires GPU.

**Input:** same fixed question splits as Experiment 1
**Output:** `results/samples_100/{model}__{dataset}/generation_results.jsonl`

```bash
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do
    uv run -m src.experiments.core_sweep.run \
        --model_id Qwen/Qwen2.5-7B-Instruct --dataset_name "$dataset" \
        --output_dir results/samples_100 \
        --n_questions 100 --n_cot_samples 100 --seed 42
done
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
