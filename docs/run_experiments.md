# Experiment Run Order

All commands are run from the repo root. Experiments 2–4 read from prior outputs — no
new model inference is needed after Experiment 1.

---

## Prerequisites

```bash
# Install base deps
uv sync

# Install GPU / inference stack (on the GPU node)
uv sync --extra inference

# Optional: authenticate with HuggingFace to avoid rate-limits
export HF_TOKEN=hf_...

# Optional: point HF cache at fast storage
export HF_HOME=/path/to/fast/storage
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/sohv/inverse-scaling-cot.git
cd inverse-scaling-cot
```

**2. Install uv**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

**3. Install all dependencies including inference stack**
```bash
uv sync --extra inference
```

**4. Set HuggingFace token (avoids rate limits + needed for Llama gated models)**
```bash
export HF_TOKEN=hf_...
```

**5. (Optional) Point HF cache at fast/persistent storage**
```bash
export HF_HOME=/path/to/fast/storage
```

**6. Run tests to verify everything is working**
```bash
uv run -m pytest tests/ -v
```

Then follow run_experiments.md for the experiment commands.

> **Note:** Llama models (`meta-llama/...`) are gated on HuggingFace — you need to accept the license on the HF website for each model before they'll download. Your `HF_TOKEN` must belong to an account that has done this.



## Experiment 1 — Core Sweep  *(requires GPU)*

Generates 20 CoT samples + 1 no-CoT answer per question for every (model, dataset)
cell. 55 cells total: 11 models × 5 datasets, 100 questions each.

Run one model at a time (loop over all 5 datasets per model):

```bash
# ── Qwen2.5-Instruct ────────────────────────────────────────────────────────
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id Qwen/Qwen2.5-0.5B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id Qwen/Qwen2.5-1.5B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id Qwen/Qwen2.5-3B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id Qwen/Qwen2.5-7B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id Qwen/Qwen2.5-14B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id Qwen/Qwen2.5-32B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id Qwen/Qwen2.5-72B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done

# ── Llama-3-Instruct ────────────────────────────────────────────────────────
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id meta-llama/Llama-3.2-1B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id meta-llama/Llama-3.2-3B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id meta-llama/Llama-3.1-8B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.core_sweep.run --model_id meta-llama/Llama-3.1-70B-Instruct --dataset_name "$dataset" --output_dir results/core_sweep --n_questions 100 --n_cot_samples 20 --seed 42; done
```

**Output:** `results/core_sweep/{model}__{dataset}/generation_results.jsonl` +
`faithfulness.json` per cell.

---

## Experiment 2 — Shuffled-CoT Null Baseline  *(no GPU needed)*

Shuffles CoT labels across questions to produce a null baseline for faithfulness.
Reads Experiment 1 outputs only.

```bash
uv run -m src.experiments.shuffled_cot.run --core_sweep_results_dir results/core_sweep --output_dir results/shuffled_cot --seed 42
```

**Output:** `results/shuffled_cot/{model}__{dataset}/faithfulness_shuffled.json`

---

## Experiment 3 — Accuracy Without CoT  *(no GPU needed)*

Scores no-CoT answers against ground truth. Reads Experiment 1 outputs only.

```bash
uv run -m src.experiments.accuracy.run --core_sweep_results_dir results/core_sweep --output_dir results/accuracy --seed 42
```

**Output:** `results/accuracy/{model}__{dataset}/accuracy.json`

---

## Experiment 4 — Confound Decomposition Regression  *(no GPU needed)*

Merges Experiments 1 + 3 into a 55-row table (11 models × 5 datasets) and runs
OLS with cluster-robust standard errors.

```bash
uv run -m src.experiments.regression.run --core_sweep_results_dir results/core_sweep --accuracy_results_dir results/accuracy --output_dir results/regression --seed 42
```

**Output:** `results/regression/regression_table.csv` + `decomposition.json`

---

## Plots

Run after the corresponding experiment completes.

```bash
# Experiment 1 — Faithfulness vs model size (main figure)
uv run -m src.experiments.core_sweep.plot --results_dir results/core_sweep --output_dir results/figures

# Experiment 2 — Real vs shuffled CoT overlay
uv run -m src.experiments.shuffled_cot.plot --core_sweep_results_dir results/core_sweep --shuffled_cot_results_dir results/shuffled_cot --output_dir results/figures

# Experiment 4 — Faithfulness vs accuracy scatter
uv run -m src.experiments.regression.plot --regression_table results/regression/regression_table.csv --output_dir results/figures
```

**Output:** `results/figures/`

---

## Optional: Experiment 5 — FUR Cross-Method Check  *(requires GPU)*

Runs an independent faithfulness measure (FUR) on the 2 smallest models × AQuA
only. Requires cloning an external repo first.

```bash
git clone https://github.com/technion-cs-nlp/parametric-faithfulness /path/to/fur
uv run -m src.experiments.fur.run --fur_repo_path /path/to/fur --core_sweep_results_dir results/core_sweep --output_dir results/fur --seed 42
```

> **Fallback:** drop if NPO+KL hyperparameter tuning is disproportionately costly;
> note in Limitations.

**Output:** `results/fur/fur_results.json`

---

## Optional: Experiment 6 — Base-Model Ablation  *(requires GPU)*

Replicates Experiment 1 on non-instruct Qwen2.5 base checkpoints with few-shot
prompts, to test whether instruction tuning is necessary for the inverse-scaling
pattern.

```bash
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do uv run -m src.experiments.base_ablation.run --model_id Qwen/Qwen2.5-0.5B --dataset_name "$dataset" --output_dir results/base_ablation --n_questions 100 --seed 42; done
# Repeat for: Qwen2.5-1.5B  3B  7B  14B  32B  72B
```

> **Fallback:** drop if >20% answer-extraction failures on few-shot prompts; note
> in Limitations.

**Output:** `results/base_ablation/{model}__{dataset}/generation_results.jsonl` +
`faithfulness.json`

