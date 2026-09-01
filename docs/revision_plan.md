# BlackboxNLP 2026 revision — experiment execution plan

Covers all 21 items of the reviewer-response plan. Ordered by dependency and cost, not by
the order they appear in the review. Every phase writes to a new `results/` subtree;
existing directories are append-only and never edited.

Hardware: 2x H100 80GB (both idle). Disk: 5.4T free. HF cache: empty.

---

## Phase 0 — Unblock the environment (required before anything)

Nothing in the repo runs today: `uv` is not on PATH, there is no `.venv`, and the
HuggingFace cache is empty.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync --extra inference
export HF_TOKEN=hf_...            # Llama checkpoints are gated
export HF_HOME=/home/ubuntu/hf_cache
uv run -m pytest tests/ -v
```

Pin `vllm` to an exact version in `pyproject.toml` before the first generation run. The
current spec is `vllm>=0.6.0`; sampling behaviour changed across minor versions and every
Phase 2-4 run must be reproducible against one version. Record it in `config.json` (add
`vllm.__version__` to `save_run_config`'s `extra_metadata`).

**Pre-register before Phase 1.** Several Phase 1 analyses have free parameters that must
be fixed before looking at results. Append to `docs/decisions.md`, dated, then commit:

- Capability-matched pairs: no-CoT accuracy tolerance `|dacc| <= 0.03`, minimum size ratio
  `>= 4x`, matched within (dataset, family).
- Capability bins for the three-regime analysis: bin edges on no-CoT accuracy at
  `[0, 0.35, 0.75, 1.0]` (near-random / intermediate / ceiling).
- Empirical shuffled null: 1000 derangements per cell, seed 42, report mean and 2.5/97.5
  percentiles.
- Leave-one-model-out and leave-one-task-out report min / median / max percent reduction;
  no per-run significance testing.
- Strict-parser agreement threshold: if the alternative parser disagrees with the current
  one on >5% of samples, the discrepancy is investigated before any metric is recomputed.

---

## Shared infrastructure to write first

Four new modules unlock nearly all of Phase 1 and are reused by Phases 2-4. Write and test
these before running any analysis.

**`src/metrics/recompute.py`** — the linchpin. Recomputes the faithfulness proxy from a
stored `generation_results.jsonl` under three orthogonal knobs, without touching a GPU:

```python
def recompute_faithfulness(
    records: list[QuestionResult],
    n_samples: int | None = None,        # use first k CoT samples (nested subsampling)
    parser: Callable = extract_answer_no_cot,
    failure_mode: str = "non_match",     # "non_match" | "exclude"
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> FaithfulnessResult
```

This single function serves item 1 (20/50/100 subsampling), item 17 (alternative parser)
and item 18 (failure treatments).

**`src/metrics/cells.py`** — builds a regression table from any results directory with
filters, replacing the hardcoded merge in `build_regression_table`:

```python
def build_cell_table(
    core_sweep_dir, accuracy_dir, *,
    exclude_models: list[str] = [], exclude_tasks: list[str] = [],
    n_samples: int | None = None, parser=..., failure_mode="non_match",
) -> pd.DataFrame
```

Every leave-one-out, per-task, per-family and sample-count variant is then one call.

**`src/metrics/robustness.py`** — the regression battery operating on a cell table:
per-task fits, leave-one-task-out, leave-one-model-out, per-family fits, residualization,
partial correlation, mixed effects (`statsmodels.MixedLM`, `proxy ~ log_params + acc` with
`groups=task`), and the capability-based reconstruction. Every estimate returns a
bootstrap CI from the existing `_bootstrap_coefs` pattern.

**`src/generation/extraction_alt.py`** — an independent strict parser. Anchors only on
`^\s*([A-E])\)` at the start of the completion plus an exact-match
`the answer is \(([A-E])\)` form, with no standalone-`(A)` fallback and no first-character
heuristic. Deliberately written to fail loudly rather than guess, so disagreement with the
permissive parser is measurable.

---

## Phase 1 — Zero-GPU analyses on the existing 55 cells

Everything here runs from files already in `results/core_sweep/` and `results/accuracy/`.
Total cost: minutes of CPU. This is roughly two thirds of the reviewer's list and should be
completed and committed before a single GPU-hour is spent.

Covers items 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, plus the mixed-effects
sensitivity analysis.

### 1a. Regression battery — `src/experiments/robustness/`

```bash
uv run -m src.experiments.robustness.run \
    --core_sweep_results_dir results/core_sweep \
    --accuracy_results_dir results/accuracy \
    --output_dir results/robustness --seed 42
```

Produces `results/robustness/`:

| File | Contents | Review item |
|---|---|---|
| `per_task.json` | 5 fits: raw beta, controlled beta, % reduction, bootstrap CIs | 3 |
| `leave_one_task_out.json` | 5 fits, one task dropped each | 4 |
| `leave_one_model_out.json` | 11 fits + min/median/max reduction, influential checkpoints | 5 |
| `per_family.json` | Qwen-only (35 cells), Llama-only (20 cells) | 6 |
| `residualized.json` | Residual slope vs log_params + bootstrap CI, pooled and per task | 8 |
| `partial_correlation.json` | Partial r(log_params, proxy \| acc), pooled / per task / per family | 9 |
| `mixed_effects.json` | `proxy ~ log_params + acc + (1\|task)`, random-intercept and random-slope | (sensitivity) |
| `nonlinear.json` | Existing quadratic check, promoted out of `decomposition.json` | 19 |

Plots: `residualized_scaling.png` (item 8, replaces the current `residual_vs_params`),
`per_task_reduction.png` (forest plot of the five per-task % reductions with CIs — this
becomes a main-paper figure), `loo_reduction.png`.

The leave-one-model-out run must be repeated after Phase 3 adds BF16 checkpoints.

### 1b. Capability-matched comparison and regime quantification — `src/experiments/capability/`

```bash
uv run -m src.experiments.capability.run \
    --cell_table results/robustness/cell_table.csv \
    --output_dir results/capability --seed 42
```

- **Matched pairs** (item 10): enumerate all (cell_i, cell_j) within the same dataset and
  family with `|dacc| <= 0.03` and size ratio `>= 4x`. Report n pairs, mean `|dproxy|`,
  mean `dlog_params`, and a paired test. Expected headline: a 7B and a 32B at equal
  no-CoT accuracy have near-equal proxy despite a >4x size gap.
- **Capability bins** (item 11): within each of the three pre-registered accuracy bins,
  report mean proxy, proxy-vs-log_params slope, and proxy-vs-accuracy slope. Quantifies
  the near-random / flat / ceiling taxonomy so it stops reading as post-hoc.
- **Position-distribution diagnostics** for the near-random bin: entropy of the no-CoT
  answer-letter distribution and max-letter share per cell. This is the quantitative
  version of the paper's current "five of the first eight AQuA questions receive D".
- **Ceiling saturation**: fit proxy vs accuracy separately above and below acc=0.75 and
  report the slope change.
- **Reconstruction** (item 20): fit `proxy ~ acc` pooled, predict each model's proxy from
  its observed no-CoT accuracy alone, and overlay predicted-vs-size on observed-vs-size.
  Report fraction of the observed scaling range reproduced. Figure:
  `reconstruction.png`. Frame as reconstruction, never as causal decomposition.

### 1c. CoT dependence and empirical null — `src/experiments/cot_dependence/`

```bash
uv run -m src.experiments.cot_dependence.run \
    --core_sweep_results_dir results/core_sweep \
    --output_dir results/cot_dependence --n_permutations 1000 --seed 42
```

- **Empirical null** (item 13): extend the current single-derangement approach in
  `shuffled_cot/run.py` to 1000 derangements per cell. Per cell report real match rate,
  mean shuffled rate, empirical 2.5/97.5 percentile interval, and the real-minus-shuffled
  difference with its position relative to the null interval. This replaces the assumed
  0.25 chance level and is cheap — the CoT texts are already on disk.
- **CoT dependence regression** (item 12): define `dependence = real - shuffled` per cell,
  regress on `log_params` pooled, per task, and per family. Report slope + bootstrap CI.

**Fix the sign convention while doing this.** The current Table 2 labels `Mean D` as
real-minus-shuffled but reports negative values, so it is actually shuffled-minus-real.
Standardise on `dependence = real - shuffled` (positive, increasing with scale)
everywhere.

Also recheck the monotonicity claims here. Recomputing from the committed cell table, 3 of
10 family-task curves are **not** monotone (Qwen AQuA 32B 0.622 -> 72B 0.610; Qwen LogiQA
14B 0.788 -> 32B 0.774 -> 72B 0.770; Llama AQuA 3B 0.307 -> 8B 0.288). Figure 2's caption
"holds without exception" is contradicted by the paper's own Table 1. Emit a
`monotonicity.json` with per-curve Spearman rho and the explicit violations so the claim
can be restated correctly.

### 1d. Extraction robustness — `src/experiments/extraction/`

```bash
uv run -m src.experiments.extraction.run \
    --core_sweep_results_dir results/core_sweep \
    --output_dir results/extraction --n_manual_sample 200 --seed 42
```

- **Parser agreement** (item 17): run `extraction_alt` over all stored `final_answer_raw`
  and `no_cot_raw_text`. Report agreement rate, disagreement counts by type, and a
  per-cell breakdown. Dump a 200-sample stratified random file
  `manual_audit_sample.jsonl` (raw text, both parsers' outputs, blank `human_label`
  column) for manual inspection — this is the only human-in-the-loop step in the plan.
- **Recomputed metrics under each treatment** (item 18): a 3x3 grid of
  {permissive, strict} parser x {non_match, exclude} failure mode, each producing a full
  cell table, Model 1 / Model 2 coefficients and % reduction. Output
  `treatment_grid.json`. Expected to move nothing given the 2.2% max failure rate, but it
  closes the concern permanently.

**Phase 1 gate.** If the % reduction stays above the pre-registered 50% threshold across
all per-task fits, all leave-one-out fits, both families, and all extraction treatments,
the core statistical claim is already reviewer-proof and Phases 2-4 are about
generalization rather than rescue. Record the gate outcome in `research_log.md` before
proceeding.

---

## Phase 2 — Generation on small/medium models

Both GPUs run different models concurrently (one model per GPU via `CUDA_VISIBLE_DEVICES`,
`tensor_parallel_size=1`). Everything at or below 32B fits on one card in BF16.

### 2a. Sample-count convergence, 20 / 50 / 100 (item 1) — highest priority

**Do not try to append 80 samples to the existing 20.** vLLM's `SamplingParams(seed=42,
n=k)` does not guarantee that the first 20 of an `n=100` draw match a prior `n=20` draw,
and the vLLM version used for the original sweep is not recorded. Generate a clean pool of
100 fresh samples per question and subsample it nestedly (first 20 / first 50 / all 100).
This isolates sample count exactly as the review asks and costs one extra pass.

Representative subset — small, medium and large from both families, all 5 datasets
(30 cells):

```
Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-32B-Instruct
meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.1-8B-Instruct
hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
```

```bash
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do
  uv run -m src.experiments.core_sweep.run \
    --model_id Qwen/Qwen2.5-7B-Instruct --dataset_name "$dataset" \
    --output_dir results/samples_100 --n_questions 100 --n_cot_samples 100 --seed 42
done
```

Then the analysis, which needs no GPU:

```bash
uv run -m src.experiments.sample_convergence.run \
    --samples_100_dir results/samples_100 --accuracy_results_dir results/accuracy \
    --output_dir results/sample_convergence --seed 42
```

Reports, at k in {20, 50, 100}: per-cell proxy and CI width, Spearman rank correlation of
model ordering against k=100, and — the analysis that actually answers the reviewer —
Model 1 beta, Model 2 beta and % reduction. Note that no-CoT accuracy is unchanged by k, so
the k=20/50/100 regressions can reuse `results/accuracy/` directly.

**Go/no-go on a full 100-sample rerun.** The 30-cell subset is roughly 30-35 GPU-hours
(~17h wall clock on 2 cards). Extending to all 55 cells at 100 samples is roughly 5x the
original sweep, on the order of 120-150 GPU-hours, about 3 days wall clock. Run the subset
first; only commit to the full rerun if convergence shows a material k-dependence.

### 2b. AQuA answer-position randomization (item 14)

New module `src/experiments/aqua_position/`. A `permute_choices(question, rng)` helper
applies one fixed permutation per question ID, seeded, and rewrites `choices`,
`choice_labels` and `correct_label` together. The **same** permutation is used for the CoT
and no-CoT conditions so the underlying task is identical.

Run on the scale-spanning subset (Qwen 0.5B / 7B / 32B, Llama 1B / 8B), AQuA only,
20 samples — 5 cells, ~1 GPU-hour.

```bash
uv run -m src.experiments.aqua_position.run \
    --model_id Qwen/Qwen2.5-7B-Instruct --output_dir results/aqua_position \
    --n_questions 100 --n_cot_samples 20 --permute_seed 7 --seed 42
```

Compare original vs permuted on: proxy, no-CoT accuracy, no-CoT answer-letter
distribution (this is the diagnostic that matters), and shuffled-CoT behaviour. If the
near-random regime survives, the interpretation strengthens; if it collapses, position
bias is itself a named mechanism of the measurement artifact. Either outcome is publishable
— do not treat a change as a failure.

Optional extension if the permuted result is interesting: a position-balanced AQuA variant
where correct answers are distributed evenly across A-E.

### 2c. Prompt-template robustness (item 15)

Parameterize `src/generation/templates.py` with a `variant` argument rather than forking
the file. Three semantically equivalent CoT instructions, each with a matched no-CoT
counterpart:

- `v0` (current): "Let's think step by step."
- `v1`: "Work through this carefully before answering." / no-CoT: "Answer with the correct
  option only."
- `v2`: "Explain your reasoning, then give the answer." / no-CoT: "Give the answer with no
  explanation."

Run v1 and v2 on Qwen 0.5B / 7B / 32B and Llama 1B / 8B, all 5 datasets, 20 samples —
50 cells, ~12-15 GPU-hours. Check three things only: does raw inverse scaling persist, does
the coefficient reduction persist, does the real-vs-shuffled gap persist. Numerical
equality across prompts is not the claim.

### 2d. Base-vs-instruct (item 16) — deprioritized

Leave `results/base_ablation/` as-is and relabel it exploratory. A clean comparison
requires giving instruct models the identical few-shot plain-text prompt, which is
6 models x 5 datasets of new generation for a result that is not central to the paper. Only
run it if Phases 2a-2c finish early:

```bash
uv run -m src.experiments.base_ablation.run --model_id Qwen/Qwen2.5-7B-Instruct \
    --dataset_name aqua --output_dir results/fewshot_instruct ...
```

That comparison — instruct-with-few-shot vs base-with-few-shot — is the only version that
isolates instruction tuning.

---

## Phase 3 — Large-model BF16 (item 2), both GPUs

Run only after Phase 2 finishes; needs both cards and exclusive use.

Memory math: Qwen2.5-72B BF16 is ~145GB of weights against 159 GiB of combined HBM,
leaving roughly 15-20 GiB for KV cache at `gpu_memory_utilization=0.95`. Llama-3.1-70B
BF16 is ~141GB, slightly more comfortable. Both are feasible with `tensor_parallel_size=2`
and a capped context. **Use vLLM, not `HFEngine`** — `HFEngine.generate_chat` loops one
conversation at a time with no batching, which would take days for the 2000-prompt
final-answer pass.

```bash
for dataset in aqua logiqa arc_challenge openbookqa hellaswag; do
  uv run -m src.experiments.core_sweep.run \
    --model_id meta-llama/Llama-3.1-70B-Instruct --dataset_name "$dataset" \
    --output_dir results/bf16_large --n_questions 100 --n_cot_samples 20 \
    --engine vllm --tensor_parallel_size 2 --gpu_memory_utilization 0.95 \
    --max_model_len 2048 --seed 42
done
# then Qwen/Qwen2.5-72B-Instruct, identical flags
```

Every other setting — questions, prompts, seed, temperature, top-p, sample count,
extraction, shuffled-CoT procedure — stays fixed, so the AWQ result already on disk is a
valid paired comparison.

Download cost is ~290GB for both; start the `hf download` in the background during Phase 2.
If 72B BF16 OOMs even at `max_model_len=1536`, fall back to Llama-70B BF16 only — one
successful BF16/AWQ pair plus the no-quantized-checkpoint regression is an adequate
response.

Analysis (`src/experiments/quantization/run.py`, no GPU):

- Paired AWQ-vs-BF16 table per dataset: proxy, no-CoT accuracy, shuffled match rate,
  real-minus-shuffled gap, with per-cell deltas and bootstrap CIs on the deltas.
- Main regression **excluding both quantized checkpoints** (45 cells, 9 models): raw beta,
  controlled beta, % reduction. This is the analysis that decouples the conclusion from
  quantization even if the BF16 runs are only partially successful.
- Main regression with AWQ endpoints **substituted** by their BF16 counterparts.
- Rerun leave-one-model-out over the expanded model set.

---

## Phase 4 — Third model family (item 7)

Recommended: **OLMo-2-Instruct** — four sizes, genuinely independent training data and
recipe (AI2), ungated, vLLM-supported:

```
allenai/OLMo-2-0425-1B-Instruct
allenai/OLMo-2-1124-7B-Instruct
allenai/OLMo-2-1124-13B-Instruct
allenai/OLMo-2-0325-32B-Instruct
```

Verify these IDs resolve on the Hub before scheduling. Fallback: Gemma-2-it (2B, 9B, 27B) —
only three sizes and gated, but a well-established independent family.

Add the four entries to `INSTRUCT_MODELS` in `src/utils/models.py`, then run the standard
sweep: 4 models x 5 datasets x 20 samples, ~12-15 GPU-hours. Everything downstream —
prompts, sampling, extraction, shuffled-CoT — is unchanged, so `core_sweep.run` works as-is.

The analysis that matters is not the pooled 75-cell regression but the **within-family**
one: run the full decomposition for OLMo-2 alone and compare its % reduction against Qwen's
and Llama's. If all three families independently show a large reduction, the generalization
claim is as strong as this design can make it. If OLMo-2 differs, that is a boundary
condition and is explicitly welcomed by the challenge — report it, do not bury it.

Then rerun, over the full 15-model set: main regression, per-task, leave-one-task-out,
leave-one-model-out, residualized analysis, CoT-dependence regression, and the scale plots
with three families.

---

## Phase 5 — Final consolidated rerun

After Phases 2-4, rerun the entire Phase 1 battery over the expanded, BF16-corrected cell
set so every number in the paper comes from one consistent table. Regenerate all figures.
Append one `research_log.md` entry per phase as it completes, with the command, the output
path, and the headline metric pulled from the structured output file.

---

## Cost summary

| Phase | GPU-hours | Wall clock (2x H100) | Blocking? |
|---|---|---|---|
| 0 — environment | 0 | ~1h (+ downloads) | yes, blocks all |
| 1 — offline analyses | 0 | ~1 day of coding | no |
| 2a — 100-sample subset (30 cells) | ~30-35 | ~17h | no |
| 2b — AQuA position (5 cells) | ~1 | ~1h | no |
| 2c — prompt variants (50 cells) | ~12-15 | ~8h | no |
| 3 — BF16 70B + 72B (10 cells) | ~10-14 | ~12h (exclusive) | needs both GPUs |
| 4 — third family (20 cells) | ~12-15 | ~8h | no |
| 5 — consolidation | 0 | ~half day | after all |

Optional full 55-cell 100-sample rerun: ~120-150 GPU-hours, ~3 days wall clock. Gated on
the Phase 2a result.

Total committed compute: roughly 65-80 GPU-hours, about 2.5-3 days of wall clock, plus
~600GB of checkpoint downloads.

---

## Reproducibility requirements for every phase

- New results go in new directories. Never edit `results/core_sweep/` or
  `results/base_ablation/`.
- `config.json` per cell records git hash, vLLM version, and full config (extend
  `save_run_config` with `extra_metadata={"vllm_version": ...}`).
- Round all floats to 4 dp on write. The committed `decomposition.json` and
  `regression_table.csv` currently carry full float64 precision, against the project
  convention; fix `write_json`/`write_jsonl` once and regenerate in Phase 5.
- Every long generation run goes in tmux, one session per model.
- Every metric printed to stdout must also exist in a structured output file.
