# Research log

## 260901 — Phase 0: environment restoration

**What:** Repo was unrunnable (no `uv`, no `.venv`, empty HF cache). Reinstalled the
toolchain, pinned `vllm==0.24.0`, materialised the fixed question splits to
`data/processed/`, and recorded the vLLM version in every `config.json`.
**Result:** 47 pre-existing tests pass after fixing 2 stale assertions in
`tests/test_templates.py` (templates gained an assistant-prefix turn; the tests still
expected 2- and 4-message conversations).
**Command:** `uv sync --extra inference && uv run -m pytest tests/ -v`
**Output:** `.venv/`, `data/processed/{dataset}_questions.jsonl`

## 260901 — Phase 1a: regression battery (revision items 3, 4, 5, 6, 8, 9, 19)

**What:** Per-task, leave-one-task-out, leave-one-model-out, per-family, residualised,
partial-correlation, quadratic and mixed-effects fits on the existing 55 cells. New
pipeline (`recompute` -> `cells` -> `robustness`) reproduces the paper's pooled numbers
exactly: raw 0.2201, controlled 0.0476, 78.4% reduction, R2 0.451 -> 0.956.
**Result:** Every subset clears the pre-registered 50% threshold. Per-task 58.4-86.7%
(min LogiQA); leave-one-task-out 65.4-89.9%; leave-one-model-out 76.8-79.3% (a 2.5-point
spread, so no influential checkpoint); Qwen-only 75.3% (matches the paper), Llama-only
84.4%. Pooled residual slope 0.031 [0.015, 0.049] excludes zero, but 4 of 5 per-task
residual CIs include zero, so the residual is not robustly present within tasks.
**Command:**
uv run -m src.experiments.robustness.run --core_sweep_results_dir results/core_sweep --output_dir results/robustness --n_bootstrap 1000 --seed 42
**Output:** `results/robustness/` (9 JSON files + `cell_table.csv`, `residual_table.csv`)

## 260901 — Phase 1c: empirical null and CoT dependence (revision items 12, 13)

**What:** Replaced the assumed 0.25 chance level with 1000 derangements per cell, defined
`dependence = real - shuffled`, and regressed it on log model size.
**Result:** 53/55 cells have a real match rate above the 97.5th percentile of their own
permutation null. Empirical null rates span 0.184-0.292, so the fixed 0.25 assumption was
roughly right but not exact. Dependence slope is positive with a CI excluding zero in
every task and both families (pooled +0.2235 [0.1605, 0.2812]). Confirmed the sign of the
paper's Table 2 `Mean D` is inverted, and that only 7/10 family-task curves are strictly
monotone (4 violations: llama/aqua 3B->8B, qwen/aqua 32B->72B, qwen/logiqa 14B->32B->72B).
**Command:**
uv run -m src.experiments.cot_dependence.run --core_sweep_results_dir results/core_sweep --output_dir results/cot_dependence --n_permutations 1000 --seed 42
**Output:** `results/cot_dependence/{null_table.csv,dependence_fits.json,monotonicity.json}`

## 260901 — Phase 1b: capability matching and regimes (revision items 10, 11, 20)

**What:** Pre-registered capability-matched pairs, three-regime bins with answer-position
diagnostics, ceiling saturation, and the capability reconstruction.
**Result:** Reconstruction is the headline: a fit on no-CoT accuracy alone reproduces 86.0%
of the observed size slope (0.189 vs 0.220, MAE 0.038). Matched pairs are thin at the
pre-registered tolerance -- only 2 within-family pairs, so the pre-registered fallback
applies and the 8 cross-family pairs are reported separately (mean |delta proxy| 0.067 vs
0.206 predicted from size alone, mean size ratio 12.1x, paired p=0.054). Regimes show a
clean monotone gradient in answer concentration: normalised entropy 0.820 / 0.883 / 0.922
and modal-answer share 0.435 / 0.380 / 0.294 across near-random / intermediate / ceiling.
Ceiling saturation is mild (accuracy slope 1.067 below 0.75 vs 0.984 above).
**Command:**
uv run -m src.experiments.capability.run --core_sweep_results_dir results/core_sweep --cell_table results/robustness/cell_table.csv --output_dir results/capability --seed 42
**Output:** `results/capability/` (5 files + `binned_cells.csv`, `answer_distribution.csv`)

## 260901 — Phase 1d: extraction robustness (revision items 17, 18)

**What:** Independent strict parser compared against the permissive one over all stored raw
completions, plus the 2x2 parser x failure-treatment grid.
**Result:** Closed. 99.36% agreement across 115,500 extractions; 0/55 cells exceed the
pre-registered 5% disagreement threshold (worst 3.05%, Qwen2.5-3B on ARC-Challenge).
Coefficient reduction is 77.9-79.4% across all four treatments, so neither the parser nor
the failure convention drives the result. Failure rates: permissive 0.28%, strict 0.92%.
**Command:**
uv run -m src.experiments.extraction.run --core_sweep_results_dir results/core_sweep --output_dir results/extraction --n_manual_sample 200 --seed 42
**Output:** `results/extraction/{parser_agreement.csv,treatment_grid.json,manual_audit_sample.jsonl}`

**Data issue found:** 3 of the 100 ARC-Challenge questions carry numeric choice labels
(1-4) rather than letters, inherited from the raw ARC dataset. They are rendered to the
model as "1) ... 2) ..." and the existing extractor handles them correctly by string
equality, so no metric is wrong -- but the prompt format is inconsistent across 3% of one
task. Worth normalising if ARC is ever regenerated.

## 260901 — Regression excluding the quantized checkpoints (revision item 2, partial)

**What:** Reran the full Phase 1a battery with both AWQ checkpoints dropped (45 cells,
9 BF16 models), to decouple the conclusion from quantization before the BF16 runs land.
**Result:** The reduction is 81.0% [68.5, 92.4] without the quantized endpoints, slightly
*higher* than the 78.4% with them. All five per-task fits, all leave-one-task-out and all
leave-one-model-out fits still clear the 50% threshold. Per-family: Qwen 77.6%,
Llama 89.4%. So the headline result is not produced by the two quantized points.
**Command:**
uv run -m src.experiments.robustness.run --core_sweep_results_dir results/core_sweep --output_dir results/robustness_no_quant --n_bootstrap 1000 --seed 42 --exclude_quantized true
**Output:** `results/robustness_no_quant/`

## 260901 — thesis chapter-4 figure style sweep

**What:** regenerated all 12 paper figures under the thesis chapter-4 style (serif, seaborn deep palette, open frame, dashed y-grid, 300 dpi) to compare against the current style
**Result:** style applies cleanly across all six plot modules with no change to plotting logic; long axis labels needed the figure widened at save time or they clip under the larger serif type
**Command:**
uv run python experiments/figure_style/260901_thesis_style_v1/1_regenerate_figures.py --output_dir results/fig_new --seed 42
**Output:** results/fig_new/

## 260901 — Phase 2a: sample-count convergence, 20 vs 50 vs 100 (revision item 1)

**What:** Generated a fresh pool of 100 CoT samples per question for 6 models spanning both
families (Qwen 0.5B/7B/32B, Llama 1B/8B, Llama-70B-AWQ) x 5 tasks = 30 cells, then
subsampled the same pool nestedly at k=20/50/100.
**Result:** 20 samples is empirically sufficient. Mean proxy 0.6176 / 0.6173 / 0.6170 at
k=20/50/100; worst per-cell deviation from k=100 is 0.0350 at k=20 and 0.0104 at k=50;
Spearman rank correlation of model ordering vs k=100 is 0.9956 at k=20. The capability-
controlled reduction is 78.7% / 77.9% / 77.5% -- a 1.2-point spread. Bootstrap CI width
narrows only from 0.1116 to 0.1090, so 5x the compute buys a 2% tighter interval.
**Command:**
uv run -m src.experiments.sample_convergence.run --samples_100_dir results/samples_100 --output_dir results/sample_convergence --n_bootstrap 1000 --seed 42
**Output:** `results/sample_convergence/convergence.json`, `results/figures/sample_convergence.png`

## 260901 — Phase 4: third model family, OLMo-2-Instruct (revision item 7)

**What:** Added OLMo-2-Instruct 1B/7B/13B/32B (ungated, independently trained by AI2) on the
same 5 tasks, same prompts, sampling and extraction. Pooled table is 75 cells, 15 models,
3 families.
**Result:** The capability control holds independently in all three families:
Qwen 75.3%, Llama 84.4%, **OLMo 97.9%** (controlled coefficient -0.005, i.e. no-CoT accuracy
absorbs essentially the entire size effect). Pooled reduction rises to 82.7% [72.2, 92.6].
All 5 per-task fits, all leave-one-task-out and all 15 leave-one-model-out fits clear the
pre-registered threshold (LOMO spans 80.0-84.7%).
CoT dependence stays positive with CI excluding zero in every task and every family
(OLMo +0.2769 [0.1540, 0.3775]). 72/75 cells beat their own permutation null.
Reconstruction from no-CoT accuracy alone now recovers 88.6% of the observed size slope.
Capability-matched pairs strengthen: 19 cross-family pairs, mean |delta proxy| 0.0447 vs
0.1873 predicted from size alone, paired p=0.009.
The near-random regime sharpens with OLMo's small models: normalised answer entropy 0.651
and modal-answer share 0.595, and the within-bin slope against model size is *negative*
(-0.011) while the slope against accuracy is +0.966.
**Command:**
uv run -m src.experiments.robustness.run --core_sweep_results_dir results/core_sweep,results/third_family --output_dir results/three_family --n_bootstrap 1000 --seed 42
**Output:** `results/three_family/`, `results/three_family_dependence/`, `results/three_family_capability/`
