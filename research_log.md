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

## 260901 — Phase 3: AWQ vs BF16 quantization (revision item 2)

**What:** Regenerated the two largest checkpoints in BF16 (Qwen2.5-72B, Llama-3.1-70B) at
tensor_parallel_size=2, everything else held fixed, and paired them against the AWQ cells.
**Result:** 10 paired cells. Mean |delta proxy| 0.023 (max 0.042), |delta accuracy| 0.018,
shuffled-CoT rate |delta| 0.0015, |delta dependence| 0.023. Deltas go in both directions
(ARC-C -0.018, LogiQA -0.028, AQuA +0.042), consistent with sampling noise rather than a
systematic quantization effect. Combined with the 81.0% reduction when the AWQ points are
dropped entirely, quantization is ruled out from two directions.
**Failure and fix:** the first BF16 Llama run OOMed -- not KV cache, but an activation
buffer: vLLM pushed a ~15k-token prefill chunk through the 28672-wide MLP needing 836 MiB
with 584 MiB free. Added max_num_seqs / max_num_batched_tokens to the engine and reran with
a 4096-token prefill cap; the cell then took 8.4 min. Lowering gpu_memory_utilization would
have made this worse, since it shrinks the same pool the activations come from.
**Command:**
uv run -m src.experiments.core_sweep.run --model_id meta-llama/Llama-3.1-70B-Instruct --dataset_name aqua,logiqa,arc_challenge,openbookqa,hellaswag --output_dir results/bf16_large --n_questions 100 --n_cot_samples 20 --tensor_parallel_size 2 --gpu_memory_utilization 0.93 --max_model_len 2048 --max_num_seqs 64 --max_num_batched_tokens 4096 --seed 42
**Output:** `results/bf16_large/`, `results/quantization/awq_vs_bf16_summary.json`

## 260901 — Phase 2b/2c: prompt and answer-position robustness (revision items 14, 15)

**What:** Two semantically equivalent prompt rewordings (v1, v2) on 5 models x 5 tasks each,
and AQuA with answer positions permuted per question (same permutation for the CoT and
no-CoT conditions) on 5 models.
**Result:** Prompt robustness -- mean |delta proxy| 0.023 (v1) and 0.026 (v2); the raw
scaling coefficient reproduces under both (0.289, 0.259 vs 0.220) and the controlled
reduction is 83.6% and 94.5% vs 78.4%. Note v2 has a max per-cell shift of 0.180, so
individual cells do move even though the aggregate conclusion does not.
Position randomisation -- mean |delta proxy| 0.023 (max 0.039), accuracy +0.056,
dependence +0.024, controlled reduction 83.9%. The near-random AQuA regime survives
answer-position randomisation, so it is not an artifact of position bias.
**Output:** `results/prompt_analysis/`, `results/aqua_position_analysis/`

## 260901 — Final consolidated analysis (all quantized checkpoints replaced by BF16)

**What:** Pooled core sweep + BF16 large + OLMo-2, excluding both AWQ checkpoints.
75 cells, 15 models, 3 families, all BF16.
**Result:** Pooled reduction **83.1%** [73.6, 92.4] (raw 0.2263 -> controlled 0.0382).
Per family: Qwen 78.0%, Llama 82.1%, OLMo 97.9%. Per task 60.7-84.9%, leave-one-task-out
69.1-92.8%, leave-one-model-out 80.6-85.2% -- every one clears the pre-registered 50%
threshold. CoT dependence slope +0.2334 [0.1819, 0.2845] pooled and positive with a CI
excluding zero in every task and every family. 72/75 cells beat their own permutation null.
Reconstruction from no-CoT accuracy alone recovers **89.0%** of the observed size slope
(R2 0.953, MAE 0.036). 13/15 family-task curves are strictly monotone.
**Command:**
uv run -m src.experiments.robustness.run --core_sweep_results_dir results/core_sweep,results/bf16_large,results/third_family --output_dir results/final --n_bootstrap 1000 --seed 42 --exclude_models "Qwen/Qwen2.5-72B-Instruct-AWQ,hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
**Output:** `results/final/`, `results/final_dependence/`, `results/final_capability/`

## 260901 — thesis style applied to the full final figure set

**What:** re-ran the style sweep over all 17 current figures, now sourced from the final
consolidated analysis (`results/final*`) plus the sample-convergence and four paired-comparison
figures
**Result:** all 17 regenerate under the thesis style with matching names and no duplicates;
repainting colours by value rather than by name was needed once the third family (OLMo) appeared,
since the old name-keyed rebuild dropped it and crashed on KeyError 'olmo'
**Command:**
uv run python experiments/figure_style/260901_thesis_style_v1/1_regenerate_figures.py --output_dir results/fig_new --seed 42
**Output:** results/fig_new/

## 260901 — Item 17: manual audit of the extraction sample, and a strict-parser bug it found

**What:** Labelled all 200 rows of `manual_audit_sample.jsonl` by hand-reading the raw
completions. Rule: the prompt ends with the assistant prefix "The answer is (", so the
option token the model emits at the start of its completion is its answer; matched
case-insensitively against the question's real `choice_labels`.
**Result:** The permissive parser is correct on **200/200** sampled disagreements; the
strict parser on **0/200**. Every disagreement was a defect in my own strict parser, not
genuine ambiguity: 188 were the 3 ARC-Challenge questions whose choice labels are 1-4
(the parser hardcoded `[A-E]`) and 12 were models emitting a lowercase `b)` (the parser was
case-sensitive). A "strict" parser whose only disagreements are its own bugs is not an
independent check, so it was rewritten to match the question's real label set,
case-insensitively, while still accepting only unambiguous parenthesised forms.
**After the fix:** the two independently written parsers agree on **99.996%** of 115,500
extractions (worst cell 0.10%), with identical failure rates (0.28% each), and the
coefficient reduction spans 77.9-78.4% across all four parser x failure-treatment cells.
**Output:** `results/extraction/manual_audit_labelled.jsonl`, `manual_audit_summary.json`,
`parser_agreement.csv`, `treatment_grid.json`

## 260901 — Item 10: matched-pair threshold sensitivity

**What:** The pre-registered thresholds (|delta acc| <= 0.03, size ratio >= 4x) yield only
2 within-family pairs, so the headline rested on cross-family pairs. Added a post-hoc sweep
over tolerance x size-ratio, with the pre-registered point kept as primary.
**Result:** The conclusion holds across the whole grid -- at every setting the observed
matched-pair proxy gap is far below what the size coefficient alone predicts. Within-family
becomes well-powered at slightly looser settings: tolerance 0.05 / ratio 2x gives n=19,
0.0591 observed vs 0.1116 predicted (p=0.005); tolerance 0.075 / ratio 4x gives n=12,
0.0929 vs 0.1808 (p=0.003). Cross-family at the pre-registered point: n=15, 0.0457 vs
0.1784 (p=0.006).
**Output:** `results/final_capability/matched_pairs_sensitivity.json`

## 260901 — crowded legends moved below the x-axis

**What:** legends with 5 or more entries now sit under the x-axis label instead of inside the
axes, via a shared `legend_below` helper called from the four plot modules that cross the
threshold; applied to both the current style (`results/figures/`) and the thesis style
(`results/fig_new/`)
**Result:** 5 of 17 figures move their legend (faithfulness_vs_size, real_vs_shuffled,
faith_vs_params, residual_vs_params, real_vs_empirical_null); panel legends are merged and
deduplicated, so repeated per-panel copies collapse to one
**Command:**
uv run -m src.experiments.core_sweep.plot --results_dir results/core_sweep --output_dir results/figures
uv run python experiments/figure_style/260901_thesis_style_v1/1_regenerate_figures.py --output_dir results/fig_new --seed 42
**Output:** results/figures/, results/fig_new/, pushed to hf.co/datasets/sohv/cot-inverse
