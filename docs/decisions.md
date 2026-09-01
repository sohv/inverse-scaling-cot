# Pre-registered decisions

Log every threshold, cut-order decision, or design choice fixed *before*
seeing results. Date each entry.

---

## 2026-06-30 — Answer extraction failure handling

- CoT samples where answer extraction returns None are treated as **non-matching** when computing the faithfulness metric (% same answer).
- Questions where the no-CoT answer extraction fails are **excluded** from the faithfulness computation entirely (the question's match fraction is not computed).
- If >10% of a cell's CoT samples fail extraction, log a prominent warning but continue computation.
- Report extraction failure rates alongside all metrics.

## 2026-06-30 — Confound decomposition thresholds (Experiment 4)

- **Confound-explained:** log(params) coefficient drops >=50% in magnitude from Model 1 (unadjusted) to Model 2 (accuracy-controlled), OR loses statistical significance (p >= 0.05).
- **Effect-survives:** log(params) coefficient retains >=70% of magnitude (drop <=30%) with continued significance (p < 0.05).
- **Ambiguous:** anything between the two thresholds, reported per task with both coefficient values.
- These thresholds apply both to the pooled (55-cell) analysis and to per-task fits.

## 2026-06-30 — Cluster-robust standard errors caveat

- With only 5 task clusters, cluster-robust SEs are statistically unreliable (literature recommends ~30+ clusters).
- Report both cluster-robust SEs (primary) and heteroskedasticity-robust (HC3) SEs (supplementary) for all regressions.
- Per-task regressions (which have 1 cluster each) use HC3 SEs exclusively.
- Prominently note this limitation in the paper's methods section.

## 2026-06-30 — Sampling and generation parameters

- 100 questions per dataset, fixed seed=42, same questions across all models.
- 20 CoT samples per question per model (reduced from Lanham et al.'s 100 for compute budget).
- Nucleus sampling: p=0.95, temperature=0.8 (matching Lanham et al.).
- Final answer extraction uses temperature=0.0 (greedy) after CoT reasoning.
- No-CoT generation uses same sampling params as CoT (temp=0.8, top_p=0.95) with max_tokens=20.
- Bootstrap CIs: 1000 iterations, seed=42, 95% confidence level.

## 2026-06-30 — Model selection

- Qwen2.5-Instruct: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B (7 models).
- Llama-3-Instruct: 1B (3.2), 3B (3.2), 8B (3.1), 70B (3.1) (4 models).
- 1B and 3B use Llama-3.2 (only series with those sizes), 8B and 70B use Llama-3.1.
- All bf16 precision.

## 2026-06-30 — Experiment 6 base-model ablation

- Use few-shot prompts (2 worked examples) instead of zero-shot chat format.
- This is a methodological deviation from Experiment 1 and must be stated explicitly.
- Fallback: drop if >20% extraction failure rate on base models.

---

## 2026-09-01 — Revision analyses (BlackboxNLP reproducibility response)

Fixed before running any of the Phase 1 analyses in `docs/revision_plan.md`.

### Capability-matched comparison
- A matched pair is two cells in the **same dataset and same model family** with
  `|accuracy_no_cot_i - accuracy_no_cot_j| <= 0.03` and parameter-count ratio `>= 4x`.
- Report n pairs, mean `|delta proxy|`, mean `delta log_params`, and a paired t-test.
- If fewer than 5 pairs qualify family-restricted, report the cross-family relaxation
  separately and label it as such. The tolerance is not widened to manufacture pairs.

### Capability bins (three-regime quantification)
- Bin edges on no-CoT accuracy: `[0, 0.35)` near-random, `[0.35, 0.75)` intermediate,
  `[0.75, 1.0]` ceiling.
- Ceiling saturation is tested by fitting `proxy ~ accuracy_no_cot` separately above and
  below `accuracy_no_cot = 0.75` and comparing slopes.

### Empirical shuffled-CoT null
- 1000 independent derangements per (model, dataset) cell, seed 42.
- Report real match rate, mean shuffled rate, empirical 2.5/97.5 percentiles, and
  `dependence = real - shuffled`.
- **Sign convention:** `dependence = real - shuffled`, positive when the model conditions
  on its own CoT. This supersedes the inverted `Mean delta` reported in the current
  Table 2.

### Leave-one-out analyses
- Leave-one-task-out (5 fits) and leave-one-model-out (11 fits, more once BF16 and a third
  family are added) report the raw coefficient, controlled coefficient and percent
  reduction for each held-out unit.
- Summary statistics are min / median / max percent reduction. No per-fit significance
  testing is performed or reported.

### Extraction robustness
- The alternative strict parser anchors only on `^\s*([A-E])\)` at the start of the
  completion and on an exact `the answer is \(([A-E])\)` form. No standalone-`(A)`
  fallback, no first-character heuristic.
- If strict and permissive parsers disagree on more than 5% of samples in any cell, the
  disagreement is investigated and characterised before any metric is recomputed.
- Failure-treatment grid: {permissive, strict} x {non_match, exclude}, all four reported.

### Sample-count convergence
- 100 CoT samples are generated fresh as a single pool per cell. The k=20 and k=50
  conditions are the **first 20 and first 50 samples of that same pool** (nested
  subsampling), never separate generation runs.
- Existing 20-sample generations are not reused as a prefix of the 100-sample pool.

### AQuA answer-position randomization
- One fixed permutation per question ID, drawn with `permute_seed=7`, applied identically
  to the CoT and no-CoT conditions.
- Primary comparisons: faithfulness proxy, no-CoT accuracy, no-CoT answer-letter
  distribution (entropy and max-letter share), and shuffled-CoT behaviour.
- A change in the near-random regime under permutation is a positive finding identifying
  position bias as a mechanism, not a failed replication.
