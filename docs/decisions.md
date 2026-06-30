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
