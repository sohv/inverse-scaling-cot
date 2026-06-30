# Experimental design

## Overview

This project replicates and extends the chain-of-thought (CoT) faithfulness
analysis from Lanham et al. (2023) and Bentham et al. (2024). The primary
question: does the fraction of with-CoT answers matching without-CoT answers
(a proxy for CoT unfaithfulness) increase with model scale?

We extend the original two-point closed-model comparison to a full scale sweep
across 11 open-weight models in two families (Qwen2.5-Instruct and Llama-3-Instruct).

See the detailed experimental procedures in the project README or the original
design document for full per-experiment specifications.

## Experiments

1. **Core metric sweep** — 11 models x 5 tasks, 20 CoT samples + 1 no-CoT answer per question
2. **Shuffled-CoT null baseline** — random CoT reassignment to test signal vs noise
3. **Accuracy-without-CoT logging** — ground-truth scoring of no-CoT answers
4. **Confound decomposition regression** — OLS analysis of faithfulness vs scale with accuracy control
5. **FUR cross-method check** (optional) — external faithfulness method on 2 models
6. **Base-model ablation** (optional) — instruct vs base comparison

## References

- Lanham, T., et al. (2023). "Measuring Faithfulness in Chain-of-Thought Reasoning."
- Bentham, J., Stringham, N., & Marasovic, A. (2024). "Chain-of-Thought Unfaithfulness as Disguised Accuracy." arXiv:2402.14897.
