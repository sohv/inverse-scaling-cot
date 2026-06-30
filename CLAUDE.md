# Project: Inverse Scaling of CoT Faithfulness

Follows the global CLAUDE.md with these project-specific overrides and notes.

## W&B

Not used. All inference is local (vLLM), not API calls. Each run saves
structured JSON/JSONL with full config and git hash to disk. Plot scripts
reconstruct dashboards from disk outputs. No value added by W&B here.

## vllm is optional

`vllm` is in the `[inference]` extras group, not in core dependencies.
Modules that touch vllm (`src/generation/engine.py`) use lazy imports so
the rest of the codebase imports cleanly without a GPU stack installed.
`src/generation/runner.py` uses `TYPE_CHECKING` for the `VLLMEngine` type.
This is a deliberate deviation from "imports at the top of every file."

## Async

Not used. vLLM offline inference is synchronous and handles its own
batching internally. No LLM API calls, so no async/await needed.

## Models

- Qwen2.5-Instruct: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B
- Llama-3-Instruct: 1B (3.2), 3B (3.2), 8B (3.1), 70B (3.1)
- Qwen2.5 base (non-instruct): same 7 sizes, for Experiment 6

## Datasets

AQuA, LogiQA, ARC-Challenge, OpenBookQA, HellaSwag.
100 questions each, fixed seed=42, same across all models.
