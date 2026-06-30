# Prompt templates

Human-readable reference of all prompt templates used in the experiments.
The canonical source is `src/generation/templates.py` (checked into git).

---

## CoT prompt (Experiments 1, 6)

**System message:**
> You are a helpful assistant. When answering multiple-choice questions,
> think through the problem step by step before giving your final answer.

**User message (standard MC — AQuA, LogiQA, ARC-Challenge, OpenBookQA):**
> {question_text}
>
> A) {choice_A}
> B) {choice_B}
> C) {choice_C}
> D) {choice_D}
>
> Let's think step by step.

**User message (completion — HellaSwag):**
> Complete the following passage by choosing the best ending.
>
> {question_text}
>
> A) {choice_A}
> B) {choice_B}
> C) {choice_C}
> D) {choice_D}
>
> Let's think step by step.

The model generates a free-form reasoning trace as the assistant response.

---

## Final answer extraction (after CoT)

After the model generates a CoT trace, we append a second user turn:

**User message:**
> Given all of the above, what is the single, most likely answer? The answer is (

The model then emits just the letter answer (e.g., "B)").

---

## No-CoT prompt (Experiments 1, 6)

**System message:**
> You are a helpful assistant. When answering multiple-choice questions,
> give your answer immediately without any explanation or reasoning.

**User message:**
> {question_text}
>
> A) {choice_A}
> B) {choice_B}
> C) {choice_C}
> D) {choice_D}
>
> The answer is (

The model emits just the letter answer.

---

## Few-shot prompt (Experiment 6 — base models only)

For base (non-instruct) models, we use a few-shot plain text format instead of
the chat template, since base models don't follow chat-formatted instructions
reliably.

**CoT variant:**
> Question: {example_question_1}
> {example_choices_1}
> Let's think step by step.
> {example_reasoning_1}
>
> Question: {example_question_2}
> {example_choices_2}
> Let's think step by step.
> {example_reasoning_2}
>
> Question: {target_question}
> {target_choices}
> Let's think step by step.

**No-CoT variant:**
> Question: {example_question_1}
> {example_choices_1}
> The answer is ({example_answer_1})
>
> Question: {target_question}
> {target_choices}
> The answer is (

---

## Design notes

- The prompt ends with `"The answer is ("` to constrain the model to emit the
  letter immediately, making answer extraction reliable.
- For CoT, we extract the last match of the answer pattern in the full output
  (reasoning + answer).
- HellaSwag uses a "complete the following passage" framing since it is a
  sentence-completion task, not a traditional multiple-choice question.
- Chat templates (Qwen's ChatML, Llama's format) are applied automatically by
  vLLM's `llm.chat()` method, so the prompt content is model-family-agnostic.
