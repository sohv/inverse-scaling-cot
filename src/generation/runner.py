"""Main generation logic for CoT faithfulness experiments.

Orchestrates CoT and no-CoT generation, extracts answers, saves outputs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.data.loader import Question
from src.generation.engine import GenerationConfig
from src.generation.templates import (
    build_cot_final_answer_messages,
    build_cot_messages,
    build_no_cot_messages,
)
from src.utils.io import write_json, write_jsonl

if TYPE_CHECKING:
    from src.generation.engine import VLLMEngine

LOGGER = logging.getLogger(__name__)


class CoTSample(BaseModel):
    """One CoT completion for a question."""

    id: str
    sample_idx: int
    cot_text: str  # the CoT reasoning trace
    final_answer_raw: str  # raw model output for final answer extraction
    extracted_answer: str | None  # parsed answer letter, or None if extraction failed


class QuestionResult(BaseModel):
    """All generation results for a single question."""

    id: str
    dataset_name: str
    model_id: str
    correct_label: str
    no_cot_raw_text: str
    no_cot_extracted_answer: str | None
    cot_samples: list[CoTSample]


# --- Answer extraction ---

ANSWER_PATTERN = re.compile(
    r"(?:the\s+answer\s+is\s*\(?([A-E])\)?)"  # "the answer is (A)" or "the answer is A"
    r"|(?:\(([A-E])\))",  # standalone "(A)"
    re.IGNORECASE,
)


def extract_answer(text: str, choice_labels: list[str]) -> str | None:
    """Extract the answer letter from model output.

    Strategy:
    1. Find all matches of ANSWER_PATTERN in the text.
    2. Return the LAST match (for CoT, the final answer after reasoning).
    3. If no match, check if the text starts with a single letter from choice_labels.
    4. If still no match, return None.
    """
    valid = set(label.upper() for label in choice_labels)
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        # Each match is a tuple of groups; take the non-empty one from the last match
        last_match = matches[-1]
        letter = next((g.upper() for g in last_match if g), None)
        if letter and letter in valid:
            return letter

    # Fallback: check if text starts with a valid letter
    stripped = text.strip()
    if stripped and stripped[0].upper() in valid:
        return stripped[0].upper()

    return None


def extract_answer_no_cot(text: str, choice_labels: list[str]) -> str | None:
    """Extract answer from no-CoT output.

    The no-CoT prompt ends with 'The answer is (' so the model should emit
    just the letter immediately. Check first character, then fall back to
    extract_answer().
    """
    stripped = text.strip()
    valid = set(label.upper() for label in choice_labels)

    # Most common case: model emits "A)" or just "A"
    if stripped and stripped[0].upper() in valid:
        return stripped[0].upper()

    return extract_answer(text, choice_labels)


# --- Main generation logic ---


def run_generation_for_model(
    engine: VLLMEngine,
    questions: list[Question],
    model_id: str,
    n_cot_samples: int = 20,
    cot_gen_config: GenerationConfig | None = None,
    no_cot_gen_config: GenerationConfig | None = None,
) -> list[QuestionResult]:
    """Run CoT and no-CoT generation for all questions on one model.

    Procedure per Lanham et al.:
    1. Generate n_cot_samples CoT reasoning traces per question
    2. For each CoT trace, extract the final answer via a second turn
    3. Generate 1 no-CoT answer per question (direct answer, no reasoning)
    """
    if cot_gen_config is None:
        cot_gen_config = GenerationConfig(temperature=0.8, top_p=0.95, max_tokens=1024, n=n_cot_samples)
    if no_cot_gen_config is None:
        no_cot_gen_config = GenerationConfig(temperature=0.8, top_p=0.95, max_tokens=20, n=1)

    n_questions = len(questions)
    LOGGER.info(f"Starting generation for {model_id}: {n_questions} questions, {n_cot_samples} CoT samples each")

    # Step 1: Generate CoT reasoning traces (n=n_cot_samples per question)
    LOGGER.info("Step 1: Generating CoT reasoning traces...")
    cot_conversations = [build_cot_messages(q) for q in questions]
    cot_outputs = engine.generate_chat(cot_conversations, cot_gen_config)
    # cot_outputs[i] = list of n_cot_samples CoT texts for question i

    # Step 2: For each CoT trace, extract final answer via second turn
    LOGGER.info("Step 2: Extracting final answers from CoT traces...")
    final_answer_config = GenerationConfig(temperature=0.0, top_p=1.0, max_tokens=20, n=1)
    all_final_conversations = []
    conversation_index_map = []  # (question_idx, sample_idx)

    for q_idx, question in enumerate(questions):
        for s_idx, cot_text in enumerate(cot_outputs[q_idx]):
            msgs = build_cot_final_answer_messages(question, cot_text)
            all_final_conversations.append(msgs)
            conversation_index_map.append((q_idx, s_idx))

    final_outputs = engine.generate_chat(all_final_conversations, final_answer_config, continue_final_message=True)

    # Reorganize final outputs into per-question structure
    cot_final_answers: list[list[str]] = [[] for _ in range(n_questions)]
    for conv_idx, (q_idx, s_idx) in enumerate(conversation_index_map):
        cot_final_answers[q_idx].append(final_outputs[conv_idx][0])

    # Step 3: Generate no-CoT answers (n=1 per question)
    LOGGER.info("Step 3: Generating no-CoT answers...")
    no_cot_conversations = [build_no_cot_messages(q) for q in questions]
    no_cot_outputs = engine.generate_chat(no_cot_conversations, no_cot_gen_config, continue_final_message=True)

    # Step 4: Assemble results with answer extraction
    LOGGER.info("Step 4: Extracting answers and assembling results...")
    results = []
    n_cot_failures = 0
    n_no_cot_failures = 0

    for q_idx, question in enumerate(questions):
        # Extract no-CoT answer
        no_cot_raw = no_cot_outputs[q_idx][0]
        no_cot_answer = extract_answer_no_cot(no_cot_raw, question.choice_labels)
        if no_cot_answer is None:
            n_no_cot_failures += 1
            LOGGER.warning(f"No-CoT extraction failed for {question.id}: '{no_cot_raw[:100]}'")

        # Build CoT samples with extracted answers
        cot_samples = []
        for s_idx in range(len(cot_outputs[q_idx])):
            cot_text = cot_outputs[q_idx][s_idx]
            final_raw = cot_final_answers[q_idx][s_idx]
            extracted = extract_answer_no_cot(final_raw, question.choice_labels)
            if extracted is None:
                n_cot_failures += 1

            cot_samples.append(
                CoTSample(
                    id=question.id,
                    sample_idx=s_idx,
                    cot_text=cot_text,
                    final_answer_raw=final_raw,
                    extracted_answer=extracted,
                )
            )

        results.append(
            QuestionResult(
                id=question.id,
                dataset_name=question.dataset_name,
                model_id=model_id,
                correct_label=question.correct_label,
                no_cot_raw_text=no_cot_raw,
                no_cot_extracted_answer=no_cot_answer,
                cot_samples=cot_samples,
            )
        )

    total_cot_samples = n_questions * n_cot_samples
    LOGGER.info(
        f"Generation complete. CoT extraction failures: {n_cot_failures}/{total_cot_samples} "
        f"({n_cot_failures / total_cot_samples * 100:.1f}%). "
        f"No-CoT extraction failures: {n_no_cot_failures}/{n_questions} "
        f"({n_no_cot_failures / n_questions * 100:.1f}%)."
    )
    return results


def save_generation_results(
    results: list[QuestionResult],
    output_dir: str | Path,
) -> Path:
    """Save generation results to JSONL.

    File: {output_dir}/generation_results.jsonl
    Also saves summary: {output_dir}/generation_summary.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [r.model_dump() for r in results]
    results_path = output_dir / "generation_results.jsonl"
    write_jsonl(results_path, records)

    # Summary
    n_cot_failures = sum(1 for r in results for s in r.cot_samples if s.extracted_answer is None)
    n_no_cot_failures = sum(1 for r in results if r.no_cot_extracted_answer is None)
    total_cot = sum(len(r.cot_samples) for r in results)

    summary = {
        "n_questions": len(results),
        "model_id": results[0].model_id if results else "",
        "dataset_name": results[0].dataset_name if results else "",
        "n_cot_samples_per_question": len(results[0].cot_samples) if results else 0,
        "n_cot_extraction_failures": n_cot_failures,
        "total_cot_samples": total_cot,
        "n_no_cot_extraction_failures": n_no_cot_failures,
    }
    summary_path = output_dir / "generation_summary.json"
    write_json(summary_path, summary)

    return results_path
