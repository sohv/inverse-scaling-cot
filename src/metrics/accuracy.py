"""Accuracy computation for no-CoT answers vs ground truth (Experiment 3)."""

import logging

from pydantic import BaseModel

from src.generation.runner import QuestionResult

LOGGER = logging.getLogger(__name__)


class AccuracyResult(BaseModel):
    """Accuracy for one (model, dataset) cell."""

    model_id: str
    dataset_name: str
    n_questions: int
    n_correct: int
    accuracy: float
    n_extraction_failures: int
    per_question_correct: list[bool]


def compute_accuracy(results: list[QuestionResult]) -> AccuracyResult:
    """Compute accuracy of no-CoT answers against ground truth.

    For each question:
    - Compare no_cot_extracted_answer to correct_label
    - None extracted answers count as incorrect
    """
    model_id = results[0].model_id
    dataset_name = results[0].dataset_name

    n_correct = 0
    n_failures = 0
    per_question_correct = []

    for r in results:
        if r.no_cot_extracted_answer is None:
            n_failures += 1
            per_question_correct.append(False)
            continue

        is_correct = r.no_cot_extracted_answer == r.correct_label
        per_question_correct.append(is_correct)
        if is_correct:
            n_correct += 1

    accuracy = n_correct / len(results) if results else 0.0

    LOGGER.info(f"{model_id}/{dataset_name}: accuracy={accuracy:.4f} ({n_correct}/{len(results)})")
    return AccuracyResult(
        model_id=model_id,
        dataset_name=dataset_name,
        n_questions=len(results),
        n_correct=n_correct,
        accuracy=accuracy,
        n_extraction_failures=n_failures,
        per_question_correct=per_question_correct,
    )
