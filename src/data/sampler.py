import logging
import random
from pathlib import Path

from src.data.loader import Question, load_dataset_questions
from src.utils.io import read_jsonl, write_jsonl

LOGGER = logging.getLogger(__name__)


def sample_questions(
    dataset_name: str,
    n: int = 100,
    seed: int = 42,
) -> list[Question]:
    """Load dataset, sample n questions with fixed seed, return sorted by raw_index.

    If dataset has fewer than n questions in the split, use all of them and log a warning.
    """
    all_questions = load_dataset_questions(dataset_name)

    if len(all_questions) <= n:
        LOGGER.warning(f"{dataset_name} has only {len(all_questions)} questions, using all of them (requested {n})")
        return sorted(all_questions, key=lambda q: q.raw_index)

    rng = random.Random(seed)
    sampled = rng.sample(all_questions, n)
    sampled.sort(key=lambda q: q.raw_index)
    LOGGER.info(f"Sampled {len(sampled)} questions from {dataset_name} with seed={seed}")
    return sampled


def save_sample_ids(questions: list[Question], output_path: str | Path) -> None:
    """Save sampled question IDs to JSONL for reproducibility audit."""
    records = [{"id": q.id, "dataset_name": q.dataset_name, "raw_index": q.raw_index} for q in questions]
    write_jsonl(output_path, records)


def load_sample_ids(path: str | Path) -> list[str]:
    """Load previously saved sample IDs. Returns list of question id strings."""
    records = read_jsonl(path)
    return [r["id"] for r in records]


def load_or_sample_questions(
    dataset_name: str,
    splits_dir: str | Path,
    n: int = 100,
    seed: int = 42,
) -> list[Question]:
    """Load questions using saved sample IDs if available, otherwise sample and save.

    This ensures all experiments use the exact same 100 questions per dataset.
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    ids_path = splits_dir / f"{dataset_name}_sample_ids.jsonl"

    if ids_path.exists():
        LOGGER.info(f"Loading existing sample IDs from {ids_path}")
        saved_ids = set(load_sample_ids(ids_path))
        all_questions = load_dataset_questions(dataset_name)
        questions = [q for q in all_questions if q.id in saved_ids]
        questions.sort(key=lambda q: q.raw_index)
        LOGGER.info(f"Loaded {len(questions)} questions from saved IDs")
        return questions

    LOGGER.info(f"No saved sample IDs found, sampling fresh for {dataset_name}")
    questions = sample_questions(dataset_name, n=n, seed=seed)
    save_sample_ids(questions, ids_path)
    return questions
