"""Tests for dataset loading and parsing.

These tests require internet access to download HuggingFace datasets.
"""

from src.data.loader import DATASET_CONFIGS, Question, load_dataset_questions


def test_load_aqua_returns_questions():
    """AQuA loads and returns Question objects with 5 choices."""
    questions = load_dataset_questions("aqua")
    assert len(questions) > 0
    q = questions[0]
    assert isinstance(q, Question)
    assert q.dataset_name == "aqua"
    assert len(q.choice_labels) == 5
    assert q.correct_label in q.choice_labels


def test_load_logiqa_returns_questions():
    """LogiQA loads with 4 choices and letter labels."""
    questions = load_dataset_questions("logiqa")
    assert len(questions) > 0
    q = questions[0]
    assert q.dataset_name == "logiqa"
    assert len(q.choices) == 4
    assert q.correct_label in ["A", "B", "C", "D"]


def test_load_arc_challenge_returns_questions():
    """ARC-Challenge loads with valid answer keys."""
    questions = load_dataset_questions("arc_challenge")
    assert len(questions) > 0
    q = questions[0]
    assert q.dataset_name == "arc_challenge"
    assert q.correct_label in q.choice_labels


def test_load_openbookqa_returns_questions():
    """OpenBookQA loads with valid answer keys."""
    questions = load_dataset_questions("openbookqa")
    assert len(questions) > 0
    q = questions[0]
    assert q.dataset_name == "openbookqa"
    assert q.correct_label in q.choice_labels


def test_load_hellaswag_label_mapping():
    """HellaSwag numeric labels get mapped to A/B/C/D."""
    questions = load_dataset_questions("hellaswag")
    assert len(questions) > 0
    for q in questions[:10]:
        assert q.correct_label in ["A", "B", "C", "D"]
        assert len(q.choices) == 4


def test_all_datasets_loadable():
    """Every configured dataset loads without error."""
    for name in DATASET_CONFIGS:
        questions = load_dataset_questions(name)
        assert len(questions) > 0, f"{name} returned no questions"


def test_question_ids_are_unique():
    """Each question within a dataset has a unique ID."""
    for name in ["aqua", "arc_challenge"]:
        questions = load_dataset_questions(name)
        ids = [q.id for q in questions]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in {name}"
