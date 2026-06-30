"""Tests for prompt template construction."""

from src.data.loader import Question
from src.generation.templates import (
    build_cot_final_answer_messages,
    build_cot_messages,
    build_no_cot_messages,
    format_choices,
)

SAMPLE_QUESTION = Question(
    id="test_0",
    dataset_name="aqua",
    question_text="What is 2 + 2?",
    choices=["3", "4", "5", "6"],
    choice_labels=["A", "B", "C", "D"],
    correct_label="B",
    raw_index=0,
)

HELLASWAG_QUESTION = Question(
    id="test_hs_0",
    dataset_name="hellaswag",
    question_text="A person picks up a guitar and",
    choices=["throws it", "plays a song", "eats it", "puts it down"],
    choice_labels=["A", "B", "C", "D"],
    correct_label="B",
    raw_index=0,
)


def test_format_choices():
    result = format_choices(["A", "B", "C", "D"], ["3", "4", "5", "6"])
    assert "A) 3" in result
    assert "D) 6" in result
    assert result.count("\n") == 3


def test_cot_messages_structure():
    messages = build_cot_messages(SAMPLE_QUESTION)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "step by step" in messages[1]["content"].lower()


def test_cot_messages_contain_question():
    messages = build_cot_messages(SAMPLE_QUESTION)
    assert "What is 2 + 2?" in messages[1]["content"]
    assert "A) 3" in messages[1]["content"]


def test_no_cot_messages_end_with_answer_prompt():
    messages = build_no_cot_messages(SAMPLE_QUESTION)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["content"].rstrip().endswith("The answer is (")


def test_no_cot_system_prompt_says_no_reasoning():
    messages = build_no_cot_messages(SAMPLE_QUESTION)
    assert "without" in messages[0]["content"].lower()


def test_cot_final_answer_messages():
    messages = build_cot_final_answer_messages(SAMPLE_QUESTION, "I think the answer is 4.")
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "I think the answer is 4."
    assert messages[3]["role"] == "user"
    assert "The answer is (" in messages[3]["content"]


def test_hellaswag_uses_completion_template():
    messages = build_cot_messages(HELLASWAG_QUESTION)
    assert "Complete the following passage" in messages[1]["content"]


def test_standard_mc_does_not_use_completion_template():
    messages = build_cot_messages(SAMPLE_QUESTION)
    assert "Complete the following passage" not in messages[1]["content"]
