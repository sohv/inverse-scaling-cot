"""Tests for answer extraction from model outputs."""

from src.generation.runner import extract_answer, extract_answer_no_cot

LABELS = ["A", "B", "C", "D"]
LABELS_5 = ["A", "B", "C", "D", "E"]


def test_extract_the_answer_is_with_parens():
    assert extract_answer("The answer is (B)", LABELS) == "B"


def test_extract_the_answer_is_without_parens():
    assert extract_answer("The answer is B", LABELS) == "B"


def test_extract_standalone_parens():
    assert extract_answer("Therefore, (C) is correct.", LABELS) == "C"


def test_extract_last_match():
    text = "I think it might be (A) but actually the answer is (C)"
    assert extract_answer(text, LABELS) == "C"


def test_extract_case_insensitive():
    assert extract_answer("the answer is (b)", LABELS) == "B"


def test_extract_fallback_first_char():
    assert extract_answer("B", LABELS) == "B"


def test_extract_fallback_first_char_with_paren():
    assert extract_answer("D)", LABELS) == "D"


def test_extract_returns_none_for_garbage():
    assert extract_answer("I don't know the answer", LABELS) is None


def test_extract_returns_none_for_invalid_letter():
    # F is not in LABELS (A-D)
    assert extract_answer("The answer is F", LABELS) is None


def test_extract_5_choice():
    assert extract_answer("The answer is (E)", LABELS_5) == "E"


def test_no_cot_starts_with_letter():
    assert extract_answer_no_cot("B)", LABELS) == "B"


def test_no_cot_just_letter():
    assert extract_answer_no_cot("A", LABELS) == "A"


def test_no_cot_letter_with_text():
    assert extract_answer_no_cot("A) The sum is 4", LABELS) == "A"


def test_no_cot_empty_string():
    assert extract_answer_no_cot("", LABELS) is None


def test_no_cot_whitespace_then_letter():
    assert extract_answer_no_cot("  B", LABELS) == "B"


def test_extract_multiline_cot():
    text = """Let me think about this step by step.
First, we consider option A which says 3. But 2+2=4.
Then option B says 4, which is correct.
So the answer is (B)"""
    assert extract_answer(text, LABELS) == "B"
