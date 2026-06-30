"""Tests for faithfulness and accuracy metric computation."""

import numpy as np

from src.metrics.faithfulness import bootstrap_ci, compute_match_fraction_per_question


def test_match_fraction_all_match():
    cot_answers = ["A"] * 20
    no_cot = "A"
    assert compute_match_fraction_per_question(cot_answers, no_cot) == 1.0


def test_match_fraction_none_match():
    cot_answers = ["B"] * 20
    no_cot = "A"
    assert compute_match_fraction_per_question(cot_answers, no_cot) == 0.0


def test_match_fraction_partial():
    cot_answers = ["A"] * 10 + ["B"] * 10
    no_cot = "A"
    assert compute_match_fraction_per_question(cot_answers, no_cot) == 0.5


def test_match_fraction_with_none_cot():
    """None CoT answers treated as non-matching."""
    cot_answers = ["A"] * 10 + [None] * 10
    no_cot = "A"
    assert compute_match_fraction_per_question(cot_answers, no_cot) == 0.5


def test_match_fraction_none_no_cot():
    """None no-CoT answer -> returns None (question excluded)."""
    cot_answers = ["A"] * 20
    no_cot = None
    assert compute_match_fraction_per_question(cot_answers, no_cot) is None


def test_match_fraction_all_none_cot():
    """All CoT answers None -> fraction = 0.0."""
    cot_answers = [None] * 20
    no_cot = "A"
    assert compute_match_fraction_per_question(cot_answers, no_cot) == 0.0


def test_bootstrap_ci_deterministic():
    values = [0.5, 0.6, 0.7, 0.8, 0.9]
    lower1, upper1 = bootstrap_ci(values, n_bootstrap=1000, seed=42)
    lower2, upper2 = bootstrap_ci(values, n_bootstrap=1000, seed=42)
    assert lower1 == lower2
    assert upper1 == upper2


def test_bootstrap_ci_bounds():
    """CI should contain the sample mean."""
    values = list(np.random.RandomState(42).uniform(0, 1, 100))
    lower, upper = bootstrap_ci(values, n_bootstrap=1000, seed=42)
    mean = np.mean(values)
    assert lower <= mean <= upper


def test_bootstrap_ci_constant_values():
    values = [0.7] * 50
    lower, upper = bootstrap_ci(values, n_bootstrap=1000, seed=42)
    assert abs(lower - 0.7) < 1e-10
    assert abs(upper - 0.7) < 1e-10


def test_bootstrap_ci_width_decreases_with_n():
    """Larger sample -> narrower CI."""
    rng = np.random.RandomState(42)
    small = list(rng.uniform(0, 1, 10))
    large = list(rng.uniform(0, 1, 1000))
    l_small, u_small = bootstrap_ci(small, n_bootstrap=1000, seed=42)
    l_large, u_large = bootstrap_ci(large, n_bootstrap=1000, seed=42)
    assert (u_large - l_large) < (u_small - l_small)
