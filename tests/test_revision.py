"""Tests for the revision analysis modules (recompute, strict parser, null baseline, capability)."""

import numpy as np
import pandas as pd
import pytest

from src.data.sampler import choice_labels_by_id
from src.generation.extraction_alt import extract_answer_strict
from src.generation.runner import QuestionResult
from src.metrics.capability import matched_pairs, reconstruct
from src.metrics.null_baseline import empirical_null, random_derangement
from src.metrics.recompute import answer_matrix, recompute_cell
from src.utils.io import read_json, read_jsonl

CELL_DIR = "results/core_sweep/Qwen_Qwen2.5-7B-Instruct__aqua"


@pytest.fixture(scope="module")
def cell():
    records = read_jsonl(f"{CELL_DIR}/generation_results.jsonl")
    results = [QuestionResult(**r) for r in records]
    return results, choice_labels_by_id(results[0].dataset_name)


def test_strict_parser_accepts_exact_forms():
    assert extract_answer_strict("B) 42", list("ABCDE")) == "B"
    assert extract_answer_strict("(C) because", list("ABCDE")) == "C"
    assert extract_answer_strict("the answer is (D)", list("ABCDE")) == "D"


def test_strict_parser_rejects_ambiguous_and_out_of_range():
    assert extract_answer_strict("I think it is C", list("ABCD")) is None
    assert extract_answer_strict("B without paren", list("ABCD")) is None
    assert extract_answer_strict("E) out of range", list("ABCD")) is None


def test_strict_parser_takes_last_phrase_match():
    assert extract_answer_strict("first the answer is (A) then the answer is (B)", list("ABCD")) == "B"


def test_recompute_reproduces_committed_faithfulness(cell):
    results, labels = cell
    stored = read_json(f"{CELL_DIR}/faithfulness.json")
    metrics = recompute_cell(results, labels_by_id=labels, n_bootstrap=0)
    assert metrics.proxy == pytest.approx(stored["mean_match_fraction"], abs=1e-6)


def test_subsampling_is_nested_and_uses_prefix(cell):
    results, labels = cell
    k5 = recompute_cell(results, labels_by_id=labels, n_samples=5, n_bootstrap=0)
    assert k5.n_cot_samples_used == 5

    expected = []
    for r in results:
        if r.no_cot_extracted_answer is None:
            continue
        answers = [s.extracted_answer for s in r.cot_samples[:5]]
        expected.append(sum(1 for a in answers if a == r.no_cot_extracted_answer) / 5)
    assert k5.proxy == pytest.approx(float(np.mean(expected)), abs=1e-9)


def test_exclude_failure_mode_never_lowers_proxy(cell):
    results, labels = cell
    non_match = recompute_cell(results, labels_by_id=labels, n_bootstrap=0, failure_mode="non_match")
    exclude = recompute_cell(results, labels_by_id=labels, n_bootstrap=0, failure_mode="exclude")
    assert exclude.proxy >= non_match.proxy - 1e-12


def test_unknown_failure_mode_raises(cell):
    results, labels = cell
    with pytest.raises(ValueError):
        recompute_cell(results, labels_by_id=labels, failure_mode="nonsense")


def test_derangement_has_no_fixed_points():
    rng = np.random.default_rng(0)
    for n in [2, 5, 50]:
        perm = random_derangement(n, rng)
        assert sorted(perm.tolist()) == list(range(n))
        assert not np.any(perm == np.arange(n))


def test_empirical_null_real_rate_matches_proxy(cell):
    results, labels = cell
    cot, no_cot, valid = answer_matrix(results, labels)
    metrics = recompute_cell(results, labels_by_id=labels, n_bootstrap=0)
    null = empirical_null(cot, no_cot, valid, "m", "aqua", n_choices=5, n_permutations=20, seed=42)
    assert null.real_match_rate == pytest.approx(metrics.proxy, abs=1e-9)
    assert null.dependence == pytest.approx(null.real_match_rate - null.shuffled_mean, abs=1e-12)


def test_answer_matrix_handles_non_letter_labels():
    """3 of the 100 ARC-Challenge questions use numeric choice labels."""
    labels = choice_labels_by_id("arc_challenge")
    assert any(set(v) == {"1", "2", "3", "4"} for v in labels.values())


def _toy_table():
    return pd.DataFrame(
        {
            "model_id": ["s", "l", "s2", "l2"],
            "dataset_name": ["t", "t", "t", "t"],
            "family": ["qwen", "qwen", "qwen", "qwen"],
            "size_b": [1.0, 32.0, 1.0, 2.0],
            "log_params": [9.0, 10.5, 9.0, 9.3],
            "accuracy_no_cot": [0.50, 0.51, 0.70, 0.90],
            "faithfulness_proxy": [0.55, 0.57, 0.72, 0.92],
        }
    )


def test_matched_pairs_respects_tolerance_and_size_ratio():
    pairs = matched_pairs(_toy_table(), raw_size_coef=0.22, acc_tolerance=0.03, min_size_ratio=4.0)
    assert pairs.n_pairs == 1
    only = pairs.pairs[0]
    assert only["small_size_b"] == 1.0 and only["large_size_b"] == 32.0
    assert only["acc_diff"] <= 0.03


def test_matched_pairs_returns_empty_when_nothing_qualifies():
    pairs = matched_pairs(_toy_table(), raw_size_coef=0.22, acc_tolerance=0.0001, min_size_ratio=100.0)
    assert pairs.n_pairs == 0
    assert pairs.pairs == []


def test_reconstruct_reports_sensible_fractions():
    df = pd.read_csv("results/robustness/cell_table.csv")
    recon = reconstruct(df)
    assert 0.0 < recon["fraction_of_slope_reproduced"] <= 1.5
    assert 0.0 <= recon["r_squared"] <= 1.0
    assert recon["mean_absolute_error"] >= 0.0
