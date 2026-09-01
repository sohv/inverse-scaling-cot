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


def test_prompt_variant_v0_is_byte_identical_to_original():
    from src.data.loader import Question
    from src.generation.templates import build_cot_final_answer_messages, build_cot_messages, build_no_cot_messages

    q = Question(
        id="x",
        dataset_name="aqua",
        question_text="2+2?",
        choices=["3", "4"],
        choice_labels=["A", "B"],
        correct_label="B",
        raw_index=0,
    )
    assert build_cot_messages(q) == build_cot_messages(q, "v0")
    assert build_no_cot_messages(q) == build_no_cot_messages(q, "v0")
    assert build_cot_final_answer_messages(q, "c") == build_cot_final_answer_messages(q, "c", "v0")


def test_prompt_variants_differ_and_reject_unknown_names():
    from src.data.loader import Question
    from src.generation.templates import PROMPT_VARIANTS, build_cot_messages

    q = Question(
        id="x",
        dataset_name="aqua",
        question_text="2+2?",
        choices=["3", "4"],
        choice_labels=["A", "B"],
        correct_label="B",
        raw_index=0,
    )
    rendered = {v: build_cot_messages(q, v)[1]["content"] for v in PROMPT_VARIANTS}
    assert len(set(rendered.values())) == len(PROMPT_VARIANTS)
    for text in rendered.values():
        assert "2+2?" in text and "A) 3" in text
    with pytest.raises(ValueError):
        build_cot_messages(q, "does_not_exist")


def test_permute_choices_preserves_the_correct_option():
    import random

    from src.data.loader import Question, permute_choices

    q = Question(
        id="x",
        dataset_name="aqua",
        question_text="q",
        choices=["w", "x", "y", "z"],
        choice_labels=list("ABCD"),
        correct_label="C",
        raw_index=0,
    )
    for seed in range(10):
        p = permute_choices(q, random.Random(seed))
        assert sorted(p.choices) == sorted(q.choices)
        assert p.choice_labels == q.choice_labels
        assert p.choices[p.choice_labels.index(p.correct_label)] == q.choices[q.choice_labels.index(q.correct_label)]


def test_permute_choices_is_seed_deterministic():
    import random

    from src.data.loader import Question, permute_choices

    q = Question(
        id="x",
        dataset_name="aqua",
        question_text="q",
        choices=["a", "b", "c", "d", "e"],
        choice_labels=list("ABCDE"),
        correct_label="A",
        raw_index=0,
    )
    assert permute_choices(q, random.Random(7)).choices == permute_choices(q, random.Random(7)).choices


def test_cells_to_table_rejects_duplicate_cells():
    """Pooling results dirs must not silently double-count a (model, dataset) cell."""
    from src.metrics.cells import cells_to_table
    from src.metrics.recompute import CellMetrics

    def make(model_id):
        return CellMetrics(
            model_id=model_id,
            dataset_name="aqua",
            n_questions=1,
            n_questions_used=1,
            n_cot_samples_used=20,
            parser="permissive",
            failure_mode="non_match",
            proxy=0.5,
            proxy_std=0.0,
            proxy_ci_lower=0.5,
            proxy_ci_upper=0.5,
            accuracy_no_cot=0.5,
            n_correct=1,
            n_cot_extraction_failures=0,
            n_no_cot_extraction_failures=0,
            per_question_fractions=[0.5],
        )

    cells = [make("Qwen/Qwen2.5-7B-Instruct"), make("Qwen/Qwen2.5-7B-Instruct")]
    with pytest.raises(ValueError, match="double-counted"):
        cells_to_table(cells)


def test_cell_dir_path_does_not_create_directories(tmp_path):
    """Existence checks must not litter the results tree with empty cell directories."""
    from src.utils.config import cell_dir_path, make_output_dir

    path = cell_dir_path(str(tmp_path), "Qwen/Qwen2.5-7B-Instruct", "aqua")
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []

    created = make_output_dir(str(tmp_path), "Qwen/Qwen2.5-7B-Instruct", "aqua")
    assert created == path and created.exists()
