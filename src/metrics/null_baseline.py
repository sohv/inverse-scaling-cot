"""Empirical shuffled-CoT null distribution (revision items 12 and 13).

Replaces the assumed 0.25 chance level with a permutation null built from the cell's own
answer distribution: CoT traces are repeatedly reassigned across questions and the
match rate recomputed. Vectorised over the encoded answer matrix, so 1000 permutations
across 55 cells is seconds of CPU.
"""

import logging

import numpy as np
from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)


class NullResult(BaseModel):
    """Real vs empirical-null match rates for one (model, dataset) cell."""

    model_id: str
    dataset_name: str
    n_questions_used: int
    n_permutations: int
    real_match_rate: float
    shuffled_mean: float
    shuffled_std: float
    null_ci_lower: float
    null_ci_upper: float
    dependence: float  # real - shuffled_mean, positive when the model uses its own CoT
    real_above_null: bool  # real rate lies above the empirical 97.5th percentile
    uniform_chance: float  # 1 / n_choices, the baseline the original analysis assumed


def random_derangement(n: int, rng: np.random.Generator, max_attempts: int = 100) -> np.ndarray:
    """Permutation with no fixed point, by rejection sampling (~2.7 draws expected)."""
    for _ in range(max_attempts):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    raise RuntimeError(f"failed to draw a derangement of size {n} in {max_attempts} attempts")


def empirical_null(
    cot: np.ndarray,
    no_cot: np.ndarray,
    valid: np.ndarray,
    model_id: str,
    dataset_name: str,
    n_choices: int,
    n_permutations: int = 1000,
    seed: int = 42,
) -> NullResult:
    """Build the permutation null for one cell.

    cot: (n_questions, n_samples) int, -1 = extraction failure (never matches).
    Failed CoT extractions stay in the denominator, matching the non_match convention.
    """
    rng = np.random.default_rng(seed)
    n = len(no_cot)
    idx = np.where(valid)[0]
    target = no_cot[idx][:, None]

    real_rate = float((cot[idx] == target).mean(axis=1).mean())

    rates = np.empty(n_permutations)
    for p in range(n_permutations):
        donors = random_derangement(n, rng)[idx]
        rates[p] = (cot[donors] == target).mean(axis=1).mean()

    return NullResult(
        model_id=model_id,
        dataset_name=dataset_name,
        n_questions_used=len(idx),
        n_permutations=n_permutations,
        real_match_rate=real_rate,
        shuffled_mean=float(rates.mean()),
        shuffled_std=float(rates.std()),
        null_ci_lower=float(np.percentile(rates, 2.5)),
        null_ci_upper=float(np.percentile(rates, 97.5)),
        dependence=real_rate - float(rates.mean()),
        real_above_null=bool(real_rate > np.percentile(rates, 97.5)),
        uniform_chance=1.0 / n_choices,
    )
