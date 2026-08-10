"""Permutation tests and multiple-testing corrections.

The permutation p-value uses the add-one estimator (Phipson & Smyth 2010):
p = (1 + #{permuted stat >= observed}) / (1 + n_permutations), which is never
exactly zero — an honest floor given finite shuffles.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from alpha_forge.config import PERMUTATION_MIN_SHUFFLES


def permutation_pvalue(
    observed_stat: float,
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    permute_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    n_permutations: int = PERMUTATION_MIN_SHUFFLES,
    seed: int = 0,
) -> dict:
    """One-sided (greater) permutation test.

    permute_fn must implement the null: it destroys exactly the structure the
    hypothesis claims (e.g. shuffle signal/outcome alignment) while preserving
    everything else (marginals, autocorrelation where possible).
    """
    if n_permutations < PERMUTATION_MIN_SHUFFLES:
        raise ValueError(
            f"minimum {PERMUTATION_MIN_SHUFFLES} shuffles required, got {n_permutations}"
        )
    rng = np.random.default_rng(seed)
    exceed = 0
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        null_stats[i] = stat_fn(permute_fn(data, rng))
        if null_stats[i] >= observed_stat:
            exceed += 1
    p = (1 + exceed) / (1 + n_permutations)
    return {
        "p_value": float(p),
        "observed": float(observed_stat),
        "null_mean": float(null_stats.mean()),
        "null_p95": float(np.percentile(null_stats, 95)),
        "n_permutations": int(n_permutations),
    }


def bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(p_values)
    return [p <= alpha / m for p in p_values]


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """BH step-up FDR control. Returns rejection mask in input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.asarray(p_values, dtype=float)[order]
    thresholds = q * (np.arange(1, m + 1) / m)
    below = np.nonzero(sorted_p <= thresholds)[0]
    k = below.max() + 1 if below.size else 0
    reject = np.zeros(m, dtype=bool)
    reject[order[:k]] = True
    return reject.tolist()
