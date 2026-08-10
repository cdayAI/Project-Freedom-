import numpy as np
import pytest

from alpha_forge.gates.permutation import benjamini_hochberg, bonferroni, permutation_pvalue


def _mean_of_selected(data: np.ndarray) -> float:
    signal, outcome = data
    return float(outcome[signal > np.median(signal)].mean())


def _shuffle_alignment(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    signal, outcome = data
    return np.stack([signal, rng.permutation(outcome)])


def test_real_signal_detected():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=400)
    outcome = 0.5 * signal + rng.normal(size=400)  # genuine link
    data = np.stack([signal, outcome])
    res = permutation_pvalue(_mean_of_selected(data), data, _mean_of_selected,
                             _shuffle_alignment, n_permutations=20000, seed=1)
    assert res["p_value"] < 0.001


def test_no_signal_not_detected():
    rng = np.random.default_rng(5)
    signal = rng.normal(size=400)
    outcome = rng.normal(size=400)  # independent
    data = np.stack([signal, outcome])
    res = permutation_pvalue(_mean_of_selected(data), data, _mean_of_selected,
                             _shuffle_alignment, n_permutations=20000, seed=2)
    assert res["p_value"] > 0.05


def test_minimum_shuffles_enforced():
    data = np.stack([np.arange(10.0), np.arange(10.0)])
    with pytest.raises(ValueError):
        permutation_pvalue(1.0, data, _mean_of_selected, _shuffle_alignment,
                           n_permutations=1000)


def test_pvalue_never_zero():
    # add-one estimator: even a perfect statistic has p >= 1/(n+1)
    rng = np.random.default_rng(9)
    signal = np.linspace(-1, 1, 100)
    outcome = signal * 100.0
    data = np.stack([signal, outcome])
    res = permutation_pvalue(_mean_of_selected(data), data, _mean_of_selected,
                             _shuffle_alignment, n_permutations=20000, seed=3)
    assert res["p_value"] >= 1.0 / 20001


def test_benjamini_hochberg_known_example():
    # classic worked example: m=6, q=0.25
    pvals = [0.010, 0.013, 0.055, 0.043, 0.894, 0.085]
    rejected = benjamini_hochberg(pvals, q=0.25)
    # sorted: .010 .013 .043 .055 .085 .894 vs k/m*q thresholds
    # .0417 .0833 .125 .1667 .2083 .25 -> largest passing k=5, so the five
    # smallest p-values are rejected (step-up rejects .085 even though it
    # exceeds its own threshold-free intuition)
    assert rejected == [True, True, True, True, False, True]


def test_bonferroni():
    assert bonferroni([0.01, 0.4, 0.02], alpha=0.05) == [True, False, False]
