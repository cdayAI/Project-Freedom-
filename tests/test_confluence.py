import numpy as np
import pytest

from alpha_forge.research.confluence import _auc, _permutation_auc_pvalue


def test_auc_perfect_separation():
    assert _auc(np.array([4.0, 5.0, 6.0]), np.array([1.0, 2.0, 3.0])) == 1.0
    assert _auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0


def test_auc_ties_are_half():
    assert _auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.5)


def test_auc_matches_probability_interpretation():
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 1.0, 500)
    neg = rng.normal(0.0, 1.0, 500)
    # analytic AUC for N(1,1) vs N(0,1) = Phi(1/sqrt(2)) ~ 0.7602
    assert _auc(pos, neg) == pytest.approx(0.7602, abs=0.03)


def test_permutation_detects_real_separation():
    rng = np.random.default_rng(1)
    pos = rng.normal(0.8, 1.0, 100)
    neg = rng.normal(0.0, 1.0, 400)
    auc, p = _permutation_auc_pvalue(pos, neg, 20_000, np.random.default_rng(2))
    assert auc > 0.6
    assert p < 0.001


def test_permutation_null_not_significant():
    rng = np.random.default_rng(3)
    pos = rng.normal(0.0, 1.0, 100)
    neg = rng.normal(0.0, 1.0, 400)
    _, p = _permutation_auc_pvalue(pos, neg, 20_000, np.random.default_rng(4))
    assert p > 0.05


def test_permutation_pvalue_never_zero():
    pos = np.arange(50.0) + 100.0
    neg = np.arange(200.0)
    _, p = _permutation_auc_pvalue(pos, neg, 20_000, np.random.default_rng(5))
    assert p >= 1.0 / 20_001
