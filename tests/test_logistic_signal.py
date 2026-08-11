import numpy as np
import pytest

from alpha_forge.research.eventbt import fold_weights, ridge_logistic


def _groups(n_groups=60, sep=(2.0, -1.0, 0.0), seed=0):
    """Matched groups where the hit is shifted by `sep` per feature."""
    rng = np.random.default_rng(seed)
    out = []
    d0 = np.datetime64("2020-01-01")
    for i in range(n_groups):
        ctrl = rng.normal(0, 1, size=(5, len(sep)))
        hit = rng.normal(0, 1, size=len(sep)) + np.asarray(sep)
        out.append((d0 + np.timedelta64(i, "D"), np.vstack([hit, ctrl])))
    return {5: out}


def test_ridge_logistic_recovers_signs():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(2000, 3))
    logit = 2.0 * X[:, 0] - 1.0 * X[:, 1]  # feature 2 irrelevant
    y = (rng.random(2000) < 1 / (1 + np.exp(-logit))).astype(float)
    w = ridge_logistic(X, y, lam=1.0)
    assert w[0] > 0.5 and w[1] < -0.2
    assert abs(w[2]) < abs(w[1])  # irrelevant feature shrunk below real ones


def test_ridge_shrinks_relative_to_weak_regularization():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, size=(300, 2))
    y = (rng.random(300) < 1 / (1 + np.exp(-3 * X[:, 0]))).astype(float)
    w_strong = ridge_logistic(X, y, lam=10.0)
    w_weak = ridge_logistic(X, y, lam=0.01)
    assert abs(w_strong[0]) < abs(w_weak[0])


def test_fold_weights_orders_features_by_strength():
    feats = ["f_strong_up", "f_weak_down", "f_noise"]
    w = fold_weights(_groups(), 5, np.datetime64("2021-01-01"), feats)
    assert w["f_strong_up"] > 0
    assert w["f_weak_down"] < 0
    assert abs(w["f_strong_up"]) > abs(w["f_weak_down"]) > abs(w.get("f_noise", 0.0))


def test_fold_weights_respects_maturation_cutoff():
    feats = ["a", "b", "c"]
    groups = _groups(n_groups=60)
    # cutoff before every group's date -> nothing matured -> no signal
    assert fold_weights(groups, 5, np.datetime64("2019-01-01"), feats) == {}
    # cutoff midway: only matured groups train (still >= 20 -> trains)
    w = fold_weights(groups, 5, np.datetime64("2020-02-01"), feats)
    assert w  # 31 matured groups suffice


def test_fold_weights_min_group_floor():
    groups = _groups(n_groups=10)
    assert fold_weights(groups, 5, np.datetime64("2021-01-01"), ["a", "b", "c"]) == {}
