"""The consistency contract: panel features == fingerprint_at to 1e-9 for
every index past warmup. If this fails, the vectorized panel is wrong."""

import numpy as np
import pytest

from alpha_forge.research.features import FEATURE_NAMES, fingerprint_at
from alpha_forge.research.features_panel import WARMUP, symbol_features


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(11)
    n = 800
    close = 20 * np.exp(np.cumsum(rng.normal(0.0004, 0.025, n)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(1e4, 5e6, n).astype(float)
    return open_, high, low, close, volume


def test_equivalence_with_fingerprint_at(series):
    open_, high, low, close, volume = series
    feats = symbol_features(open_, high, low, close, volume)
    rng = np.random.default_rng(0)
    idxs = list(rng.integers(WARMUP, close.size, 40)) + [WARMUP, close.size - 1]
    for i in idxs:
        fp = fingerprint_at(int(i), open_, high, low, close, volume)
        for name in FEATURE_NAMES:
            a, b = fp[name], feats[name][i]
            if np.isnan(a) or np.isnan(b):
                assert np.isnan(a) and np.isnan(b), f"{name}@{i}: {a} vs {b}"
            else:
                assert b == pytest.approx(a, abs=1e-9), f"{name}@{i}: {a} vs {b}"


def test_warmup_region_is_flagged(series):
    open_, high, low, close, volume = series
    feats = symbol_features(open_, high, low, close, volume)
    # ret_252d must be NaN before a year of history exists
    assert np.all(np.isnan(feats["ret_252d"][:252]))


def test_no_lookahead_in_panel(series):
    open_, high, low, close, volume = series
    feats = symbol_features(open_, high, low, close, volume)
    i = 500
    c2, o2, h2, l2, v2 = (a.copy() for a in (close, open_, high, low, volume))
    c2[i + 1 :] *= 13.0
    h2[i + 1 :] *= 17.0
    v2[i + 1 :] = 9e9
    feats2 = symbol_features(o2, h2, l2, c2, v2)
    for name in FEATURE_NAMES:
        a, b = feats[name][i], feats2[name][i]
        if np.isnan(a):
            assert np.isnan(b)
        else:
            assert b == pytest.approx(a, abs=1e-12), f"lookahead in {name}"
