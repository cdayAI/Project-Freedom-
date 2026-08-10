import numpy as np
import pytest

from alpha_forge.costs.slippage import (
    abdi_ranaldo_spread,
    corwin_schultz_spread,
    effective_half_spread,
)


def _spread_series(s: float, n: int = 4000, seed: int = 0):
    """Random-walk mid observed through proportional spread s."""
    rng = np.random.default_rng(seed)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
    half = s / 2
    high = mid * (1 + half)
    low = mid * (1 - half) * np.exp(-np.abs(rng.normal(0, 1e-4, n)))
    close = np.where(rng.random(n) > 0.5, high, low)
    return high, low, close


def test_ar_estimator_recovers_synthetic_spread():
    h, l, c = _spread_series(0.02)
    ar2 = abdi_ranaldo_spread(h, l, c)
    est = float(np.sqrt(max(np.nanmean(ar2), 0.0)))
    assert est == pytest.approx(0.02, rel=0.5)  # right order of magnitude


def test_blend_is_at_least_each_estimator():
    h, l, c = _spread_series(0.03, seed=2)
    half = effective_half_spread(h, l, c)
    cs = corwin_schultz_spread(h, l, c)
    # spot check away from warmup: blend >= CS median/2 at the same index
    i = 200
    w = cs[i - 20 : i + 1]
    w = w[~np.isnan(w)]
    assert half[i] >= np.median(w) / 2.0 - 1e-12


def test_blend_never_below_tick_floor():
    n = 300
    c = np.full(n, 50.0)
    h = c * 1.0001
    l = c * 0.9999
    half = effective_half_spread(h, l, c)
    assert np.all(half[~np.isnan(half)] >= 0.005 / 50.0 - 1e-15)


def test_ar_slot_zero_is_nan_and_length_preserved():
    h, l, c = _spread_series(0.01, n=50, seed=3)
    ar2 = abdi_ranaldo_spread(h, l, c)
    assert np.isnan(ar2[0])
    assert ar2.size == 50
