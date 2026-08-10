import numpy as np
import pytest

from alpha_forge.research.sizing import kelly_fraction, sizing_frontier


def test_kelly_binary_bet():
    # win +100% w.p. 0.6, lose -100% w.p. 0.4 -> f* = 2p-1 = 0.2
    rng = np.random.default_rng(0)
    r = np.where(rng.random(200_000) < 0.6, 1.0, -1.0)
    f = kelly_fraction(r)
    assert f == pytest.approx(0.2, abs=0.02)


def test_frontier_shapes_and_constraint():
    rng = np.random.default_rng(1)
    r = rng.normal(0.02, 0.10, 500)  # decent edge
    out = sizing_frontier(r, n_trades_per_path=100, n_paths=20_000, seed=3)
    fr = out["frontier"]
    assert len(fr) >= 10
    # ruin probability is monotone nondecreasing in fraction
    ruins = [row["p_ruin_90pct_loss"] for row in fr]
    assert all(b >= a - 1e-9 for a, b in zip(ruins, ruins[1:]))
    # constrained optimum respects the constraint, unconstrained may not
    c, u = out["constrained_optimum"], out["unconstrained_optimum"]
    assert c is not None
    assert c["p_ruin_90pct_loss"] < 0.05
    assert u["median_terminal_wealth"] >= c["median_terminal_wealth"] - 1e-9


def test_min_paths_enforced():
    with pytest.raises(ValueError):
        sizing_frontier(np.random.default_rng(0).normal(0.01, 0.05, 100), n_paths=1000)


def test_unsizeable_edge_reported():
    rng = np.random.default_rng(2)
    r = rng.normal(-0.05, 0.30, 300)  # strongly negative edge
    out = sizing_frontier(r, n_paths=20_000, seed=4)
    # honest note when nothing satisfies the ruin constraint, or the
    # constrained optimum sits at the smallest fraction
    if out["constrained_optimum"] is None:
        assert "unsizeable" in out["note"]
    else:
        assert out["constrained_optimum"]["fraction"] <= 0.05
