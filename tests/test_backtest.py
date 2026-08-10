import numpy as np
import polars as pl
import pytest

from alpha_forge.research.backtest import build_month_panel, momentum_scores, run_config


def _tiny_panel(n_days=560, n_syms=25, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = np.datetime64("2024-01-02")
    all_days = np.arange(start, start + np.timedelta64(2 * n_days, "D"))
    bdays = all_days[(all_days.astype("datetime64[D]").view("int64") % 7) < 5][:n_days]
    frames = []
    for s in range(n_syms):
        drift = rng.normal(0.0002, 0.0004)
        close = 30 * np.exp(np.cumsum(rng.normal(drift, 0.02, n_days)))
        open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.003, n_days))
        frames.append(
            pl.DataFrame(
                {
                    "symbol": [f"S{s:02d}"] * n_days,
                    "date": bdays.astype("datetime64[us]"),
                    "open": open_,
                    "high": np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days))),
                    "low": np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days))),
                    "close": close,
                    "volume": np.full(n_days, 2e6),
                }
            )
        )
    return pl.concat(frames)


@pytest.fixture(scope="module")
def mp():
    # dates start 2024 so every sell date has verified SEC/TAF rates
    return build_month_panel(_tiny_panel())


def test_next_open_fill_arithmetic(mp):
    # r_gross[m, s] must equal open[entry_{m+1}]/open[entry_m] - 1 exactly
    m, s = 3, 7
    e_in, e_out = mp.entry_idx[3], mp.entry_idx[4]
    expected = mp.open_[e_out, s] / mp.open_[e_in, s] - 1.0
    assert mp.r_gross[m, s] == pytest.approx(expected, abs=1e-12)


def test_costs_always_positive_and_applied(mp):
    diff = mp.r_gross - mp.r_net
    d = diff[~np.isnan(diff)]
    assert d.size > 0
    # every traded name-month pays at least a tick of half-spread twice
    assert np.all(d > 0)


def test_run_config_months_align(mp):
    scores = momentum_scores(mp, formation_months=3, skip_months=0)
    res = run_config(mp, scores, top_k=3)
    assert res["net_returns"].size == res["month_indices"].size
    assert res["n_trade_events"] >= res["net_returns"].size * 3
    # every traded month index must be valid for the panel
    assert res["month_indices"].max() < mp.r_net.shape[0]


def test_momentum_score_uses_only_past_closes(mp):
    scores = momentum_scores(mp, formation_months=3, skip_months=1)
    m = 8
    si = mp.signal_idx[m]
    s = 4
    i_end, i_start = si - 21, si - 21 - 63
    expected = mp.close_hist[i_end, s] / mp.close_hist[i_start, s] - 1.0
    assert scores[m, s] == pytest.approx(expected, abs=1e-12)
