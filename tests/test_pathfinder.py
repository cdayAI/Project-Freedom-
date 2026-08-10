import numpy as np
import polars as pl
import pytest

from alpha_forge.research.features import fingerprint_at
from alpha_forge.research.pathfinder import enumerate_paths


def _panel_from_close(symbol: str, close: np.ndarray, start="2020-01-01") -> pl.DataFrame:
    n = close.size
    dates = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(2 * n, "D"))
    bdays = dates[(dates.astype("datetime64[D]").view("int64") % 7) < 5][:n]
    open_ = np.concatenate([[close[0]], close[:-1]])  # opens at prior close
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": bdays.astype("datetime64[us]"),
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": np.full(n, 1e6),
        }
    )


def test_engineered_10x_path_detected():
    n = 400
    close = np.full(n, 10.0)
    # 12x ramp inside days 150..210 (fits easily in one 126-day window)
    ramp = np.linspace(10, 120, 61)
    close[150:211] = ramp
    close[211:] = 120.0
    panel = _panel_from_close("MOON", close)
    hits, counts = enumerate_paths(panel)
    assert hits.height > 0
    tens = hits.filter(pl.col("n_multiple") == 10)
    assert tens.height > 0
    row = tens.row(0, named=True)
    assert row["achieved_multiple"] >= 10.0
    assert row["catalyst_class"] == "NONE"
    # 50x must NOT be claimed
    assert hits.filter(pl.col("n_multiple") == 50).height == 0


def test_flat_series_has_no_paths():
    close = np.full(300, 25.0) * np.exp(np.linspace(0, 0.1, 300))  # gentle drift
    panel = _panel_from_close("FLAT", close)
    hits, counts = enumerate_paths(panel)
    assert hits.height == 0


def test_entry_basis_is_next_open_never_signal_close():
    # flash dip: the series trades at 50, closes ONE bar at 10 (bar 200),
    # gaps back to 50 at the next open, then runs to 120. The only 5x+
    # arithmetic available is 120/10 — but 10 was a CLOSE, and signals on a
    # close fill at the NEXT session's open (50, the recovery gap). Honest
    # best multiple is 120/50 = 2.4x, so a correct enumerator finds nothing;
    # an enumerator that uses the signal bar's close as the entry basis
    # manufactures a fake 12x.
    n = 400
    close = np.full(n, 50.0)
    close[200] = 10.0
    close[201:] = 120.0
    panel = _panel_from_close("DIP", close).with_columns(
        pl.when(pl.arange(0, n) == 201)
        .then(50.0)  # overnight recovery: next open never traded at 10
        .otherwise(pl.col("open"))
        .alias("open")
    )
    hits, _ = enumerate_paths(panel)
    assert hits.height == 0


def test_fingerprint_no_lookahead():
    rng = np.random.default_rng(0)
    n = 300
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    vol = rng.integers(1e5, 1e6, n).astype(float)
    i = 150
    fp_before = fingerprint_at(i, open_, high, low, close, vol)
    # mutate ALL future bars; fingerprint at i must not move
    close2, open2, high2, low2, vol2 = (a.copy() for a in (close, open_, high, low, vol))
    close2[i + 1 :] *= 37.0
    high2[i + 1 :] *= 41.0
    low2[i + 1 :] *= 0.01
    vol2[i + 1 :] = 9e9
    fp_after = fingerprint_at(i, open2, high2, low2, close2, vol2)
    for k, v in fp_before.items():
        if isinstance(v, float) and np.isnan(v):
            assert np.isnan(fp_after[k])
        else:
            assert fp_after[k] == pytest.approx(v, abs=1e-12), f"lookahead in {k}"
