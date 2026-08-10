import polars as pl

from alpha_forge.data.regime import regime_on


def _regime_frame():
    return pl.DataFrame(
        {
            "date": pl.date_range(pl.date(2025, 1, 1), pl.date(2025, 1, 10), "1d", eager=True),
            "spy_close": [100.0] * 10,
            "vix_close": [15.0] * 10,
            "trend": ["BULL"] * 5 + ["BEAR"] * 5,
            "vol_state": ["LOW"] * 10,
            "chop": ["SMOOTH"] * 10,
        }
    )


def test_regime_on_exact_and_between_dates():
    r = _regime_frame()
    assert regime_on(r, "2025-01-03")["trend"] == "BULL"
    assert regime_on(r, "2025-01-08")["trend"] == "BEAR"
    # a date past the series end uses the last known regime
    assert regime_on(r, "2025-02-01")["trend"] == "BEAR"


def test_regime_before_series_is_unknown():
    r = _regime_frame()
    assert regime_on(r, "2024-12-01")["trend"] == "UNKNOWN"
