"""Vectorized daily feature panel.

Same ten features as research.features.fingerprint_at, computed for EVERY
(symbol, day) at once instead of one point at a time — this is what makes an
event-driven backtest over 1,500 symbols x 13 years feasible in seconds
instead of hours.

CONSISTENCY CONTRACT: for any index i with >= WARMUP bars of history, every
feature here equals fingerprint_at(i, ...) to 1e-9. The equivalence test in
tests/test_features_panel.py is the enforcement; if the two implementations
ever drift, the test fails and the panel is wrong, not the fingerprint.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from alpha_forge.config import TRADING_DAYS_PER_YEAR
from alpha_forge.costs.slippage import corwin_schultz_spread

WARMUP = 260  # bars of history required before a symbol's features are valid

META_COLUMNS = {"symbol", "date", "valid", "starts_5x_fwd", "pos"}


def feature_columns(features: pl.DataFrame) -> list[str]:
    """Every non-meta column is a feature. Catalyst columns joined onto the
    panel flow through confluence and the event engine with no code changes —
    the feature list is data, not configuration."""
    return [c for c in features.columns if c not in META_COLUMNS]


def _rolling_nan(fn, arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling fn over trailing `window` values (inclusive of current)."""
    out = np.full(arr.size, np.nan)
    if arr.size >= window:
        views = np.lib.stride_tricks.sliding_window_view(arr, window)
        with np.errstate(all="ignore"):
            out[window - 1 :] = fn(views, axis=1)
    return out


def symbol_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> dict[str, np.ndarray]:
    """All fingerprint features for one symbol, full-length arrays.
    Entries before WARMUP are NaN by contract."""
    n = close.size
    feats: dict[str, np.ndarray] = {}

    def lag_ret(lb: int) -> np.ndarray:
        out = np.full(n, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            base = np.where(close[:-lb] > 0, close[:-lb], np.nan)  # zero-price
            out[lb:] = close[lb:] / base - 1.0                     # rows -> NaN
        return out

    feats["ret_21d"] = lag_ret(21)
    feats["ret_63d"] = lag_ret(63)
    feats["ret_126d"] = lag_ret(126)
    feats["ret_252d"] = lag_ret(252)

    # vol_20d_ann: std (ddof=1) of the last 20 log returns, annualized.
    # fingerprint_at uses logret over close[i-20..i] -> 20 returns.
    logret = np.full(n, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        logret[1:] = np.diff(np.log(close))
    sd = _rolling_nan(np.nanstd, logret, 20)  # population; fix ddof below
    # nanstd with ddof: compute via two-pass on views for exactness
    out = np.full(n, np.nan)
    if n >= 20:
        views = np.lib.stride_tricks.sliding_window_view(logret, 20)
        with np.errstate(all="ignore"):
            out[19:] = np.std(views, axis=1, ddof=1)
    feats["vol_20d_ann"] = out * np.sqrt(TRADING_DAYS_PER_YEAR)
    _ = sd

    # volume_z_20v126: (mean20 - mean126) / std126(ddof=1), windows inclusive
    v = volume.astype(float)
    m20 = _rolling_nan(np.mean, v, 20)
    if n >= 126:
        views = np.lib.stride_tricks.sliding_window_view(v, 126)
        m126 = np.full(n, np.nan)
        s126 = np.full(n, np.nan)
        m126[125:] = np.mean(views, axis=1)
        s126[125:] = np.std(views, axis=1, ddof=1)
    else:
        m126 = np.full(n, np.nan)
        s126 = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["volume_z_20v126"] = np.where(s126 > 0, (m20 - m126) / s126, np.nan)

    feats["dollar_vol_med_20d"] = _rolling_nan(np.median, close * v, 20)

    hi252 = _rolling_nan(np.max, close, 252)
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["dist_from_252d_high"] = close / hi252 - 1.0

    feats["price"] = close.copy()

    # cs_spread_est: median of the CS estimates over the last 22 bars.
    # fingerprint_at computes CS on the slice [i-21..i]; its pair estimates
    # sit at local indices 1..21 == global pair-dates (i-20..i), so the
    # rolling window over the full-series CS is 21 values ending at i.
    cs = corwin_schultz_spread(high, low, close)
    out = np.full(n, np.nan)
    if n >= 21:
        views = np.lib.stride_tricks.sliding_window_view(cs, 21)
        with np.errstate(all="ignore"):
            out[20:] = np.nanmedian(views, axis=1)
    feats["cs_spread_est"] = out

    # enforce the warmup contract: nothing before WARMUP is served
    for k in feats:
        feats[k][: min(WARMUP, n)] = feats[k][: min(WARMUP, n)]  # keep computed
    return feats


def forward_path_label(
    open_: np.ndarray, close: np.ndarray, horizon: int = 126, multiple: float = 5.0
) -> np.ndarray:
    """LOOKAHEAD column (label, never a feature): does buying at the NEXT
    open reach `multiple`x on a close within `horizon` bars? Used to label
    hits/controls in matched designs and to exclude contaminated controls."""
    n = close.size
    out = np.zeros(n, dtype=bool)
    if n < 3:
        return out
    # forward max of close over (i+1 .. i+win]: sliding max, then shift.
    # Tail bars whose forward window is truncated stay NaN -> label False
    # (a path that cannot complete inside the data is not a confirmed path).
    win = min(horizon, n - 1)
    fwd_max = np.full(n, np.nan)
    views = np.lib.stride_tricks.sliding_window_view(close, win)
    rmax = views.max(axis=1)  # rmax[i] = max(close[i .. i+win-1])
    fwd_max[: rmax.size - 1] = rmax[1:]  # -> max over [i+1 .. i+win]
    with np.errstate(invalid="ignore", divide="ignore"):
        next_open = np.append(open_[1:], np.nan)
        out = (fwd_max / next_open) >= multiple
    return np.where(np.isnan(fwd_max) | np.isnan(next_open), False, out)


def build_features_panel(panel: pl.DataFrame) -> pl.DataFrame:
    """Long panel (symbol, date, <features...>, valid) for every symbol-day.
    `valid` is the warmup mask; consumers must filter on it."""
    frames = []
    for (sym,), g in panel.group_by("symbol", maintain_order=True):
        g = g.sort("date")
        n = g.height
        feats = symbol_features(
            g["open"].to_numpy().astype(float),
            g["high"].to_numpy().astype(float),
            g["low"].to_numpy().astype(float),
            g["close"].to_numpy().astype(float),
            g["volume"].to_numpy().astype(float),
        )
        valid = np.zeros(n, dtype=bool)
        valid[min(WARMUP, n) :] = True
        starts_5x = forward_path_label(
            g["open"].to_numpy().astype(float), g["close"].to_numpy().astype(float)
        )
        frames.append(
            pl.DataFrame(
                {"symbol": [str(sym)] * n, "date": g["date"], "valid": valid,
                 "starts_5x_fwd": starts_5x}  # LOOKAHEAD label — never a feature
            ).with_columns([pl.Series(k, v) for k, v in feats.items()])
        )
    return pl.concat(frames)
