"""Spread and slippage, modeled from observed price data — never assumed zero.

Without intraday quote data, the bid-ask spread is estimated from daily
high/low ranges via Corwin & Schultz (2012), "A Simple Way to Estimate
Bid-Ask Spreads from Daily High and Low Prices", Journal of Finance 67(2).
The estimator exploits that daily ranges reflect both variance (scales with
time) and spread (does not). Negative two-day estimates are floored at zero,
as the authors recommend. See SOURCES.md.

At a $2,000 account trading liquid names, the half-spread is the dominant
cost; the estimator's known bias (understates spreads in low-volatility
regimes) is recorded in the data manifest as a limitation until real NBBO
snapshots are ingested.
"""

from __future__ import annotations

import numpy as np


def corwin_schultz_spread(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Per-day proportional spread estimates (spread / price), same length as
    input with NaN in slot 0. Overnight-gap adjusted per the paper: shift the
    second day's range by the gap between its low/high and prior close.
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = h.size
    out = np.full(n, np.nan)
    if n < 2:
        return out

    h1, l1 = h[:-1], l[:-1]
    h2, l2 = h[1:].copy(), l[1:].copy()
    # gap adjustment: if day-2 opens beyond day-1 close, slide day-2's range
    prev_close = c[:-1]
    gap_up = np.maximum(l2 - prev_close, 0)
    gap_dn = np.maximum(prev_close - h2, 0)
    h2 = h2 - gap_up + gap_dn
    l2 = l2 - gap_up + gap_dn

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
        hh = np.maximum(h1, h2)
        ll = np.minimum(l1, l2)
        gamma = np.log(hh / ll) ** 2
        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    out[1:] = np.maximum(s, 0.0)  # negative estimates floored, per the paper
    return out


def effective_half_spread(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 21,
) -> np.ndarray:
    """Rolling-median CS spread / 2, floored at half a tick ($0.005/price).
    This is the per-side proportional cost applied to every fill.
    """
    c = np.asarray(close, dtype=float)
    s = corwin_schultz_spread(high, low, close)
    half = np.full(c.size, np.nan)
    if c.size >= lookback:
        windows = np.lib.stride_tricks.sliding_window_view(s, lookback)
        with np.errstate(invalid="ignore"):
            half[lookback - 1 :] = np.nanmedian(windows, axis=1) / 2.0
    # warm-up rows: expanding median over what exists so far
    for i in range(min(lookback - 1, c.size)):
        w = s[: i + 1]
        w = w[~np.isnan(w)]
        if w.size:
            half[i] = np.median(w) / 2.0
    tick_floor = 0.005 / np.maximum(c, 1e-12)  # half of the $0.01 minimum increment
    return np.fmax(half, tick_floor)
