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


def abdi_ranaldo_spread(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Per-pair squared-spread estimates via Abdi & Ranaldo (2017), "A Simple
    Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices",
    Review of Financial Studies 30(12). See SOURCES.md.

    s2_t = 4 * (c_t - eta_t) * (c_t - eta_{t+1}), eta = midpoint of daily
    log range, c = log close. Returns the SQUARED estimates (may be negative
    on single pairs; the caller averages before taking the root, per the
    paper). Slot t of the output is the pair (t-1, t), NaN at slot 0.
    """
    h = np.log(np.asarray(high, dtype=float))
    l = np.log(np.asarray(low, dtype=float))
    c = np.log(np.asarray(close, dtype=float))
    n = c.size
    out = np.full(n, np.nan)
    if n < 2:
        return out
    eta = (h + l) / 2.0
    out[1:] = 4.0 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    return out


def _rolling_stat(vals: np.ndarray, lookback: int, fn) -> np.ndarray:
    out = np.full(vals.size, np.nan)
    if vals.size >= lookback:
        windows = np.lib.stride_tricks.sliding_window_view(vals, lookback)
        with np.errstate(invalid="ignore"):
            out[lookback - 1 :] = fn(windows, axis=1)
    for i in range(min(lookback - 1, vals.size)):
        w = vals[: i + 1]
        w = w[~np.isnan(w)]
        if w.size:
            out[i] = fn(w.reshape(1, -1), axis=1)[0]
    return out


def effective_half_spread(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    lookback: int = 21,
) -> np.ndarray:
    """Per-side proportional cost applied to every fill: the CONSERVATIVE
    blend max(CS, AR)/2, floored at half a tick ($0.005/price).

    Two independent estimators (Corwin-Schultz range-based; Abdi-Ranaldo
    close-vs-midrange) disagree most exactly where estimation is hardest —
    taking the elementwise max means the model may overstate spread cost but
    never quietly understates it. Overstated costs kill marginal edges at
    the gates; understated costs graduate fictions. The asymmetry is policy.
    """
    c = np.asarray(close, dtype=float)
    cs = corwin_schultz_spread(high, low, close)
    half_cs = _rolling_stat(cs, lookback, np.nanmedian) / 2.0

    ar2 = abdi_ranaldo_spread(high, low, close)
    with np.errstate(invalid="ignore"):
        ar = np.sqrt(np.maximum(_rolling_stat(ar2, lookback, np.nanmean), 0.0))
    half_ar = ar / 2.0

    half = np.fmax(half_cs, half_ar)  # fmax: NaN in one estimator defers to the other
    tick_floor = 0.005 / np.maximum(c, 1e-12)  # half of the $0.01 minimum increment
    return np.fmax(half, tick_floor)
