"""Pre-move feature fingerprints.

Everything is computed from data STRICTLY at or before the signal bar's
close — the entry happens at the next session's open, so a fingerprint may
use bar i but never bar i+1. Feature lookahead is the cheapest way to fake
an edge; this module is the only place fingerprints are computed, and its
tests assert the no-lookahead property.
"""

from __future__ import annotations

import numpy as np

from alpha_forge.config import TRADING_DAYS_PER_YEAR
from alpha_forge.costs.slippage import corwin_schultz_spread

FEATURE_NAMES = [
    "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "vol_20d_ann", "volume_z_20v126", "dollar_vol_med_20d",
    "dist_from_252d_high", "price", "cs_spread_est",
]


def fingerprint_at(
    i: int,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> dict[str, float]:
    """Features known at close of bar i. NaN where history is insufficient."""

    def ret(lb: int) -> float:
        return close[i] / close[i - lb] - 1.0 if i >= lb else float("nan")

    logret = np.diff(np.log(close[max(0, i - 20) : i + 1]))
    vol20 = float(logret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if logret.size >= 5 else float("nan")

    v20 = volume[max(0, i - 19) : i + 1].astype(float)
    v126 = volume[max(0, i - 125) : i + 1].astype(float)
    if v126.size >= 21 and v126.std(ddof=1) > 0:
        volume_z = float((v20.mean() - v126.mean()) / v126.std(ddof=1))
    else:
        volume_z = float("nan")

    dv20 = close[max(0, i - 19) : i + 1] * volume[max(0, i - 19) : i + 1]
    hi252 = close[max(0, i - 251) : i + 1].max()

    cs = corwin_schultz_spread(
        high[max(0, i - 21) : i + 1], low[max(0, i - 21) : i + 1], close[max(0, i - 21) : i + 1]
    )
    cs_med = float(np.nanmedian(cs)) if np.any(~np.isnan(cs)) else float("nan")

    return {
        "ret_21d": ret(21),
        "ret_63d": ret(63),
        "ret_126d": ret(126),
        "ret_252d": ret(252),
        "vol_20d_ann": vol20,
        "volume_z_20v126": volume_z,
        "dollar_vol_med_20d": float(np.median(dv20)),
        "dist_from_252d_high": float(close[i] / hi252 - 1.0),
        "price": float(close[i]),
        "cs_spread_est": cs_med,
    }
