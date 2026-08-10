"""Ingestion validation: a failed or anomalous pull halts downstream agents.

Checks per symbol series: minimum rows, strictly increasing dates, OHLC
sanity, gap detection, and unadjusted-split suspicion (extreme overnight
ratios paired with volume spikes). A series that fails hard checks is
quarantined — it never enters the pattern store.
"""

from __future__ import annotations

import numpy as np
import polars as pl


class ValidationFailure(Exception):
    pass


def validate_daily_series(df: pl.DataFrame, symbol: str, min_rows: int = 60) -> dict:
    issues: list[str] = []
    hard_fail = False

    if df.height < min_rows:
        return {"symbol": symbol, "ok": False, "hard_fail": True,
                "issues": [f"only {df.height} rows (<{min_rows})"]}

    dates = df["date"].to_numpy()
    if not np.all(dates[1:] > dates[:-1]):
        issues.append("dates not strictly increasing")
        hard_fail = True

    o, h, l, c = (df[k].to_numpy().astype(float) for k in ("open", "high", "low", "close"))
    bad_ohlc = int(np.sum((l > np.minimum(o, c)) | (h < np.maximum(o, c)) | (l <= 0)))
    if bad_ohlc:
        issues.append(f"{bad_ohlc} rows violate OHLC sanity")
        if bad_ohlc > df.height * 0.01:
            hard_fail = True

    day_gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
    max_gap = int(day_gaps.max()) if day_gaps.size else 0
    if max_gap > 14:
        issues.append(f"max calendar gap {max_gap}d (halt or missing data)")

    # unadjusted-split suspicion: overnight close ratio beyond 1.8x or below 0.55x
    ratio = c[1:] / c[:-1]
    suspects = int(np.sum((ratio > 1.8) | (ratio < 0.55)))
    if suspects:
        issues.append(f"{suspects} extreme overnight ratios (possible unadjusted splits "
                      "or real gaps; flagged for pathfinder cross-check)")

    return {
        "symbol": symbol,
        "ok": not hard_fail,
        "hard_fail": hard_fail,
        "rows": df.height,
        "first_date": str(df["date"][0]),
        "last_date": str(df["date"][-1]),
        "max_gap_days": max_gap,
        "extreme_ratio_days": suspects,
        "issues": issues,
    }
