"""Market regime series: trend / volatility / chop, computed from data.

Benchmarks: SPY (S&P 500 proxy) and ^VIX from the same Yahoo chart API as
the equity panel. Regime definitions are structural research choices
(documented in ARCHITECTURE.md), all computed from the series themselves:

  trend: BULL if SPY close > its trailing 200-bar mean, else BEAR
  vol:   LOW/MID/HIGH by trailing-1260-bar (5y) terciles of VIX close
  chop:  CHOPPY if SPY 20-bar realized vol > its trailing median, else SMOOTH

Every pathfinder hit and prediction is stamped with the regime in effect on
its signal date.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from alpha_forge.config import STORE_DIR
from alpha_forge.data.equities import YahooDailyAdapter

REGIME_PATH = STORE_DIR / "regime.parquet"


def build_regime_series(adapter: YahooDailyAdapter | None = None) -> pl.DataFrame:
    adapter = adapter or YahooDailyAdapter()
    spy = adapter.fetch_symbol("SPY")
    vix = adapter.fetch_symbol("^VIX")
    if spy is None or vix is None:
        raise RuntimeError("regime: SPY/^VIX fetch failed — halting (no silent default regime)")

    spy = spy.select(["date", "close"]).rename({"close": "spy_close"})
    vix = vix.select(["date", "close"]).rename({"close": "vix_close"})
    df = spy.join(vix, on="date", how="left").sort("date")

    c = df["spy_close"].to_numpy().astype(float)
    v = df["vix_close"].to_numpy().astype(float)
    n = c.size

    ma200 = np.full(n, np.nan)
    if n >= 200:
        cs = np.cumsum(np.insert(c, 0, 0.0))
        ma200[199:] = (cs[200:] - cs[:-200]) / 200.0

    logret = np.diff(np.log(c), prepend=np.nan)
    rv20 = np.full(n, np.nan)
    for i in range(20, n):
        rv20[i] = np.nanstd(logret[i - 19 : i + 1], ddof=1)

    trend = np.where(np.isnan(ma200), "UNKNOWN", np.where(c > ma200, "BULL", "BEAR"))

    vol_state = np.full(n, "UNKNOWN", dtype=object)
    for i in range(n):
        if np.isnan(v[i]):
            continue
        hist = v[max(0, i - 1259) : i + 1]
        hist = hist[~np.isnan(hist)]
        if hist.size < 252:
            continue
        t1, t2 = np.percentile(hist, [33.33, 66.67])
        vol_state[i] = "LOW" if v[i] <= t1 else ("HIGH" if v[i] > t2 else "MID")

    chop = np.full(n, "UNKNOWN", dtype=object)
    for i in range(n):
        if np.isnan(rv20[i]):
            continue
        hist = rv20[max(0, i - 1259) : i + 1]
        hist = hist[~np.isnan(hist)]
        if hist.size < 252:
            continue
        chop[i] = "CHOPPY" if rv20[i] > np.median(hist) else "SMOOTH"

    out = df.with_columns(
        pl.Series("trend", trend.astype(str)),
        pl.Series("vol_state", vol_state.astype(str)),
        pl.Series("chop", chop.astype(str)),
    )
    out.write_parquet(REGIME_PATH)
    return out


def load_regime() -> pl.DataFrame:
    if not REGIME_PATH.exists():
        raise FileNotFoundError("regime series not built — run the daily job")
    return pl.read_parquet(REGIME_PATH)


def regime_on(regime: pl.DataFrame, date_str: str) -> dict:
    """Regime in effect on the last bar at or before date_str."""
    d = regime.filter(pl.col("date") <= pl.lit(date_str).str.to_date())
    if d.height == 0:
        return {"trend": "UNKNOWN", "vol_state": "UNKNOWN", "chop": "UNKNOWN"}
    row = d.row(-1, named=True)
    return {"trend": row["trend"], "vol_state": row["vol_state"], "chop": row["chop"]}
