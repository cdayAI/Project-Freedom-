"""Options chain snapshots from Cboe's public delayed-quote feed.

Source: https://cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json —
the exchange's own published delayed (~15 min) chain: real bids, asks, IV,
open interest, volume. This is REAL market data, watermarked REAL_DELAYED;
nothing here is ever model-priced, per the operating rule that synthetic
option prices never enter a backtest.

Why archive daily: no free source provides HISTORICAL chains, so the system
builds its own — every nightly snapshot appends to a parquet archive that
becomes a genuine (if young) chain history. Until the archive is deep enough
to backtest, it already serves two purposes tonight:

  1. MEASURED SPREADS: per-underlying spread-as-fraction-of-premium stats,
     the dominant options cost at small size, measured from real NBBO-style
     quotes instead of assumed (mission Section 2 requirement).
  2. Paper-trading option structures against real quotes (Agent 10).

Archived per contract: expiry, strike, right, bid, ask, last, volume, OI,
IV (exchange-computed; stored as vendor data, never used to price anything).
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import requests

from alpha_forge.config import STORE_DIR

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

CHAINS_DIR = STORE_DIR / "options_chains"
SPREADS_PATH = STORE_DIR / "options_spread_stats.parquet"

# OCC option symbology: ROOT + YYMMDD + C/P + strike*1000, 8 digits
_OSI = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")

DATA_WATERMARK = "REAL_DELAYED_CBOE"


def fetch_chain(symbol: str, timeout: int = 30) -> pl.DataFrame | None:
    """One underlying's full delayed chain, normalized. None if unlisted."""
    r = requests.get(CBOE_URL.format(symbol=symbol.upper()), headers=UA, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json().get("data", {})
    opts = data.get("options", [])
    if not opts:
        return None
    rows = []
    for o in opts:
        m = _OSI.match(o.get("option", ""))
        if not m:
            continue
        root, ymd, right, strike_ms = m.groups()
        rows.append(
            {
                "underlying": symbol.upper(),
                "osi": o["option"],
                "expiry": f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}",
                "right": right,
                "strike": int(strike_ms) / 1000.0,
                "bid": float(o.get("bid") or 0.0),
                "ask": float(o.get("ask") or 0.0),
                "last": float(o.get("last_trade_price") or 0.0),
                "volume": float(o.get("volume") or 0.0),
                "open_interest": float(o.get("open_interest") or 0.0),
                "iv_vendor": float(o.get("iv") or 0.0),
            }
        )
    if not rows:
        return None
    return pl.DataFrame(rows).with_columns(
        pl.lit(float(data.get("close") or np.nan)).alias("underlying_close"),
        pl.lit(DATA_WATERMARK).alias("source"),
    )


def snapshot_chains(symbols: list[str], pause_s: float = 0.30) -> dict:
    """Archive today's chains for `symbols`. Idempotent per (day, symbol):
    an existing snapshot file for today is never refetched or overwritten —
    snapshots are immutable observations, like predictions."""
    CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_path = CHAINS_DIR / f"chains_{today}.parquet"
    have: set[str] = set()
    frames: list[pl.DataFrame] = []
    if out_path.exists():
        old = pl.read_parquet(out_path)
        have = set(old["underlying"].unique().to_list())
        frames.append(old)

    fetched, empty, errors = 0, 0, 0
    for sym in symbols:
        if sym.upper() in have:
            continue
        try:
            df = fetch_chain(sym)
        except requests.RequestException:
            errors += 1
            continue
        time.sleep(pause_s)
        if df is None or df.height == 0:
            empty += 1
            continue
        frames.append(
            df.with_columns(
                pl.lit(today).alias("snapshot_date"),
                pl.lit(datetime.now(timezone.utc).isoformat()).alias("captured_utc"),
            )
        )
        fetched += 1
    if frames:
        pl.concat(frames, how="diagonal").write_parquet(out_path)
    summary = {
        "snapshot_date": today,
        "requested": len(symbols),
        "fetched_now": fetched,
        "no_chain": empty,
        "errors": errors,
        "file": str(out_path) if frames else None,
    }
    if frames:
        _update_spread_stats()
    return summary


def _update_spread_stats() -> None:
    """Per-underlying spread economics from every archived snapshot:
    median relative spread (ask-bid)/mid among quoted contracts, split by
    moneyness bucket — the measured input for the options cost model."""
    files = sorted(CHAINS_DIR.glob("chains_*.parquet"))
    if not files:
        return
    frames = [pl.read_parquet(f) for f in files]
    df = pl.concat(frames, how="diagonal")
    df = df.filter((pl.col("bid") > 0) & (pl.col("ask") > pl.col("bid")))
    if df.height == 0:
        return
    df = df.with_columns(
        ((pl.col("ask") - pl.col("bid")) / ((pl.col("ask") + pl.col("bid")) / 2))
        .alias("rel_spread"),
        (pl.col("strike") / pl.col("underlying_close") - 1.0).abs().alias("moneyness_dist"),
    ).with_columns(
        pl.when(pl.col("moneyness_dist") <= 0.05)
        .then(pl.lit("ATM"))
        .when(pl.col("moneyness_dist") <= 0.20)
        .then(pl.lit("NEAR"))
        .otherwise(pl.lit("FAR"))
        .alias("moneyness")
    )
    stats = (
        df.group_by(["underlying", "moneyness"])
        .agg(
            pl.len().alias("n_quotes"),
            pl.col("rel_spread").median().alias("median_rel_spread"),
            pl.col("rel_spread").quantile(0.75).alias("p75_rel_spread"),
            pl.col("open_interest").median().alias("median_oi"),
            pl.col("snapshot_date").n_unique().alias("n_snapshot_days"),
        )
        .sort(["underlying", "moneyness"])
    )
    stats.write_parquet(SPREADS_PATH)


def load_spread_stats() -> pl.DataFrame | None:
    return pl.read_parquet(SPREADS_PATH) if SPREADS_PATH.exists() else None


def archive_depth() -> dict:
    """How much chain history the system has built for itself so far."""
    files = sorted(CHAINS_DIR.glob("chains_*.parquet"))
    if not files:
        return {"snapshot_days": 0, "underlyings": 0, "note": "archive starts on first nightly run"}
    latest = pl.read_parquet(files[-1])
    return {
        "snapshot_days": len(files),
        "underlyings": int(latest["underlying"].n_unique()),
        "first_day": files[0].stem.replace("chains_", ""),
        "last_day": files[-1].stem.replace("chains_", ""),
    }
