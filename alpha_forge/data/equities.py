"""Daily US equity OHLCV ingestion.

Default adapter: Yahoo Finance v8 chart API (no key required). Prices are
split/dividend adjusted by scaling OHLC with the vendor's adjclose/close
factor — the standard total-return adjustment. Vendor claims are recorded,
not trusted: validation runs split-detection on every series, and the
manifest records "adjustment quality unverified" as a known bias.

If Norgate/Sharadar/CRSP credentials are present in .env the
survivorship-free adapters take over (interfaces below; DESIGN mode without
credentials).

All prices stored as pulled; timestamps are dates (daily bars); everything
downstream treats the close as information available only after the close,
and fills happen at the NEXT session's open (see research.backtest).

Note: a Stooq adapter was tried first; stooq.com now fronts its CSV endpoint
with a JavaScript proof-of-work browser check, which this system will not
script around. Yahoo's chart API serves the same daily bars without a wall.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import requests

from alpha_forge.config import STORE_DIR
from alpha_forge.data.validation import validate_daily_series

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}


def _drop_incomplete_session(df: pl.DataFrame) -> pl.DataFrame:
    """A pull during US market hours includes TODAY'S PARTIAL bar — a
    mid-session snapshot masquerading as a daily close. Drop the current
    ET session's bar unless the session has closed (>= 16:15 ET buffer)."""
    from datetime import datetime, timedelta, timezone

    now_utc = datetime.now(timezone.utc)
    # ET offset: -5 standard, -4 daylight. Approximate DST by month (Mar-Nov)
    # is not acceptable for a data-integrity rule; use the exact zoneinfo.
    from zoneinfo import ZoneInfo

    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
    if now_et.hour * 60 + now_et.minute >= 16 * 60 + 15:
        return df
    return df.filter(pl.col("date") < now_et.date())


class YahooDailyAdapter:
    vendor = "yahoo_finance_chart_api"
    survivorship_free = False
    known_biases = [
        "current listings only: delisted names absent (survivorship bias)",
        "vendor adjclose used for split/dividend adjustment; quality not independently verified",
        "volume left unadjusted for splits (incomparable across split dates)",
        "no intraday data: overnight gap risk measured from open/close only",
    ]

    def __init__(self, pause_s: float = 0.25, max_retries: int = 3, range_: str = "15y"):
        self.pause_s = pause_s
        self.max_retries = max_retries
        self.range_ = range_

    def fetch_symbol(self, symbol: str) -> pl.DataFrame | None:
        params = {"interval": "1d", "range": self.range_, "events": "div,splits"}
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(
                    YAHOO_URL.format(symbol=symbol), params=params, headers=UA, timeout=30
                )
                if resp.status_code == 404:
                    return None  # unknown/delisted on this vendor
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                resp.raise_for_status()
                result = resp.json().get("chart", {}).get("result")
                if not result:
                    return None
                r0 = result[0]
                ts = r0.get("timestamp")
                quote = r0["indicators"]["quote"][0]
                adj = r0["indicators"].get("adjclose", [{}])[0].get("adjclose")
                if not ts or adj is None:
                    return None
                df = pl.DataFrame(
                    {
                        "ts": ts,
                        "open": quote["open"],
                        "high": quote["high"],
                        "low": quote["low"],
                        "close": quote["close"],
                        "volume": quote["volume"],
                        "adjclose": adj,
                    }
                ).drop_nulls()
                if df.height == 0:
                    return None
                # session date from the exchange's own timestamp (UTC date of a
                # US session open == the session date)
                df = df.with_columns(
                    pl.from_epoch("ts", time_unit="s").dt.date().alias("date"),
                    (pl.col("adjclose") / pl.col("close")).alias("factor"),
                )
                df = df.with_columns(
                    (pl.col("open") * pl.col("factor")).alias("open"),
                    (pl.col("high") * pl.col("factor")).alias("high"),
                    (pl.col("low") * pl.col("factor")).alias("low"),
                    pl.col("adjclose").alias("close"),
                    pl.col("volume").cast(pl.Float64),
                )
                df = df.select(["date", "open", "high", "low", "close", "volume"]).sort("date")
                return _drop_incomplete_session(df)
            except (requests.RequestException, KeyError, ValueError):
                time.sleep(2**attempt)
        return None

    def ingest(self, symbols: list[str], out_dir: Path | None = None,
               resume: bool = True) -> dict:
        """Fetch, validate, quarantine failures, write one parquet per symbol
        plus a combined long-format parquet. Returns ingestion summary.

        resume=True reuses per-symbol parquet already pulled TODAY (same
        vendor, same vintage), so a rate-limit interruption or a sample
        widening never refetches what is already on disk."""
        import datetime as _dt

        out_dir = out_dir or (STORE_DIR / "equities_daily")
        out_dir.mkdir(parents=True, exist_ok=True)
        ok_frames, reports, quarantined = [], [], []
        today = _dt.date.today()
        for sym in symbols:
            cached = out_dir / f"{sym}.parquet"
            if (
                resume
                and cached.exists()
                and _dt.date.fromtimestamp(cached.stat().st_mtime) == today
            ):
                ok_frames.append(pl.read_parquet(cached))
                continue
            df = self.fetch_symbol(sym)
            time.sleep(self.pause_s)
            if df is None or df.height == 0:
                quarantined.append({"symbol": sym, "reason": "no data returned"})
                continue
            report = validate_daily_series(df, sym)
            reports.append(report)
            if not report["ok"]:
                quarantined.append({"symbol": sym, "reason": "; ".join(report["issues"])})
                continue
            df = df.with_columns(pl.lit(sym).alias("symbol"))
            df.write_parquet(out_dir / f"{sym}.parquet")
            ok_frames.append(df)
        if ok_frames:
            combined = pl.concat(ok_frames)
            combined.write_parquet(STORE_DIR / "equities_daily.parquet")
        return {
            "requested": len(symbols),
            "ingested": len(ok_frames),
            "quarantined": quarantined,
            "validation_reports": reports,
        }


class SurvivorshipFreeAdapterInterface:
    """DESIGN-mode interface for Norgate / Sharadar / CRSP.

    With credentials in .env (NORGATE_*, SHARADAR_API_KEY, ...), implementers
    must return the same schema as YahooDailyAdapter and set
    survivorship_free=True. Without credentials, design_queries() documents
    exactly what would be pulled so the switch is a config change, not a
    research change.
    """

    survivorship_free = True

    @staticmethod
    def design_queries() -> list[str]:
        return [
            "Sharadar SEP: SELECT ticker,date,open,high,low,close,volume FROM SEP "
            "WHERE date >= '2010-01-01'  -- includes delisted tickers",
            "Norgate: watchlist 'US All Securities incl Delisted', daily bars, "
            "padding=NONE, capital-adjusted",
        ]


def load_equities_panel() -> pl.DataFrame:
    """Long-format panel (symbol, date, open, high, low, close, volume)."""
    path = STORE_DIR / "equities_daily.parquet"
    if not path.exists():
        raise FileNotFoundError("no equities panel ingested yet — run the daily job")
    return pl.read_parquet(path)
