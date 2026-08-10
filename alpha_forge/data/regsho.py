"""FINRA Reg SHO daily short-sale volume — free, primary, DAILY, per symbol.

Source: https://cdn.finra.org/equity/regsho/daily/CNMSshvolYYYYMMDD.txt
(consolidated NMS file; pipe-delimited Date|Symbol|ShortVolume|
ShortExemptVolume|TotalVolume|Market). This is CAUSE-side data the OHLCV
fingerprint cannot see: short-sale pressure, the raw material of squeezes.

Timing discipline: FINRA publishes day D's file on the EVENING of D — after
the close. Features stamped at signal-close D therefore use files through
D-1 only (LAG_DAYS=1). Conservative by one session, never lookahead.

Backfill is resumable and budgeted: each call ingests at most
`max_files_per_run` missing days, newest first, so the nightly loop extends
one day forward and eats history backward until the archive reaches
`target_days`. Raw files are aggregated into one parquet (date, symbol,
short_ratio, total_volume) and discarded — ~2,000 trading days compact to
tens of MB.
"""

from __future__ import annotations

import io
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import requests

from alpha_forge.config import STORE_DIR

URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ymd}.txt"
UA = {"User-Agent": "AlphaForge Research daychristopher91@gmail.com"}

REGSHO_PATH = STORE_DIR / "regsho_daily.parquet"
LAG_DAYS = 1
TARGET_DAYS = 2000  # ~8 years of trading days, matching the 2018+ file era

SHORT_FEATURES = ["short_ratio_5d", "short_ratio_z"]


def _fetch_day(d: date, timeout: int = 30) -> pl.DataFrame | None:
    ymd = d.strftime("%Y%m%d")
    r = requests.get(URL.format(ymd=ymd), headers=UA, timeout=timeout)
    if r.status_code != 200 or not r.text.startswith("Date|"):
        return None  # weekend/holiday or pre-era date
    df = pl.read_csv(
        io.StringIO(r.text),
        separator="|",
        schema_overrides={
            "Date": pl.Utf8,
            "ShortVolume": pl.Float64,
            "ShortExemptVolume": pl.Float64,
            "TotalVolume": pl.Float64,
        },
    )
    if df.height == 0 or "ShortVolume" not in df.columns:
        return None
    # the file ends with a record-count trailer row; keep only real records
    df = df.filter(pl.col("Date").str.contains(r"^\d{8}$") & pl.col("Symbol").is_not_null())
    return (
        df.select(
            pl.col("Date").str.to_date("%Y%m%d").alias("date"),
            pl.col("Symbol").alias("symbol"),
            (pl.col("ShortVolume") / pl.col("TotalVolume").clip(lower_bound=1.0))
            .alias("short_ratio"),
            pl.col("TotalVolume").alias("total_volume"),
        )
        .filter(pl.col("symbol").str.len_chars() <= 5)
    )


def ingest_regsho(
    max_files_per_run: int = 120,
    target_days: int = TARGET_DAYS,
    pause_s: float = 0.15,
) -> dict:
    """Extend the archive: today backward, skipping days already stored."""
    have: set = set()
    frames: list[pl.DataFrame] = []
    if REGSHO_PATH.exists():
        old = pl.read_parquet(REGSHO_PATH)
        have = set(old["date"].unique().to_list())
        frames.append(old)

    fetched, misses = 0, 0
    d = date.today()
    scanned = 0
    # scan back far enough to cover target_days trading days plus weekends
    while fetched < max_files_per_run and scanned < int(target_days * 1.6):
        scanned += 1
        cur = d
        d = d - timedelta(days=1)
        if cur in have or cur.weekday() >= 5:
            continue
        try:
            df = _fetch_day(cur)
        except requests.RequestException:
            misses += 1
            continue
        time.sleep(pause_s)
        if df is None:
            misses += 1
            continue
        frames.append(df)
        fetched += 1

    if frames:
        out = pl.concat(frames, how="diagonal").unique(subset=["date", "symbol"])
        out.write_parquet(REGSHO_PATH)
        days = out["date"].n_unique()
    else:
        days = 0
    return {"fetched_now": fetched, "days_in_archive": int(days), "misses": misses}


def load_regsho() -> pl.DataFrame | None:
    return pl.read_parquet(REGSHO_PATH) if REGSHO_PATH.exists() else None


def attach_short_features(features: pl.DataFrame, regsho: pl.DataFrame | None) -> pl.DataFrame:
    """Ex-ante short-pressure features on the features panel:
      short_ratio_5d  mean daily short ratio over the 5 files through D-1
      short_ratio_z   (5d mean - 63d mean) / 63d std, same lag
    NaN before the archive era or where the symbol is absent."""
    if regsho is None or regsho.height == 0:
        return features.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("short_ratio_5d"),
            pl.lit(None, dtype=pl.Float64).alias("short_ratio_z"),
        )
    by_symbol: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (sym,), g in regsho.group_by("symbol"):
        g = g.sort("date")
        by_symbol[str(sym)] = (
            g["date"].to_numpy().astype("datetime64[D]"),
            g["short_ratio"].to_numpy().astype(float),
        )

    frames = []
    for (sym,), g in features.group_by("symbol", maintain_order=True):
        g = g.sort("date")
        dates = g["date"].to_numpy().astype("datetime64[D]")
        n = dates.size
        f5 = np.full(n, np.nan)
        fz = np.full(n, np.nan)
        if str(sym) in by_symbol:
            sdates, sratio = by_symbol[str(sym)]
            # index of last regsho row at or before each panel date MINUS lag
            idx = np.searchsorted(sdates, dates - np.timedelta64(LAG_DAYS, "D"), side="right") - 1
            for i in range(n):
                j = idx[i]
                if j < 4:
                    continue
                w5 = sratio[j - 4 : j + 1]
                f5[i] = float(np.mean(w5))
                lo = max(0, j - 62)
                w63 = sratio[lo : j + 1]
                if w63.size >= 30:
                    sd = float(np.std(w63, ddof=1))
                    # strict-positive is not enough: a constant series has
                    # sd ~ 2e-16 of float residue, and noise/noise = garbage
                    if sd > 1e-9:
                        fz[i] = (f5[i] - float(np.mean(w63))) / sd
        frames.append(
            g.with_columns(
                pl.Series("short_ratio_5d", f5), pl.Series("short_ratio_z", fz)
            )
        )
    return pl.concat(frames)
