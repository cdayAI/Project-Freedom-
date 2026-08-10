"""Catalyst layer v1 — SEC EDGAR 8-K filing histories (free, primary source).

Most extreme moves are catalyst-driven; a pattern store built from OHLCV
alone fingerprints the aftermath and misses the cause. This module supplies
the first cause data:

  - CIK mapping from the SEC's official ticker file
  - per-company 8-K filing histories from data.sec.gov submissions
    (Item 2.02 = Results of Operations — the earnings announcement)
  - hit tagging: every pathfinder hit gets a catalyst class
        EARNINGS_8K  an Item-2.02 8-K within [-10, +30] calendar days of entry
        OTHER_8K     any other 8-K in that window
        NONE         no 8-K near the move (or no EDGAR coverage)
  - EX-ANTE features (knowable at the signal close, no lookahead):
        days_since_earnings_8k   calendar days since the last Item-2.02 8-K
        days_since_any_8k        calendar days since the last 8-K of any kind
    Quarterly reporters cluster near ~91 days, so "earnings likely soon" is
    learnable from days_since_earnings_8k without any forward calendar.

Known limitations (recorded in the manifest): the submissions API's
"recent" window caps at ~1000 filings per company (ample for small caps,
truncated history for mega-filers); short interest and halt feeds are
DESIGN-mode until their sources are wired.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import requests

from alpha_forge.config import STORE_DIR

CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
UA = {"User-Agent": "AlphaForge Research daychristopher91@gmail.com"}

CATALYST_PATH = STORE_DIR / "catalysts_8k.parquet"

WINDOW_BEFORE_DAYS = 10   # catalyst window around a hit's entry (calendar days)
WINDOW_AFTER_DAYS = 30


def fetch_cik_map(timeout: int = 60) -> dict[str, int]:
    r = requests.get(CIK_MAP_URL, headers=UA, timeout=timeout)
    r.raise_for_status()
    doc = r.json()
    return {e["ticker"].upper(): int(e["cik_str"]) for e in doc.values()}


TRACKED_FORM_PREFIXES = (
    "8-K",            # material events (items parsed; 2.02 = earnings)
    "4",              # insider transactions (statement of changes)
    "S-1", "S-3",     # shelf/IPO registration — dilution pipeline
    "F-1", "F-3",     # foreign-issuer equivalents
    "424B",           # prospectus filed = offering priced
)


def _tracked(form: str) -> bool:
    return any(
        form == p or form.startswith(p + "/") or (p == "424B" and form.startswith("424B"))
        for p in TRACKED_FORM_PREFIXES
    )


def fetch_filing_history(cik: int, timeout: int = 30) -> list[dict]:
    """(form, filing_date, items) for every TRACKED filing in the company's
    recent window — one request already being made for 8-Ks now yields the
    insider and dilution series too."""
    r = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=UA, timeout=timeout)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    recent = r.json().get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    items = recent.get("items", [""] * len(forms))
    out = []
    for f, d, it in zip(forms, dates, items):
        if _tracked(f):
            out.append({"form": f, "filing_date": d, "items": it or ""})
    return out


def fetch_8k_history(cik: int, timeout: int = 30) -> list[dict]:
    """Back-compat: 8-K rows only."""
    return [
        {k: v for k, v in row.items() if k != "form"}
        for row in fetch_filing_history(cik, timeout)
        if row["form"].startswith("8-K")
    ]


def ingest_8k_catalysts(symbols: list[str], pause_s: float = 0.12) -> dict:
    """Pull 8-K histories for the research sample; write the catalog parquet.
    Resumable: symbols already in the catalog are not refetched."""
    have: set[str] = set()
    frames = []
    if CATALYST_PATH.exists():
        old = pl.read_parquet(CATALYST_PATH)
        have = set(old["symbol"].unique().to_list())
        frames.append(old)

    cik_map = fetch_cik_map()
    matched = [s for s in symbols if s.upper() in cik_map]
    todo = [s for s in matched if s not in have]
    rows = []
    fetched = 0
    for sym in todo:
        try:
            filings = fetch_filing_history(cik_map[sym.upper()])
            fetched += 1
        except requests.RequestException:
            continue
        for f in filings:
            rows.append(
                {
                    "symbol": sym,
                    "form": f["form"],
                    "filing_date": f["filing_date"],
                    "items": f["items"],
                    "is_earnings": f["form"].startswith("8-K") and "2.02" in f["items"],
                }
            )
        time.sleep(pause_s)
    if rows:
        frames.append(
            pl.DataFrame(rows).with_columns(
                pl.col("filing_date").str.to_date().alias("filing_date")
            )
        )
    if frames:
        catalog = pl.concat(frames, how="diagonal")
        if "form" not in catalog.columns:
            catalog = catalog.with_columns(pl.lit("8-K").alias("form"))
        # pre-v2 archives stored 8-K rows without a form column
        catalog = catalog.with_columns(pl.col("form").fill_null("8-K")).unique(
            subset=["symbol", "form", "filing_date", "items"]
        )
        catalog.write_parquet(CATALYST_PATH)
    else:
        catalog = pl.DataFrame()
    return {
        "symbols_requested": len(symbols),
        "symbols_with_cik": len(matched),
        "symbols_fetched_now": fetched,
        "filings_total": catalog.height if catalog.height else 0,
    }


def load_catalog() -> pl.DataFrame | None:
    return pl.read_parquet(CATALYST_PATH) if CATALYST_PATH.exists() else None


# ---------------------------------------------------------------- tagging


def tag_hits(hits: pl.DataFrame, catalog: pl.DataFrame | None) -> pl.DataFrame:
    """Replace the placeholder catalyst_class with EDGAR-derived classes."""
    if catalog is None or catalog.height == 0 or hits is None or hits.height == 0:
        return hits
    if "form" in catalog.columns:  # v2 catalogs carry all forms; tag on 8-Ks
        catalog = catalog.filter(pl.col("form").str.starts_with("8-K"))
    by_symbol: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (sym,), g in catalog.group_by("symbol"):
        g = g.sort("filing_date")
        d = g["filing_date"].to_numpy()
        by_symbol[str(sym)] = (d, g["is_earnings"].to_numpy())

    classes = []
    for row in hits.iter_rows(named=True):
        sym = row["symbol"]
        entry = np.datetime64(row["entry_date"][:10])
        cat = "NONE"
        if sym in by_symbol:
            dates, earn = by_symbol[sym]
            lo = entry - np.timedelta64(WINDOW_BEFORE_DAYS, "D")
            hi = entry + np.timedelta64(WINDOW_AFTER_DAYS, "D")
            in_win = (dates >= lo) & (dates <= hi)
            if in_win.any():
                cat = "EARNINGS_8K" if earn[in_win].any() else "OTHER_8K"
        classes.append(cat)
    return hits.with_columns(pl.Series("catalyst_class", classes))


# ------------------------------------------------------- ex-ante features

CATALYST_FEATURES = [
    "days_since_earnings_8k",
    "days_since_any_8k",
    "days_until_expected_earnings",  # forward clock from own filing cadence
    "days_since_form4",              # insider transaction recency
    "n_form4_90d",                   # insider filing intensity
    "days_since_dilution_filing",    # S-1/S-3/F-1/F-3/424B recency
]

MIN_EARNINGS_FOR_CADENCE = 4


def _dilution_mask(forms: np.ndarray) -> np.ndarray:
    out = np.zeros(forms.size, dtype=bool)
    for i, f in enumerate(forms):
        s = str(f)
        out[i] = s.startswith(("S-1", "S-3", "F-1", "F-3", "424B"))
    return out


def attach_catalyst_features(features: pl.DataFrame, catalog: pl.DataFrame | None) -> pl.DataFrame:
    """Add ex-ante catalyst features to the features panel. Strictly
    backward-looking: a filing dated D is knowable from D onward. The
    forward-looking earnings clock uses only PAST filings: expected next
    earnings = last earnings 8-K + the symbol's own median inter-earnings
    gap (>= MIN_EARNINGS_FOR_CADENCE past reports required), so quarterly
    reporters read ~"-60 ... 0 ... +91" with negative = overdue."""
    null_cols = [pl.lit(None, dtype=pl.Float64).alias(c) for c in CATALYST_FEATURES]
    if catalog is None or catalog.height == 0:
        return features.with_columns(null_cols)
    if "form" not in catalog.columns:
        catalog = catalog.with_columns(pl.lit("8-K").alias("form"))

    by_symbol: dict[str, dict] = {}
    for (sym,), g in catalog.group_by("symbol"):
        g = g.sort("filing_date")
        fdates = g["filing_date"].to_numpy().astype("datetime64[D]")
        forms = np.asarray(g["form"].to_list(), dtype=object)
        earn = g["is_earnings"].to_numpy().astype(bool)
        is_8k = np.array([str(f).startswith("8-K") for f in forms])
        by_symbol[str(sym)] = {
            "any8k": fdates[is_8k],
            "earn": fdates[earn],
            "form4": fdates[np.array([str(f) == "4" or str(f).startswith("4/") for f in forms])],
            "dilution": fdates[_dilution_mask(forms)],
        }

    frames = []
    for (sym,), g in features.group_by("symbol", maintain_order=True):
        g = g.sort("date")
        dates = g["date"].to_numpy().astype("datetime64[D]")
        n = dates.size
        cols = {c: np.full(n, np.nan) for c in CATALYST_FEATURES}
        info = by_symbol.get(str(sym))
        if info is not None:
            def _days_since(series: np.ndarray, out: np.ndarray) -> None:
                if series.size == 0:
                    return
                idx = np.searchsorted(series, dates, side="right") - 1
                ok = idx >= 0
                out[ok] = (dates[ok] - series[idx[ok]]).astype(float)

            _days_since(info["any8k"], cols["days_since_any_8k"])
            _days_since(info["earn"], cols["days_since_earnings_8k"])
            _days_since(info["form4"], cols["days_since_form4"])
            _days_since(info["dilution"], cols["days_since_dilution_filing"])

            f4 = info["form4"]
            if f4.size:
                hi = np.searchsorted(f4, dates, side="right")
                lo = np.searchsorted(f4, dates - np.timedelta64(90, "D"), side="right")
                cols["n_form4_90d"] = (hi - lo).astype(float)

            edates = info["earn"]
            if edates.size >= MIN_EARNINGS_FOR_CADENCE:
                # expected next report, using only earnings filings <= each date
                idx = np.searchsorted(edates, dates, side="right") - 1
                for i in range(n):
                    j = idx[i]
                    if j < MIN_EARNINGS_FOR_CADENCE - 1:
                        continue
                    gaps = np.diff(edates[: j + 1]).astype(float)
                    med_gap = float(np.median(gaps))
                    expected = edates[j] + np.timedelta64(int(round(med_gap)), "D")
                    cols["days_until_expected_earnings"][i] = float(
                        (expected - dates[i]).astype(float)
                    )
        frames.append(g.with_columns([pl.Series(c, v) for c, v in cols.items()]))
    return pl.concat(frames)
