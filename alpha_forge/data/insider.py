"""SEC DERA insider transactions — quarterly Form 3/4/5 data sets (free, primary).

Source: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
Quarterly ZIPs of the structured Form 3/4/5 tables; each contains
SUBMISSION.tsv (ACCESSION_NUMBER, FILING_DATE, DOCUMENT_TYPE,
ISSUERTRADINGSYMBOL, ...) and NONDERIV_TRANS.tsv (ACCESSION_NUMBER,
TRANS_CODE, TRANS_ACQUIRED_DISP_CD, TRANS_SHARES, TRANS_PRICEPERSHARE, ...).
Quarters 2006q1-2026q1 live under /files/structureddata/..., newer ones
under /files/datastandardsinnovation/... — both prefixes are tried.

Signal content: NET INSIDER BUYING. Only open-market discretionary trades
are counted — TRANS_CODE 'P' (open-market purchase, cross-checked
TRANS_ACQUIRED_DISP_CD='A') and 'S' (open-market sale, cross-checked 'D').
Awards/grants/exercises/tax-withholding (A/M/F/G/...) are compensation
plumbing, not conviction, and are ignored. Dollar value is
TRANS_SHARES * TRANS_PRICEPERSHARE; rows with a null price or share count
are skipped, and single transactions above MAX_TXN_USD are dropped as
filer data errors (see the constant). Amendments (4/A, 5/A) are dropped:
they restate transactions already counted from the original filing and
would double-count.

Timing discipline: the FILING_DATE is the knowledge date (same convention
as catalysts.py) — a transaction is knowable to the market only once the
form hits EDGAR, so features at panel date D use filings with
filing_date <= D only. TRANS_DATE (the trade itself, up to 2 business days
earlier, sometimes months for Form 5) is deliberately NOT used for
alignment: that would be lookahead. Panel dates before the earliest
filing_date in the archive are NaN (archive absent, not "no buying");
symbols never seen in the archive are NaN everywhere.

Backfill is resumable and budgeted: each call ingests at most
`max_quarters_per_run` missing quarterly ZIPs, newest first, so the
archive extends forward each quarter and eats history backward until
FIRST_QUARTER. Raw ZIPs (~15 MB each) are deleted after aggregation; the
whole 2018+ era compacts to a few MB of parquet keyed (symbol, filing_date).
"""

from __future__ import annotations

import time
import zipfile
from datetime import date

import numpy as np
import polars as pl
import requests

from alpha_forge.config import RAW_DIR, STORE_DIR

URL_TEMPLATES = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{q}_form345.zip",
    "https://www.sec.gov/files/datastandardsinnovation/data/insider-transactions-data-sets/{q}_form345.zip",
)
UA = {"User-Agent": "AlphaForge Research daychristopher91@gmail.com"}

INSIDER_PATH = STORE_DIR / "insider_daily.parquet"
FIRST_QUARTER = "2018q1"
WINDOW_DAYS = 90

# Filer-error guard: forms occasionally carry the TOTAL value in the
# price-per-share field (observed 2026q1: a $24,035,774.40 "price" on an OTC
# microcap -> a $2.4e15 "purchase"). No genuine open-market insider trade
# approaches $10B (the largest real ones in the data are ~$2.3B), so single
# transactions above this are dropped as data errors, not winsorized.
MAX_TXN_USD = 1e10

INSIDER_FEATURES = ["insider_net_buy_90d_usd", "insider_buy_ratio_90d"]

_SUB_COLS = ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERTRADINGSYMBOL"]
_TRANS_COLS = [
    "ACCESSION_NUMBER",
    "TRANS_CODE",
    "TRANS_ACQUIRED_DISP_CD",
    "TRANS_SHARES",
    "TRANS_PRICEPERSHARE",
]


# ------------------------------------------------------------- aggregation


def aggregate_form345(submissions: pl.DataFrame, transactions: pl.DataFrame) -> pl.DataFrame:
    """Pure aggregation of one quarter's raw tables to per-(symbol, filing_date)
    net insider buying. Expects the DERA column names (see _SUB_COLS /
    _TRANS_COLS); FILING_DATE may be Utf8 in DERA's DD-MON-YYYY form or an
    already-parsed Date. Returns columns (symbol, filing_date, net_buy_usd,
    buy_usd, sell_usd, n_buy_txns, n_sell_txns)."""
    filing = pl.col("FILING_DATE")
    if submissions.schema["FILING_DATE"] == pl.Utf8:
        filing = filing.str.to_date("%d-%b-%Y")
    sub = (
        submissions.select(
            pl.col("ACCESSION_NUMBER"),
            filing.alias("filing_date"),
            pl.col("ISSUERTRADINGSYMBOL").str.strip_chars().str.to_uppercase().alias("symbol"),
            pl.col("DOCUMENT_TYPE"),
        )
        # amendments restate the original filing's rows -> double count
        .filter(~pl.col("DOCUMENT_TYPE").str.ends_with("/A"))
        # real tickers only: the raw field carries junk like "NYSE: KRC",
        # "Z AND ZG", "[NONE]" — keep <=5 alpha chars, matching the panel;
        # bare "NONE" is the no-symbol placeholder, pooling unrelated issuers
        .filter(
            pl.col("symbol").str.contains(r"^[A-Z]{1,5}$") & (pl.col("symbol") != "NONE")
        )
    )
    trans = (
        transactions.select(
            pl.col("ACCESSION_NUMBER"),
            pl.col("TRANS_CODE"),
            pl.col("TRANS_ACQUIRED_DISP_CD"),
            pl.col("TRANS_SHARES").cast(pl.Float64, strict=False),
            pl.col("TRANS_PRICEPERSHARE").cast(pl.Float64, strict=False),
        )
        # open-market only, direction cross-checked against the A/D flag
        # (a handful of rows per quarter carry P-with-D or S-with-A; drop them)
        .filter(
            ((pl.col("TRANS_CODE") == "P") & (pl.col("TRANS_ACQUIRED_DISP_CD") == "A"))
            | ((pl.col("TRANS_CODE") == "S") & (pl.col("TRANS_ACQUIRED_DISP_CD") == "D"))
        )
        .filter(
            pl.col("TRANS_PRICEPERSHARE").is_not_null() & pl.col("TRANS_SHARES").is_not_null()
        )
        .with_columns(
            (pl.col("TRANS_SHARES") * pl.col("TRANS_PRICEPERSHARE")).alias("value_usd"),
            (pl.col("TRANS_CODE") == "P").alias("is_buy"),
        )
        .filter(pl.col("value_usd") <= MAX_TXN_USD)
    )
    joined = trans.join(sub, on="ACCESSION_NUMBER", how="inner")
    return (
        joined.group_by("symbol", "filing_date")
        .agg(
            pl.col("value_usd").filter(pl.col("is_buy")).sum().alias("buy_usd"),
            pl.col("value_usd").filter(~pl.col("is_buy")).sum().alias("sell_usd"),
            pl.col("is_buy").sum().cast(pl.Int64).alias("n_buy_txns"),
            (~pl.col("is_buy")).sum().cast(pl.Int64).alias("n_sell_txns"),
        )
        .with_columns((pl.col("buy_usd") - pl.col("sell_usd")).alias("net_buy_usd"))
        .select(
            "symbol", "filing_date", "net_buy_usd", "buy_usd", "sell_usd",
            "n_buy_txns", "n_sell_txns",
        )
        .sort("symbol", "filing_date")
    )


# --------------------------------------------------------------- ingestion


def _quarter_str(y: int, q: int) -> str:
    return f"{y}q{q}"


def _default_quarters() -> list[str]:
    """FIRST_QUARTER through the last complete quarter (a quarter's data set
    is published shortly after the quarter ends)."""
    y0, q0 = int(FIRST_QUARTER[:4]), int(FIRST_QUARTER[-1])
    today = date.today()
    y1, q1 = today.year, (today.month - 1) // 3 + 1
    q1 -= 1
    if q1 == 0:
        y1, q1 = y1 - 1, 4
    out = []
    y, q = y0, q0
    while (y, q) <= (y1, q1):
        out.append(_quarter_str(y, q))
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return out


def _fetch_quarter(quarter: str, timeout: int = 180) -> pl.DataFrame | None:
    """Download one quarterly ZIP (trying both URL prefixes), aggregate the
    two needed TSVs, delete the raw ZIP. None if the quarter is unavailable."""
    zpath = RAW_DIR / f"{quarter}_form345.zip"
    for tmpl in URL_TEMPLATES:
        r = requests.get(tmpl.format(q=quarter), headers=UA, timeout=timeout)
        if r.status_code == 200 and r.content[:2] == b"PK":
            zpath.write_bytes(r.content)
            break
    else:
        return None
    try:
        with zipfile.ZipFile(zpath) as z:
            sub = pl.read_csv(
                z.read("SUBMISSION.tsv"),
                separator="\t",
                columns=_SUB_COLS,
                infer_schema_length=0,
                quote_char=None,
                encoding="utf8-lossy",
            )
            trans = pl.read_csv(
                z.read("NONDERIV_TRANS.tsv"),
                separator="\t",
                columns=_TRANS_COLS,
                infer_schema_length=0,
                quote_char=None,
                encoding="utf8-lossy",
            )
    finally:
        zpath.unlink(missing_ok=True)
    return aggregate_form345(sub, trans)


def ingest_insider(
    quarters: list[str] | None = None,
    max_quarters_per_run: int = 4,
    pause_s: float = 0.5,
) -> dict:
    """Extend the archive: newest missing quarter first, at most
    `max_quarters_per_run` downloads per call. Quarters already present
    (inferred from filing_date — a quarter is ingested atomically) are
    skipped, so the nightly loop converges without refetching."""
    have: set[str] = set()
    frames: list[pl.DataFrame] = []
    if INSIDER_PATH.exists():
        old = pl.read_parquet(INSIDER_PATH)
        frames.append(old)
        if old.height:
            have = set(
                old.select(
                    (
                        pl.col("filing_date").dt.year().cast(pl.Utf8)
                        + pl.lit("q")
                        + ((pl.col("filing_date").dt.month() - 1) // 3 + 1).cast(pl.Utf8)
                    ).alias("q")
                )["q"]
                .unique()
                .to_list()
            )

    if quarters is None:
        quarters = _default_quarters()
    todo = [q for q in sorted(set(quarters), reverse=True) if q not in have]
    todo = todo[:max_quarters_per_run]

    fetched, misses = 0, 0
    for q in todo:
        try:
            agg = _fetch_quarter(q)
        except requests.RequestException:
            misses += 1
            continue
        time.sleep(pause_s)  # fair-access politeness (sec.gov cap: 10 req/s)
        if agg is None or agg.height == 0:
            misses += 1
            continue
        frames.append(agg)
        fetched += 1

    rows = n_quarters = 0
    if frames:
        out = (
            pl.concat(frames, how="diagonal")
            .unique(subset=["symbol", "filing_date"])
            .sort("symbol", "filing_date")
        )
        out.write_parquet(INSIDER_PATH)
        rows = out.height
        n_quarters = (
            out.select(
                pl.col("filing_date").dt.year() * 4 + (pl.col("filing_date").dt.month() - 1) // 3
            )
            .n_unique()
        )
    return {
        "fetched_now": fetched,
        "misses": misses,
        "rows_in_archive": int(rows),
        "quarters_in_archive": int(n_quarters),
    }


def load_insider() -> pl.DataFrame | None:
    return pl.read_parquet(INSIDER_PATH) if INSIDER_PATH.exists() else None


# ------------------------------------------------------- ex-ante features


def attach_insider_features(features: pl.DataFrame, insider: pl.DataFrame | None) -> pl.DataFrame:
    """Ex-ante net-insider-buying features on the features panel:
      insider_net_buy_90d_usd  sum of net_buy_usd over the trailing 90
                               calendar days (D-90, D] — filing_date <= D only
      insider_buy_ratio_90d    buy_usd / (buy_usd + sell_usd) over the same
                               window; NaN when the denominator is 0
    NaN before the archive era or where the symbol never appears; a covered
    symbol with an empty window has net_buy = 0 (known quiet, not unknown)."""
    null_cols = [pl.lit(None, dtype=pl.Float64).alias(c) for c in INSIDER_FEATURES]
    if insider is None or insider.height == 0:
        return features.with_columns(null_cols)

    era_start = np.datetime64(insider["filing_date"].min(), "D")
    by_symbol: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (sym,), g in insider.group_by("symbol"):
        g = g.sort("filing_date")
        by_symbol[str(sym)] = (
            g["filing_date"].to_numpy().astype("datetime64[D]"),
            g["buy_usd"].to_numpy().astype(float),
            g["sell_usd"].to_numpy().astype(float),
        )

    frames = []
    for (sym,), g in features.group_by("symbol", maintain_order=True):
        g = g.sort("date")
        dates = g["date"].to_numpy().astype("datetime64[D]")
        n = dates.size
        net90 = np.full(n, np.nan)
        ratio90 = np.full(n, np.nan)
        info = by_symbol.get(str(sym))
        if info is not None:
            fdates, buy, sell = info
            cbuy = np.concatenate([[0.0], np.cumsum(buy)])
            csell = np.concatenate([[0.0], np.cumsum(sell)])
            # filings with date-90 < filing_date <= date (searchsorted style
            # mirrors catalysts.attach_catalyst_features n_form4_90d)
            hi = np.searchsorted(fdates, dates, side="right")
            lo = np.searchsorted(fdates, dates - np.timedelta64(WINDOW_DAYS, "D"), side="right")
            b = cbuy[hi] - cbuy[lo]
            s = csell[hi] - csell[lo]
            net90 = b - s
            denom = b + s
            with np.errstate(invalid="ignore"):
                ratio90 = np.where(denom > 0, b / denom, np.nan)
            # before the archive era, "no filings" is unknowable, not zero
            pre = dates < era_start
            net90[pre] = np.nan
            ratio90[pre] = np.nan
        frames.append(
            g.with_columns(
                pl.Series("insider_net_buy_90d_usd", net90),
                pl.Series("insider_buy_ratio_90d", ratio90),
            )
        )
    return pl.concat(frames)
