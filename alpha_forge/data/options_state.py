"""Options as sensors (BREAKTHROUGH_PHASE1 §6): archived Cboe delayed
chains become Genome fields long before options are ever traded. Per
(underlying, snapshot_date):

  iv_atm_near         mean vendor IV of near-expiry contracts within 5% of
                      spot (7-45 DTE preferred; the shortest tenor >= 7 DTE
                      otherwise — 0-2 DTE noise never defines "the" IV)
  iv_term_slope       far ATM IV minus near ATM IV (far = first expiry
                      >= 21 days beyond near); inverted slopes into a filed
                      catalyst date are the §6 sensor case
  iv_skew_asym        mean IV of 80-95%-moneyness puts minus mean IV of
                      105-120%-moneyness calls, near expiry
  oi_conc_top_strike  share of the chain's total open interest sitting at
                      the single most-loaded strike

Data is REAL_DELAYED (source column watermark); snapshots are captured
intraday, so every value here was public before that session's close —
lag 0 at a close-of-session decision, recorded in the genome registry.
Coverage begins with the archive (2026-08-10); everything earlier is null,
and null is never evidence.
"""

from __future__ import annotations

import polars as pl

from alpha_forge.config import STORE_DIR

CHAINS_DIR = STORE_DIR / "options_chains"
NEAR_DTE_MIN, NEAR_DTE_MAX = 7, 45
ATM_BAND = 0.05
PUT_BAND = (0.80, 0.95)
CALL_BAND = (1.05, 1.20)
MIN_FAR_GAP_DAYS = 21

OPTIONS_STATE_FIELDS = (
    "iv_atm_near", "iv_term_slope", "iv_skew_asym", "oi_conc_top_strike",
)


def compute_options_state(chains: pl.DataFrame) -> pl.DataFrame:
    """(symbol, date, *OPTIONS_STATE_FIELDS) from one or many snapshots."""
    c = chains.with_columns(
        (pl.col("expiry").str.to_date() - pl.col("snapshot_date").str.to_date())
        .dt.total_days().alias("dte"),
        (pl.col("strike") / pl.col("underlying_close")).alias("m"),
        pl.col("iv_vendor").fill_nan(None).alias("iv"),
    ).filter(pl.col("dte") >= NEAR_DTE_MIN)

    rows = []
    for (sym, snap), g in c.group_by(["underlying", "snapshot_date"]):
        expiries = (
            g.group_by("dte").len().sort("dte")["dte"].to_list())
        if not expiries:
            continue
        preferred = [d for d in expiries if d <= NEAR_DTE_MAX]
        near = preferred[0] if preferred else expiries[0]
        fars = [d for d in expiries if d >= near + MIN_FAR_GAP_DAYS]
        far = fars[0] if fars else None

        def _atm_iv(dte):
            s = g.filter((pl.col("dte") == dte)
                         & (pl.col("m") - 1.0).abs().le(ATM_BAND))["iv"]
            return float(s.mean()) if s.drop_nulls().len() else None

        near_iv = _atm_iv(near)
        far_iv = _atm_iv(far) if far is not None else None
        slope = (far_iv - near_iv) if (near_iv is not None
                                       and far_iv is not None) else None

        nearg = g.filter(pl.col("dte") == near)
        puts = nearg.filter((pl.col("right") == "P")
                            & pl.col("m").is_between(*PUT_BAND))["iv"]
        calls = nearg.filter((pl.col("right") == "C")
                             & pl.col("m").is_between(*CALL_BAND))["iv"]
        skew = (float(puts.mean()) - float(calls.mean())) if (
            puts.drop_nulls().len() and calls.drop_nulls().len()) else None

        oi = g.group_by("strike").agg(pl.col("open_interest").sum().alias("o"))
        tot = float(oi["o"].sum())
        conc = float(oi["o"].max()) / tot if tot > 0 else None

        rows.append({"symbol": sym, "date": snap, "iv_atm_near": near_iv,
                     "iv_term_slope": slope, "iv_skew_asym": skew,
                     "oi_conc_top_strike": conc})
    schema = {"symbol": pl.Utf8, "date": pl.Utf8,
              **{f: pl.Float64 for f in OPTIONS_STATE_FIELDS}}
    out = (pl.DataFrame(rows, schema=schema) if rows
           else pl.DataFrame(schema=schema))
    return out.with_columns(pl.col("date").str.to_date())


def load_options_state() -> pl.DataFrame | None:
    """Compute the sensor table from every archived snapshot; None when the
    archive is empty (builder then carries null options fields)."""
    if not CHAINS_DIR.exists():
        return None
    files = sorted(CHAINS_DIR.glob("chains_*.parquet"))
    if not files:
        return None
    return compute_options_state(
        pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed"))
