"""Market Genome: knowability is mechanical, labels are not features."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from alpha_forge.genome.builder import (
    GenomeSchemaError,
    _last_published_quarter_end,
    audit_no_leak,
    build_genome,
)
from alpha_forge.genome.schema import FIELD_BY_NAME


def _busdays(start: str, n: int) -> list[dt.date]:
    days = []
    d = np.datetime64(start)
    while len(days) < n:
        if np.is_busday(d):
            days.append(d.astype(dt.date))
        d += 1
    return days


def _features(n_days=260, symbols=("AAA", "BBB")) -> pl.DataFrame:
    """Deterministic values: every field = day index (+ symbol offset), so a
    lag of k sessions is exactly a difference of k."""
    days = _busdays("2026-01-02", n_days)
    rows = []
    for si, sym in enumerate(symbols):
        for i, d in enumerate(days):
            v = float(i + 1000 * si)
            row = {"symbol": sym, "date": d, "valid": True}
            for name, f in FIELD_BY_NAME.items():
                if f.source_frame != "panel":
                    continue  # joined from the regime/options frames
                row[name] = bool(i % 7 == 0) if name == "starts_5x_fwd" else v
            rows.append(row)
    return pl.DataFrame(rows)


def _regime(n_days=260) -> pl.DataFrame:
    days = _busdays("2026-01-02", n_days)
    return pl.DataFrame(
        {
            "date": days,
            "spy_close": [400.0 + i for i in range(n_days)],
            "vix_close": [15.0] * n_days,
            "trend": ["UP"] * n_days,
            "vol_state": ["LOW"] * n_days,
            "chop": [False] * n_days,
        }
    )


def test_last_published_quarter_end():
    # Q2 file (Jun 30 + 35d ~ Aug 4): visible Aug 11, not Jul 20
    assert _last_published_quarter_end(dt.date(2026, 8, 11)) == dt.date(2026, 6, 30)
    assert _last_published_quarter_end(dt.date(2026, 7, 20)) == dt.date(2026, 3, 31)
    # Feb 1 is BEFORE the Q4 drop (Dec 31 + 35d ~ Feb 4): only Q3 is out
    assert _last_published_quarter_end(dt.date(2026, 2, 1)) == dt.date(2025, 9, 30)
    assert _last_published_quarter_end(dt.date(2026, 2, 10)) == dt.date(2025, 12, 31)


def test_lagged_field_is_exactly_the_prior_session(tmp_path):
    genome, meta = build_genome(_features(), _regime(), out_dir=tmp_path)
    g = genome.filter(pl.col("symbol") == "AAA").sort("date")
    src = _features().filter(pl.col("symbol") == "AAA").sort("date")
    # days_since_any_8k is declared lag 1: row i must equal source row i-1
    got = g["days_since_any_8k"].to_list()
    exp = src["days_since_any_8k"].to_list()
    assert got[0] is None
    assert got[1:] == exp[:-1]
    # price is lag 0: identical to source
    assert g["price"].to_list() == src["price"].to_list()


def test_quarterly_source_is_frozen_at_the_published_quarter(tmp_path):
    genome, _ = build_genome(_features(), _regime(), out_dir=tmp_path)
    g = genome.filter(pl.col("symbol") == "AAA").sort("date")
    src = _features().filter(pl.col("symbol") == "AAA").sort("date")
    dates = g["date"].to_list()
    vals = dict(zip(dates, g["insider_net_buy_90d_usd"].to_list()))
    src_by_date = dict(zip(src["date"].to_list(), src["insider_net_buy_90d_usd"].to_list()))
    # pick a decision date in August: only the Q2 (Jun 30) file is published
    d = next(x for x in dates if x >= dt.date(2026, 8, 10))
    cutoff = _last_published_quarter_end(d)
    last_src_date = max(x for x in dates if x <= cutoff)
    # deep inside the published-quarter window the frozen value rules (the
    # extra declared session of lag only shifts the publication boundary)
    assert vals[d] == src_by_date[last_src_date]
    # and the same-day filing-date value (the optimistic one) must NOT leak
    assert vals[d] != src_by_date[d]


def test_missing_registered_field_refuses(tmp_path):
    feats = _features().drop("short_ratio_5d")
    with pytest.raises(GenomeSchemaError, match="short_ratio_5d"):
        build_genome(feats, _regime(), out_dir=tmp_path)


def test_labels_ride_unlagged_and_are_flagged_as_outcomes(tmp_path):
    genome, meta = build_genome(_features(), _regime(), out_dir=tmp_path)
    g = genome.filter(pl.col("symbol") == "AAA").sort("date")
    src = _features().filter(pl.col("symbol") == "AAA").sort("date")
    assert g["starts_5x_fwd"].to_list() == src["starts_5x_fwd"].to_list()
    assert "starts_5x_fwd" in meta["outcome_fields"]
    assert meta["universe_survivorship_biased"] is True
    assert any(r["field"] == "insider_net_buy_90d_usd" and "published" in r["rule"]
               for r in meta["knowability"])


def test_audit_passes_on_honest_build_and_catches_a_leak(tmp_path):
    feats = _features()
    genome, _ = build_genome(feats, _regime(), out_dir=tmp_path)
    audited = audit_no_leak(feats, genome)
    assert audited["days_since_any_8k"]["checked"] > 0
    # manufacture a leak: overwrite the lagged column with same-session values
    leaked = genome.drop("days_since_any_8k").join(
        feats.select(["symbol", "date", "days_since_any_8k"]),
        on=["symbol", "date"], how="left",
    )
    with pytest.raises(GenomeSchemaError, match="knowability leak"):
        audit_no_leak(feats, leaked)
