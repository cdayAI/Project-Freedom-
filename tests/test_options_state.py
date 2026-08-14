"""Options sensors: field math from a hand-built chain, and the genome's
null-before-coverage join semantics."""

import datetime as dt

import polars as pl

from alpha_forge.data.options_state import compute_options_state


def _chain() -> pl.DataFrame:
    """XYZ at 100 on 2026-08-12: near expiry 30d (ATM IV .50 both rights,
    90%-m put IV .70, 110%-m call IV .40), far expiry 58d (ATM IV .60).
    OI: 600 of 800 total at the 100 strike (near 500 + far 100). Plus a
    2-DTE expiry that must be ignored."""
    snap, near, far, short = "2026-08-12", "2026-09-11", "2026-10-09", "2026-08-14"
    rows = [
        # 2-DTE noise: absurd IV that must not contaminate anything
        dict(expiry=short, right="C", strike=100.0, iv_vendor=5.0, open_interest=0),
        dict(expiry=near, right="C", strike=100.0, iv_vendor=0.50, open_interest=300),
        dict(expiry=near, right="P", strike=100.0, iv_vendor=0.50, open_interest=200),
        dict(expiry=near, right="P", strike=90.0, iv_vendor=0.70, open_interest=100),
        dict(expiry=near, right="C", strike=110.0, iv_vendor=0.40, open_interest=100),
        dict(expiry=far, right="C", strike=100.0, iv_vendor=0.60, open_interest=50),
        dict(expiry=far, right="P", strike=100.0, iv_vendor=0.60, open_interest=50),
    ]
    return pl.DataFrame([
        {"underlying": "XYZ", "snapshot_date": snap, "underlying_close": 100.0,
         **r} for r in rows
    ])


def test_sensor_math():
    got = compute_options_state(_chain())
    assert got.height == 1
    r = got.row(0, named=True)
    assert r["symbol"] == "XYZ" and r["date"] == dt.date(2026, 8, 12)
    assert abs(r["iv_atm_near"] - 0.50) < 1e-9   # 2-DTE 5.0 IV excluded
    assert abs(r["iv_term_slope"] - 0.10) < 1e-9  # .60 far - .50 near
    assert abs(r["iv_skew_asym"] - 0.30) < 1e-9   # .70 put - .40 call
    assert abs(r["oi_conc_top_strike"] - 600 / 800) < 1e-9


def test_missing_far_expiry_yields_null_slope_not_zero():
    c = _chain().filter(pl.col("expiry") != "2026-10-09")
    r = compute_options_state(c).row(0, named=True)
    assert r["iv_term_slope"] is None
    assert r["iv_atm_near"] is not None


def test_genome_join_null_before_coverage(tmp_path):
    from tests.test_genome import _features, _regime

    from alpha_forge.genome.builder import build_genome

    feats = _features()
    # no archive at all -> null columns, coverage recorded as empty
    g0, m0 = build_genome(feats, _regime(), out_dir=tmp_path)
    assert g0["iv_atm_near"].is_null().all()
    assert m0["options_state_coverage"]["rows"] == 0

    d = dt.date(2026, 8, 12)
    opts = pl.DataFrame({
        "symbol": ["AAA"], "date": [d], "iv_atm_near": [0.5],
        "iv_term_slope": [0.1], "iv_skew_asym": [0.3],
        "oi_conc_top_strike": [0.6],
    })
    g1, m1 = build_genome(feats, _regime(), out_dir=tmp_path,
                          options_state=opts)
    hit = g1.filter((pl.col("symbol") == "AAA") & (pl.col("date") == d))
    assert hit["iv_atm_near"].to_list() == [0.5]
    # every other row stays null — coverage does not smear backward
    assert g1.filter(~((pl.col("symbol") == "AAA") & (pl.col("date") == d))
                     )["iv_atm_near"].is_null().all()
    assert m1["options_state_coverage"]["rows"] == 1
