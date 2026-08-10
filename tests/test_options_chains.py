import numpy as np
import polars as pl
import pytest

import alpha_forge.data.options_chains as oc


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(oc, "CHAINS_DIR", tmp_path / "chains")
    monkeypatch.setattr(oc, "SPREADS_PATH", tmp_path / "spreads.parquet")
    return tmp_path


def _fake_chain(symbol: str) -> pl.DataFrame:
    rows = []
    for strike in (90, 100, 110, 150):
        rows.append(
            {
                "underlying": symbol,
                "osi": f"{symbol}261218C{int(strike*1000):08d}",
                "expiry": "2026-12-18",
                "right": "C",
                "strike": float(strike),
                "bid": max(0.05, 100 - strike) if strike < 100 else 1.0,
                "ask": (max(0.05, 100 - strike) if strike < 100 else 1.0) * 1.1,
                "last": 1.0,
                "volume": 10.0,
                "open_interest": 100.0,
                "iv_vendor": 0.5,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.lit(100.0).alias("underlying_close"),
        pl.lit(oc.DATA_WATERMARK).alias("source"),
    )


def test_osi_parsing():
    m = oc._OSI.match("AAPL261120C00165000")
    assert m
    root, ymd, right, strike = m.groups()
    assert root == "AAPL" and right == "C" and int(strike) / 1000.0 == 165.0


def test_snapshot_idempotent_and_spread_stats(sandbox, monkeypatch):
    calls = {"n": 0}

    def fake_fetch(symbol, timeout=30):
        calls["n"] += 1
        return _fake_chain(symbol)

    monkeypatch.setattr(oc, "fetch_chain", fake_fetch)
    s1 = oc.snapshot_chains(["AAA", "BBB"], pause_s=0)
    assert s1["fetched_now"] == 2
    # second run same day: nothing refetched (snapshots are immutable)
    s2 = oc.snapshot_chains(["AAA", "BBB"], pause_s=0)
    assert s2["fetched_now"] == 0
    assert calls["n"] == 2

    stats = oc.load_spread_stats()
    assert stats is not None
    assert set(stats["underlying"].unique().to_list()) == {"AAA", "BBB"}
    # relative spreads were constructed at ~10% of mid
    atm = stats.filter(pl.col("moneyness") == "ATM")
    assert atm.height > 0
    assert atm["median_rel_spread"][0] == pytest.approx(0.095, abs=0.02)


def test_no_synthetic_watermark_ever():
    """The archive carries the REAL_DELAYED watermark; the word SYNTHETIC
    must never appear as a data source in this module."""
    import inspect

    src = inspect.getsource(oc)
    assert "REAL_DELAYED" in src
    assert 'source", "SYNTHETIC' not in src


def test_archive_depth_empty(sandbox):
    d = oc.archive_depth()
    assert d["snapshot_days"] == 0
