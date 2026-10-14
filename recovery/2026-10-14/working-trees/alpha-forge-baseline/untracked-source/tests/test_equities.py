from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from alpha_forge.data.equities import _drop_incomplete_session
from alpha_forge.data.equities import YahooDailyAdapter


def test_new_york_timezone_database_is_available() -> None:
    """Clean installs must support the exchange-time data-integrity rule."""
    assert ZoneInfo("America/New_York").key == "America/New_York"


def test_drop_incomplete_session_accepts_daily_frame() -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2020, 1, 2)],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000.0],
        }
    )

    assert _drop_incomplete_session(frame).height == 1


def test_ingest_parallel_fetch_preserves_symbol_order(tmp_path, monkeypatch) -> None:
    adapter = YahooDailyAdapter(pause_s=0, max_workers=3)
    rows = 60
    frame = pl.DataFrame(
        {
            "date": [date(2020, 1, 2) + timedelta(days=i) for i in range(rows)],
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [1_000.0] * rows,
        }
    )
    monkeypatch.setattr(adapter, "fetch_symbol", lambda _symbol: frame.clone())
    monkeypatch.setattr("alpha_forge.data.equities.STORE_DIR", tmp_path)

    result = adapter.ingest(["CCC", "AAA", "BBB"], out_dir=tmp_path / "symbols")
    combined = pl.read_parquet(tmp_path / "equities_daily.parquet")

    assert result["ingested"] == 3
    assert combined["symbol"].unique(maintain_order=True).to_list() == ["CCC", "AAA", "BBB"]


def test_ingest_refetches_corrupt_same_day_cache(tmp_path, monkeypatch) -> None:
    adapter = YahooDailyAdapter(pause_s=0, max_workers=1)
    rows = 60
    frame = pl.DataFrame(
        {
            "date": [date(2020, 1, 2) + timedelta(days=i) for i in range(rows)],
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [1_000.0] * rows,
        }
    )
    calls: list[str] = []

    def fetch(symbol: str) -> pl.DataFrame:
        calls.append(symbol)
        return frame.clone()

    out_dir = tmp_path / "symbols"
    out_dir.mkdir()
    (out_dir / "BROKEN.parquet").write_bytes(b"truncated")
    monkeypatch.setattr(adapter, "fetch_symbol", fetch)
    monkeypatch.setattr("alpha_forge.data.equities.STORE_DIR", tmp_path)

    result = adapter.ingest(["BROKEN"], out_dir=out_dir)

    assert result["ingested"] == 1
    assert calls == ["BROKEN"]
    assert pl.read_parquet(out_dir / "BROKEN.parquet").height == rows
