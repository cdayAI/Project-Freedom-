import pytest

import alpha_forge.execution.marketdata as md


@pytest.fixture
def fake_creds(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")


def test_requires_credentials(monkeypatch):
    with pytest.raises(RuntimeError):
        md.MarketDataClient()


def test_bars_newest_window_oldest_first(fake_creds, monkeypatch):
    calls = []

    def fake_get(self, path, params):
        calls.append((path, dict(params)))
        # sort=desc: newest first from the API
        return {"bars": [{"t": "2026-08-10", "c": 3}, {"t": "2026-08-09", "c": 2},
                         {"t": "2026-08-08", "c": 1}], "next_page_token": None}

    monkeypatch.setattr(md.MarketDataClient, "_get", fake_get)
    bars = md.MarketDataClient().bars("AAPL", "1Day", limit=3)
    assert calls[0][1]["sort"] == "desc"
    assert [b["t"] for b in bars] == ["2026-08-08", "2026-08-09", "2026-08-10"]


def test_bars_rejects_bad_timeframe(fake_creds):
    with pytest.raises(ValueError):
        md.MarketDataClient().bars("AAPL", "3Sec")


def test_snapshot_change_pct(fake_creds, monkeypatch):
    def fake_get(self, path, params):
        return {
            "AAPL": {
                "latestTrade": {"p": 110.0, "t": "2026-08-10T15:00:00Z"},
                "latestQuote": {"bp": 109.9, "ap": 110.1},
                "dailyBar": {"o": 108, "h": 111, "l": 107, "v": 1000},
                "prevDailyBar": {"c": 100.0},
            }
        }

    monkeypatch.setattr(md.MarketDataClient, "_get", fake_get)
    snap = md.MarketDataClient().snapshots(["AAPL"])
    assert snap["AAPL"]["change_pct"] == pytest.approx(10.0)
    assert snap["AAPL"]["feed"] == "iex"


def test_empty_symbol_list(fake_creds):
    assert md.MarketDataClient.__init__ is not None
    client = md.MarketDataClient()
    assert client.snapshots([]) == {}


def test_deep_history_missing_symbol_empty():
    assert md.deep_history("NOSUCHTICKERXYZ") == []
