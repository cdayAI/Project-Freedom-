import pytest

from alpha_forge.execution.alpaca import (
    LIVE_URL,
    PAPER_URL,
    AlpacaClient,
    AlpacaCredentialsMissing,
    AlpacaLiveEndpointBlocked,
)
from alpha_forge.execution.broker import AlpacaAdapter, LiveTradingDisabledError, Order


@pytest.fixture
def paper_env(monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL", PAPER_URL)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "test-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "test-secret")


def test_missing_credentials_raise(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(AlpacaCredentialsMissing):
        AlpacaClient()


def test_live_endpoint_blocked_by_default(monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL", LIVE_URL)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    with pytest.raises(AlpacaLiveEndpointBlocked):
        AlpacaClient()
    # allow_live is reachable only through broker.place_order's double-lock
    client = AlpacaClient(allow_live=True)
    assert not client.is_paper


def test_paper_client_constructs(paper_env):
    client = AlpacaClient()
    assert client.is_paper
    assert client.base == PAPER_URL


def test_paper_order_requires_paper_endpoint(monkeypatch):
    monkeypatch.setenv("ALPACA_BASE_URL", LIVE_URL)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    adapter = AlpacaAdapter()
    # constructing the paper path against a live URL must refuse before any
    # HTTP happens (AlpacaClient raises at construction without allow_live)
    with pytest.raises((LiveTradingDisabledError, AlpacaLiveEndpointBlocked)):
        adapter.paper_order(Order(symbol="AAPL", side="BUY", quantity=1, order_type="MARKET_ON_OPEN"))


def test_paper_order_payload_shape(paper_env, monkeypatch):
    captured = {}

    def fake_post(self, path, payload):
        captured["path"] = path
        captured["payload"] = payload
        return {"id": "fake", "status": "accepted"}

    monkeypatch.setattr(AlpacaClient, "_post", fake_post)
    adapter = AlpacaAdapter()
    out = adapter.paper_order(
        Order(symbol="AAPL", side="BUY", quantity=1.5, order_type="MARKET_ON_OPEN")
    )
    assert out["status"] == "accepted"
    assert captured["path"] == "/v2/orders"
    assert captured["payload"] == {
        "symbol": "AAPL",
        "qty": "1.5",
        "side": "buy",
        "type": "market",
        "time_in_force": "day",
    }


def test_place_order_still_double_locked_on_paper(paper_env, tmp_path, monkeypatch):
    # place_order is the LIVE path; even with paper keys it demands the
    # human double-lock — paper flow must use paper_order explicitly
    monkeypatch.delenv("ALPHA_FORGE_LIVE_TRADING", raising=False)
    from alpha_forge.ledger import Ledger

    ledger = Ledger(path=tmp_path / "l.jsonl")
    with pytest.raises(LiveTradingDisabledError):
        AlpacaAdapter().place_order(
            Order(symbol="AAPL", side="BUY", quantity=1, order_type="MARKET_ON_OPEN"), ledger
        )
