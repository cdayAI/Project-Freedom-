import pytest

from alpha_forge.execution.broker import (
    LIVE_FLAG_ENV,
    LIVE_FLAG_VALUE,
    AlpacaAdapter,
    LiveTradingDisabledError,
    Order,
    live_trading_enabled,
)
from alpha_forge.ledger import Ledger

ORDER = Order(symbol="AAPL", side="BUY", quantity=1, order_type="MARKET_ON_OPEN")


def test_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv(LIVE_FLAG_ENV, raising=False)
    ledger = Ledger(path=tmp_path / "l.jsonl")
    with pytest.raises(LiveTradingDisabledError):
        AlpacaAdapter().place_order(ORDER, ledger)


def test_env_flag_alone_insufficient(tmp_path, monkeypatch):
    monkeypatch.setenv(LIVE_FLAG_ENV, LIVE_FLAG_VALUE)
    ledger = Ledger(path=tmp_path / "l.jsonl")  # no HUMAN_DECISION entry
    with pytest.raises(LiveTradingDisabledError):
        AlpacaAdapter().place_order(ORDER, ledger)


def test_ledger_decision_alone_insufficient(tmp_path, monkeypatch):
    monkeypatch.delenv(LIVE_FLAG_ENV, raising=False)
    ledger = Ledger(path=tmp_path / "l.jsonl")
    ledger.append("HUMAN_DECISION", {"action": "enable_live_trading"})
    assert not live_trading_enabled(ledger)


def test_both_required_then_credentials_gate(tmp_path, monkeypatch):
    monkeypatch.setenv(LIVE_FLAG_ENV, LIVE_FLAG_VALUE)
    ledger = Ledger(path=tmp_path / "l.jsonl")
    ledger.append("HUMAN_DECISION", {"action": "enable_live_trading"})
    assert live_trading_enabled(ledger)
    # even fully enabled, no API credentials (conftest strips them) means no
    # order can be constructed — and the conftest network guard would fail
    # the test loudly if an HTTP call were ever attempted here
    from alpha_forge.execution.alpaca import AlpacaCredentialsMissing

    with pytest.raises(AlpacaCredentialsMissing):
        AlpacaAdapter().place_order(ORDER, ledger)


def test_code_never_sets_the_flag():
    """Grep guard: no source file assigns the live-trading env variable."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "alpha_forge"
    for py in root.rglob("*.py"):
        text = py.read_text()
        assert f'environ["{LIVE_FLAG_ENV}"] =' not in text
        assert f"environ['{LIVE_FLAG_ENV}'] =" not in text
        assert "putenv" not in text
