"""Agent 10 — broker adapter interface. BUILT BUT DISABLED.

The system never places a live order on its own. Live order flow requires
BOTH of, at call time:
  1. environment variable ALPHA_FORGE_LIVE_TRADING=I_UNDERSTAND_THE_RISKS
     (set by a human, never by code — grep guard in tests)
  2. a HUMAN_DECISION ledger entry with payload {"action": "enable_live_trading"}

Absent either, place_order raises. Paper trading uses the same adapter
surface with fills from the Agent-5 stress-fill model, so the runbook that
paper-trades is byte-identical to the one that would trade live.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from alpha_forge.ledger import Ledger

LIVE_FLAG_ENV = "ALPHA_FORGE_LIVE_TRADING"
LIVE_FLAG_VALUE = "I_UNDERSTAND_THE_RISKS"


class LiveTradingDisabledError(Exception):
    pass


@dataclass
class Order:
    symbol: str
    side: str          # BUY | SELL
    quantity: float
    order_type: str    # MARKET_ON_OPEN | LIMIT | STOP
    limit_price: float | None = None
    stop_price: float | None = None


def live_trading_enabled(ledger: Ledger) -> bool:
    if os.environ.get(LIVE_FLAG_ENV) != LIVE_FLAG_VALUE:
        return False
    return any(
        e["kind"] == "HUMAN_DECISION"
        and e["payload"].get("action") == "enable_live_trading"
        for e in ledger.entries()
    )


class BrokerAdapter(ABC):
    """Alpaca / IBKR / Tradier implement this surface."""

    name: str = "abstract"

    @abstractmethod
    def submit(self, order: Order) -> dict: ...

    @abstractmethod
    def positions(self) -> list[dict]: ...

    def place_order(self, order: Order, ledger: Ledger) -> dict:
        if not live_trading_enabled(ledger):
            raise LiveTradingDisabledError(
                "live trading is disabled: requires the human-set env flag AND a "
                "ledgered HUMAN_DECISION. The system never enables itself."
            )
        return self.submit(order)


class AlpacaAdapter(BrokerAdapter):
    """DESIGN mode without ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY."""

    name = "alpaca"

    def submit(self, order: Order) -> dict:
        raise NotImplementedError(
            "design: POST /v2/orders {symbol, qty, side, type, time_in_force} "
            "(fractional qty supported; commission $0 per Alpaca pricing page)"
        )

    def positions(self) -> list[dict]:
        raise NotImplementedError("design: GET /v2/positions")


class IBKRAdapter(BrokerAdapter):
    name = "ibkr"

    def submit(self, order: Order) -> dict:
        raise NotImplementedError("design: Client Portal API POST /iserver/account/orders")

    def positions(self) -> list[dict]:
        raise NotImplementedError("design: GET /portfolio/accounts + /positions")


class TradierAdapter(BrokerAdapter):
    name = "tradier"

    def submit(self, order: Order) -> dict:
        raise NotImplementedError("design: POST /v1/accounts/{id}/orders")

    def positions(self) -> list[dict]:
        raise NotImplementedError("design: GET /v1/accounts/{id}/positions")
