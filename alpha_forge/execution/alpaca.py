"""Alpaca Trading API client (paper by default).

Credentials come from the environment (.env is loaded by config, and .env is
gitignored — keys never enter the repo). The client refuses to construct
against the LIVE endpoint unless the caller passes allow_live=True, which
only broker.place_order does after the human double-lock; everything else in
the system talks to the paper endpoint only.

API reference: https://docs.alpaca.markets/reference (Trading API v2).
"""

from __future__ import annotations

import os

import requests

import alpha_forge.config  # noqa: F401  (side effect: loads .env)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"


class AlpacaCredentialsMissing(Exception):
    pass


class AlpacaLiveEndpointBlocked(Exception):
    pass


class AlpacaClient:
    def __init__(self, allow_live: bool = False, timeout: int = 20):
        self.base = os.environ.get("ALPACA_BASE_URL", PAPER_URL).rstrip("/")
        key = os.environ.get("ALPACA_API_KEY_ID")
        secret = os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise AlpacaCredentialsMissing(
                "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set (see .env.example)"
            )
        self.is_paper = "paper-api" in self.base
        if not self.is_paper and not allow_live:
            raise AlpacaLiveEndpointBlocked(
                "ALPACA_BASE_URL points at the LIVE endpoint; this code path only "
                "accepts the paper endpoint. Live order flow goes exclusively "
                "through broker.place_order's human double-lock."
            )
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None):
        r = requests.get(f"{self.base}{path}", headers=self._headers,
                         params=params or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict):
        r = requests.post(f"{self.base}{path}", headers=self._headers,
                          json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---- read-only ----

    def account(self) -> dict:
        return self._get("/v2/account")

    def clock(self) -> dict:
        return self._get("/v2/clock")

    def positions(self) -> list[dict]:
        return self._get("/v2/positions")

    def orders(self, status: str = "open", limit: int = 50) -> list[dict]:
        return self._get("/v2/orders", {"status": status, "limit": limit})

    # ---- order flow (paper: allowed; live: only via broker double-lock) ----

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> dict:
        payload: dict = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": order_type.lower(),
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if stop_price is not None:
            payload["stop_price"] = str(stop_price)
        return self._post("/v2/orders", payload)


def account_snapshot_public() -> dict | None:
    """Sanitized account state for dashboards/exports: no account number,
    no identifiers — only balances and trading-state flags. Returns None
    when credentials are absent or the API is unreachable (dashboards must
    degrade gracefully, not crash the nightly loop)."""
    try:
        client = AlpacaClient()
        acct = client.account()
        clock = client.clock()
        positions = client.positions()
        orders = client.orders("open")
    except (AlpacaCredentialsMissing, AlpacaLiveEndpointBlocked, requests.RequestException):
        return None
    return {
        "mode": "PAPER" if client.is_paper else "LIVE",
        "status": acct.get("status"),
        "equity": float(acct.get("equity", 0)),
        "cash": float(acct.get("cash", 0)),
        "buying_power": float(acct.get("buying_power", 0)),
        "pattern_day_trader": bool(acct.get("pattern_day_trader", False)),
        "daytrade_count": int(acct.get("daytrade_count", 0)),
        "market_open": bool(clock.get("is_open", False)),
        "next_open": clock.get("next_open"),
        "next_close": clock.get("next_close"),
        "positions": [
            {
                "symbol": p.get("symbol"),
                "qty": float(p.get("qty", 0)),
                "avg_entry_price": float(p.get("avg_entry_price", 0)),
                "market_value": float(p.get("market_value", 0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0)),
            }
            for p in positions
        ],
        "open_orders": [
            {
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "qty": o.get("qty"),
                "type": o.get("type"),
                "status": o.get("status"),
                "submitted_at": o.get("submitted_at"),
            }
            for o in orders
        ],
    }
