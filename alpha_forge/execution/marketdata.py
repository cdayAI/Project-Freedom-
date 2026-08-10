"""Alpaca Market Data API client (data.alpaca.markets).

Feed honesty: paper/free accounts get the IEX feed — real-time but covering
IEX's slice of consolidated volume (~2-3%). Full-market SIP data is a paid
subscription (Algo Trader Plus) and a human purchasing decision; until then
every quote this module serves is stamped feed="iex". Latency is dominated
by feed tier, not transport.

Daily history beyond Alpaca's range is served from our own 15-year Yahoo
parquet store, so charts get deep history plus fresh intraday in one call.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import requests

import alpha_forge.config  # noqa: F401  (side effect: loads .env)
from alpha_forge.config import STORE_DIR

DATA_URL = "https://data.alpaca.markets"
FEED = "iex"

TIMEFRAMES = {"1Min", "5Min", "15Min", "1Hour", "1Day"}


class MarketDataClient:
    def __init__(self, timeout: int = 20):
        key = os.environ.get("ALPACA_API_KEY_ID")
        secret = os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("market data needs ALPACA_API_KEY_ID/SECRET (see .env.example)")
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.timeout = timeout

    def _get(self, path: str, params: dict):
        r = requests.get(f"{DATA_URL}{path}", headers=self._headers, params=params,
                         timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def bars(self, symbol: str, timeframe: str = "1Day", limit: int = 500) -> list[dict]:
        """Recent bars, oldest first: [{t,o,h,l,c,v}]. Split-adjusted."""
        if timeframe not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {sorted(TIMEFRAMES)}")
        lookback_days = {"1Min": 5, "5Min": 14, "15Min": 30, "1Hour": 90, "1Day": 1500}
        start = (datetime.now(timezone.utc) - timedelta(days=lookback_days[timeframe]))
        # sort=desc: first page is the NEWEST bars — charts want the recent
        # window, not the oldest page of the range
        out: list[dict] = []
        page_token = None
        while len(out) < limit:
            params = {
                "timeframe": timeframe,
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": min(1000, limit - len(out)),
                "adjustment": "split",
                "feed": FEED,
                "sort": "desc",
            }
            if page_token:
                params["page_token"] = page_token
            doc = self._get(f"/v2/stocks/{symbol}/bars", params)
            out.extend(doc.get("bars") or [])
            page_token = doc.get("next_page_token")
            if not page_token:
                break
        return out[:limit][::-1]  # newest-first pages -> oldest-first series

    def snapshots(self, symbols: list[str]) -> dict:
        """Latest trade/quote/minute-bar/day-bar per symbol, feed-stamped."""
        if not symbols:
            return {}
        doc = self._get("/v2/stocks/snapshots", {"symbols": ",".join(symbols), "feed": FEED})
        out = {}
        for sym, s in doc.items():
            if not isinstance(s, dict):
                continue
            trade = s.get("latestTrade") or {}
            quote = s.get("latestQuote") or {}
            day = s.get("dailyBar") or {}
            prev = s.get("prevDailyBar") or {}
            last = trade.get("p")
            prev_close = prev.get("c")
            out[sym] = {
                "last": last,
                "last_ts": trade.get("t"),
                "bid": quote.get("bp"),
                "ask": quote.get("ap"),
                "day_open": day.get("o"),
                "day_high": day.get("h"),
                "day_low": day.get("l"),
                "day_volume": day.get("v"),
                "prev_close": prev_close,
                "change_pct": (last / prev_close - 1.0) * 100.0
                if last and prev_close else None,
                "feed": FEED,
            }
        return out


def deep_history(symbol: str) -> list[dict]:
    """Daily candles from our 15y parquet store, oldest first."""
    import polars as pl

    path = STORE_DIR / "equities_daily" / f"{symbol.upper()}.parquet"
    if not path.exists():
        return []
    df = pl.read_parquet(path).sort("date")
    return [
        {"t": str(r["date"]), "o": r["open"], "h": r["high"], "l": r["low"],
         "c": r["close"], "v": r["volume"]}
        for r in df.iter_rows(named=True)
    ]
