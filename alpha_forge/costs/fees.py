"""Date-aware regulatory fee model.

Fees are applied at the rate in effect on the TRADE DATE, never today's rate.
Rate tables live in data/fees/*.json; every entry carries its primary-source
URL and a `verified` flag. Looking up a rate for a date not covered by a
verified entry raises UnverifiedFeeError — an unverified fee blocks any
backtest that touches it, by design (see Operating Rules).

Current tables: SEC Section 31 (per $1M covered sales), FINRA TAF on covered
equity sales (per share, capped per trade), FINRA TAF on option sales (per
contract), NFA assessment (per side, futures). Sources in SOURCES.md and in
each JSON entry.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from alpha_forge.config import FEES_DIR


class UnverifiedFeeError(Exception):
    """A backtest touched a fee rate that has no verified primary source."""


class FeeSchedule:
    """One fee's history: sorted (effective_date, rate, verified, source)."""

    def __init__(self, name: str, unit: str, entries: list[dict]):
        self.name = name
        self.unit = unit
        self.entries = sorted(entries, key=lambda e: e["effective_date"])
        if not self.entries:
            raise ValueError(f"fee table {name} is empty")

    @classmethod
    def load(cls, name: str, fees_dir: Path = FEES_DIR) -> "FeeSchedule":
        path = fees_dir / f"{name}.json"
        if not path.exists():
            raise UnverifiedFeeError(f"no fee table on disk for {name} ({path})")
        doc = json.loads(path.read_text())
        return cls(name=doc["fee"], unit=doc["unit"], entries=doc["entries"])

    def entry_on(self, trade_date: date | str) -> dict:
        if isinstance(trade_date, str):
            trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
        governing = None
        for e in self.entries:
            eff = datetime.strptime(e["effective_date"], "%Y-%m-%d").date()
            if eff <= trade_date:
                governing = e
            else:
                break
        if governing is None:
            raise UnverifiedFeeError(
                f"{self.name}: no rate on record for {trade_date} (earliest verified "
                f"entry is {self.entries[0]['effective_date']}); backtests touching "
                "this date are blocked until the historical rate is sourced"
            )
        if not governing.get("verified", False):
            raise UnverifiedFeeError(
                f"{self.name}: governing rate for {trade_date} "
                f"(effective {governing['effective_date']}) is UNVERIFIED"
            )
        return governing

    def rate_on(self, trade_date: date | str) -> float:
        return float(self.entry_on(trade_date)["rate"])


_cache: dict[str, FeeSchedule] = {}


def _schedule(name: str) -> FeeSchedule:
    if name not in _cache:
        _cache[name] = FeeSchedule.load(name)
    return _cache[name]


def earliest_verified_equity_fee_date() -> date:
    """First date from which BOTH SEC 31 and equity TAF are verified — the
    hard lower bound for any equities backtest that pays sell-side fees."""
    bounds = []
    for name in ("sec_section31", "finra_taf_equity"):
        sched = _schedule(name)
        verified = [e for e in sched.entries if e.get("verified")]
        if not verified:
            raise UnverifiedFeeError(f"{name}: no verified entries at all")
        bounds.append(datetime.strptime(verified[0]["effective_date"], "%Y-%m-%d").date())
    return max(bounds)


def equity_sell_fees(notional_usd: float, shares: int, trade_date: date | str) -> dict:
    """Regulatory fees on a US covered equity SELL (buys carry none of these).

    Returns component breakdown plus total, all in USD.
    """
    sec = _schedule("sec_section31")
    taf = _schedule("finra_taf_equity")
    sec_fee = notional_usd / 1_000_000.0 * sec.rate_on(trade_date)
    taf_entry = taf.entry_on(trade_date)
    taf_fee = min(shares * float(taf_entry["rate"]), float(taf_entry["cap_per_trade_usd"]))
    return {
        "sec_section31": sec_fee,
        "finra_taf": taf_fee,
        "total": sec_fee + taf_fee,
    }


def option_sell_fees(contracts: int, trade_date: date | str) -> dict:
    """FINRA TAF on option sells. ORF/OCC schedules are not yet verified and
    will raise here until their tables land with primary sources."""
    taf = _schedule("finra_taf_options")
    taf_fee = contracts * taf.rate_on(trade_date)
    orf = FeeSchedule.load("orf_by_exchange")  # raises UnverifiedFeeError until sourced
    _ = orf
    return {"finra_taf": taf_fee, "total": taf_fee}
