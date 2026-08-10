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


def _orf_table() -> dict:
    path = FEES_DIR / "orf_by_exchange.json"
    if not path.exists():
        raise UnverifiedFeeError("no ORF table on disk")
    return json.loads(path.read_text())


def orf_rate_on(exchange: str, trade_date: date | str) -> float:
    """Per-contract-side ORF for the executing exchange on trade_date.
    Dates before the earliest verified entry raise: several exchanges publish
    rates without ORF-specific effective dates, so those rates are anchored
    as-of the fetch date and carry NO verified history (see table notes —
    the ORF assessment model itself changed industry-wide on 2026-07-01)."""
    if isinstance(trade_date, str):
        trade_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
    table = _orf_table()
    entries = table.get("exchanges", {}).get(exchange)
    if not entries:
        raise UnverifiedFeeError(
            f"orf_by_exchange: no verified schedule for exchange {exchange!r}; "
            f"known: {sorted(table.get('exchanges', {}).keys())}"
        )
    governing = None
    for e in sorted(entries, key=lambda x: x["effective_date"]):
        if datetime.strptime(e["effective_date"], "%Y-%m-%d").date() <= trade_date:
            governing = e
    if governing is None:
        raise UnverifiedFeeError(
            f"orf_by_exchange[{exchange}]: no rate on record for {trade_date} "
            f"(earliest verified {entries[0]['effective_date']})"
        )
    if not governing.get("verified", False):
        raise UnverifiedFeeError(f"orf_by_exchange[{exchange}]: governing rate UNVERIFIED")
    return float(governing["rate"])


def option_sell_fees(
    contracts: int,
    premium_usd: float,
    trade_date: date | str,
    exchange: str = "CBOE",
    is_index_option: bool = False,
) -> dict:
    """Regulatory + clearing fees on a US-listed option SELL.

    premium_usd: total premium of the sale (contracts x price x 100).
    Components (each verified-or-raise, per primary sources in the tables):
      - FINRA TAF per contract, NO cap for options; index options TAF-exempt
      - OCC clearing per contract (also assessed on buys; callers model the
        buy side by calling option_buy_fees)
      - ORF per contract side, by EXECUTING exchange
      - SEC Section 31 on the premium (covered sale); INDEX options exempt
        (17 CFR 240.31(a)(11)(vi))
    """
    occ = _schedule("occ_clearing")
    occ_fee = contracts * occ.rate_on(trade_date)
    orf_fee = contracts * orf_rate_on(exchange, trade_date)
    if is_index_option:
        taf_fee = 0.0  # index options are TAF-exempt (Schedule A Section 1)
        sec_fee = 0.0  # and Section 31-exempt (240.31(a)(11)(vi))
    else:
        taf = _schedule("finra_taf_options")
        taf_fee = contracts * taf.rate_on(trade_date)
        sec = _schedule("sec_section31")
        sec_fee = premium_usd / 1_000_000.0 * sec.rate_on(trade_date)
    return {
        "finra_taf": taf_fee,
        "occ_clearing": occ_fee,
        "orf": orf_fee,
        "sec_section31": sec_fee,
        "total": taf_fee + occ_fee + orf_fee + sec_fee,
    }


def option_buy_fees(
    contracts: int, trade_date: date | str, exchange: str = "CBOE"
) -> dict:
    """Buy side: OCC clearing + ORF apply; TAF and Section 31 are sell-side."""
    occ = _schedule("occ_clearing")
    occ_fee = contracts * occ.rate_on(trade_date)
    orf_fee = contracts * orf_rate_on(exchange, trade_date)
    return {"occ_clearing": occ_fee, "orf": orf_fee, "total": occ_fee + orf_fee}
