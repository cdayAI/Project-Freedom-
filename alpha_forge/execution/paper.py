"""Paper-trading harness with stress fills.

Every EXECUTABLE-track strategy paper-trades for a minimum of 60 trading
days before the flag is granted. Fills use the Agent-5 stress model against
the daily panel (next-open entry, half-spread + fees, gap-through-stop),
not clean prices. State is a JSONL journal per strategy; the reconciler
grades paper fills exactly like predictions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import polars as pl

from alpha_forge.config import STORE_DIR
from alpha_forge.costs.fees import equity_sell_fees
from alpha_forge.costs.slippage import effective_half_spread

PAPER_DIR = STORE_DIR / "paper"
MIN_PAPER_TRADING_DAYS = 60


@dataclass
class PaperFill:
    strategy_id: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    fees_usd: float
    fill_date: str
    note: str


def paper_fill_next_open(
    strategy_id: str,
    symbol: str,
    side: str,
    quantity: float,
    signal_date: str,
    panel: pl.DataFrame,
) -> PaperFill | None:
    """Fill at the next session's open after signal_date, spread-adjusted,
    date-aware fees on sells. Returns None if no next session exists yet."""
    g = panel.filter(pl.col("symbol") == symbol).sort("date")
    dates = g["date"].to_numpy()
    after = np.nonzero(dates > np.datetime64(signal_date))[0]
    if after.size == 0:
        return None
    i = int(after[0])
    open_ = g["open"].to_numpy().astype(float)
    hs = effective_half_spread(
        g["high"].to_numpy().astype(float),
        g["low"].to_numpy().astype(float),
        g["close"].to_numpy().astype(float),
    )
    px = open_[i]
    half = hs[i] if np.isfinite(hs[i]) else 0.005 / px
    fill = px * (1 + half) if side == "BUY" else px * (1 - half)
    fees = 0.0
    if side == "SELL":
        fees = equity_sell_fees(fill * quantity, int(np.ceil(quantity)), str(dates[i])[:10])["total"]
    pf = PaperFill(
        strategy_id=strategy_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        fill_price=float(fill),
        fees_usd=float(fees),
        fill_date=str(dates[i])[:10],
        note="stress fill: next open +/- half-spread; sells pay SEC31+TAF at date-aware rates",
    )
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with (PAPER_DIR / f"{strategy_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({**asdict(pf), "ts_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
    return pf


def paper_trading_days(strategy_id: str) -> int:
    path = PAPER_DIR / f"{strategy_id}.jsonl"
    if not path.exists():
        return 0
    days = {json.loads(l)["fill_date"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}
    return len(days)


def executable_flag(strategy_id: str) -> dict:
    days = paper_trading_days(strategy_id)
    return {
        "strategy_id": strategy_id,
        "paper_trading_days": days,
        "executable": days >= MIN_PAPER_TRADING_DAYS,
        "required_days": MIN_PAPER_TRADING_DAYS,
    }


def emit_runbook(
    strategy_id: str,
    instrument_universe: str,
    order_types: list[str],
    sizing_formula: str,
    stop_logic: str,
    invalidation_conditions: list[str],
    kill_criteria: list[str],
) -> str:
    """Runbook for an EXECUTABLE strategy: exact mechanics, no discretion."""
    flag = executable_flag(strategy_id)
    lines = [
        f"# RUNBOOK — {strategy_id}",
        "",
        f"Executable: **{flag['executable']}** "
        f"({flag['paper_trading_days']}/{flag['required_days']} paper days)",
        "",
        f"## Instruments\n{instrument_universe}",
        f"## Order types\n" + "\n".join(f"- {o}" for o in order_types),
        f"## Sizing\n{sizing_formula}",
        f"## Stop logic\n{stop_logic}",
        "## Invalidation conditions\n" + "\n".join(f"- {c}" for c in invalidation_conditions),
        "## Kill criteria (mechanical, Agent 8b enforces)\n"
        + "\n".join(f"- {c}" for c in kill_criteria),
        "",
        "_Live order flow additionally requires the human-set env flag and a "
        "ledgered HUMAN_DECISION; see execution/broker.py._",
    ]
    path = STORE_DIR.parent.parent / "reports" / f"runbook-{strategy_id}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
