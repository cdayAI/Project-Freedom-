"""Agent 5 — TRAP DETECTOR.

Adversarial simulation of the failure modes clean backtests hide. At $2,000,
ACCOUNT MECHANICS bind before market microstructure does, so they are
first-class simulations here, not footnotes:

  - FINRA Pattern Day Trader rule (FINRA 4210(f)(8)(B)): margin accounts
    under $25,000 are limited to 3 day trades per 5 rolling business days.
  - Cash accounts escape PDT but must respect T+1 settlement (SEC Rel.
    34-96930, eff. 2024-05-28) and free-riding/good-faith-violation rules.
  - Positions are INTEGER shares (or broker-supported fractional lots);
    no strategy may assume divisible capital.
  - Stops are not fills: a gap through the stop fills at the next open.
  - Halt days (no volume / missing bars) are forced holds.

A strategy whose trade plan is infeasible under its account type is killed
here regardless of its backtest. Reported performance is performance UNDER
stress fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PDT_EQUITY_THRESHOLD = 25_000.0   # FINRA 4210(f)(8)(B)(iv)
PDT_MAX_DAY_TRADES = 3            # per 5 rolling business days
SETTLEMENT_DAYS = 1               # T+1, SEC Rel. 34-96930


@dataclass
class Trade:
    """One round trip in business-day indices (entry open, exit open/close)."""
    entry_day: int
    exit_day: int
    notional: float
    symbol: str = "?"

    @property
    def is_day_trade(self) -> bool:
        return self.entry_day == self.exit_day


@dataclass
class AccountCheckResult:
    feasible: bool
    violations: list[str] = field(default_factory=list)
    day_trades_used: int = 0


def check_pdt(trades: list[Trade], account_equity: float) -> AccountCheckResult:
    """Margin-account PDT feasibility."""
    if account_equity >= PDT_EQUITY_THRESHOLD:
        return AccountCheckResult(feasible=True)
    day_trades = sorted(t.entry_day for t in trades if t.is_day_trade)
    violations = []
    for i, d in enumerate(day_trades):
        window = [x for x in day_trades if d - 4 <= x <= d]
        if len(window) > PDT_MAX_DAY_TRADES:
            violations.append(
                f"day {d}: {len(window)} day trades in rolling 5-day window "
                f"(limit {PDT_MAX_DAY_TRADES} under $25k equity) — account would be flagged"
            )
            break
    return AccountCheckResult(
        feasible=not violations, violations=violations, day_trades_used=len(day_trades)
    )


def check_cash_settlement(trades: list[Trade], starting_cash: float) -> AccountCheckResult:
    """Cash-account T+1 settlement and good-faith-violation simulation.

    Settled cash ledger: buys spend settled cash only; sale proceeds settle
    entry_day+1 relative to the sale day. Buying with unsettled funds and
    selling that position before the funding sale settles = GFV.
    """
    violations = []
    settled = starting_cash
    pending: list[tuple[int, float]] = []  # (settle_day, amount)
    events = []
    for t in trades:
        events.append((t.entry_day, "BUY", t))
        events.append((t.exit_day, "SELL", t))
    # within a day: sells of prior-day positions free cash first, then buys,
    # then same-day sells (a day trade's SELL cannot precede its own BUY)
    def _sub(e):
        day, kind, t = e
        if kind == "SELL":
            return 0 if t.entry_day < day else 2
        return 1
    events.sort(key=lambda e: (e[0], _sub(e)))

    unsettled_funded_positions: dict[int, int] = {}  # id(trade) -> funding settle day
    for day, kind, t in events:
        # settle matured proceeds
        matured = [p for p in pending if p[0] <= day]
        for p in matured:
            settled += p[1]
            pending.remove(p)
        if kind == "BUY":
            if settled >= t.notional:
                settled -= t.notional
            else:
                unsettled_total = sum(a for _, a in pending)
                if settled + unsettled_total >= t.notional:
                    # funded by unsettled proceeds: legal, but selling before
                    # those proceeds settle is a GFV
                    latest_settle = max(sd for sd, _ in pending)
                    need_from_unsettled = t.notional - settled
                    settled = 0.0
                    remaining = need_from_unsettled
                    new_pending = []
                    for sd, amt in pending:
                        take = min(amt, remaining)
                        remaining -= take
                        if amt - take > 0:
                            new_pending.append((sd, amt - take))
                    pending = new_pending
                    unsettled_funded_positions[id(t)] = latest_settle
                else:
                    violations.append(
                        f"day {day}: buy of ${t.notional:,.0f} exceeds settled+unsettled "
                        "cash — free-riding / insufficient funds"
                    )
        else:  # SELL
            pending.append((day + SETTLEMENT_DAYS, t.notional))
            fund_day = unsettled_funded_positions.pop(id(t), None)
            if fund_day is not None and day < fund_day:
                violations.append(
                    f"day {day}: sold a position bought with unsettled funds before "
                    f"those funds settled (day {fund_day}) — good-faith violation"
                )
    return AccountCheckResult(feasible=not violations, violations=violations)


def integer_lot_weights(
    capital: float, prices: np.ndarray, target_weights: np.ndarray, fractional_ok: bool
) -> dict:
    """Achievable weights under whole-share (or broker fractional) lots."""
    prices = np.asarray(prices, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    dollars = capital * target
    if fractional_ok:
        achieved = target.copy()
        unallocated = 0.0
    else:
        shares = np.floor(dollars / prices)
        spent = shares * prices
        achieved = spent / capital
        unallocated = float(capital - spent.sum())
    tracking_err = float(np.abs(achieved - target).sum() / 2.0)
    return {
        "achieved_weights": achieved,
        "unallocated_cash": unallocated if not fractional_ok else 0.0,
        "tracking_error_l1": tracking_err,
        "infeasible_names": int(np.sum((dollars < prices) & (target > 0)))
        if not fractional_ok
        else 0,
    }


def stop_fill_price(stop: float, next_open: float, low_after: float) -> float:
    """A stop is a trigger, not a guarantee: if price gaps through the stop,
    the fill is the next open (worse), else the stop price."""
    if next_open <= stop:
        return next_open
    return stop if low_after <= stop else np.nan  # NaN = stop never triggered


def stress_overnight_gaps(
    trade_returns: np.ndarray,
    panel_overnight_gaps: np.ndarray,
    worst_quantile: float = 0.01,
    n_injected: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Inject worst-tail overnight gaps (measured from the panel's own
    open/close series) into a trade-return sample: each injected trade takes
    an additional gap drawn from the worst `worst_quantile` of observed
    gaps. Models earnings-style gaps without a calendar; replaced by real
    catalyst-conditioned gaps when the calendar feed lands."""
    r = np.asarray(trade_returns, dtype=float).copy()
    gaps = np.asarray(panel_overnight_gaps, dtype=float)
    gaps = gaps[~np.isnan(gaps)]
    tail = np.sort(gaps)[: max(1, int(gaps.size * worst_quantile))]
    rng = np.random.default_rng(seed)
    k = n_injected if n_injected is not None else max(1, r.size // 20)
    idx = rng.choice(r.size, size=min(k, r.size), replace=False)
    r[idx] = (1 + r[idx]) * (1 + rng.choice(tail, size=idx.size)) - 1
    return r


def run_account_mechanics(
    trades: list[Trade],
    account_type: str,
    account_equity: float,
    fractional_ok: bool = False,
) -> AccountCheckResult:
    """Composite feasibility verdict for a trade plan under an account type."""
    if account_type == "margin":
        return check_pdt(trades, account_equity)
    if account_type == "cash":
        pdt_free = AccountCheckResult(feasible=True)
        settle = check_cash_settlement(trades, account_equity)
        return AccountCheckResult(
            feasible=pdt_free.feasible and settle.feasible,
            violations=settle.violations,
        )
    raise ValueError(f"unknown account type {account_type}")
