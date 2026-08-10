"""Cost drag analysis: where does trading frequency kill the edge?

`cost_drag_report` takes a strategy's realized per-trade GROSS returns and its
full per-round-trip cost (regulatory fees + two half-spread crossings) and
shows: gross vs net equity curves, and — assuming the strategy's per-day edge
is what it measured — the holding-period / trade-frequency frontier at which
the net edge crosses zero. Spread dominates fees at small account sizes; this
report is where that fact becomes visible.
"""

from __future__ import annotations

import numpy as np


def cost_drag_report(
    gross_trade_returns: np.ndarray,
    holding_days: np.ndarray,
    cost_per_roundtrip: float,
) -> dict:
    """gross_trade_returns: simple returns per trade (gross of costs).
    holding_days: bars held per trade, same length.
    cost_per_roundtrip: proportional cost per round trip (e.g. 0.002 = 20bp).
    """
    g = np.asarray(gross_trade_returns, dtype=float)
    h = np.asarray(holding_days, dtype=float)
    if g.size == 0:
        raise ValueError("no trades")
    if np.any(h <= 0):
        raise ValueError("holding_days must be positive")

    net = g - cost_per_roundtrip
    gross_curve = np.cumprod(1 + g)
    net_curve = np.cumprod(1 + net)

    edge_per_day = np.log1p(g).sum() / h.sum()  # realized log edge per holding day
    cost_log = np.log1p(-cost_per_roundtrip)    # negative

    # Net log growth per year if the same per-day edge were harvested at
    # holding period H: 252/H trades x (edge_per_day*H + cost_log).
    horizon = np.arange(1, 253)
    annual_net_log = (252.0 / horizon) * (edge_per_day * horizon + cost_log)
    breakeven_days = -cost_log / edge_per_day if edge_per_day > 0 else np.inf

    return {
        "n_trades": int(g.size),
        "gross_total_return": float(gross_curve[-1] - 1),
        "net_total_return": float(net_curve[-1] - 1),
        "gross_curve": gross_curve,
        "net_curve": net_curve,
        "cost_per_roundtrip": float(cost_per_roundtrip),
        "edge_log_per_holding_day": float(edge_per_day),
        "breakeven_holding_days": float(breakeven_days),
        "max_trades_per_year_before_edge_dies": (
            float(252.0 / breakeven_days) if np.isfinite(breakeven_days) and breakeven_days > 0 else float("inf")
        ),
        "annual_net_log_growth_by_holding_days": annual_net_log,
    }
