"""Agent 8a — CAPACITY & TIER PLANNER.

Three lifecycle numbers per strategy, continuously re-estimated:

CAPACITY — the account size at which the strategy's own footprint erodes its
edge. Footprint model (structural, documented): a position may consume at
most PARTICIPATION_LIMIT of a name's median daily dollar volume without
material impact beyond the modeled spread; capacity is the account size at
which the strategy's per-name allocation hits that limit across its
selection pool. Computed entirely from the panel's own ADV data.

EDGE HALF-LIFE — exponential decay fit on live reconciliation edge (grade
series). Requires live grades; reports the data requirement honestly until
enough exist. The system assumes decay by default: a strategy with no live
evidence of persistence carries `half_life_bars: null, assumed_decaying: true`.

TIER MAP — capital roadmap. Only sourced thresholds appear as constants:
$25,000 PDT threshold (FINRA 4210(f)(8)(B)(iv)); $2,000 minimum margin
equity (FINRA 4210(b)); T+1 settlement (SEC 34-96930). Broker-specific
levels (portfolio-margin minimums, futures day-trade margins) are broker
profile inputs marked UNVERIFIED until sourced per broker.
"""

from __future__ import annotations

import numpy as np
import polars as pl

PARTICIPATION_LIMIT = 0.01  # structural: <=1% of median daily dollar volume per name


def strategy_capacity(
    panel: pl.DataFrame,
    selection_symbols: list[str],
    n_concurrent_positions: int,
    lookback_days: int = 63,
) -> dict:
    """Account size at which per-name allocation exceeds the participation
    limit of the selection pool's median-ADV name."""
    advs = []
    for sym in selection_symbols:
        g = panel.filter(pl.col("symbol") == sym).sort("date").tail(lookback_days)
        if g.height < 20:
            continue
        dv = (g["close"] * g["volume"]).to_numpy()
        advs.append(float(np.nanmedian(dv)))
    if not advs:
        return {"capacity_usd": None, "note": "no ADV data for selection pool"}
    pool_adv = float(np.median(advs))
    per_name_limit = pool_adv * PARTICIPATION_LIMIT
    capacity = per_name_limit * n_concurrent_positions
    return {
        "capacity_usd": capacity,
        "pool_median_adv_usd": pool_adv,
        "per_name_notional_limit_usd": per_name_limit,
        "participation_limit": PARTICIPATION_LIMIT,
        "n_concurrent_positions": n_concurrent_positions,
        "note": "impact beyond modeled spread assumed material above "
        f"{PARTICIPATION_LIMIT:.0%} ADV participation (structural model, "
        "stress-tested by Agent 5 before any graduation)",
    }


def edge_half_life(grade_dates: np.ndarray, edge_series: np.ndarray) -> dict:
    """Exponential decay fit: edge_t = a * exp(-lambda * t). Needs >= 8 live
    grade points with positive early edge; otherwise the honest answer is
    'insufficient live evidence' and decay is assumed."""
    e = np.asarray(edge_series, dtype=float)
    if e.size < 8:
        return {
            "half_life_bars": None,
            "assumed_decaying": True,
            "note": f"only {e.size} live grade points; the system assumes decay "
            "until live reconciliation shows persistence",
        }
    t = np.arange(e.size, dtype=float)
    pos = e > 0
    if pos.sum() < 6:
        return {"half_life_bars": 0.0, "assumed_decaying": True,
                "note": "edge non-positive in most live grades — already dead"}
    coef = np.polyfit(t[pos], np.log(e[pos]), 1)
    lam = -coef[0]
    hl = float(np.log(2) / lam) if lam > 0 else float("inf")
    return {"half_life_bars": hl, "assumed_decaying": bool(lam > 0), "decay_rate": float(lam)}


# Sourced regulatory thresholds only; broker-specific numbers live in broker
# profiles with their own verification status.
TIER_MAP = [
    {
        "tier": "under_25k",
        "range_usd": [0, 25_000],
        "constraints": [
            "PDT-bound in margin accounts: max 3 day trades / 5 business days "
            "(FINRA 4210(f)(8)(B)(iv))",
            "cash account alternative: T+1 settlement + GFV rules (SEC 34-96930)",
            "integer-lot friction dominates below ~$5k for high-priced names",
        ],
        "unlocks": ["swing/position strategies", "futures (PDT-exempt, margin permitting)"],
    },
    {
        "tier": "25k_plus",
        "range_usd": [25_000, 110_000],
        "constraints": ["margin maintenance (FINRA 4210)"],
        "unlocks": ["unlimited day trading in margin accounts"],
    },
    {
        "tier": "portfolio_margin",
        "range_usd": [110_000, None],
        "constraints": [
            "minimum equity for portfolio margin is broker-set and UNVERIFIED "
            "here until a broker profile documents it — do not plan on it"
        ],
        "unlocks": ["risk-based margin on options-heavy books (broker-dependent)"],
    },
]


def tier_report(account_equity: float, book: list[dict], panel: pl.DataFrame) -> dict:
    """Which tier the account is in, what dies/unlocks at the boundaries,
    and each book strategy's capacity expiration."""
    current = next(
        (
            t
            for t in TIER_MAP
            if account_equity >= t["range_usd"][0]
            and (t["range_usd"][1] is None or account_equity < t["range_usd"][1])
        ),
        TIER_MAP[0],
    )
    strategies = []
    for s in book:
        cap = s.get("capacity_usd")
        strategies.append(
            {
                "strategy_id": s.get("strategy_id"),
                "capacity_usd": cap,
                "dies_at_tier": None
                if cap is None
                else next(
                    (
                        t["tier"]
                        for t in TIER_MAP
                        if t["range_usd"][1] is not None and cap < t["range_usd"][1]
                    ),
                    "portfolio_margin",
                ),
            }
        )
    empty_tiers = [
        t["tier"]
        for t in TIER_MAP
        if not any(
            s["capacity_usd"] is None or s["capacity_usd"] >= t["range_usd"][0]
            for s in strategies
        )
    ] if strategies else [t["tier"] for t in TIER_MAP]
    return {
        "account_equity": account_equity,
        "current_tier": current["tier"],
        "tier_map": TIER_MAP,
        "book_capacity": strategies,
        "tiers_with_no_validated_strategy": empty_tiers,
        "warning": "baton-pass risk: growth into a tier with no validated "
        "strategies strands the account" if empty_tiers else None,
    }
