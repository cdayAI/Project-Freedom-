"""Agent 4 — SIZING LAB.

For a validated edge (an empirical per-trade net return sample), sweep the
bet fraction from 1% through full Kelly and beyond, Monte Carlo >= 20,000
bootstrap wealth paths per fraction, and report:

  - the frontier: median terminal wealth AND P(drawdown below X) vs fraction
  - the CONSTRAINED optimum: fraction maximizing median terminal wealth
    subject to P(losing 90% of the account) < 5%
  - the UNCONSTRAINED optimum, so the trade-off is visible in one chart

Log-wealth is the objective; the constrained/unconstrained gap is the price
of the ruin constraint, and the system always shows it.
"""

from __future__ import annotations

import numpy as np

from alpha_forge.config import MONTE_CARLO_MIN_PATHS, RUIN_DRAWDOWN_LEVEL


def kelly_fraction(returns: np.ndarray, grid: np.ndarray | None = None) -> float:
    """argmax_f E[log(1+f*r)] on the empirical distribution."""
    r = np.asarray(returns, dtype=float)
    grid = grid if grid is not None else np.linspace(0.01, 2.0, 200)
    best_f, best_g = 0.0, -np.inf
    for f in grid:
        w = 1.0 + f * r
        if np.any(w <= 0):
            continue
        g = float(np.mean(np.log(w)))
        if g > best_g:
            best_f, best_g = float(f), g
    return best_f


def sizing_frontier(
    trade_returns: np.ndarray,
    n_trades_per_path: int = 100,
    n_paths: int = MONTE_CARLO_MIN_PATHS,
    fractions: np.ndarray | None = None,
    ruin_loss: float = 0.90,
    ruin_prob_limit: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap Monte Carlo over the empirical trade-return sample."""
    r = np.asarray(trade_returns, dtype=float)
    if r.size < 30:
        raise ValueError("need >= 30 trades for a sizing frontier")
    if n_paths < MONTE_CARLO_MIN_PATHS:
        raise ValueError(f"minimum {MONTE_CARLO_MIN_PATHS} Monte Carlo paths")
    f_kelly = kelly_fraction(r)
    if fractions is None:
        # 1% absolute through 1.5x Kelly (and past it, as demanded)
        top = max(f_kelly * 1.5, 0.25)
        fractions = np.unique(np.round(np.linspace(0.01, top, 30), 4))

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, r.size, size=(n_paths, n_trades_per_path))
    sampled = r[draws]  # shared across fractions: common random numbers

    rows = []
    for f in fractions:
        growth = np.log1p(np.clip(f * sampled, -0.9999, None))
        log_wealth = np.cumsum(growth, axis=1)
        terminal = log_wealth[:, -1]
        running_max = np.maximum.accumulate(np.hstack([np.zeros((n_paths, 1)), log_wealth]), axis=1)
        dd = log_wealth - running_max[:, 1:]
        min_dd = dd.min(axis=1)  # most negative log drawdown per path
        p_ruin = float(np.mean(min_dd <= np.log(1 - ruin_loss)))
        p_half = float(np.mean(min_dd <= np.log(1 - RUIN_DRAWDOWN_LEVEL)))
        rows.append(
            {
                "fraction": float(f),
                "median_terminal_wealth": float(np.exp(np.median(terminal))),
                "p5_terminal_wealth": float(np.exp(np.percentile(terminal, 5))),
                "p95_terminal_wealth": float(np.exp(np.percentile(terminal, 95))),
                "p_ruin_90pct_loss": p_ruin,
                "p_drawdown_below_50pct": p_half,
                "mean_log_growth_per_trade": float(np.mean(growth)),
            }
        )

    feasible = [row for row in rows if row["p_ruin_90pct_loss"] < ruin_prob_limit]
    constrained = max(feasible, key=lambda x: x["median_terminal_wealth"]) if feasible else None
    unconstrained = max(rows, key=lambda x: x["median_terminal_wealth"])
    return {
        "kelly_fraction": f_kelly,
        "frontier": rows,
        "constrained_optimum": constrained,
        "unconstrained_optimum": unconstrained,
        "constraint": f"P(losing {ruin_loss:.0%}) < {ruin_prob_limit:.0%}",
        "n_paths": int(n_paths),
        "n_trades_per_path": int(n_trades_per_path),
        "note": None
        if feasible
        else "NO fraction satisfies the ruin constraint — this edge is unsizeable as measured",
    }
