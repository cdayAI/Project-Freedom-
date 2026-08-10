"""Gate 9: null baseline comparison.

A strategy graduates only if its net metric beats the 95th percentile of
>= 1,000 random-entry strategies matched on turnover, holding periods,
universe, and full cost model. Null *generation* is strategy-shaped and lives
next to the backtester (alpha_forge.research.backtest.matched_null_draws);
this module owns the statistical verdict so the threshold logic is in one
audited place.
"""

from __future__ import annotations

import numpy as np

from alpha_forge.config import NULL_BASELINE_MIN_DRAWS, NULL_BASELINE_PERCENTILE


def null_baseline_verdict(strategy_net_metric: float, null_net_metrics: np.ndarray) -> dict:
    nulls = np.asarray(null_net_metrics, dtype=float)
    if nulls.size < NULL_BASELINE_MIN_DRAWS:
        raise ValueError(
            f"null baseline needs >= {NULL_BASELINE_MIN_DRAWS} draws, got {nulls.size}"
        )
    threshold = float(np.percentile(nulls, NULL_BASELINE_PERCENTILE))
    exceed = float((nulls >= strategy_net_metric).mean())
    return {
        "passed": bool(strategy_net_metric > threshold),
        "strategy_net_metric": float(strategy_net_metric),
        "null_p95": threshold,
        "null_median": float(np.median(nulls)),
        "empirical_p_value": exceed,
        "n_draws": int(nulls.size),
    }
