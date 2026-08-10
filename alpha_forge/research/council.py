"""Agent 8b — STRATEGY COUNCIL. Keep/kill is mechanical, never argued.

Kill criteria (any one suffices):
  - PBO > 0.5 on re-estimation
  - DSR probability < 0.95 given the ledger's CURRENT cumulative trial count
    (a strategy that passed at N=100 trials can die at N=1000 — deflation
    is retroactive by design)
  - live calibration outside the backtest confidence band for 30 trading days

Survivors get projected win rates with confidence intervals derived from
LIVE reconciliation data only; backtest numbers are never projected forward.
"""

from __future__ import annotations

import numpy as np

from alpha_forge.config import DSR_MIN_PROBABILITY, PBO_KILL
from alpha_forge.gates.dsr import deflated_sharpe_ratio
from alpha_forge.ledger import Ledger

CALIBRATION_DRIFT_DAYS = 30


def review_strategy(
    strategy_id: str,
    net_returns: np.ndarray,
    current_pbo: float,
    ledger: Ledger,
    live_calibration_drift_days: int,
    live_win_flags: np.ndarray | None,
) -> dict:
    """One council pass over one book strategy. Ledgers KILL if triggered."""
    reasons = []
    trial_srs = ledger.trial_sharpes()
    var_trial = float(np.var(trial_srs, ddof=1)) if len(trial_srs) >= 2 else 0.0
    dsr = deflated_sharpe_ratio(net_returns, max(ledger.trial_count(), 1), var_trial)
    if dsr["dsr_probability"] < DSR_MIN_PROBABILITY:
        reasons.append(
            f"DSR {dsr['dsr_probability']:.4f} < {DSR_MIN_PROBABILITY} at the "
            f"ledger's current N={ledger.trial_count()}"
        )
    if current_pbo > PBO_KILL:
        reasons.append(f"PBO {current_pbo:.3f} > {PBO_KILL}")
    if live_calibration_drift_days >= CALIBRATION_DRIFT_DAYS:
        reasons.append(
            f"live calibration outside backtest band for "
            f"{live_calibration_drift_days} trading days (limit {CALIBRATION_DRIFT_DAYS})"
        )

    verdict = {"strategy_id": strategy_id, "kill": bool(reasons), "reasons": reasons}
    if reasons:
        ledger.append("KILL", verdict)
        return verdict

    # projected win rate: live data only, Wilson interval
    if live_win_flags is not None and live_win_flags.size >= 10:
        wins = int(live_win_flags.sum())
        n = int(live_win_flags.size)
        p = wins / n
        z = 1.959963984540054  # 95%
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        verdict["projected_win_rate"] = {
            "point": p,
            "ci95": [max(0.0, center - half), min(1.0, center + half)],
            "n_live_trades": n,
            "source": "live reconciliation only",
        }
    else:
        verdict["projected_win_rate"] = {
            "point": None,
            "note": "insufficient live trades; no projection is made from backtest data",
        }
    return verdict
