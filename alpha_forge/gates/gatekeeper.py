"""Runs every Section-4 gate against a candidate and renders one verdict.

The gate thresholds are non-negotiable constants in config.py. A candidate
that has not passed ALL gates never reaches the Strategy Council. Gross
results are never reported without net beside them (gate 6): the report
structure makes the pair mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alpha_forge.config import (
    DSR_MIN_PROBABILITY,
    MIN_INDEPENDENT_TRADES,
    PBO_FLAG,
    PBO_KILL,
)
from alpha_forge.gates.dsr import deflated_sharpe_ratio
from alpha_forge.gates.nullbaseline import null_baseline_verdict
from alpha_forge.gates.pbo import cscv_pbo
from alpha_forge.ledger import Ledger


@dataclass
class GateReport:
    strategy_id: str
    reg_id: str
    verdict: str = "PENDING"  # PASS | FLAG | KILL
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "reg_id": self.reg_id,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "checks": self.checks,
        }


def run_gates(
    strategy_id: str,
    reg_id: str,
    ledger: Ledger,
    net_returns: np.ndarray,
    gross_returns: np.ndarray,
    config_returns_matrix: np.ndarray,
    n_trade_events: int,
    permutation_result: dict,
    null_result_inputs: tuple[float, np.ndarray],
    survivorship_free: bool,
    walkforward_summary: dict,
    extra_checks: dict | None = None,
) -> GateReport:
    """Evaluate gates 1-9. Writes the full report to the ledger.

    net_returns / gross_returns: per-period strategy returns, same length.
    config_returns_matrix: (T, N) net returns of every config tried (for PBO).
    null_result_inputs: (strategy_net_metric, null_net_metrics) for gate 9.
    extra_checks: caller metadata (e.g. data vintage) that must land in the
    LEDGERED copy of the report, not just the returned object.
    """
    report = GateReport(strategy_id=strategy_id, reg_id=reg_id)
    if extra_checks:
        report.checks.update(extra_checks)
    kill, flag = [], []

    # Gate 1 — preregistration exists and predates this evaluation.
    reg = ledger._find_registration(reg_id)
    if reg is None:
        kill.append("gate1: no preregistration found — inadmissible")
        report.checks["preregistration"] = {"found": False}
    else:
        report.checks["preregistration"] = {"found": True, "ts_utc": reg["ts_utc"]}

    # Gate 2 — permutation test (already corrected across the day's features
    # by the caller; the corrected rejection flag rides in the dict).
    report.checks["permutation"] = permutation_result
    if not permutation_result.get("rejected_after_correction", False):
        kill.append(
            f"gate2: permutation p={permutation_result.get('p_value'):.4g} "
            "not significant after multiple-testing correction"
        )

    # Gate 3 — Deflated Sharpe with TRUE ledger trial count.
    n_trials = ledger.trial_count()
    trial_srs = ledger.trial_sharpes()
    if len(trial_srs) >= 2:
        var_trial = float(np.var(trial_srs, ddof=1))
    else:
        # A single recorded trial gives no cross-trial variance; deflation
        # degenerates to PSR vs 0. Flag it — this only happens on day one.
        var_trial = 0.0
        flag.append("gate3: <2 ledgered trial Sharpes; deflation benchmark degenerate")
    dsr = deflated_sharpe_ratio(net_returns, n_trials=max(n_trials, 1), var_trial_sr=var_trial)
    report.checks["dsr"] = dsr
    if dsr["dsr_probability"] < DSR_MIN_PROBABILITY:
        kill.append(
            f"gate3: DSR probability {dsr['dsr_probability']:.4f} < {DSR_MIN_PROBABILITY} "
            f"at N={n_trials} ledgered trials"
        )

    # Gate 4 — PBO via CSCV.
    pbo = cscv_pbo(config_returns_matrix)
    report.checks["pbo"] = pbo
    if pbo["pbo"] > PBO_KILL:
        kill.append(f"gate4: PBO {pbo['pbo']:.3f} > {PBO_KILL}")
    elif pbo["pbo"] > PBO_FLAG:
        flag.append(f"gate4: PBO {pbo['pbo']:.3f} > {PBO_FLAG} (flag)")

    # Gate 5 — walk-forward already executed by caller with purged splits.
    report.checks["walkforward"] = walkforward_summary
    if not walkforward_summary.get("all_folds_evaluated", False):
        kill.append("gate5: walk-forward incomplete")

    # Gate 6 — net-of-cost display: report both, always.
    gross_total = float(np.prod(1 + np.asarray(gross_returns)) - 1)
    net_total = float(np.prod(1 + np.asarray(net_returns)) - 1)
    report.checks["net_vs_gross"] = {"gross_total_return": gross_total, "net_total_return": net_total}

    # Gate 7 — survivorship disclosure.
    report.checks["survivorship_free_universe"] = survivorship_free
    if not survivorship_free:
        flag.append(
            "gate7: universe includes only surviving listings — results are "
            "biased and marked as such in every display"
        )

    # Gate 8 — minimum sample.
    report.checks["n_trade_events"] = int(n_trade_events)
    if n_trade_events < MIN_INDEPENDENT_TRADES:
        kill.append(f"gate8: {n_trade_events} trade events < {MIN_INDEPENDENT_TRADES} minimum")

    # Gate 9 — null baseline.
    strat_metric, null_metrics = null_result_inputs
    nb = null_baseline_verdict(strat_metric, null_metrics)
    report.checks["null_baseline"] = nb
    if not nb["passed"]:
        kill.append(
            f"gate9: net metric {nb['strategy_net_metric']:.4f} does not beat "
            f"null 95th percentile {nb['null_p95']:.4f}"
        )

    if kill:
        report.verdict = "KILL"
    elif flag:
        report.verdict = "FLAG"
    else:
        report.verdict = "PASS"
    report.reasons = kill + flag

    ledger.append("GATE_REPORT", report.to_payload())
    return report
