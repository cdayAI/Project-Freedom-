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
    stress_sizing_inputs: dict | None = None,
    net_vs_gross_override: dict | None = None,
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

    # Gate 6 — net-of-cost display: report both, always. Event candidates
    # pass per-trade totals via the override (their daily series is net-only).
    if net_vs_gross_override is not None:
        report.checks["net_vs_gross"] = net_vs_gross_override
    else:
        gross_total = float(np.prod(1 + np.asarray(gross_returns)) - 1)
        net_total = float(np.prod(1 + np.asarray(net_returns)) - 1)
        report.checks["net_vs_gross"] = {
            "gross_total_return": gross_total, "net_total_return": net_total,
        }

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

    # Gates 10-11 — stress fills + account mechanics, then sizing. Only
    # meaningful for candidates with a real trade ledger (event strategies);
    # portfolio-style candidates pass stress inputs when they graduate to one.
    if stress_sizing_inputs is not None:
        k10, k11 = _gates_10_11(report, stress_sizing_inputs)
        kill.extend(k10)
        kill.extend(k11)

    if kill:
        report.verdict = "KILL"
    elif flag:
        report.verdict = "FLAG"
    else:
        report.verdict = "PASS"
    report.reasons = kill + flag

    ledger.append("GATE_REPORT", report.to_payload())
    return report


def _gates_10_11(report: GateReport, inputs: dict) -> tuple[list[str], list[str]]:
    """Gate 10: performance under stress fills + account-mechanics
    feasibility. Gate 11: a ruin-constrained position size must exist.

    inputs:
      trade_net_returns  simple returns per trade (OOS)
      trade_entry_days / trade_exit_days  business-day indices
      overnight_gaps     pooled panel gap distribution for tail injection
      null_p95_total_log gate-9 threshold the STRESSED strategy must still beat
      account_equity, per_trade_notional
    """
    from alpha_forge.research.sizing import sizing_frontier
    from alpha_forge.research.traps import Trade, run_account_mechanics, stress_overnight_gaps

    kill10: list[str] = []
    kill11: list[str] = []
    r = np.asarray(inputs["trade_net_returns"], dtype=float)

    # ---- gate 10a: worst-tail overnight gaps injected into the trade sample
    stressed = stress_overnight_gaps(
        r, np.asarray(inputs["overnight_gaps"], dtype=float), seed=7
    )
    stressed_total_log = float(np.sum(np.log1p(np.maximum(stressed, -0.9999))))
    report.checks["stress_fills"] = {
        "clean_total_log": float(np.sum(np.log1p(np.maximum(r, -0.9999)))),
        "stressed_total_log": stressed_total_log,
        "null_p95_total_log": inputs["null_p95_total_log"],
    }
    if stressed_total_log <= inputs["null_p95_total_log"]:
        kill10.append(
            f"gate10: stressed net log {stressed_total_log:.4f} no longer beats "
            f"the null p95 {inputs['null_p95_total_log']:.4f} — the edge is a "
            "clean-fill artifact"
        )

    # ---- gate 10b: account mechanics at the target equity
    trades = [
        Trade(entry_day=int(e), exit_day=int(x), notional=float(inputs["per_trade_notional"]))
        for e, x in zip(inputs["trade_entry_days"], inputs["trade_exit_days"])
    ]
    cash = run_account_mechanics(trades, "cash", inputs["account_equity"])
    margin = run_account_mechanics(trades, "margin", inputs["account_equity"])
    report.checks["account_mechanics"] = {
        "cash_feasible": cash.feasible,
        "cash_violations": cash.violations[:3],
        "margin_feasible": margin.feasible,
        "margin_violations": margin.violations[:3],
    }
    if not cash.feasible and not margin.feasible:
        kill10.append(
            "gate10: trade plan infeasible in BOTH cash (settlement/GFV) and "
            "margin (PDT) accounts at this equity — killed regardless of backtest"
        )

    # ---- gate 11: ruin-constrained sizing must exist
    if r.size >= 30:
        frontier = sizing_frontier(r, n_trades_per_path=max(50, r.size), seed=11)
        c = frontier["constrained_optimum"]
        report.checks["sizing"] = {
            "kelly_fraction": frontier["kelly_fraction"],
            "constrained_optimum": c,
            "unconstrained_optimum": frontier["unconstrained_optimum"],
            "constraint": frontier["constraint"],
        }
        if c is None:
            kill11.append(
                "gate11: no bet fraction satisfies P(losing 90%) < 5% — the "
                "edge is unsizeable as measured"
            )
        elif c["median_terminal_wealth"] <= 1.0:
            kill11.append(
                f"gate11: the best ruin-safe fraction ({c['fraction']:.2f}) still "
                f"has median terminal wealth {c['median_terminal_wealth']:.3f} <= 1 "
                "— no sizing grows this edge inside the ruin constraint"
            )
    else:
        kill11.append(f"gate11: {r.size} OOS trades < 30 — cannot size")

    return kill10, kill11
