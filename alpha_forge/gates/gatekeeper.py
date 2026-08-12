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
    STATUS_CEILING,
    SURVIVORSHIP_FREE_DATA_INSTALLED,
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
    stats_ledger: Ledger | None = None,
) -> GateReport:
    """Evaluate gates 1-9. Writes the full report to the ledger.

    net_returns / gross_returns: per-period strategy returns, same length.
    config_returns_matrix: (T, N) net returns of every config tried (for PBO).
    null_result_inputs: (strategy_net_metric, null_net_metrics) for gate 9.
    extra_checks: caller metadata (e.g. data vintage) that must land in the
    LEDGERED copy of the report, not just the returned object.
    stats_ledger: where the DSR trial statistics come from, when the report
    is written elsewhere (the Alpha-v0 acceptance replay writes to an
    isolated acceptance ledger but deflates by the RESEARCH ledger's audited
    trial history — replay trials must not shrink the deflation).
    """
    stats = stats_ledger if stats_ledger is not None else ledger
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

    # Gate 3 — Deflated Sharpe with audited trial semantics. The PRIMARY DSR
    # uses the Sharpe-bearing trial population — count and variance from the
    # SAME population, per Bailey & Lopez de Prado (2014). The CONSERVATIVE
    # sensitivity deflates by the raw cumulative ledger count (which includes
    # Sharpe-less trials, e.g. confluence cells). The gate BINDS on the
    # conservative number; the effective independent count is reported as
    # bounds only (per-trial return series are not retained yet).
    audit = stats.trial_audit()
    trial_srs = stats.trial_sharpes()
    if len(trial_srs) >= 2:
        var_trial = float(np.var(trial_srs, ddof=1))
    else:
        # A single recorded trial gives no cross-trial variance; deflation
        # degenerates to PSR vs 0. Flag it — this only happens on day one.
        var_trial = 0.0
        flag.append("gate3: <2 ledgered trial Sharpes; deflation benchmark degenerate")
    n_raw = max(audit["raw_cumulative_trials"], 1)
    n_sharpe = max(audit["sharpe_bearing_trials"], 1)
    dsr_primary = deflated_sharpe_ratio(net_returns, n_trials=n_sharpe, var_trial_sr=var_trial)
    dsr = (
        dsr_primary
        if n_raw == n_sharpe
        else deflated_sharpe_ratio(net_returns, n_trials=n_raw, var_trial_sr=var_trial)
    )
    report.checks["dsr"] = dsr  # the binding (conservative) computation
    report.checks["dsr_audit"] = {
        **audit,
        "dsr_probability_primary_sharpe_bearing_n": dsr_primary["dsr_probability"],
        "dsr_probability_conservative_raw_n": dsr["dsr_probability"],
        "method": "preregistered: primary N = Sharpe-bearing trials (count and "
        "variance from the same population); the gate binds on the conservative "
        "raw-count sensitivity; effective independent N reported as bounds",
    }
    if dsr["dsr_probability"] < DSR_MIN_PROBABILITY:
        kill.append(
            f"gate3: DSR probability {dsr['dsr_probability']:.4f} < {DSR_MIN_PROBABILITY} "
            f"at conservative N={n_raw} raw ledgered trials "
            f"(primary N={n_sharpe} Sharpe-bearing: {dsr_primary['dsr_probability']:.4f})"
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

    # Status ceiling: on survivorship-biased data no verdict — including PASS —
    # can confer VALIDATED/GRADUATED/EXECUTABLE status. Stamped on the report
    # so every downstream display carries it; enforced by the ledger itself,
    # which refuses GRADUATION entries while the ceiling holds.
    if not SURVIVORSHIP_FREE_DATA_INSTALLED:
        report.checks["status_ceiling"] = STATUS_CEILING

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
    from alpha_forge.research.sizing2 import sizing_frontier_v2
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

    # ---- gate 11: drawdown-constrained sizing must exist under the WORST of
    # the v2 path generators (stationary blocks / Bayesian bootstrap / regime
    # chain when labels exist); the iid number rides along as the baseline
    # it is — measured, never voting. The binding constraint is
    # P(maxDD >= 50% over the 20y horizon) <= epsilon; the strategy's own
    # trade frequency maps the horizon onto path length.
    if r.size >= 30:
        entry_days = np.asarray(inputs["trade_entry_days"], dtype=float)
        exit_days = np.asarray(inputs["trade_exit_days"], dtype=float)
        span_years = max(
            (float(exit_days.max()) - float(entry_days.min()) + 1.0) / 252.0,
            1.0 / 252.0,
        )
        frontier = sizing_frontier_v2(
            r,
            regime_labels=inputs.get("trade_regime_labels"),
            trades_per_year=r.size / span_years,
            seed=11,
        )
        c = frontier["constrained_optimum"]
        report.checks["sizing"] = {
            "kelly": frontier["kelly"],
            "generators_used": frontier["generators_used"],
            "constrained_optimum": c,
            "unconstrained_optimum": frontier["unconstrained_optimum"],
            "constraint": frontier["constraint"],
            "horizon": frontier["horizon"],
        }
        if c is None:
            kill11.append(
                f"gate11: no bet fraction satisfies {frontier['constraint']} — unsizeable"
            )
        elif c["median_terminal_wealth_bayes"] <= 1.0:
            kill11.append(
                f"gate11: the best ruin-safe fraction ({c['fraction']:.2f}) still "
                f"has median terminal wealth {c['median_terminal_wealth_bayes']:.3f} "
                "<= 1 under parameter uncertainty — no sizing grows this edge "
                "inside the ruin constraint"
            )
    else:
        kill11.append(f"gate11: {r.size} OOS trades < 30 — cannot size")

    return kill10, kill11
