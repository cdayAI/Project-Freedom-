import numpy as np
import pytest

from alpha_forge.gates.gatekeeper import run_gates
from alpha_forge.ledger import Ledger


def _base_inputs(ledger, rng, strong=True):
    reg = ledger.preregister("h", "u", {})
    ledger.record_trial(reg, {"a": 1}, 0.1)
    ledger.record_trial(reg, {"a": 2}, 0.12)
    mu = 0.004 if strong else 0.0
    net = rng.normal(mu, 0.01, 700)
    cfgm = rng.normal(0, 0.01, (700, 6))
    cfgm[:, 2] += mu  # one config carries the edge
    return dict(
        strategy_id="s",
        reg_id=reg,
        ledger=ledger,
        net_returns=net,
        gross_returns=net + 0.0005,
        config_returns_matrix=cfgm,
        n_trade_events=150,
        permutation_result={"p_value": 0.0001, "rejected_after_correction": True},
        null_result_inputs=(float(np.sum(np.log1p(net))), rng.normal(0, 0.5, 2000)),
        survivorship_free=True,
        walkforward_summary={"all_folds_evaluated": True},
    )


def _stress_inputs(rng, trade_mu=0.05, n=120):
    trades = rng.normal(trade_mu, 0.15, n)
    return {
        "trade_net_returns": trades,
        "trade_entry_days": list(range(0, n * 30, 30)),
        "trade_exit_days": [d + 20 for d in range(0, n * 30, 30)],
        "overnight_gaps": rng.normal(-0.001, 0.03, 5000),
        "null_p95_total_log": 0.5,
        "account_equity": 2000.0,
        "per_trade_notional": 200.0,
    }


def test_gates_10_11_pass_for_robust_edge(tmp_path):
    rng = np.random.default_rng(0)
    ledger = Ledger(path=tmp_path / "l.jsonl")
    report = run_gates(**_base_inputs(ledger, rng),
                       stress_sizing_inputs=_stress_inputs(rng))
    assert "stress_fills" in report.checks
    assert "account_mechanics" in report.checks
    assert "sizing" in report.checks
    assert report.checks["sizing"]["constrained_optimum"] is not None
    assert not any("gate10" in r or "gate11" in r for r in report.reasons)


def test_gate10_kills_clean_fill_artifact(tmp_path):
    rng = np.random.default_rng(1)
    ledger = Ledger(path=tmp_path / "l.jsonl")
    stress = _stress_inputs(rng, trade_mu=0.006, n=100)
    # edge so thin the stressed record falls below the null threshold
    stress["null_p95_total_log"] = 0.55
    report = run_gates(**_base_inputs(ledger, rng), stress_sizing_inputs=stress)
    assert report.verdict == "KILL"
    assert any("gate10" in r and "clean-fill artifact" in r for r in report.reasons)


def test_gate10_kills_account_infeasible_plan(tmp_path):
    rng = np.random.default_rng(2)
    ledger = Ledger(path=tmp_path / "l.jsonl")
    stress = _stress_inputs(rng)
    # 10 same-day round trips in one 5-day window and buys outrunning cash:
    stress["trade_entry_days"] = [10] * 10
    stress["trade_exit_days"] = [10] * 10
    stress["per_trade_notional"] = 5000.0  # > equity: free-riding in cash too
    report = run_gates(**_base_inputs(ledger, rng), stress_sizing_inputs=stress)
    assert report.verdict == "KILL"
    assert any("infeasible in BOTH" in r for r in report.reasons)


def test_gate11_kills_unsizeable_edge(tmp_path):
    rng = np.random.default_rng(3)
    ledger = Ledger(path=tmp_path / "l.jsonl")
    stress = _stress_inputs(rng)
    # decisively negative-EV, violent trades: every ruin-safe fraction loses
    stress["trade_net_returns"] = np.where(rng.random(200) < 0.4, -0.95, 0.4)
    stress["null_p95_total_log"] = -999.0  # keep gate 10a quiet
    report = run_gates(**_base_inputs(ledger, rng), stress_sizing_inputs=stress)
    killed = [r for r in report.reasons if "gate11" in r]
    assert killed and ("unsizeable" in killed[0] or "no sizing grows" in killed[0])


def test_too_few_trades_cannot_size(tmp_path):
    rng = np.random.default_rng(4)
    ledger = Ledger(path=tmp_path / "l.jsonl")
    stress = _stress_inputs(rng, n=10)
    stress["null_p95_total_log"] = -999.0
    report = run_gates(**_base_inputs(ledger, rng), stress_sizing_inputs=stress)
    assert any("cannot size" in r for r in report.reasons)
