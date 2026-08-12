import json

import pytest

from alpha_forge.ledger.ledger import ChainBrokenError, Ledger, LedgerError


@pytest.fixture
def ledger(tmp_path):
    return Ledger(path=tmp_path / "ledger.jsonl")


def test_chain_verifies_and_counts_trials(ledger):
    reg = ledger.preregister("h1", "u1", {"p": 1})
    ledger.record_trial(reg, {"cfg": 1}, sharpe=0.1)
    ledger.record_trial(reg, {"cfg": 2}, sharpe=None)
    ledger.record_result(reg, {"outcome": "meh"})
    assert ledger.verify_chain() == 4
    assert ledger.trial_count() == 2
    assert ledger.trial_sharpes() == [0.1]


def test_tampering_breaks_chain(ledger):
    reg = ledger.preregister("h1", "u1", {})
    ledger.record_trial(reg, {"cfg": 1}, sharpe=0.5)
    lines = ledger.path.read_text().strip().split("\n")
    entry = json.loads(lines[0])
    entry["payload"]["hypothesis"] = "rewritten after seeing results"
    lines[0] = json.dumps(entry, sort_keys=True)
    ledger.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ChainBrokenError):
        ledger.verify_chain()


def test_truncation_detected(ledger):
    reg = ledger.preregister("h1", "u1", {})
    ledger.record_trial(reg, {"cfg": 1}, sharpe=0.5)
    lines = ledger.path.read_text().strip().split("\n")
    ledger.path.write_text("\n".join(lines[1:]) + "\n")  # drop the preregistration
    with pytest.raises(ChainBrokenError):
        ledger.verify_chain()


def test_result_without_preregistration_is_inadmissible(ledger):
    with pytest.raises(LedgerError):
        ledger.record_result("deadbeef00000000", {"outcome": "great"})
    with pytest.raises(LedgerError):
        ledger.record_trial("deadbeef00000000", {"cfg": 1}, sharpe=1.0)


def test_trial_count_persists_across_instances(ledger):
    reg = ledger.preregister("h1", "u1", {})
    ledger.record_trial(reg, {"cfg": 1}, sharpe=0.5)
    reopened = Ledger(path=ledger.path)
    assert reopened.trial_count() == 1
    assert reopened.verify_chain() == 2


def test_trial_audit_separates_populations(ledger):
    """Raw trials, Sharpe-bearing trials, and registration families are three
    different numbers; the effective independent count is bounds-only."""
    r1 = ledger.preregister("h1", "u1", {})
    r2 = ledger.preregister("h2", "u1", {})
    ledger.record_trial(r1, {"cfg": 1}, sharpe=0.3)
    ledger.record_trial(r1, {"cfg": 2}, sharpe=None)  # e.g. confluence cell
    ledger.record_trial(r2, {"cfg": 1}, sharpe=-0.1)
    audit = ledger.trial_audit()
    assert audit["raw_cumulative_trials"] == 3
    assert audit["sharpe_bearing_trials"] == 2
    assert audit["distinct_registrations"] == 2
    assert audit["effective_independent_trials"] == "NOT YET ESTIMABLE"
    lo, hi = audit["effective_independent_bounds"]
    assert lo == 2 and hi == 3


def test_killed_strategies_are_distinct_identities(ledger):
    reg = ledger.preregister("h1", "u1", {})
    for _ in range(2):
        ledger.append("GATE_REPORT", {"strategy_id": "s1", "verdict": "KILL", "reg_id": reg})
    ledger.append("KILL", {"strategy_id": "s2"})
    assert ledger.killed_strategies() == {"s1", "s2"}
