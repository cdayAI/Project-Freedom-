"""Alpha-v0 harness: run-twice identity, isolated ledger, research read-only."""

import json

import numpy as np
import polars as pl
import pytest

import alpha_forge.alpha_v0 as av0
from alpha_forge.ledger import Ledger


def _synthetic_store(tmp_path, n_symbols=30, n_days=1300, seed=5):
    """A small but gate-runnable panel: ~38 months of daily bars, prices a
    positive-drift walk so every month has a rankable cross-section."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2020-01-01")
    days = []
    d = start
    while len(days) < n_days:
        if np.is_busday(d.astype("datetime64[D]")):
            days.append(d)
        d = d + 1
    rows = []
    for s in range(n_symbols):
        px = 20.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n_days)))
        for i, day in enumerate(days):
            o = px[i] * (1 + rng.normal(0, 0.002))
            rows.append(
                {
                    "symbol": f"T{s:03d}",
                    "date": day.astype("datetime64[D]").item(),
                    "open": o,
                    "high": max(o, px[i]) * 1.01,
                    "low": min(o, px[i]) * 0.99,
                    "close": px[i],
                    "volume": float(rng.integers(200_000, 2_000_000)),
                }
            )
    store = tmp_path / "store"
    store.mkdir()
    pl.DataFrame(rows).write_parquet(store / "equities_daily.parquet")
    return store


def _research_ledger_with_kill(tmp_path):
    """The research history Alpha-v0 replays against: a preregistration,
    trials (mixed Sharpe-bearing and not), and the original KILL."""
    led = Ledger(path=tmp_path / "research.jsonl")
    reg = led.preregister("xsmom demo", "sample", {"p": 1})
    led.record_trial(reg, {"cfg": 1}, sharpe=0.05)
    led.record_trial(reg, {"cfg": 2}, sharpe=-0.10)
    led.record_trial(reg, {"cfg": 3}, sharpe=None)
    led.append(
        "GATE_REPORT",
        {
            "strategy_id": "xsmom_demo_v1",
            "verdict": "KILL",
            "reg_id": reg,
            "checks": {"data_vintage": "2026-01-01"},
        },
    )
    return led


@pytest.fixture
def sandbox(tmp_path):
    store = _synthetic_store(tmp_path)
    research = _research_ledger_with_kill(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"dataset_id": "equities_daily_yahoo", "vendor": "test"}) + "\n",
        encoding="utf-8",
    )
    kw = dict(
        store_dir=store,
        research_ledger_path=research.path,
        acceptance_ledger_path=tmp_path / "acceptance.jsonl",
        reports_dir=tmp_path / "reports",
        manifest_path=manifest,
        n_permutations=300,  # fast for tests; the real run uses 20k (config)
        n_null_draws=1000,   # the gate's own floor is non-negotiable
        log=lambda *_: None,
    )
    return kw, research


def test_run_twice_identical_evidence_and_no_new_entries(sandbox):
    kw, research = sandbox
    r1 = av0.run_once(**kw)
    n_after_first = sum(1 for _ in Ledger(path=kw["acceptance_ledger_path"]).entries())
    r2 = av0.run_once(**kw)
    n_after_second = sum(1 for _ in Ledger(path=kw["acceptance_ledger_path"]).entries())

    assert r1["first_run"] and not r2["first_run"]
    assert r1["evidence_hash"] == r2["evidence_hash"]
    assert r2["evidence_reproduced"]
    assert n_after_second == n_after_first  # zero appends on the rerun
    # a KILL reproduced is a PASS of the harness, not a failure
    assert r1["verdict"] == r2["verdict"]


def test_research_ledger_is_read_only(sandbox):
    kw, research = sandbox
    before = research.path.read_bytes()
    av0.run_once(**kw)
    av0.run_once(**kw)
    assert research.path.read_bytes() == before  # byte-identical, not just count


def test_acceptance_chain_verifies_and_is_isolated(sandbox):
    kw, _ = sandbox
    av0.run_once(**kw)
    acc = Ledger(path=kw["acceptance_ledger_path"])
    assert acc.verify_chain() >= 4  # DATA_PULL, PREREG, trials..., GATE, RESULT
    kinds = {e["kind"] for e in acc.entries()}
    assert {"DATA_PULL", "PREREGISTRATION", "TRIAL", "GATE_REPORT", "RESULT"} <= kinds
    for e in acc.entries():
        if e["kind"] in ("DATA_PULL", "PREREGISTRATION", "GATE_REPORT", "RESULT"):
            payload = e["payload"]
            mode = payload.get("mode") or payload.get("checks", {}).get("mode")
            assert mode == "ACCEPTANCE_REPLAY"


def test_report_is_deterministic_and_watermarked(sandbox):
    kw, _ = sandbox
    r1 = av0.run_once(**kw)
    first = open(r1["report_path"], encoding="utf-8").read()
    r2 = av0.run_once(**kw)
    second = open(r2["report_path"], encoding="utf-8").read()
    assert first == second  # byte-identical rerun
    assert "PIPELINE_PROOF_ONLY" in first
    assert "NOT YET ESTIMABLE" in first
    assert r1["evidence_hash"] in first


def test_refuses_to_run_without_an_original_kill(sandbox, tmp_path):
    kw, _ = sandbox
    empty = Ledger(path=tmp_path / "empty_research.jsonl")
    empty.preregister("h", "u", {})  # a ledger with no xsmom gate report
    kw = {**kw, "research_ledger_path": empty.path}
    with pytest.raises(RuntimeError, match="replays a KILLED candidate"):
        av0.run_once(**kw)
