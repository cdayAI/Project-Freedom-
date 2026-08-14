"""Discovery: bounded deterministic generation, every candidate ledgered,
anti-overlap collapse, battery statuses."""

import datetime as dt

import polars as pl

from alpha_forge.ledger.ledger import Ledger
from alpha_forge.mechanism.discovery import (
    collapse_overlaps,
    generate_candidates,
    mechanism_key,
    run_discovery,
)


def test_generation_is_bounded_deterministic_distinct():
    full = generate_candidates(n_stages=2, max_candidates=None)
    capped = generate_candidates(n_stages=2, max_candidates=10)
    assert [mechanism_key(m) for m in capped] == [
        mechanism_key(m) for m in full[:10]]  # capped = prefix, not sample
    assert len({mechanism_key(m) for m in full}) == len(full)
    for m in full:
        names = [n for n, _ in m.stages]
        assert len(set(names)) == len(names)  # distinct predicates


def test_collapse_overlaps_merges_near_completions():
    eps = pl.DataFrame({
        "symbol": ["A", "A", "A", "B"],
        "stage1_date": [dt.date(2026, 1, 5)] * 3 + [dt.date(2026, 1, 5)],
        "stage2_date": [dt.date(2026, 1, 6), dt.date(2026, 1, 10),
                        dt.date(2026, 3, 2), dt.date(2026, 1, 6)],
    })
    got = collapse_overlaps(eps, horizon=10)
    # A's Jan 6 and Jan 10 completions are one observation; March survives
    assert got.filter(pl.col("symbol") == "A").height == 2
    assert got.filter(pl.col("symbol") == "B").height == 1


def _toy_genome(n_days=40) -> pl.DataFrame:
    days = [dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(n_days)]
    rows = []
    for sym in ("AAA", "BBB", "CCC"):
        for i, d in enumerate(days):
            rows.append({
                "symbol": sym, "date": d, "valid": True,
                "insider_net_buy_90d_usd": 1.0 if i % 5 == 0 else -1.0,
                "short_ratio_z": -1.0,
                "days_since_any_8k": 0.0 if i % 7 == 0 else 99.0,
                "starts_5x_fwd": sym == "AAA" and i % 9 == 0,
                "regime_trend": "UP" if i < n_days // 2 else "DOWN",
            })
    return pl.DataFrame(rows)


def test_run_discovery_ledgers_every_candidate(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    res = run_discovery(_toy_genome(), led, n_stages=2, max_candidates=6)
    assert res.height == 6
    assert led.trial_count() == 6
    assert set(res["status"].to_list()) <= {"SURVIVOR", "KILLED", "UNDECIDABLE"}
    # a preregistration and a result bracket the trials
    kinds = [e["kind"] for e in led.entries()]
    assert kinds[0] == "PREREGISTRATION" and kinds[-1] == "RESULT"
    assert led.verify_chain() > 0

    # killed keys are never re-run
    led2 = Ledger(tmp_path / "ledger2.jsonl")
    skip = {res["mechanism"][0]}
    res2 = run_discovery(_toy_genome(), led2, n_stages=2, max_candidates=6,
                         killed_keys=skip)
    assert res2.height == 5
    assert skip.isdisjoint(set(res2["mechanism"].to_list()))
