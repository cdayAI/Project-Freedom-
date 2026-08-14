"""Strategy compiler: family preregistered, every config a ledgered trial,
full gate battery invoked, UNDECIDABLE below minimum support."""

import datetime as dt

import numpy as np
import polars as pl

from alpha_forge.ledger.ledger import Ledger
from alpha_forge.mechanism.compiler import EXIT_GRID, compile_and_gate
from alpha_forge.mechanism.episodes import Mechanism
from alpha_forge.research.eventbt import EventPanel


def _busdays(n):
    out, d = [], np.datetime64("2024-01-02")
    while len(out) < n:
        if np.is_busday(d):
            out.append(d)
        d += 1
    return np.array(out, dtype="datetime64[D]")


def _panel(n_days=320, symbols=("AAA", "BBB", "CCC")) -> EventPanel:
    D, S = n_days, len(symbols)
    px = np.full((D, S), 10.0)
    px += np.cumsum(np.full((D, S), 0.001), axis=0)  # gentle drift
    return EventPanel(
        dates=_busdays(n_days), symbols=list(symbols),
        open_=px.copy(), high=px * 1.01, low=px * 0.99, close=px.copy(),
        half_spread=np.full((D, S), 0.001),
        eligible=np.ones((D, S), dtype=bool),
        feat_pctl={},
        sell_fee_prop=np.zeros((D, S)),
        overnight_gaps=np.array([-0.02, 0.0, 0.02]),
    )


def _genome(n_days=320, symbols=("AAA", "BBB", "CCC"), every=25) -> pl.DataFrame:
    days = [d.astype(dt.date) for d in _busdays(n_days)]
    rows = []
    for si, sym in enumerate(symbols):
        for i, d in enumerate(days):
            # staggered per symbol so same-session non-completers exist
            # (the matched-null pool must never be empty by construction)
            rows.append({
                "symbol": sym, "date": d, "valid": True,
                "days_since_any_8k": 0.0 if (i + 7 * si) % every == 0 else 99.0,
                "dist_from_252d_high": 0.0,
            })
    return pl.DataFrame(rows)


MECH = Mechanism(
    (("CATALYST_SHOCK", (("k", 1),)), ("THRESHOLD_BREAK", (("b", 0.05),))),
    (3,),
)


def test_compile_and_gate_full_run(tmp_path):
    # CSCV PBO needs >= 32 monthly periods: ~36 months of business days
    led = Ledger(tmp_path / "ledger.jsonl")
    out = compile_and_gate(_panel(760), _genome(760), MECH, led,
                           data_vintage="2024-test")
    assert out["verdict"] in {"PASS", "FLAG", "KILL"}
    assert out["n_signals"] >= 30
    assert out["primary"]["n_trades"] > 0
    # one preregistration, one TRIAL per exit config, gate report ledgered
    kinds = [e["kind"] for e in led.entries()]
    assert kinds.count("PREREGISTRATION") == 1
    assert kinds.count("TRIAL") == len(EXIT_GRID)
    assert "GATE_REPORT" in kinds or "RESULT" in kinds
    assert led.verify_chain() > 0
    # flat-drift tape with matched nulls must not produce a significant p
    assert out["p_matched_null"] > 0.01


def test_compile_undecidable_below_min_support(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    out = compile_and_gate(_panel(60), _genome(60, every=999), MECH, led,
                           data_vintage="2024-test")
    assert out["verdict"] == "UNDECIDABLE"
    # the refusal is itself ledgered against the preregistration
    kinds = [e["kind"] for e in led.entries()]
    assert kinds.count("PREREGISTRATION") == 1 and "RESULT" in kinds
