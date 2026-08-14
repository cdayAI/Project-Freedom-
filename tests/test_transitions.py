"""Transition statistics: hazard vs matched base, and a planted field
effect that survives BH while noise does not."""

import datetime as dt

import numpy as np
import polars as pl

from alpha_forge.ledger.ledger import Ledger
from alpha_forge.mechanism.episodes import Mechanism
from alpha_forge.mechanism.transitions import transition_stats


def _genome(n_days=120, n_syms=12, seed=3) -> pl.DataFrame:
    """CATALYST_SHOCK fires day 10 of each 20-day cycle for every symbol;
    THRESHOLD_BREAK follows within 3 sessions ONLY for symbols whose
    volume_z is high (planted transition condition). vol_20d_ann is pure
    noise. Half the symbols are high-volume advancers."""
    rng = np.random.default_rng(seed)
    days = [dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(n_days)]
    rows = []
    for s in range(n_syms):
        sym = f"S{s:02d}"
        advancer = s < n_syms // 2
        phase = (10 + 3 * s) % 20  # staggered: controls exist every session
        for i, d in enumerate(days):
            shock = i % 20 == phase
            brk = advancer and (i % 20 in ((phase + 1) % 20, (phase + 2) % 20))
            rows.append({
                "symbol": sym, "date": d, "valid": True,
                "days_since_any_8k": 0.0 if shock else 99.0,
                "dist_from_252d_high": 0.0 if brk else -0.5,
                "volume_z_20v126": (1.5 if advancer else -1.5)
                + float(rng.normal(0, 0.05)),
                "short_ratio_z": 0.0, "vol_20d_ann": float(rng.normal(1, 0.1)),
                "dollar_vol_med_20d": 1e6,
            })
    return pl.DataFrame(rows)


MECH = Mechanism(
    (("CATALYST_SHOCK", (("k", 1),)), ("THRESHOLD_BREAK", (("b", 0.05),))),
    (3,),
)


def test_hazard_lift_and_planted_field_effect(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    out = transition_stats(_genome(), MECH, ledger=led)
    assert len(out) == 1
    t = out[0]
    # half the prefix-holders advance; matched controls (no shock) barely do
    assert t["n"] >= 30
    assert 0.35 <= t["hazard"] <= 0.65
    assert t["lift"] is not None and t["lift"] > 1.0
    by_field = {f["field"]: f for f in t["fields"]}
    # the planted condition is found and survives BH...
    assert by_field["volume_z_20v126"]["bh_significant"]
    assert by_field["volume_z_20v126"]["rank_gap"] > 0.2
    # ...pure noise does not
    assert not by_field["vol_20d_ann"]["bh_significant"]
    # family preregistered, one trial per field tested
    kinds = [e["kind"] for e in led.entries()]
    assert kinds.count("PREREGISTRATION") == 1
    assert kinds.count("TRIAL") == len(t["fields"])
    assert kinds[-1] == "RESULT"


def test_runs_without_ledger_for_diagnostics():
    out = transition_stats(_genome(60, 6), MECH)
    assert out and out[0]["n"] > 0
