"""Mechanism layer: predicates are honest claims, episodes are
deterministic, matched controls are same-session nearest neighbors."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from alpha_forge.mechanism.episodes import Mechanism, extract_episodes, match_ignitions
from alpha_forge.mechanism.predicates import PREDICATES, validate_registry


def _mini_genome() -> pl.DataFrame:
    """Two symbols, 12 sessions, hand-set fields."""
    days = [dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(12)]
    rows = []
    for sym in ("AAA", "BBB"):
        for i, d in enumerate(days):
            rows.append({
                "symbol": sym, "date": d, "valid": True,
                "days_since_any_8k": 99.0, "volume_z_20v126": 0.0,
                "cs_spread_est": 0.01, "dist_from_252d_high": -0.5,
                "short_ratio_z": 0.0, "insider_net_buy_90d_usd": float("nan"),
                "starts_5x_fwd": False,
            })
    return pl.DataFrame(rows)


def test_registry_reads_only_registered_fields_and_no_labels_in_states():
    validate_registry()


def test_predicate_null_is_never_evidence():
    g = _mini_genome().with_columns(
        pl.lit(None, dtype=pl.Float64).alias("days_since_any_8k"))
    got = PREDICATES["CATALYST_SHOCK"].evaluate(g, {"k": 3})
    assert not got.any()


def test_unknown_param_refused():
    with pytest.raises(ValueError, match="unknown params"):
        PREDICATES["CATALYST_SHOCK"].evaluate(_mini_genome(), {"days": 3})


def test_labels_cannot_enter_a_prefix():
    with pytest.raises(ValueError, match="label"):
        Mechanism((("CATALYST_SHOCK", (("k", 3),)), ("IGNITION", ())), (5,))


def test_episode_greedy_earliest_and_window():
    g = _mini_genome()
    # AAA: 8-K at session 2; threshold break at sessions 4 and 6
    g = g.with_columns([
        pl.when((pl.col("symbol") == "AAA")
                & (pl.col("date") == dt.date(2026, 1, 7)))
        .then(0.0).otherwise(pl.col("days_since_any_8k"))
        .alias("days_since_any_8k"),
        pl.when((pl.col("symbol") == "AAA")
                & pl.col("date").is_in([dt.date(2026, 1, 9), dt.date(2026, 1, 11)]))
        .then(0.0).otherwise(pl.col("dist_from_252d_high"))
        .alias("dist_from_252d_high"),
    ])
    mech = Mechanism(
        (("CATALYST_SHOCK", (("k", 1),)), ("THRESHOLD_BREAK", (("b", 0.02),))),
        (3,),
    )
    eps = extract_episodes(g, mech)
    # CATALYST_SHOCK k=1 holds at sessions 2 AND 3 (days_since <= 1);
    # both bind greedily to the earliest break (session 4) and collapse
    assert eps.height == 1
    assert eps["symbol"].to_list() == ["AAA"]
    assert eps["stage2_date"][0] == dt.date(2026, 1, 9)
    # a 1-session window can no longer reach the break -> no episode
    assert extract_episodes(g, Mechanism(mech.stages, (1,))).height == 0
    assert "then within 3 sessions" in mech.sentence()


def test_matched_controls_are_nearest_same_session():
    g = _mini_genome()
    d = dt.date(2026, 1, 5)
    day = pl.DataFrame({
        "symbol": ["IGN", "C1", "C2", "C3"], "date": [d] * 4,
        "valid": [True] * 4,
        "days_since_any_8k": [99.0] * 4, "volume_z_20v126": [0.0] * 4,
        "cs_spread_est": [0.01] * 4,
        "dist_from_252d_high": [-0.10, -0.11, -0.50, -0.90],
        "short_ratio_z": [1.0, 0.9, -1.0, -2.0],
        "insider_net_buy_90d_usd": [float("nan")] * 4,
        "starts_5x_fwd": [True, False, False, False],
    })
    g = pl.concat([g.filter(pl.col("date") != d), day]).sort(["symbol", "date"])
    m = match_ignitions(g, ("dist_from_252d_high", "short_ratio_z"), k=2)
    got = m.sort("match_rank")
    assert got["ignition_symbol"].to_list() == ["IGN", "IGN"]
    assert got["control_symbol"].to_list() == ["C1", "C2"]  # nearest first
    assert got["distance"][0] < got["distance"][1]
    # controls only ever come from the ignition's own session
    assert set(m["date"].to_list()) == {d}


def test_thinness_is_trailing_and_nan_is_not_the_top_of_the_tape():
    days = [dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(12)]
    spreads = {"A": 0.05, "B": 0.04, "C": 0.01, "D": float("nan")}
    rows = []
    for sym, sp in spreads.items():
        for i, d in enumerate(days):
            rows.append({
                "symbol": sym, "date": d, "valid": True,
                "volume_z_20v126": 2.0 if i == len(days) - 1 else 0.0,
                "cs_spread_est": sp,
            })
    g = pl.DataFrame(rows)
    got = g.with_columns(
        PREDICATES["VOLUME_SURGE_THIN"].evaluate(
            g, {"min_vol_z": 1.5, "min_spread_pctl": 0.80}).alias("__f"))
    last = got.filter(pl.col("date") == days[-1]).sort("symbol")["__f"].to_list()
    # only A's trailing median clears the 80th pctl of the 3 KNOWN names;
    # D (all-NaN spread) must not fire and must not crowd A out of the top
    assert last == [True, False, False, False]
    # no surge, no firing — earlier sessions stay quiet
    assert not got.filter(pl.col("date") != days[-1])["__f"].any()


def test_matched_controls_nan_margin_is_neutral_not_extreme():
    d = dt.date(2026, 1, 5)
    day = pl.DataFrame({
        "symbol": ["IGN", "CNAN", "CHI"], "date": [d] * 3,
        "valid": [True] * 3,
        "short_ratio_z": [0.5, float("nan"), 50.0],
        "starts_5x_fwd": [True, False, False],
    })
    m = match_ignitions(day, ("short_ratio_z",), k=1)
    # NaN ranks neutral (0.5): nearer to the mid-ranked ignition than the
    # extreme control — NaN must never masquerade as the top of the tape
    assert m["control_symbol"].to_list() == ["CNAN"]
