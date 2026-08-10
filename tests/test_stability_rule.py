import polars as pl

from alpha_forge.ledger import Ledger
from alpha_forge.research.loop import generate_hypotheses, stable_survivor_features


def _family_result(ledger: Ledger, features: list[str], vintage: str):
    reg = ledger.preregister("family", "u", {})
    ledger.record_result(
        reg,
        {
            "family": "confluence_identifiability_v3_1_episodes",
            "data_vintage": vintage,
            "survivor_cells": [{"feature": f, "n_multiple": 5, "matched_auc": 0.7} for f in features],
        },
    )


def _conf(features: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "n_multiple": [5] * len(features),
            "feature": features,
            "auc": [0.75] * len(features),
            "p_value": [0.001] * len(features),
            "verdict": ["IDENTIFIABLE"] * len(features),
        }
    )


def test_no_filter_with_fewer_than_two_families(tmp_path):
    ledger = Ledger(path=tmp_path / "l.jsonl")
    _family_result(ledger, ["vol_20d_ann"], "2026-08-01")
    assert stable_survivor_features(ledger) is None
    hyps = generate_hypotheses(_conf(["vol_20d_ann", "price"]), ledger)
    assert len(hyps) == 2  # bootstrapping: everything eligible


def test_intersection_of_last_two_families(tmp_path):
    ledger = Ledger(path=tmp_path / "l.jsonl")
    _family_result(ledger, ["vol_20d_ann", "price"], "2026-08-01")
    _family_result(ledger, ["vol_20d_ann", "ret_21d"], "2026-08-08")
    stable = stable_survivor_features(ledger)
    assert stable == {"vol_20d_ann"}
    hyps = generate_hypotheses(_conf(["vol_20d_ann", "price", "ret_21d"]), ledger)
    assert [h["feature"] for h in hyps] == ["vol_20d_ann"]


def test_catalyst_family_does_not_count(tmp_path):
    ledger = Ledger(path=tmp_path / "l.jsonl")
    _family_result(ledger, ["vol_20d_ann"], "2026-08-01")
    reg = ledger.preregister("cat family", "u", {})
    ledger.record_result(
        reg,
        {"family": "confluence_identifiability_catalyst_v1",
         "data_vintage": "2026-08-08",
         "survivor_cells": [{"feature": "price"}]},
    )
    # still only ONE main-family run -> no filter yet
    assert stable_survivor_features(ledger) is None
