"""Confluence, conditioned on CATALYST CLASS (mission Section 1: "every
pathfinder hit is tagged with its catalyst class ... and identifiability is
measured per class").

Pooling earnings-driven and non-catalyst hits dilutes both: an 8-K-driven
runup and a flowless squeeze may carry different (even opposing) pre-move
fingerprints. This family re-runs the episode-level time-matched design of
confluence v3.1 separately per catalyst class within the 5x cohort
(EARNINGS_8K / OTHER_8K / NONE), so the generator can screen on the
class-conditional signature instead of the pooled average.

Cells: (catalyst_class, feature). Same matched-group statistic, same 20k
stratified permutations, BH across THIS family's cells. Every tested cell
is a ledgered trial. Underpowered classes are reported, never tested.
"""

from __future__ import annotations

import polars as pl

from alpha_forge.config import PERMUTATION_MIN_SHUFFLES, STORE_DIR
from alpha_forge.gates.permutation import benjamini_hochberg
from alpha_forge.ledger import Ledger
from alpha_forge.research.confluence2 import (
    MIN_GROUPS_PER_CLASS,
    _matched_stats,
    _stratified_permutation_p,
    build_matched_groups,
)
from alpha_forge.research.features_panel import feature_columns

import numpy as np

FAMILY = "confluence_identifiability_catalyst_v1"
TARGET_CLASS = 5  # 5x cohort: the only class with per-catalyst power today
CONFLUENCE_CAT_PATH = STORE_DIR / "confluence_catalyst_results.parquet"


def _last_vintage(ledger: Ledger) -> str | None:
    last = None
    for e in ledger.entries():
        if (
            e["kind"] == "RESULT"
            and e["payload"].get("result", {}).get("family") == FAMILY
        ):
            last = e["payload"]["result"].get("data_vintage")
    return last


def should_retest(ledger: Ledger, features: pl.DataFrame, data_vintage: str,
                  retest_days: int = 5) -> bool:
    from datetime import datetime as _dt

    last = _last_vintage(ledger)
    if last is None:
        return True
    if last == data_vintage:
        return False
    last_d = _dt.strptime(last[:10], "%Y-%m-%d").date()
    dates = features.get_column("date").unique()
    return int((dates > last_d).sum()) >= retest_days


def run_confluence_by_catalyst(
    features: pl.DataFrame,
    hits: pl.DataFrame,
    ledger: Ledger,
    data_vintage: str,
    seed: int = 29,
    max_date=None,
) -> pl.DataFrame | None:
    """hits must already carry EDGAR catalyst_class tags (tag_hits)."""
    if hits is None or hits.height == 0 or "catalyst_class" not in hits.columns:
        return None
    if not should_retest(ledger, features, data_vintage):
        return pl.read_parquet(CONFLUENCE_CAT_PATH) if CONFLUENCE_CAT_PATH.exists() else None

    classes = sorted(hits["catalyst_class"].unique().to_list())
    feat_cols = feature_columns(features)
    reg_id = ledger.preregister(
        hypothesis="Catalyst-conditional identifiability: within episode-level "
        "time-matched groups, the pre-move fingerprint of 5x hits differs from "
        "same-era controls SEPARATELY per catalyst class (EARNINGS_8K / "
        "OTHER_8K / NONE). Pooling classes averages away class-specific "
        "signatures; this family measures each on its own.",
        universe="ingested equity panel (survivorship-biased) + EDGAR 8-K catalog",
        parameters={
            "family": FAMILY,
            "target_n_multiple": TARGET_CLASS,
            "catalyst_classes": classes,
            "features": feat_cols,
            "statistic": "mean_normalized_within_group_rank",
            "n_permutations": PERMUTATION_MIN_SHUFFLES,
            "min_groups_per_cell": MIN_GROUPS_PER_CLASS,
            "data_vintage": data_vintage,
        },
    )

    rng = np.random.default_rng(seed)
    results, p_values, cells = [], [], []
    for cat in classes:
        sub = hits.filter(
            (pl.col("catalyst_class") == cat) & (pl.col("n_multiple") == TARGET_CLASS)
        )
        groups_by_class = build_matched_groups(features, sub, seed, max_date=max_date)
        groups = groups_by_class.get(TARGET_CLASS, [])
        for fi, feat in enumerate(feat_cols):
            stats = _matched_stats(groups, fi)
            cell = {
                "catalyst_class": cat,
                "feature": feat,
                "n_groups": 0 if stats is None else int(stats[0].size),
            }
            if stats is None or stats[0].size < MIN_GROUPS_PER_CLASS:
                cell.update({"matched_auc": None, "mean_u": None, "p_value": None,
                             "verdict": "UNDERPOWERED"})
                results.append(cell)
                continue
            us, sizes, auc = stats
            p = _stratified_permutation_p(us, sizes, PERMUTATION_MIN_SHUFFLES, rng)
            cell.update({"matched_auc": auc, "mean_u": float(us.mean()),
                         "p_value": p, "verdict": "PENDING_BH"})
            ledger.record_trial(
                reg_id, {"family": FAMILY, "catalyst_class": cat, "feature": feat}, None
            )
            p_values.append(p)
            cells.append(cell)
            results.append(cell)

    if p_values:
        rejected = benjamini_hochberg(p_values, q=0.05)
        for cell, rej in zip(cells, rejected):
            cell["verdict"] = "IDENTIFIABLE" if rej else "NOT_SIGNIFICANT"

    df = pl.DataFrame(
        results,
        schema_overrides={"matched_auc": pl.Float64, "mean_u": pl.Float64,
                          "p_value": pl.Float64},
    )
    df.write_parquet(CONFLUENCE_CAT_PATH)
    survivors = df.filter(pl.col("verdict") == "IDENTIFIABLE")
    ledger.record_result(
        reg_id,
        {
            "family": FAMILY,
            "data_vintage": data_vintage,
            "cells_tested": len(cells),
            "identifiable": survivors.height,
            "survivor_cells": survivors.select(
                ["catalyst_class", "feature", "matched_auc"]
            ).to_dicts(),
        },
    )
    return df
