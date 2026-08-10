"""Agent 3 v2 — CONFLUENCE with time-matched controls.

Correction over v1 (documented in the ledger family note): v1 compared hit
fingerprints against controls drawn from ALL of history, so any feature that
merely co-moves with the era hits cluster in (volatility above all) scored
as "identifiable" partly through time-confounding. v2 is a matched design:

  each deduplicated hit gets K controls drawn from OTHER symbols within
  +/- MATCH_WINDOW trading days of the hit's signal date, past feature
  warmup, that do NOT begin a >=5x path in the following 126 bars.

Statistic per (feature, class): mean normalized within-group rank of the
hit, u_g = (rank - 0.5)/(group size). Under the null the hit is exchangeable
within its group, so E[u] = 0.5 REGARDLESS of era effects — time confounds
cancel inside each group. Permutation: redraw which group member is "the
hit", 20k stratified draws, two-sided on |mean_u - 0.5|. BH across the
feature x class family. matched_auc = P(hit > control) with ties at 0.5.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from alpha_forge.config import PERMUTATION_MIN_SHUFFLES, STORE_DIR
from alpha_forge.gates.permutation import benjamini_hochberg
from alpha_forge.ledger import Ledger
from alpha_forge.research.features import FEATURE_NAMES  # noqa: F401 (legacy consumers)
from alpha_forge.research.features_panel import feature_columns

K_CONTROLS = 5
MATCH_WINDOW = 10          # trading days on either side of the hit's date
MIN_GROUPS_PER_CLASS = 20  # below this a cell is UNDERPOWERED, not tested

CONFLUENCE2_PATH = STORE_DIR / "confluence_v2_results.parquet"
# Family history (each bump is a ledgered re-preregistration):
#   v2    time-matched controls (superseded: entry-day join leaked one day)
#   v2_1  signal-day join
#   v3    feature set extended with ex-ante catalyst features
#         (days_since_earnings_8k / days_since_any_8k from EDGAR 8-K)
FAMILY = "confluence_identifiability_v3_catalyst"


def last_family_vintage(ledger: Ledger) -> str | None:
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

    last = last_family_vintage(ledger)
    if last is None:
        return True
    if last == data_vintage:
        return False
    last_d = _dt.strptime(last[:10], "%Y-%m-%d").date()
    dates = features.get_column("date").unique()
    return int((dates > last_d).sum()) >= retest_days


def build_matched_groups(
    features: pl.DataFrame,
    hits: pl.DataFrame,
    seed: int,
) -> dict[int, list[np.ndarray]]:
    """class -> list of groups; each group is a (K+1, n_features) float
    matrix with the HIT in row 0 and its controls below. Fully numpy-indexed:
    the pool is sorted by trading-date position once, and each hit's control
    window is a binary-searched slice."""
    rng = np.random.default_rng(seed)
    feat_cols = feature_columns(features)
    dedup = hits.unique(subset=["symbol", "entry_date", "n_multiple"]).with_columns(
        pl.col("entry_date").str.slice(0, 10).str.to_date().alias("entry_dt")
    )

    all_dates_series = features.get_column("date").unique().sort()
    all_dates_list = all_dates_series.to_list()
    date_pos = {d: i for i, d in enumerate(all_dates_list)}

    # THE SIGNAL DAY, not the entry day: pathfinder entries fill at the open
    # of entry_date, so the last information available was the close of the
    # PRIOR trading day. Joining features at entry_date would hand the hit
    # its own first up-day — a one-day lookahead that inflates every cell.
    prior_date = {all_dates_list[i]: all_dates_list[i - 1] for i in range(1, len(all_dates_list))}
    dedup = dedup.with_columns(
        pl.col("entry_dt").replace_strict(prior_date, default=None).alias("signal_dt")
    ).drop_nulls("signal_dt")

    # hit rows joined to their own panel features (same source as controls)
    hit_feats = dedup.join(
        features.filter(pl.col("valid")),
        left_on=["symbol", "signal_dt"],
        right_on=["symbol", "date"],
        how="inner",
    )

    # control pool: valid, not starting a 5x path; sorted by date position
    pool = features.filter(pl.col("valid") & ~pl.col("starts_5x_fwd")).with_columns(
        pl.col("date").replace_strict(date_pos, default=None).alias("pos")
    ).drop_nulls("pos").sort("pos")
    pool_pos = pool.get_column("pos").to_numpy()
    pool_sym = np.asarray(pool.get_column("symbol").to_list(), dtype=object)
    pool_mat = pool.select(feat_cols).to_numpy().astype(float)

    groups: dict[int, list[tuple[object, np.ndarray]]] = {}
    for row in hit_feats.iter_rows(named=True):
        hpos = date_pos.get(row["signal_dt"])
        if hpos is None:
            continue
        lo = int(np.searchsorted(pool_pos, hpos - MATCH_WINDOW, side="left"))
        hi = int(np.searchsorted(pool_pos, hpos + MATCH_WINDOW, side="right"))
        if hi - lo < K_CONTROLS + 1:
            continue
        idx = np.arange(lo, hi)
        idx = idx[pool_sym[idx] != row["symbol"]]
        if idx.size < K_CONTROLS:
            continue
        take = rng.choice(idx, size=K_CONTROLS, replace=False)
        hit_vec = np.array([row[c] for c in feat_cols], dtype=float)
        group = np.vstack([hit_vec, pool_mat[take]])
        # SIGNAL date rides along: walk-forward training must exclude groups
        # whose hit's label had not matured by a fold's cutoff
        groups.setdefault(int(row["n_multiple"]), []).append((row["signal_dt"], group))
    return groups


def _matched_stats(
    groups: list[tuple[object, np.ndarray]], feature_idx: int
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Per-group normalized hit rank u_g, per-group sizes, and matched AUC.
    Groups where the hit or too many controls are NaN are dropped."""
    us, sizes = [], []
    auc_num, auc_den = 0.0, 0
    for _date, group in groups:
        vals = group[:, feature_idx]
        if np.isnan(vals[0]):
            continue
        ctrl = vals[1:]
        ctrl = ctrl[~np.isnan(ctrl)]
        if ctrl.size < 2:
            continue
        below = float((ctrl < vals[0]).sum()) + 0.5 * float((ctrl == vals[0]).sum())
        m = ctrl.size + 1
        rank = below + 1.0  # 1-based rank of the hit
        us.append((rank - 0.5) / m)
        sizes.append(float(m))
        auc_num += below
        auc_den += ctrl.size
    if len(us) == 0 or auc_den == 0:
        return None
    return np.asarray(us), np.asarray(sizes), auc_num / auc_den


def _stratified_permutation_p(
    us_obs: np.ndarray, group_sizes: np.ndarray, n_perm: int, rng: np.random.Generator
) -> float:
    """Under the null the hit's rank is uniform over its group positions:
    draw each group's u from {(r-0.5)/m : r=1..m}, mean across groups,
    two-sided on |mean - 0.5|."""
    obs = abs(float(us_obs.mean()) - 0.5)
    G = us_obs.size
    exceed = 0
    chunk = 4000
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        ranks = np.floor(rng.random((b, G)) * group_sizes).astype(int) + 1  # 1..m
        u = (ranks - 0.5) / group_sizes
        exceed += int(np.sum(np.abs(u.mean(axis=1) - 0.5) >= obs - 1e-15))
        done += b
    return (1 + exceed) / (1 + n_perm)


def run_confluence_v2(
    features: pl.DataFrame,
    hits: pl.DataFrame,
    ledger: Ledger,
    data_vintage: str,
    seed: int = 17,
    groups_by_class: dict | None = None,
) -> pl.DataFrame | None:
    """groups_by_class may be passed in when the caller (orchestrator) also
    needs the matched groups for walk-forward training — one build, two
    consumers, identical groups."""
    if hits is None or hits.height == 0:
        return None
    if not should_retest(ledger, features, data_vintage):
        return pl.read_parquet(CONFLUENCE2_PATH) if CONFLUENCE2_PATH.exists() else None

    reg_id = ledger.preregister(
        hypothesis="TIME-MATCHED identifiability: within groups of one N-x hit "
        "and K same-era controls (other symbols, +/-10 trading days), the hit's "
        "fingerprint rank differs from uniform. Corrects v1's era confound "
        "(v1 controls spanned all of history; features that co-move with the "
        "hit-rich era scored as signal).",
        universe="ingested equity panel (survivorship-biased, see manifest)",
        parameters={
            "family": FAMILY,
            "features": feature_columns(features),
            "k_controls": K_CONTROLS,
            "match_window_trading_days": MATCH_WINDOW,
            "statistic": "mean_normalized_within_group_rank",
            "n_permutations": PERMUTATION_MIN_SHUFFLES,
            "min_groups_per_class": MIN_GROUPS_PER_CLASS,
            "data_vintage": data_vintage,
        },
    )

    rng = np.random.default_rng(seed)
    if groups_by_class is None:
        groups_by_class = build_matched_groups(features, hits, seed)

    feat_cols = feature_columns(features)
    results, p_values, cells = [], [], []
    for n_x in sorted(groups_by_class):
        groups = groups_by_class[n_x]
        for fi, feat in enumerate(feat_cols):
            stats = _matched_stats(groups, fi)
            cell = {
                "n_multiple": n_x,
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
                reg_id, {"family": FAMILY, "class": n_x, "feature": feat}, None
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
    df.write_parquet(CONFLUENCE2_PATH)
    survivors = df.filter(pl.col("verdict") == "IDENTIFIABLE")
    ledger.record_result(
        reg_id,
        {
            "family": FAMILY,
            "data_vintage": data_vintage,
            "cells_tested": len(cells),
            "identifiable": survivors.height,
            "survivor_cells": survivors.select(
                ["n_multiple", "feature", "matched_auc"]
            ).to_dicts(),
            "note": "supersedes confluence v1 for all downstream consumers",
        },
    )
    return df
