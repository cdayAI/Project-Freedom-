"""Agent 3 — CONFLUENCE.

Question: are the pre-move fingerprints of N-x path hits IDENTIFIABLE — do
they differ from the fingerprints of matched ordinary symbol-days? If yes,
in which features and which direction? Survivors become screening features
for the hypothesis generator; everything tested is a ledgered trial.

Method (pre-registered as a family before any statistic is computed):
  For each N-x class and each fingerprint feature:
    statistic = AUC (Mann-Whitney) of hits vs matched baseline days
    matched baseline: for every hit, K random (symbol, day) draws from the
    same panel with >= 60 bars of history that did NOT begin a >=5x path in
    the following 126 bars — same feature computation, no lookahead.
    permutation test: shuffle hit/baseline labels, 20k shuffles
    correction: Benjamini-Hochberg at q=0.05 across ALL feature x class
    cells tested that day.

AUC is rank-based: robust to the fat tails these fingerprints live in.
Regime tags are attached descriptively (counts per regime), not tested,
so they add no trials until a regime hypothesis is pre-registered.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from alpha_forge.config import PERMUTATION_MIN_SHUFFLES, STORE_DIR
from alpha_forge.gates.permutation import benjamini_hochberg
from alpha_forge.ledger import Ledger
from alpha_forge.research.features import FEATURE_NAMES, fingerprint_at

BASELINE_PER_HIT = 5
MIN_HITS_PER_CLASS = 20  # below this, the cell is reported UNDERPOWERED, not tested
RETEST_TRADING_DAYS = 5  # a family re-runs only after this many new trading days:
                         # re-testing on 1-day-shifted data would ledger ~30 trials
                         # per night for near-zero new information (over-deflation
                         # is safe but wasteful; cadence is the honest fix)

CONFLUENCE_PATH = STORE_DIR / "confluence_results.parquet"


def last_family_vintage(ledger: Ledger) -> str | None:
    """Data vintage of the most recent confluence family, if any."""
    last = None
    for e in ledger.entries():
        if (
            e["kind"] == "RESULT"
            and e["payload"].get("result", {}).get("family") == "confluence_identifiability"
        ):
            last = e["payload"]["result"].get("data_vintage")
    return last


def should_retest(ledger: Ledger, panel: pl.DataFrame, data_vintage: str) -> bool:
    """True when no family exists yet or >= RETEST_TRADING_DAYS trading days
    of new data have arrived since the last family's vintage."""
    from datetime import datetime as _dt

    last = last_family_vintage(ledger)
    if last is None:
        return True
    if last == data_vintage:
        return False
    last_d = _dt.strptime(last[:10], "%Y-%m-%d").date()
    dates = panel.get_column("date").unique()
    n_new = int((dates > last_d).sum())
    return n_new >= RETEST_TRADING_DAYS


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random hit value > random baseline value), midranks for ties."""
    from scipy.stats import rankdata

    combined = np.concatenate([pos, neg])
    ranks = rankdata(combined)
    n1, n2 = pos.size, neg.size
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2))


def _permutation_auc_pvalue(
    pos: np.ndarray, neg: np.ndarray, n_perm: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Two-sided permutation p for AUC != 0.5. Ranks are invariant under
    label shuffles, so we rank once and draw which n1 ranks are 'positive' —
    vectorized in chunks."""
    from scipy.stats import rankdata

    combined = np.concatenate([pos, neg])
    ranks = rankdata(combined)
    n1, n2 = pos.size, neg.size
    denom = n1 * n2
    offset = n1 * (n1 + 1) / 2.0
    auc_obs = float((ranks[:n1].sum() - offset) / denom)
    obs = abs(auc_obs - 0.5)

    exceed = 0
    chunk = 2000
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        keys = rng.random((b, combined.size))
        sel = np.argpartition(keys, n1 - 1, axis=1)[:, :n1]
        sums = ranks[sel].sum(axis=1)
        auc_p = (sums - offset) / denom
        exceed += int(np.sum(np.abs(auc_p - 0.5) >= obs - 1e-15))
        done += b
    p = (1 + exceed) / (1 + n_perm)
    return auc_obs, float(p)


def sample_baselines(
    panel: pl.DataFrame,
    n_draws: int,
    exclude: set[tuple[str, str]],
    seed: int,
) -> pl.DataFrame:
    """Random (symbol, day) fingerprints that did not begin a >=5x path.

    exclude: (symbol, entry_date) pairs of known hits — a baseline draw whose
    symbol-day starts any 5x+ path within 126 bars is rejected.
    """
    rng = np.random.default_rng(seed)
    rows = []
    by_symbol = {}
    for (sym,), g in panel.group_by("symbol"):
        g = g.sort("date")
        if g.height >= 200:
            by_symbol[str(sym)] = (
                [str(d) for d in g["date"].to_list()],
                g["open"].to_numpy().astype(float),
                g["high"].to_numpy().astype(float),
                g["low"].to_numpy().astype(float),
                g["close"].to_numpy().astype(float),
                g["volume"].to_numpy().astype(float),
            )
    symbols = list(by_symbol.keys())
    if not symbols:
        return pl.DataFrame()
    attempts = 0
    while len(rows) < n_draws and attempts < n_draws * 20:
        attempts += 1
        sym = symbols[rng.integers(len(symbols))]
        dates, open_, high, low, close, volume = by_symbol[sym]
        n = close.size
        i = int(rng.integers(60, n - 130))  # need history behind and 126 bars ahead
        date_i = dates[i]
        if (sym, date_i) in exclude:
            continue
        # reject if a >=5x path starts here: max close in (i+1..i+126] / open[i+1]
        fwd = close[i + 1 : i + 127]
        if fwd.size and open_[i + 1] > 0 and fwd.max() / open_[i + 1] >= 5.0:
            continue
        fp = fingerprint_at(i, open_, high, low, close, volume)
        fp["symbol"] = sym
        fp["date"] = date_i
        rows.append(fp)
    return pl.DataFrame(rows)


def run_confluence(
    panel: pl.DataFrame,
    hits: pl.DataFrame,
    ledger: Ledger,
    data_vintage: str,
    seed: int = 7,
) -> pl.DataFrame | None:
    """Returns per-(class, feature) results with BH-corrected verdicts."""
    if hits is None or hits.height == 0:
        return None

    # cadence guard: one family per vintage AND >= RETEST_TRADING_DAYS of new
    # data before a re-test; otherwise serve the cached results
    if not should_retest(ledger, panel, data_vintage):
        return pl.read_parquet(CONFLUENCE_PATH) if CONFLUENCE_PATH.exists() else None

    reg_id = ledger.preregister(
        hypothesis="Pre-move fingerprints of N-x path hits are distinguishable "
        "from matched non-hit symbol-days (AUC != 0.5), per feature and class, "
        "BH q=0.05 across the full feature x class family.",
        universe="ingested equity panel (survivorship-biased, see manifest)",
        parameters={
            "features": FEATURE_NAMES,
            "classes": sorted(hits["n_multiple"].unique().to_list()),
            "baseline_per_hit": BASELINE_PER_HIT,
            "statistic": "mann_whitney_auc",
            "n_permutations": PERMUTATION_MIN_SHUFFLES,
            "min_hits_per_class": MIN_HITS_PER_CLASS,
            "data_vintage": data_vintage,
        },
    )

    # deduplicate hits: one fingerprint per (symbol, entry_date, class) —
    # overlapping windows re-report the same move
    dedup = hits.unique(subset=["symbol", "entry_date", "n_multiple"])
    exclude = set(zip(dedup["symbol"].to_list(), dedup["entry_date"].to_list()))

    baselines = sample_baselines(
        panel, n_draws=max(200, dedup.height * BASELINE_PER_HIT), exclude=exclude, seed=seed
    )

    rng = np.random.default_rng(seed + 1)
    results = []
    p_values, cells = [], []
    for n_x in sorted(dedup["n_multiple"].unique().to_list()):
        class_hits = dedup.filter(pl.col("n_multiple") == n_x)
        for feat in FEATURE_NAMES:
            col = f"fp_{feat}"
            pos = class_hits[col].to_numpy().astype(float)
            pos = pos[~np.isnan(pos)]
            neg = baselines[feat].to_numpy().astype(float)
            neg = neg[~np.isnan(neg)]
            cell = {
                "n_multiple": n_x,
                "feature": feat,
                "n_hits": int(pos.size),
                "n_baseline": int(neg.size),
            }
            if pos.size < MIN_HITS_PER_CLASS or neg.size < 50:
                cell.update({"auc": None, "p_value": None, "verdict": "UNDERPOWERED"})
                results.append(cell)
                continue
            auc, p = _permutation_auc_pvalue(pos, neg, PERMUTATION_MIN_SHUFFLES, rng)
            cell.update({"auc": auc, "p_value": p, "verdict": "PENDING_BH"})
            ledger.record_trial(reg_id, {"family": "confluence", "class": n_x, "feature": feat}, None)
            p_values.append(p)
            cells.append(cell)
            results.append(cell)

    if p_values:
        rejected = benjamini_hochberg(p_values, q=0.05)
        for cell, rej in zip(cells, rejected):
            cell["verdict"] = "IDENTIFIABLE" if rej else "NOT_SIGNIFICANT"

    df = pl.DataFrame(
        [{k: v for k, v in r.items()} for r in results],
        schema_overrides={"auc": pl.Float64, "p_value": pl.Float64},
    )
    df.write_parquet(CONFLUENCE_PATH)
    survivors = df.filter(pl.col("verdict") == "IDENTIFIABLE")
    ledger.record_result(
        reg_id,
        {
            "family": "confluence_identifiability",
            "data_vintage": data_vintage,
            "cells_tested": len(cells),
            "identifiable": survivors.height,
            "survivor_cells": survivors.select(["n_multiple", "feature", "auc"]).to_dicts(),
        },
    )
    return df
