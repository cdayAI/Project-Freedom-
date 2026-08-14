"""Transition statistics — layer 3 of the mechanism grammar
(BREAKTHROUGH_PHASE1 §3): for each mechanism prefix, the conditional
hazard of ADVANCING to the next state versus dying, against matched
same-session controls, plus the §4 discovery question "what was different
immediately before the ones that advanced?" answered mechanically:
per-field rank gaps between advancers and stallers, permutation-tested
WITHIN date strata (era and regime never masquerade as a field effect),
Benjamini-Hochberg corrected per family.

The learned object is the set of transition conditions — which fields
separate sequences that advance from sequences that stall — not a score.
Each analyzed transition family is preregistered and every field test is
a ledgered trial when a ledger is supplied; diagnostics run without one,
but nothing unledgered may seed generation (§5.1a reads THESE tables)."""

from __future__ import annotations

import numpy as np
import polars as pl

from alpha_forge.gates.permutation import benjamini_hochberg
from alpha_forge.mechanism.discovery import collapse_overlaps
from alpha_forge.mechanism.episodes import Mechanism, extract_episodes
from alpha_forge.mechanism.predicates import PREDICATES

N_PERMUTATIONS = 500
BH_Q = 0.05

DEFAULT_MARGINS = (
    "volume_z_20v126", "dist_from_252d_high", "short_ratio_z",
    "vol_20d_ann", "dollar_vol_med_20d",
)


def _rank_pctl(genome: pl.DataFrame, fields) -> pl.DataFrame:
    """Per-date rank percentile of each margin field over valid rows; NaN
    ranks as null and lands neutral at 0.5 — never evidence, never extreme."""
    day = genome.filter(pl.col("valid"))
    return day.with_columns([
        ((pl.col(f).fill_nan(None).rank("average").over("date")
          / pl.col(f).fill_nan(None).count().over("date"))
         .fill_null(0.5).alias(f"__r_{f}"))
        for f in fields
    ]).select(["symbol", "date", *[f"__r_{f}" for f in fields]])


def _stratified_perm_p(gap: float, ranks: np.ndarray, advanced: np.ndarray,
                       dates: np.ndarray, rng) -> float:
    """Two-sided permutation p for the advancer-staller mean rank gap,
    shuffling advance labels WITHIN each date stratum."""
    order = np.argsort(dates, kind="stable")
    ranks, advanced, dates = ranks[order], advanced[order], dates[order]
    bounds = np.flatnonzero(np.concatenate(
        [[True], dates[1:] != dates[:-1], [True]]))
    draws = np.empty(N_PERMUTATIONS)
    for b in range(N_PERMUTATIONS):
        lab = advanced.copy()
        for i in range(bounds.size - 1):
            s, e = bounds[i], bounds[i + 1]
            rng.shuffle(lab[s:e])
        a, st = ranks[lab], ranks[~lab]
        draws[b] = a.mean() - st.mean() if a.size and st.size else 0.0
    return float((1 + np.sum(np.abs(draws) >= abs(gap)))
                 / (1 + N_PERMUTATIONS))


def transition_stats(
    genome: pl.DataFrame,
    mech: Mechanism,
    margin_fields: tuple[str, ...] = DEFAULT_MARGINS,
    ledger=None,
    seed: int = 11,
) -> list[dict]:
    """One dict per transition j -> j+1: hazard among prefix-holders,
    matched same-session base, lift, and BH-corrected per-field rank gaps
    between advancers and stallers."""
    rng = np.random.default_rng(seed)
    genome = genome.sort(["symbol", "date"])
    flags = np.column_stack([
        PREDICATES[name].evaluate(genome, dict(params)).to_numpy()
        for name, params in mech.stages
    ])
    sym = genome["symbol"].to_numpy()
    dates_all = genome["date"].to_numpy()
    valid = genome["valid"].to_numpy()
    change = np.flatnonzero(sym[1:] != sym[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [sym.size]])
    block_of = np.zeros(sym.size, dtype=int)
    for i, (s, e) in enumerate(zip(starts, ends)):
        block_of[s:e] = i

    ranks_df = _rank_pctl(genome, margin_fields)
    rk = genome.select(["symbol", "date"]).join(
        ranks_df, on=["symbol", "date"], how="left")
    rank_arr = {f: rk[f"__r_{f}"].fill_null(0.5).to_numpy()
                for f in margin_fields}

    reg_id = None
    if ledger is not None:
        reg_id = ledger.preregister(
            hypothesis=(f"transition conditions for {mech.sentence()}: "
                        "which margin fields separate advancers from "
                        "stallers at each prefix boundary"),
            universe="genome_v1 (survivorship-biased; PIPELINE_PROOF_ONLY)",
            parameters={"margins": list(margin_fields),
                        "n_permutations": N_PERMUTATIONS, "bh_q": BH_Q},
            author="transition_stats",
        )

    out = []
    n_stage = len(mech.stages)
    for j in range(1, n_stage):
        w = mech.windows[j - 1]
        # prefix-holder rows (global indices), anti-overlap collapsed
        if j == 1:
            hold = np.flatnonzero(flags[:, 0] & valid)
            eps = pl.DataFrame({"symbol": sym[hold],
                                "stage1_date": dates_all[hold]})
            eps = collapse_overlaps(
                eps.with_columns(pl.col("stage1_date").cast(pl.Date)), w)
            key = set(zip(eps["symbol"].to_list(),
                          eps["stage1_date"].to_list()))
            hold = np.array([i for i in hold
                             if (sym[i], dates_all[i].astype("datetime64[D]")
                                 .item()) in key], dtype=int)
        else:
            pre = Mechanism(mech.stages[:j], mech.windows[:j - 1])
            eps = collapse_overlaps(extract_episodes(genome, pre,
                                                     flags=flags[:, :j]), w)
            last = eps.columns[-1]
            ix = {(s_, d_): i for i, (s_, d_) in enumerate(
                zip(sym.tolist(), [d.astype("datetime64[D]").item()
                                   for d in dates_all]))}
            hold = np.array([ix[(s_, d_)] for s_, d_ in
                             eps.select(["symbol", last]).iter_rows()
                             if (s_, d_) in ix], dtype=int)
        if hold.size == 0:
            out.append({"transition": j, "n": 0})
            continue

        nxt = flags[:, j]
        advanced = np.zeros(hold.size, dtype=bool)
        for i, g in enumerate(hold):
            e = ends[block_of[g]]
            advanced[i] = bool(nxt[g + 1: min(g + 1 + w, e)].any())

        # matched base: same-session valid rows NOT holding the prefix
        hold_set = set(hold.tolist())
        base_hits = base_n = 0
        for d in np.unique(dates_all[hold]):
            day_rows = np.flatnonzero((dates_all == d) & valid)
            ctrl = [r for r in day_rows if r not in hold_set]
            if not ctrl:
                continue
            take = rng.choice(ctrl, size=min(50, len(ctrl)), replace=False)
            for r in take:
                e = ends[block_of[r]]
                base_hits += bool(nxt[r + 1: min(r + 1 + w, e)].any())
                base_n += 1
        hazard = float(advanced.mean())
        base = base_hits / base_n if base_n else None
        fields = []
        p_vals = []
        if advanced.any() and (~advanced).any():
            for f in margin_fields:
                r = rank_arr[f][hold]
                gap = float(r[advanced].mean() - r[~advanced].mean())
                p = _stratified_perm_p(gap, r, advanced, dates_all[hold], rng)
                fields.append({"field": f, "rank_gap": round(gap, 4),
                               "p": p})
                p_vals.append(p)
                if ledger is not None:
                    ledger.record_trial(reg_id, {
                        "transition": j, "field": f, "rank_gap": gap}, None)
            for rec, ok in zip(fields, benjamini_hochberg(p_vals, q=BH_Q)):
                rec["bh_significant"] = bool(ok)
        out.append({"transition": j, "window": int(w), "n": int(hold.size),
                    "hazard": round(hazard, 4),
                    "base": round(base, 4) if base is not None else None,
                    "lift": round(hazard / base, 3) if base else None,
                    "fields": fields})
    if ledger is not None:
        ledger.record_result(reg_id, {
            "transitions": [{k: v for k, v in t.items() if k != "fields"}
                            for t in out]})
    return out
