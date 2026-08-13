"""Mechanism prefixes and ignition-versus-failure matching
(BREAKTHROUGH_PHASE1 §3-4).

A mechanism is an ordered sequence of 2-5 state predicates with timing
windows measured in SESSIONS (row offsets within a symbol — calendar gaps
from halts or delistings therefore never stretch a window):

    P1 --(within w1 sessions)--> P2 --(within w2)--> ...

extract_episodes() scans each symbol greedily: every session where P1
holds opens a candidate episode, each later stage binds to the EARLIEST
qualifying session inside its window, and episodes that complete on
identical stage dates collapse to one. Greedy-earliest is deterministic,
so an episode count is reproducible from (genome, mechanism) alone.

match_ignitions() builds the matched sets the discovery questions run on:
for every valid ignition row, the k nearest same-session non-ignitions by
mean absolute rank-percentile distance on the mechanism's own margin
fields. Same-session matching makes era and regime stratification exact by
construction (regime is a date-level fact), which is the strictest reading
of "same era, same regime stratum". Null margins rank at 0.5 — neutral,
never disqualifying — so thinly covered fields widen matches rather than
silently shrinking the control pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from alpha_forge.mechanism.predicates import OUTCOME_PREDICATES, PREDICATES

MAX_STAGES = 5


@dataclass(frozen=True)
class Mechanism:
    """stages: ((predicate_name, params), ...); windows[i] = max sessions
    between stage i and stage i+1 (len == len(stages) - 1)."""

    stages: tuple
    windows: tuple

    def __post_init__(self):
        if not 2 <= len(self.stages) <= MAX_STAGES:
            raise ValueError(f"mechanisms take 2-{MAX_STAGES} stages")
        if len(self.windows) != len(self.stages) - 1:
            raise ValueError("need exactly len(stages)-1 windows")
        if any(w < 1 for w in self.windows):
            raise ValueError("windows are >=1 session")
        for name, _ in self.stages:
            if name in OUTCOME_PREDICATES:
                raise ValueError(f"{name} is a label — not usable in a prefix")
            if name not in PREDICATES:
                raise ValueError(f"unknown predicate {name}")

    def sentence(self) -> str:
        parts = [f"[{n} {p}]" if p else f"[{n}]" for n, p in self.stages]
        joins = [f" then within {w} sessions " for w in self.windows]
        out = parts[0]
        for j, p in zip(joins, parts[1:]):
            out += j + p
        return out


def extract_episodes(genome: pl.DataFrame, mech: Mechanism) -> pl.DataFrame:
    """One row per completed episode: symbol, stage1_date..stageN_date."""
    genome = genome.sort(["symbol", "date"])
    flags = np.column_stack([
        PREDICATES[name].evaluate(genome, dict(params)).to_numpy()
        for name, params in mech.stages
    ])
    sym = genome["symbol"].to_numpy()
    dates = genome["date"].to_list()  # python dates: polars infers pl.Date
    n_stage = flags.shape[1]

    rows: list[tuple] = []
    seen: set[tuple] = set()
    # symbol block boundaries (sorted input)
    change = np.flatnonzero(sym[1:] != sym[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [sym.size]])
    for s, e in zip(starts, ends):
        stage_idx = [np.flatnonzero(flags[s:e, j]) for j in range(n_stage)]
        for i1 in stage_idx[0]:
            chain = [i1]
            ok = True
            for j in range(1, n_stage):
                nxt = stage_idx[j]
                pos = np.searchsorted(nxt, chain[-1], side="right")
                if pos == nxt.size or nxt[pos] - chain[-1] > mech.windows[j - 1]:
                    ok = False
                    break
                chain.append(nxt[pos])
            if ok:
                key = (sym[s], *(int(c) for c in chain[1:]))
                if key not in seen:  # same completion path, later start
                    seen.add(key)
                    rows.append((sym[s], *(dates[s + c] for c in chain)))
    cols = ["symbol"] + [f"stage{j + 1}_date" for j in range(n_stage)]
    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 if c == "symbol" else pl.Date
                                    for c in cols})
    schema = {c: (pl.Utf8 if c == "symbol" else pl.Date) for c in cols}
    return pl.DataFrame(rows, schema=schema, orient="row")


def match_ignitions(
    genome: pl.DataFrame,
    margin_fields: tuple[str, ...],
    k: int = 5,
) -> pl.DataFrame:
    """(ignition_symbol, date, control_symbol, distance, match_rank) — the
    k nearest valid same-session non-ignitions per valid ignition row."""
    need = {"symbol", "date", "valid", "starts_5x_fwd", *margin_fields}
    missing = need - set(genome.columns)
    if missing:
        raise ValueError(f"genome lacks {sorted(missing)}")

    day = genome.filter(pl.col("valid"))
    # rank-percentile per date, nulls neutral at 0.5
    day = day.with_columns([
        ((pl.col(f).fill_nan(None).rank("average").over("date")
          / pl.col(f).fill_nan(None).count().over("date"))
         .fill_null(0.5).alias(f"__r_{f}"))
        for f in margin_fields
    ])
    ign_dates = (
        day.filter(pl.col("starts_5x_fwd").cast(pl.Boolean))["date"].unique()
    )
    out: list[tuple] = []
    rcols = [f"__r_{f}" for f in margin_fields]
    for d in ign_dates:
        g = day.filter(pl.col("date") == d)
        r = g.select(rcols).to_numpy()
        is_ign = g["starts_5x_fwd"].cast(pl.Boolean).to_numpy()
        syms = g["symbol"].to_numpy()
        ctrl = np.flatnonzero(~is_ign)
        if ctrl.size == 0:
            continue
        order_syms = np.argsort(syms[ctrl], kind="stable")  # tie-break: symbol
        ctrl = ctrl[order_syms]
        for i in np.flatnonzero(is_ign):
            dist = np.abs(r[ctrl] - r[i]).mean(axis=1)
            take = np.argsort(dist, kind="stable")[:k]
            for rank, t in enumerate(take):
                out.append((syms[i], d, syms[ctrl[t]],
                            float(dist[t]), rank + 1))
    schema = {"ignition_symbol": pl.Utf8, "date": pl.Date,
              "control_symbol": pl.Utf8, "distance": pl.Float64,
              "match_rank": pl.Int64}
    return (pl.DataFrame(out, schema=schema, orient="row") if out
            else pl.DataFrame(schema=schema))
