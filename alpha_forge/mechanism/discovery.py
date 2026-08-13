"""Generate-then-kill discovery over the mechanism grammar
(BREAKTHROUGH_PHASE1 §5). Cold-layer only: nothing here runs in the
nightly loop — discovery is invoked explicitly, and every evaluated
candidate is a ledgered trial before its score is looked at. The LLM
seeding path (§5.1b) enters through the same door: a proposed mechanism
becomes a grammar-conforming candidate here or it does not exist.

Scoring is matched lift: among a mechanism's completions (anti-overlap
collapsed), the share whose symbol starts a >=5x path within the horizon,
divided by the same-session base rate over all valid rows — the base is
computed per completion date, so era drift in the base rate cannot
manufacture lift.

The falsification battery kills, it never rescues:
  era stability      lift >= min_lift in BOTH date-halves of completions
  regime stability   completions beat their base in >= 2 regime strata
  order necessity    reversed stages must score materially worse, else
                     the "sequence" is a static screen wearing a costume
  ablation necessity (3+ stages) every drop-one-stage ablation must score
                     materially worse, else the shorter mechanism wins
  min support        too few collapsed completions = UNDECIDABLE, not a
                     survivor (absence of evidence is not evidence)
Kill reasons are ledgered per candidate; a later generation round reads
the result store and never re-runs a killed configuration.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import polars as pl

from alpha_forge.mechanism.episodes import Mechanism, extract_episodes
from alpha_forge.mechanism.predicates import OUTCOME_PREDICATES, PREDICATES

WINDOW_GRID = (3, 5, 10)
HORIZON_SESSIONS = 10
MIN_COMPLETIONS = 30
MIN_LIFT = 1.5
NECESSITY_MARGIN = 0.8  # ablated/reordered must score < margin * full lift


def state_predicate_names() -> list[str]:
    return sorted(n for n in PREDICATES if n not in OUTCOME_PREDICATES)


def generate_candidates(
    n_stages: int = 2,
    windows: tuple[int, ...] = WINDOW_GRID,
    max_candidates: int | None = None,
) -> list[Mechanism]:
    """Deterministic enumeration: ordered tuples of DISTINCT predicates x
    their preregistered grids x one shared window size. Bounded by
    construction; max_candidates truncates the deterministic order so a
    capped run is a prefix of the full run, never a sample of it."""
    out: list[Mechanism] = []
    for names in permutations(state_predicate_names(), n_stages):
        grids = [PREDICATES[n].grid_points() for n in names]
        idx = [0] * n_stages
        while True:
            stages = tuple(
                (n, tuple(sorted(grids[j][idx[j]].items())))
                for j, n in enumerate(names)
            )
            for w in windows:
                out.append(Mechanism(stages, (w,) * (n_stages - 1)))
            for j in range(n_stages - 1, -1, -1):
                idx[j] += 1
                if idx[j] < len(grids[j]):
                    break
                idx[j] = 0
            else:
                break
    if max_candidates is not None:
        out = out[:max_candidates]
    return out


def mechanism_key(mech: Mechanism) -> str:
    return repr((mech.stages, mech.windows))


def _forward_ignition(genome: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """symbol/date/fwd_ign: does a >=5x path start within (t, t+horizon]
    sessions for this symbol? Label-derived — cold layer only."""
    g = genome.sort(["symbol", "date"])
    sym = g["symbol"].to_numpy()
    ign = g["starts_5x_fwd"].cast(pl.Boolean).fill_null(False).to_numpy()
    fwd = np.zeros(sym.size, dtype=bool)
    change = np.flatnonzero(sym[1:] != sym[:-1]) + 1
    for s, e in zip(np.concatenate([[0], change]), np.concatenate([change, [sym.size]])):
        c = np.concatenate([[0], np.cumsum(ign[s:e])])
        n = e - s
        hi = np.minimum(np.arange(1, n + 1) + horizon, n)
        fwd[s:e] = (c[hi] - c[np.arange(1, n + 1)]) > 0
    return g.select(["symbol", "date", "valid"]).with_columns(
        pl.Series("fwd_ign", fwd))


def collapse_overlaps(eps: pl.DataFrame, horizon: int) -> pl.DataFrame:
    """Anti-overlap: per symbol, a completion within `horizon` sessions of
    the previously KEPT completion is the same observation. Calendar-day
    proxy horizon*2 covers weekends; conservative (collapses more)."""
    if eps.height == 0:
        return eps
    last_col = eps.columns[-1]
    keep_rows = []
    for (_,), g in eps.sort(["symbol", last_col]).group_by(
            ["symbol"], maintain_order=True):
        last_kept = None
        for row in g.iter_rows(named=True):
            d = row[last_col]
            if last_kept is None or (d - last_kept).days > horizon * 2:
                keep_rows.append(row)
                last_kept = d
    return pl.DataFrame(keep_rows, schema=eps.schema)


def score_mechanism(
    genome: pl.DataFrame,
    mech: Mechanism,
    fwd: pl.DataFrame,
    base_by_date: dict,
    horizon: int = HORIZON_SESSIONS,
    flags: np.ndarray | None = None,
    eps: pl.DataFrame | None = None,
) -> dict:
    """Matched lift over anti-overlap-collapsed completions."""
    if eps is None:
        eps = extract_episodes(genome, mech, flags=flags)
    eps = collapse_overlaps(eps, horizon)
    if eps.height == 0:
        return {"n": 0, "hits": 0, "hit_rate": None, "base": None, "lift": None}
    last_col = eps.columns[-1]
    joined = eps.rename({last_col: "date"}).join(
        fwd.select(["symbol", "date", "fwd_ign"]), on=["symbol", "date"],
        how="left")
    hits = int(joined["fwd_ign"].fill_null(False).sum())
    n = eps.height
    bases = [base_by_date.get(d, None) for d in joined["date"].to_list()]
    bases = [b for b in bases if b is not None]
    base = float(np.mean(bases)) if bases else None
    hit_rate = hits / n
    lift = (hit_rate / base) if base else None
    return {"n": n, "hits": hits, "hit_rate": hit_rate, "base": base,
            "lift": lift, "dates": joined["date"].to_list(),
            "eps": eps}


def falsification_battery(
    genome: pl.DataFrame,
    mech: Mechanism,
    score: dict,
    fwd: pl.DataFrame,
    base_by_date: dict,
    horizon: int = HORIZON_SESSIONS,
    flag_cache: dict | None = None,
) -> tuple[str, str]:
    """(status, reason): SURVIVOR / KILLED / UNDECIDABLE."""
    if score["n"] < MIN_COMPLETIONS:
        return "UNDECIDABLE", f"support {score['n']} < {MIN_COMPLETIONS}"
    if not score["lift"] or score["lift"] < MIN_LIFT:
        return "KILLED", f"lift {score['lift']} < {MIN_LIFT}"

    def _lift_of(sub_eps: pl.DataFrame) -> float | None:
        if sub_eps.height == 0:
            return None
        last = sub_eps.columns[-1]
        j = sub_eps.rename({last: "date"}).join(
            fwd.select(["symbol", "date", "fwd_ign"]), on=["symbol", "date"],
            how="left")
        bases = [base_by_date.get(d) for d in j["date"].to_list()]
        bases = [b for b in bases if b is not None]
        if not bases:
            return None
        hr = float(j["fwd_ign"].fill_null(False).mean())
        return hr / float(np.mean(bases)) if np.mean(bases) else None

    # era stability: both date-halves of the collapsed completions
    eps = score["eps"]
    last = eps.columns[-1]
    dates = sorted(eps[last].to_list())
    mid = dates[len(dates) // 2]
    for tag, half in (("early", eps.filter(pl.col(last) < mid)),
                      ("late", eps.filter(pl.col(last) >= mid))):
        hl = _lift_of(half)
        if hl is None or hl < MIN_LIFT:
            return "KILLED", f"era instability: {tag} half lift {hl}"

    # regime stability: completions beat base in >= 2 regime_trend strata
    reg = eps.rename({last: "date"}).join(
        genome.select(["symbol", "date", "regime_trend"]).unique(),
        on=["symbol", "date"], how="left").join(
        fwd.select(["symbol", "date", "fwd_ign"]), on=["symbol", "date"],
        how="left")
    strata_beating = 0
    for (_,), g in reg.group_by(["regime_trend"]):
        bases = [base_by_date.get(d) for d in g["date"].to_list()]
        bases = [b for b in bases if b is not None]
        if bases and float(g["fwd_ign"].fill_null(False).mean()) > np.mean(bases):
            strata_beating += 1
    if strata_beating < 2:
        return "KILLED", f"regime-confined: beats base in {strata_beating} stratum"

    def _cached_score(m2: Mechanism) -> float | None:
        fl = None
        if flag_cache is not None:
            fl = np.column_stack([flag_cache[(n, p)] for n, p in m2.stages])
        s2 = score_mechanism(genome, m2, fwd, base_by_date, horizon, flags=fl)
        return s2["lift"]

    # order necessity: reversed stages must be materially worse
    rev = Mechanism(tuple(reversed(mech.stages)), mech.windows)
    rl = _cached_score(rev)
    if rl is not None and rl >= score["lift"] * NECESSITY_MARGIN:
        return "KILLED", f"order-free: reversed lift {rl:.2f} ~ full {score['lift']:.2f}"

    # ablation necessity (3+ stages): every drop-one must be materially worse
    if len(mech.stages) >= 3:
        for j in range(len(mech.stages)):
            stages = mech.stages[:j] + mech.stages[j + 1:]
            windows = mech.windows[:-1] if j == len(mech.stages) - 1 \
                else mech.windows[1:] if j == 0 else \
                mech.windows[:j - 1] + (mech.windows[j - 1] + mech.windows[j],
                                        ) + mech.windows[j + 1:]
            windows = windows[:len(stages) - 1]
            al = _cached_score(Mechanism(stages, windows))
            if al is not None and al >= score["lift"] * NECESSITY_MARGIN:
                return "KILLED", (f"stage {j} unnecessary: ablated lift "
                                  f"{al:.2f} ~ full {score['lift']:.2f}")
    return "SURVIVOR", ""


def run_discovery(
    genome: pl.DataFrame,
    ledger,
    n_stages: int = 2,
    max_candidates: int | None = 50,
    horizon: int = HORIZON_SESSIONS,
    killed_keys: set[str] | None = None,
) -> pl.DataFrame:
    """Preregister the family, ledger every candidate as a trial, run the
    battery, return the result table (one row per candidate). Survivors are
    NOT strategies: compilation and the Section-4 battery (C10+) still gate
    anything that would ever touch sizing. Cold layer; label-reading is
    confined to fwd-ignition scoring."""
    genome = genome.sort(["symbol", "date"])
    cands = generate_candidates(n_stages=n_stages, max_candidates=max_candidates)
    killed_keys = killed_keys or set()
    cands = [m for m in cands if mechanism_key(m) not in killed_keys]

    reg_id = ledger.preregister(
        hypothesis=(f"mechanism grammar sweep: {n_stages}-stage ordered "
                    f"predicate sequences lift >=5x ignition odds vs "
                    f"same-session base within {horizon} sessions"),
        universe="genome_v1 (survivorship-biased; PIPELINE_PROOF_ONLY)",
        parameters={"n_stages": n_stages, "windows": list(WINDOW_GRID),
                    "n_candidates": len(cands), "min_lift": MIN_LIFT,
                    "necessity_margin": NECESSITY_MARGIN},
        author="discovery_engine",
    )

    fwd = _forward_ignition(genome, horizon)
    base_by_date = dict(
        fwd.filter(pl.col("valid")).group_by("date")
        .agg(pl.col("fwd_ign").mean().alias("b"))
        .iter_rows())

    flag_cache: dict = {}
    for m in cands:
        for name, params in m.stages:
            if (name, params) not in flag_cache:
                flag_cache[(name, params)] = (
                    PREDICATES[name].evaluate(genome, dict(params)).to_numpy())

    rows = []
    for m in cands:
        ledger.record_trial(reg_id, {"mechanism": mechanism_key(m)}, None)
        fl = np.column_stack([flag_cache[(n, p)] for n, p in m.stages])
        sc = score_mechanism(genome, m, fwd, base_by_date, horizon, flags=fl)
        if sc["n"] == 0:
            status, reason = "UNDECIDABLE", "no completions"
        else:
            status, reason = falsification_battery(
                genome, m, sc, fwd, base_by_date, horizon, flag_cache)
        rows.append({
            "mechanism": mechanism_key(m), "sentence": m.sentence(),
            "n": sc["n"], "hits": sc["hits"], "hit_rate": sc["hit_rate"],
            "base": sc["base"], "lift": sc["lift"],
            "status": status, "kill_reason": reason,
        })
    results = pl.DataFrame(rows)
    ledger.record_result(reg_id, {
        "candidates": len(cands),
        "survivors": int((results["status"] == "SURVIVOR").sum()),
        "killed": int((results["status"] == "KILLED").sum()),
        "undecidable": int((results["status"] == "UNDECIDABLE").sum()),
    })
    return results
