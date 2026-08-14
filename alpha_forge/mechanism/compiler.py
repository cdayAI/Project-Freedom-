"""Strategy compiler (BREAKTHROUGH_PHASE1 §7): a forecast is not a
strategy. A mechanism that survives discovery is compiled into a small
PREREGISTERED policy family and pushed through the same event engine,
the same costs, and the full Section-4 gate battery as every other
strategy — the discovery engine gets no exemptions from the immune
system.

Family bounds in this build:
  instrument   shares only. Option policies are eligible only when the
               archived chain history covers the backtest era; with days
               of coverage against decades of episodes they are
               mechanically ineligible, and that ineligibility is
               recorded in the family metadata rather than silently
               worked around.
  entry        immediate next-open after mechanism completion (the
               confirmation variant belongs to prefix compilation, which
               needs transition statistics — C11 territory).
  exit         fixed target/stop grid (preregistered below) plus the
               engine's 126-bar time stop.
  staging      single entry; account-tier mechanics (min ticket, PDT,
               settlement) live in the execution layer and are simulated
               there, not re-modeled here.

The whole family is ONE multiplicity unit: one preregistration, each
config a ledgered trial, one gate report for the family's primary config
(chosen by net log-wealth growth). Matched nulls: every completion is
paired with same-session eligible non-completing symbols traded under
the identical policy — the null holds era, regime, and market state
fixed and asks only whether THESE symbols mattered."""

from __future__ import annotations

import numpy as np
import polars as pl

from alpha_forge.gates.dsr import sharpe_ratio
from alpha_forge.gates.gatekeeper import run_gates
from alpha_forge.gates.walkforward import purged_walk_forward_splits
from alpha_forge.mechanism.discovery import (
    HORIZON_SESSIONS,
    collapse_overlaps,
    mechanism_key,
)
from alpha_forge.mechanism.episodes import Mechanism, extract_episodes

from alpha_forge.config import NULL_BASELINE_MIN_DRAWS

# preregistered exit grid: (target_mult, stop_frac)
EXIT_GRID: tuple[tuple[float, float], ...] = (
    (3.0, 0.35), (3.0, 0.50), (5.0, 0.35), (5.0, 0.50),
)
NULL_DRAWS = NULL_BASELINE_MIN_DRAWS  # gate 9 refuses fewer — same bar here
NULLS_PER_COMPLETION = 5
MIN_TRADES = 30


def _signal_indices(ep, eps: pl.DataFrame) -> list[tuple[int, int]]:
    """(symbol_idx, signal_day) for completions that exist and are eligible
    in the event panel on their completion day."""
    sym_ix = {s: i for i, s in enumerate(ep.symbols)}
    date_ix = {d: i for i, d in enumerate(ep.dates.astype("datetime64[D]").tolist())}
    last_col = eps.columns[-1]
    out = []
    for sym, d in eps.select(["symbol", last_col]).iter_rows():
        s, t = sym_ix.get(sym), date_ix.get(d)
        if s is not None and t is not None and ep.eligible[t, s]:
            out.append((s, t))
    return out


def _trades_for(ep, signals, target, stop) -> tuple[np.ndarray, np.ndarray, int]:
    """(net_log_rets, gross_log_rets, n) chronological by exit day."""
    from alpha_forge.research.eventbt import simulate_trade

    rows = []
    for s, t in signals:
        tr = simulate_trade(ep, s, t, target, stop)
        if tr is not None:
            rows.append((tr.exit_day, tr.net_log_ret, tr.gross_log_ret))
    rows.sort()
    if not rows:
        return np.array([]), np.array([]), 0
    _, net, gross = zip(*rows)
    return np.asarray(net), np.asarray(gross), len(rows)


def _monthly(ep, signals, target, stop) -> np.ndarray:
    """Equal-length monthly net simple-return series for the PBO matrix:
    per month, the summed net log return of trades EXITING that month."""
    from alpha_forge.research.eventbt import simulate_trade

    months = ep.dates.astype("datetime64[M]")
    uniq = np.unique(months)
    acc = np.zeros(uniq.size)
    ix = {m: i for i, m in enumerate(uniq.tolist())}
    for s, t in signals:
        tr = simulate_trade(ep, s, t, target, stop)
        if tr is not None:
            acc[ix[months[tr.exit_day].tolist()]] += tr.net_log_ret
    return np.expm1(acc)


def _matched_null_signals(ep, signals, rng) -> list[tuple[int, int]]:
    """Same-session eligible non-completing symbols, NULLS_PER_COMPLETION
    per real signal — the matched null of §7."""
    out = []
    taken = set(signals)
    for _, t in signals:
        elig = np.flatnonzero(ep.eligible[t])
        elig = [s for s in elig if (s, t) not in taken]
        if not elig:
            continue
        k = min(NULLS_PER_COMPLETION, len(elig))
        for s in rng.choice(elig, size=k, replace=False):
            out.append((int(s), t))
    return out


def compile_and_gate(
    ep,
    genome: pl.DataFrame,
    mech: Mechanism,
    ledger,
    data_vintage: str,
    seed: int = 7,
    stats_ledger=None,
) -> dict:
    """Compile one mechanism into the shares policy family and run the full
    Section-4 battery on the primary config. Returns the gate payload plus
    family metadata; every config is ledgered before any verdict is read."""
    rng = np.random.default_rng(seed)
    eps = collapse_overlaps(
        extract_episodes(genome, mech), HORIZON_SESSIONS)
    signals = _signal_indices(ep, eps)

    reg_id = ledger.preregister(
        hypothesis=(f"compiled policy family for mechanism {mech.sentence()} "
                    "— shares, next-open entry, fixed target/stop exits; "
                    "family is one multiplicity unit"),
        universe="genome_v1 + event panel (survivorship-biased)",
        parameters={"mechanism": mechanism_key(mech),
                    "exit_grid": [list(x) for x in EXIT_GRID],
                    "instrument": "shares",
                    "options_ineligible_reason":
                        "chain archive does not cover the backtest era"},
        author="strategy_compiler",
    )

    if len(signals) < MIN_TRADES:
        reason = f"{len(signals)} tradable completions < {MIN_TRADES}"
        ledger.record_result(reg_id, {"verdict": "UNDECIDABLE",
                                      "reason": reason})
        return {"verdict": "UNDECIDABLE", "reason": reason,
                "n_signals": len(signals), "reg_id": reg_id}

    per_config = []
    monthly_cols = []
    for target, stop in EXIT_GRID:
        net, gross, n = _trades_for(ep, signals, target, stop)
        monthly = _monthly(ep, signals, target, stop)
        growth = float(net.sum())
        sr = sharpe_ratio(monthly) if monthly.std() > 0 else 0.0
        ledger.record_trial(reg_id, {"target": target, "stop": stop,
                                     "n_trades": n}, float(sr))
        per_config.append({"target": target, "stop": stop, "n": n,
                           "net": net, "gross": gross, "growth": growth,
                           "monthly": monthly})
        monthly_cols.append(monthly)

    primary = max(per_config, key=lambda c: c["growth"])
    config_matrix = np.column_stack(monthly_cols)

    # matched nulls under the primary policy: NULL_DRAWS growth draws, each
    # a same-size resample of null trades
    null_sig = _matched_null_signals(ep, signals, rng)
    null_net, _, _ = _trades_for(
        ep, null_sig, primary["target"], primary["stop"])
    if null_net.size == 0:
        reason = "no tradable matched nulls"
        ledger.record_result(reg_id, {"verdict": "UNDECIDABLE",
                                      "reason": reason})
        return {"verdict": "UNDECIDABLE", "reason": reason,
                "n_signals": len(signals), "reg_id": reg_id}
    n_tr = primary["n"]
    null_growth = np.array([
        rng.choice(null_net, size=n_tr, replace=True).sum()
        for _ in range(NULL_DRAWS)])
    strat_growth = primary["growth"]
    p_perm = float((1 + np.sum(null_growth >= strat_growth))
                   / (1 + null_growth.size))
    permutation_result = {
        "p_value": p_perm, "observed": strat_growth,
        "null_mean": float(null_growth.mean()),
        "n_permutations": int(null_growth.size),
        "rejected_after_correction": bool(p_perm <= 0.05 / len(EXIT_GRID)),
        "correction": f"bonferroni alpha=0.05/{len(EXIT_GRID)} across the "
        "family's exit grid",
    }

    net_m = primary["monthly"]
    folds = purged_walk_forward_splits(net_m.size, n_folds=5, purge=1)
    fold_srs = []
    for _, test_idx in folds:
        seg = net_m[test_idx]
        if seg.size >= 6 and seg.std() > 0:
            fold_srs.append(round(sharpe_ratio(seg), 3))
    walkforward_summary = {
        "all_folds_evaluated": len(fold_srs) == len(folds),
        "fold_net_sharpes": fold_srs,
        "purge_months": 1,
    }

    import hashlib

    mech_id = hashlib.sha256(
        mechanism_key(mech).encode("utf-8")).hexdigest()[:8]
    report = run_gates(
        strategy_id=f"mech_{mech_id}",
        reg_id=reg_id,
        ledger=ledger,
        net_returns=primary["net"],
        gross_returns=primary["gross"],
        config_returns_matrix=config_matrix,
        n_trade_events=primary["n"],
        permutation_result=permutation_result,
        null_result_inputs=(strat_growth, null_growth),
        survivorship_free=False,
        walkforward_summary=walkforward_summary,
        extra_checks={"data_vintage": data_vintage,
                      "family": "mechanism_shares_v1",
                      "mechanism": mech.sentence()},
        stats_ledger=stats_ledger,
    )
    payload = report.to_payload()
    payload.update({"n_signals": len(signals),
                    "primary": {"target": primary["target"],
                                "stop": primary["stop"],
                                "n_trades": primary["n"],
                                "net_log_growth": strat_growth},
                    "p_matched_null": p_perm})
    return payload
