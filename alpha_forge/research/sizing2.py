"""Sizing engine v2 — path generation without the i.i.d. lie.

The v1 frontier bootstrapped trades independently, which assumes markets
have no memory: no loss clustering, no regimes, no uncertainty about the
distribution itself. Each assumption biases risk DOWN — the dangerous
direction for a sizing engine. v2 replaces the generator, keeps the
objective (median log-wealth growth under a hard ruin constraint), and
reports the v1 number beside its replacements so the understatement is
measured, not asserted.

Three generators, layered by which lie they remove:

  IID          the v1 baseline (kept for comparison, never for sizing)
  STATIONARY   Politis & Romano (1994) stationary bootstrap: resamples the
               trade SEQUENCE in geometric-length blocks, preserving serial
               dependence — clustered losses survive into the paths
  BAYESIAN     Rubin (1981) Bayesian bootstrap: each path draws Dirichlet
               weights over the observed trades, so every path lives under a
               DIFFERENT plausible return distribution — parameter
               uncertainty propagates instead of being ignored
  REGIME       (when per-trade regime labels exist) empirical regime
               transition chain at trade frequency; each simulated trade
               draws from its regime's own return pool — bear sequences
               arrive in runs, as they do in data

Decision rule: the constrained optimum maximizes median terminal wealth
under the BAYESIAN generator subject to the ruin constraint holding under
the WORST generator available. Optimism must survive every model on the
table, pessimism only needs one.

Sizing under parameter uncertainty also demotes point-estimate Kelly: the
engine reports the POSTERIOR of the Kelly fraction (via Dirichlet-weighted
distributions) and its 5th percentile — the principled fractional-Kelly
answer, derived rather than hand-waved.

References in SOURCES.md. Path count floor: MONTE_CARLO_MIN_PATHS per
generator (mission Section 3, Agent 4).
"""

from __future__ import annotations

import numpy as np

from alpha_forge.config import MONTE_CARLO_MIN_PATHS, RUIN_DRAWDOWN_LEVEL
from alpha_forge.research.sizing import kelly_fraction

GENERATORS = ("iid", "stationary", "bayesian", "regime")


# ------------------------------------------------------------- generators


def iid_paths(r: np.ndarray, n_paths: int, T: int, rng: np.random.Generator) -> np.ndarray:
    return r[rng.integers(0, r.size, size=(n_paths, T))]


def stationary_bootstrap_paths(
    r: np.ndarray, n_paths: int, T: int, rng: np.random.Generator,
    expected_block: float | None = None,
) -> np.ndarray:
    """Politis-Romano: geometric block lengths (mean L), wrap-around indexing.
    Default L = max(5, n^(1/3)) — the standard rate; a structural choice,
    recorded in the output."""
    n = r.size
    L = expected_block if expected_block is not None else max(5.0, n ** (1.0 / 3.0))
    p_new = 1.0 / L
    idx = np.empty((n_paths, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_paths)
    for t in range(1, T):
        new_block = rng.random(n_paths) < p_new
        fresh = rng.integers(0, n, size=n_paths)
        idx[:, t] = np.where(new_block, fresh, (idx[:, t - 1] + 1) % n)
    return r[idx]


def bayesian_bootstrap_paths(
    r: np.ndarray, n_paths: int, T: int, rng: np.random.Generator, chunk: int = 2000
) -> np.ndarray:
    """Rubin: per-path Dirichlet(1,...,1) weights over the observed trades,
    then T draws from that path's weighted distribution."""
    n = r.size
    out = np.empty((n_paths, T))
    done = 0
    while done < n_paths:
        b = min(chunk, n_paths - done)
        w = rng.gamma(1.0, size=(b, n))
        w /= w.sum(axis=1, keepdims=True)
        cdf = np.cumsum(w, axis=1)
        u = rng.random((b, T))
        # row-wise searchsorted via broadcasting (chunked to bound memory)
        idx = (u[:, :, None] > cdf[:, None, :]).sum(axis=2)
        out[done : done + b] = r[np.minimum(idx, n - 1)]
        done += b
    return out


def estimate_transition_matrix(labels: np.ndarray, states: list[str]) -> np.ndarray:
    """Row-stochastic transition matrix at trade frequency, Laplace-smoothed."""
    k = len(states)
    pos = {s: i for i, s in enumerate(states)}
    counts = np.ones((k, k))  # Laplace prior: unseen transitions stay possible
    for a, b in zip(labels[:-1], labels[1:]):
        counts[pos[a], pos[b]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def regime_conditional_paths(
    r: np.ndarray,
    labels: np.ndarray,
    n_paths: int,
    T: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Simulate regime chains from the empirical transition matrix; each
    trade draws from its regime's own return pool. None when any regime has
    too few trades to form a pool (honesty over interpolation)."""
    states = sorted(set(labels.tolist()))
    if len(states) < 2:
        return None
    pools = {s: r[labels == s] for s in states}
    if any(p.size < 10 for p in pools.values()):
        return None
    P = estimate_transition_matrix(labels, states)
    k = len(states)
    start_dist = np.bincount([states.index(s) for s in labels], minlength=k).astype(float)
    start_dist /= start_dist.sum()

    chain = np.empty((n_paths, T), dtype=np.int64)
    chain[:, 0] = rng.choice(k, size=n_paths, p=start_dist)
    cum = np.cumsum(P, axis=1)
    for t in range(1, T):
        u = rng.random(n_paths)
        prev_rows = cum[chain[:, t - 1]]
        chain[:, t] = (u[:, None] > prev_rows).sum(axis=1)
    out = np.empty((n_paths, T))
    for si, s in enumerate(states):
        mask = chain == si
        out[mask] = rng.choice(pools[s], size=int(mask.sum()), replace=True)
    return out


# --------------------------------------------------------------- frontier


def _frontier_for_matrix(sampled: np.ndarray, fractions: np.ndarray,
                         ruin_loss: float) -> list[dict]:
    n_paths = sampled.shape[0]
    rows = []
    for f in fractions:
        growth = np.log1p(np.clip(f * sampled, -0.9999, None))
        log_wealth = np.cumsum(growth, axis=1)
        terminal = log_wealth[:, -1]
        running_max = np.maximum.accumulate(
            np.hstack([np.zeros((n_paths, 1)), log_wealth]), axis=1
        )
        dd = log_wealth - running_max[:, 1:]
        min_dd = dd.min(axis=1)
        rows.append(
            {
                "fraction": float(f),
                "median_terminal_wealth": float(np.exp(np.median(terminal))),
                "p5_terminal_wealth": float(np.exp(np.percentile(terminal, 5))),
                "p_ruin": float(np.mean(min_dd <= np.log(1 - ruin_loss))),
                "p_drawdown_below_50pct": float(
                    np.mean(min_dd <= np.log(1 - RUIN_DRAWDOWN_LEVEL))
                ),
            }
        )
    return rows


def kelly_posterior(r: np.ndarray, rng: np.random.Generator, n_dists: int = 2000,
                    grid: np.ndarray | None = None) -> dict:
    """Posterior of the Kelly fraction under Dirichlet-weighted resamples of
    the observed trades. p5 is the derived fractional-Kelly answer."""
    grid = grid if grid is not None else np.linspace(0.01, 2.0, 100)
    n = r.size
    w = rng.gamma(1.0, size=(n_dists, n))
    w /= w.sum(axis=1, keepdims=True)
    # E_w[log(1+f r)] for every (dist, f): (n_dists, G)
    lg = np.log1p(np.clip(grid[None, :, None] * r[None, None, :], -0.9999, None))
    obj = np.einsum("dn,dgn->dg", w, np.broadcast_to(lg, (n_dists,) + lg.shape[1:]))
    best = grid[np.argmax(obj, axis=1)]
    return {
        "kelly_p5": float(np.percentile(best, 5)),
        "kelly_p50": float(np.percentile(best, 50)),
        "kelly_p95": float(np.percentile(best, 95)),
        "kelly_point": kelly_fraction(r),
    }


def sizing_frontier_v2(
    trade_returns: np.ndarray,
    regime_labels: np.ndarray | None = None,
    n_trades_per_path: int = 100,
    n_paths: int = MONTE_CARLO_MIN_PATHS,
    fractions: np.ndarray | None = None,
    ruin_loss: float = 0.90,
    ruin_prob_limit: float = 0.05,
    seed: int = 0,
) -> dict:
    """The v2 decision surface. See module docstring for the decision rule."""
    r = np.asarray(trade_returns, dtype=float)
    if r.size < 30:
        raise ValueError("need >= 30 trades for a sizing frontier")
    if n_paths < MONTE_CARLO_MIN_PATHS:
        raise ValueError(f"minimum {MONTE_CARLO_MIN_PATHS} Monte Carlo paths per generator")
    rng = np.random.default_rng(seed)
    kp = kelly_posterior(r, rng)
    if fractions is None:
        top = max(kp["kelly_p95"] * 1.25, 0.25)
        fractions = np.unique(np.round(np.linspace(0.01, top, 30), 4))

    T = n_trades_per_path
    matrices: dict[str, np.ndarray] = {
        "iid": iid_paths(r, n_paths, T, rng),
        "stationary": stationary_bootstrap_paths(r, n_paths, T, rng),
        "bayesian": bayesian_bootstrap_paths(r, n_paths, T, rng),
    }
    if regime_labels is not None:
        reg = regime_conditional_paths(
            r, np.asarray(regime_labels), n_paths, T, rng
        )
        if reg is not None:
            matrices["regime"] = reg

    frontiers = {
        name: _frontier_for_matrix(m, fractions, ruin_loss) for name, m in matrices.items()
    }

    # decision surface: worst-generator ruin (EXCLUDING the iid baseline —
    # it exists to be measured against, not to vote), bayesian median growth
    risk_gens = [g for g in frontiers if g != "iid"]
    combined = []
    for i, f in enumerate(fractions):
        worst_ruin = max(frontiers[g][i]["p_ruin"] for g in risk_gens)
        worst_dd50 = max(frontiers[g][i]["p_drawdown_below_50pct"] for g in risk_gens)
        combined.append(
            {
                "fraction": float(f),
                "median_terminal_wealth_bayes": frontiers["bayesian"][i][
                    "median_terminal_wealth"
                ],
                "p_ruin_worst": worst_ruin,
                "p_drawdown_below_50pct_worst": worst_dd50,
                "p_ruin_iid_baseline": frontiers["iid"][i]["p_ruin"],
            }
        )

    feasible = [row for row in combined if row["p_ruin_worst"] < ruin_prob_limit]
    constrained = (
        max(feasible, key=lambda x: x["median_terminal_wealth_bayes"]) if feasible else None
    )
    unconstrained = max(combined, key=lambda x: x["median_terminal_wealth_bayes"])

    # how much the old model understated risk at the chosen fraction
    understatement = None
    if constrained is not None:
        understatement = {
            "fraction": constrained["fraction"],
            "p_ruin_iid": constrained["p_ruin_iid_baseline"],
            "p_ruin_worst_model": constrained["p_ruin_worst"],
            "note": "iid is reported as the baseline it is; it never votes on size",
        }

    return {
        "kelly": kp,
        "fractions": [float(f) for f in fractions],
        "frontiers": frontiers,
        "combined": combined,
        "constrained_optimum": constrained,
        "unconstrained_optimum": unconstrained,
        "constraint": f"P(losing {ruin_loss:.0%}) < {ruin_prob_limit:.0%} under the "
        f"WORST of {sorted(risk_gens)}",
        "generators_used": sorted(matrices.keys()),
        "n_paths_per_generator": int(n_paths),
        "n_trades_per_path": int(T),
        "note": None
        if feasible
        else "NO fraction satisfies the ruin constraint under the worst model — "
        "this edge is unsizeable as measured",
    }
