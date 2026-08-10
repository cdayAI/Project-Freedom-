"""Probability of Backtest Overfitting via CSCV.

Reference: Bailey, Borwein, Lopez de Prado & Zhu (2015), "The Probability of
Backtest Overfitting", Journal of Computational Finance. See SOURCES.md.

Combinatorially Symmetric Cross-Validation with S=16 contiguous blocks:
for each of C(16,8)=12,870 in-sample block subsets, pick the config with the
best in-sample Sharpe and record its relative out-of-sample rank. PBO is the
fraction of splits where the in-sample winner lands in the bottom half
out-of-sample (logit lambda <= 0).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from alpha_forge.config import CSCV_BLOCKS


def cscv_pbo(returns_matrix: np.ndarray, n_blocks: int = CSCV_BLOCKS) -> dict:
    """returns_matrix: shape (T, N) — T periods, N strategy configurations.

    Every column must be a config actually evaluated (and ledgered as a
    trial). Feeding a pruned subset of configs understates PBO.
    """
    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T periods x N configs)")
    t, n = m.shape
    if n < 2:
        raise ValueError("need at least 2 configurations for PBO")
    if t < n_blocks * 2:
        raise ValueError(f"need at least {n_blocks * 2} periods for {n_blocks} blocks")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even")

    # Contiguous blocks preserve serial structure inside each block.
    bounds = np.linspace(0, t, n_blocks + 1, dtype=int)
    blk_sum = np.empty((n_blocks, n))
    blk_sumsq = np.empty((n_blocks, n))
    blk_cnt = np.empty(n_blocks)
    for b in range(n_blocks):
        seg = m[bounds[b] : bounds[b + 1]]
        blk_sum[b] = seg.sum(axis=0)
        blk_sumsq[b] = (seg**2).sum(axis=0)
        blk_cnt[b] = seg.shape[0]

    def sharpe_from_blocks(idx: tuple[int, ...]) -> np.ndarray:
        cnt = blk_cnt[list(idx)].sum()
        s = blk_sum[list(idx)].sum(axis=0)
        ss = blk_sumsq[list(idx)].sum(axis=0)
        mean = s / cnt
        var = (ss - cnt * mean**2) / (cnt - 1)
        sd = np.sqrt(np.maximum(var, 1e-300))
        return mean / sd

    all_blocks = frozenset(range(n_blocks))
    lambdas = []
    for is_idx in combinations(range(n_blocks), n_blocks // 2):
        oos_idx = tuple(sorted(all_blocks - set(is_idx)))
        sr_is = sharpe_from_blocks(is_idx)
        sr_oos = sharpe_from_blocks(oos_idx)
        best = int(np.argmax(sr_is))
        # ascending relative rank of the IS winner in the OOS distribution
        rank = float((sr_oos < sr_oos[best]).sum() + 1)
        omega = rank / (n + 1)
        lambdas.append(np.log(omega / (1.0 - omega)))

    lam = np.asarray(lambdas)
    return {
        "pbo": float((lam <= 0).mean()),
        "n_splits": int(lam.size),
        "lambda_mean": float(lam.mean()),
        "n_configs": int(n),
        "n_periods": int(t),
        "n_blocks": int(n_blocks),
    }
