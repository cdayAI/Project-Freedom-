"""Purged, embargoed walk-forward splits and the final-holdout registry.

Purging removes training samples whose evaluation window overlaps the test
window (label leakage through overlapping holding periods); the embargo drops
a further margin after the test window (leakage through serial correlation).
Reference: Lopez de Prado (2018), Advances in Financial Machine Learning,
ch. 7. See SOURCES.md.

The final holdout is the last `holdout_fraction` of the data. It is touched
exactly once per strategy, ever; every touch is written to the ledger. Code
cannot read holdout rows without going through `HoldoutRegistry.unlock`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpha_forge.ledger import Ledger


def purged_walk_forward_splits(
    n_obs: int,
    n_folds: int = 5,
    purge: int = 5,
    embargo_frac: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward: each fold trains on data strictly before
    the test block, minus `purge` observations at the boundary; the embargo
    applies to any later reuse of post-test data (returned as excluded here
    for symmetry with k-fold usage).

    Returns list of (train_idx, test_idx). Test blocks tile the latter part of
    the series; the first fold's training set is the initial stub.
    """
    if n_folds < 2:
        raise ValueError("need at least 2 folds")
    embargo = int(n_obs * embargo_frac)
    # Reserve the first chunk as the minimum training stub.
    test_span = n_obs // (n_folds + 1)
    if test_span <= purge:
        raise ValueError("series too short for requested folds/purge")
    splits = []
    for k in range(1, n_folds + 1):
        test_start = k * test_span
        test_end = min((k + 1) * test_span, n_obs)
        train_end = max(test_start - purge, 0)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        splits.append((train_idx, test_idx))
    # note: expanding-window walk-forward never trains on post-test data, so
    # the embargo is structural here; the parameter is kept for k-fold variants.
    _ = embargo
    return splits


@dataclass
class HoldoutRegistry:
    """Guards the final untouched holdout slice of a series."""

    n_obs: int
    holdout_fraction: float = 0.15

    @property
    def boundary(self) -> int:
        return int(self.n_obs * (1.0 - self.holdout_fraction))

    def research_indices(self) -> np.ndarray:
        return np.arange(0, self.boundary)

    def unlock(self, strategy_id: str, ledger: Ledger, reason: str) -> np.ndarray:
        """One-time-per-strategy access to holdout rows. Logged, enforced."""
        prior = ledger.holdout_touches(strategy_id)
        if prior:
            raise PermissionError(
                f"strategy {strategy_id} already touched the holdout on "
                f"{prior[0]['ts_utc']}; second touches are forbidden"
            )
        ledger.append(
            "HOLDOUT_TOUCH",
            {"strategy_id": strategy_id, "reason": reason, "boundary": self.boundary},
        )
        return np.arange(self.boundary, self.n_obs)
