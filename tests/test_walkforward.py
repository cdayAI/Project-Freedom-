import numpy as np
import pytest

from alpha_forge.gates.walkforward import HoldoutRegistry, purged_walk_forward_splits
from alpha_forge.ledger import Ledger


def test_purge_gap_respected():
    splits = purged_walk_forward_splits(n_obs=600, n_folds=5, purge=10)
    assert len(splits) == 5
    for train, test in splits:
        assert train.max() <= test.min() - 10
        assert np.all(np.diff(test) == 1)


def test_expanding_window_never_trains_on_future():
    splits = purged_walk_forward_splits(n_obs=300, n_folds=4, purge=5)
    for train, test in splits:
        assert train.max() < test.min()


def test_holdout_single_touch(tmp_path):
    ledger = Ledger(path=tmp_path / "ledger.jsonl")
    reg = HoldoutRegistry(n_obs=1000, holdout_fraction=0.2)
    assert reg.boundary == 800
    assert reg.research_indices().max() == 799
    idx = reg.unlock("strat_a", ledger, reason="final validation")
    assert idx.min() == 800 and idx.max() == 999
    with pytest.raises(PermissionError):
        reg.unlock("strat_a", ledger, reason="peeking again")
    # a different strategy gets its own single touch
    idx_b = reg.unlock("strat_b", ledger, reason="final validation")
    assert idx_b.size == 200
    assert len(ledger.holdout_touches()) == 2
