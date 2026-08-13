import numpy as np
import pytest

from alpha_forge.research.eventbt import EventPanel, fold_hazard_weights


def _panel(D=400, S=80, seed=0):
    rng = np.random.default_rng(seed)
    px = np.full((D, S), 10.0)
    sig = rng.random((D, S)).astype(np.float32)
    noise = rng.random((D, S)).astype(np.float32)
    ep = EventPanel(
        dates=np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + D),
        symbols=[f"S{i}" for i in range(S)],
        open_=px, high=px * 1.01, low=px * 0.99, close=px,
        half_spread=np.full((D, S), 0.01),
        eligible=np.ones((D, S), dtype=bool),
        feat_pctl={"sig": sig, "noise": noise},
        sell_fee_prop=np.zeros((D, S)),
        overnight_gaps=np.zeros(10),
        feat_cols=["sig", "noise"],
    )
    return ep, rng


def _plant(ep, rng, feature: str, day_lo: int, day_hi: int, n: int,
           label: np.ndarray | None = None) -> np.ndarray:
    """Plant n labeled start-days in [day_lo, day_hi) with the named feature
    pushed into its top decile on those days. Distinct (day, symbol) cells."""
    D, S = ep.close.shape
    if label is None:
        label = np.zeros((D, S), dtype=bool)
    placed = 0
    while placed < n:
        d, s = int(rng.integers(day_lo, day_hi)), int(rng.integers(0, S))
        if label[d, s]:
            continue
        label[d, s] = True
        ep.feat_pctl[feature][d, s] = 0.9 + 0.1 * rng.random()
        placed += 1
    return label


def test_hazard_learns_predictive_feature():
    ep, rng = _panel()
    label = _plant(ep, rng, "sig", 0, 300, 120)
    w = fold_hazard_weights(ep, label, train_end_idx=350, purge=10,
                            rng=np.random.default_rng(1))
    assert w["sig"] > 0.5                     # predictive feature: strong +
    assert abs(w.get("noise", 0.0)) < w["sig"] / 3  # noise shrunk well below


def test_hazard_excludes_labels_beyond_maturation_cutoff():
    """THE maturation regression: labels planted ONLY after the cutoff
    (train_end_idx - purge) must be invisible — no pre-cutoff positives
    exist, so the trainer must refuse rather than peek forward."""
    ep, rng = _panel()
    label = _plant(ep, rng, "sig", 345, 400, 120)  # all beyond 350 - 10 = 340
    w = fold_hazard_weights(ep, label, train_end_idx=350, purge=10,
                            rng=np.random.default_rng(2))
    assert w == {}


def test_hazard_poison_labels_after_cutoff_do_not_train():
    """Pre-cutoff labels ride on 'sig'; a LARGER batch of post-cutoff labels
    rides on 'noise'. If the cutoff leaks, 'noise' dominates; if maturation
    is honored, 'sig' wins and 'noise' stays shrunk."""
    ep, rng = _panel(seed=3)
    label = _plant(ep, rng, "sig", 0, 300, 120)
    label = _plant(ep, rng, "noise", 341, 400, 300, label=label)  # poison
    w = fold_hazard_weights(ep, label, train_end_idx=350, purge=10,
                            rng=np.random.default_rng(4))
    assert w["sig"] > 0.5
    assert abs(w.get("noise", 0.0)) < w["sig"] / 3


def test_hazard_rejects_misaligned_label_matrix():
    """A label matrix pivoted over the wrong symbol set must fail loudly,
    never silently train on the wrong symbols' labels."""
    ep, rng = _panel()
    bad = np.zeros((ep.close.shape[0], ep.close.shape[1] - 3), dtype=bool)
    with pytest.raises(ValueError, match="does not match the panel"):
        fold_hazard_weights(ep, bad, train_end_idx=350, purge=10,
                            rng=np.random.default_rng(5))


def test_hazard_collapses_overlapping_same_symbol_starts():
    """Consecutive labeled start-days on one symbol describe ONE forward
    episode. 25 clusters of 4 consecutive days = 100 raw positive cells but
    only 25 independent observations — below the 30-positive floor, so the
    trainer must refuse rather than treat duplicates as sample size."""
    ep, rng = _panel(seed=6)
    D, S = ep.close.shape
    label = np.zeros((D, S), dtype=bool)
    for c in range(25):
        s = c % S
        d0 = int(rng.integers(0, 250))
        for d in range(d0, d0 + 4):  # 4-day cluster, one episode
            label[d, s] = True
            ep.feat_pctl["sig"][d, s] = 0.95
    assert int(label.sum()) >= 95  # raw cells would clear the floor
    w = fold_hazard_weights(ep, label, train_end_idx=350, purge=10,
                            rng=np.random.default_rng(7))
    assert w == {}


def test_hazard_needs_minimum_positives():
    ep, rng = _panel()
    label = _plant(ep, rng, "sig", 0, 300, 120)
    thin = np.zeros_like(label)
    thin[:20, 0] = label[:20, 0]  # a handful at most
    w = fold_hazard_weights(ep, thin, train_end_idx=350, purge=10,
                            rng=np.random.default_rng(8))
    assert w == {}
