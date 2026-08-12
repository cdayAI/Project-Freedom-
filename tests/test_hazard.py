import numpy as np

from alpha_forge.research.eventbt import EventPanel, fold_hazard_weights


def _panel_with_predictive_feature(D=400, S=80, seed=0):
    """Feature 'sig' percentile HIGH on days that start a labeled run-up;
    feature 'noise' independent. Labels only in the first 300 days."""
    rng = np.random.default_rng(seed)
    px = np.full((D, S), 10.0)
    label = np.zeros((D, S), dtype=bool)
    sig = rng.random((D, S)).astype(np.float32)
    for _ in range(120):
        d, s = rng.integers(0, 300), rng.integers(0, S)
        label[d, s] = True
        sig[d, s] = 0.9 + 0.1 * rng.random()  # hits sit in the top decile
    ep = EventPanel(
        dates=np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + D),
        symbols=[f"S{i}" for i in range(S)],
        open_=px, high=px * 1.01, low=px * 0.99, close=px,
        half_spread=np.full((D, S), 0.01),
        eligible=np.ones((D, S), dtype=bool),
        feat_pctl={"sig": sig, "noise": rng.random((D, S)).astype(np.float32)},
        sell_fee_prop=np.zeros((D, S)),
        overnight_gaps=np.zeros(10),
        feat_cols=["sig", "noise"],
    )
    return ep, label


def test_hazard_learns_predictive_feature():
    ep, label = _panel_with_predictive_feature()
    w = fold_hazard_weights(ep, label, train_end_idx=350, purge=10,
                            rng=np.random.default_rng(1))
    assert w["sig"] > 0.5                     # predictive feature: strong +
    assert abs(w.get("noise", 0.0)) < w["sig"] / 3  # noise shrunk well below


def test_hazard_respects_label_maturation():
    ep, label = _panel_with_predictive_feature()
    # labels live in days < 300; a cutoff before any labels -> no signal
    w = fold_hazard_weights(ep, np.zeros_like(label), train_end_idx=350,
                            purge=10, rng=np.random.default_rng(2))
    assert w == {}  # no positives -> refuses to train


def test_hazard_needs_minimum_positives():
    ep, label = _panel_with_predictive_feature()
    thin = np.zeros_like(label)
    thin[:20, 0] = label[:20, 0]  # a handful at most
    w = fold_hazard_weights(ep, thin, train_end_idx=350, purge=10,
                            rng=np.random.default_rng(3))
    assert w == {}
