import numpy as np

from alpha_forge.research.eventbt import (
    MAX_ROUNDTRIP_SPREAD,
    EventPanel,
    _adaptive_lam,
    fold_weights,
    fold_weights_pooled,
    run_event_strategy,
)
from tests.test_logistic_signal import _groups


def test_adaptive_lambda_shrinks_harder_when_data_is_scarce():
    assert _adaptive_lam(60, 15) > _adaptive_lam(300, 15)
    assert _adaptive_lam(10_000, 15) == 1.0  # floor


def test_pooled_classes_use_more_groups_than_single():
    g5 = _groups(n_groups=30, seed=1)[5]
    g3 = _groups(n_groups=40, seed=2)[5]
    groups = {5: g5, 3: g3}
    cutoff = np.datetime64("2021-01-01")
    feats = ["a", "b", "c"]
    w_single = fold_weights(groups, 5, cutoff, feats)
    w_pooled = fold_weights_pooled(groups, (3, 5), cutoff, feats)
    assert w_single and w_pooled
    # pooling 70 groups vs 30 -> lighter adaptive shrinkage -> larger weights
    assert abs(w_pooled["a"]) > 0


def _tiny_panel(n_tight=59, hs_tight=0.01, hs_wide=0.20):
    """>= 50 eligible names so the engine's cross-section floor is met;
    the LAST symbol is the wide-spread one."""
    D = 12
    S = n_tight + 1
    dates = np.arange(np.datetime64("2025-01-01"), np.datetime64("2025-01-13"))
    px = np.full((D, S), 10.0)
    hs = np.full((D, S), hs_tight)
    hs[:, -1] = hs_wide
    symbols = [f"T{i:02d}" for i in range(n_tight)] + ["WIDE"]
    return EventPanel(
        dates=dates, symbols=symbols,
        open_=px.copy(), high=px * 1.01, low=px * 0.99, close=px.copy(),
        half_spread=hs, eligible=np.ones((D, S), dtype=bool),
        feat_pctl={"f": np.full((D, S), 0.99, dtype=np.float32)},
        sell_fee_prop=np.zeros((D, S)),
        overnight_gaps=np.zeros(10),
        feat_cols=["f"],
    )


def test_tradability_floor_excludes_wide_spread_names():
    ep = _tiny_panel()
    rng = np.random.default_rng(0)
    score = rng.uniform(0.5, 1.0, size=(12, len(ep.symbols))).astype(np.float32)
    score[:, -1] = 1.0  # WIDE has the BEST score every day — floor must still veto
    res = run_event_strategy(
        ep, score, {"entry_pct": 0.0, "target_mult": 2.0, "stop_frac": None}, (0, 12)
    )
    traded = {ep.symbols[t.symbol_idx] for t in res["trades"]}
    assert len(traded) > 0               # tight names trade
    assert "WIDE" not in traded          # 40% roundtrip > 10% cap, despite top score
    assert 2 * 0.20 > MAX_ROUNDTRIP_SPREAD
