import numpy as np
import pytest

from alpha_forge.research.eventbt import (
    EventPanel,
    composite_score,
    fold_directions,
    run_event_strategy,
    simulate_trade,
)
from alpha_forge.research.features import FEATURE_NAMES


def _panel(close_rows: np.ndarray, open_rows: np.ndarray | None = None) -> EventPanel:
    """Tiny hand-built panel: (D, S) closes; opens default to prior close."""
    C = close_rows.astype(float)
    D, S = C.shape
    O = open_rows.astype(float) if open_rows is not None else np.vstack([C[:1], C[:-1]])
    H = np.maximum(O, C) * 1.001
    L = np.minimum(O, C) * 0.999
    dates = np.arange(np.datetime64("2024-01-02"), np.datetime64("2024-01-02") + np.timedelta64(D, "D"))
    return EventPanel(
        dates=dates, symbols=[f"S{i}" for i in range(S)],
        open_=O, high=H, low=L, close=C,
        half_spread=np.full((D, S), 0.001),
        eligible=np.ones((D, S), dtype=bool),
        feat_pctl={f: np.full((D, S), 0.5, dtype=np.float32) for f in FEATURE_NAMES},
        sell_fee_prop=np.full((D, S), 1e-5),
        overnight_gaps=np.array([-0.01, 0.0, 0.01]),
    )


class TestSimulateTrade:
    def test_target_hit_fills_at_target(self):
        D = 20
        C = np.full((D, 1), 10.0)
        C[5:, 0] = 25.0  # jumps after entry
        ep = _panel(C)
        # signal day 2 -> entry open day 3 (= close day 2 = 10) * (1+hs)
        tr = simulate_trade(ep, 0, 2, target_mult=2.0, stop_frac=None)
        assert tr is not None
        assert tr.exit_reason in ("target", "target_gap")
        # entry ~10.01, target ~20.02; net must be close to +100% minus costs
        assert 0.65 < tr.net_log_ret < 0.72  # log(2) minus costs

    def test_gap_through_stop_fills_at_open(self):
        D = 20
        C = np.full((D, 1), 10.0)
        C[6:, 0] = 3.0  # collapse
        O = np.vstack([C[:1], C[:-1]])
        O[6, 0] = 3.5  # overnight gap far below the 50% stop (5.0)
        ep = _panel(C, O)
        tr = simulate_trade(ep, 0, 2, 5.0, 0.5)
        assert tr is not None
        assert tr.exit_reason == "stop_gap"
        # fill at the gapped open (3.5), NOT at the stop price (5.0)
        assert tr.exit_px < 5.0 * 0.999

    def test_time_exit_at_open(self):
        D = 200
        C = np.full((D, 1), 10.0)  # nothing ever happens
        ep = _panel(C)
        tr = simulate_trade(ep, 0, 2, 5.0, None)
        assert tr is not None
        assert tr.exit_reason == "time"
        assert tr.holding_days == 126
        # flat price: net is just the round-trip cost drag (negative)
        assert -0.01 < tr.net_log_ret < 0.0

    def test_halt_extends_hold(self):
        D = 200
        C = np.full((D, 1), 10.0)
        ep = _panel(C)
        ep.open_[10:15, 0] = np.nan  # halt week
        tr = simulate_trade(ep, 0, 8, 5.0, 0.5)
        assert tr is not None  # survives the halt, exits later

    def test_gap_above_target_fills_at_open_not_stop(self):
        # bar opens ABOVE target then collapses through the stop intraday:
        # the resting limit filled at the open, chronologically before any
        # intraday decline — must be target_gap at the open, never a stop
        D = 30
        C = np.full((D, 1), 10.0)
        C[6:, 0] = 4.0
        O = np.vstack([C[:1], C[:-1]])
        O[6, 0] = 55.0  # gap far above the 2x target (~20)
        ep = _panel(C, O)
        ep.low[6, 0] = 4.0  # collapses through the 50% stop (~5) intraday
        tr = simulate_trade(ep, 0, 2, 2.0, 0.5)
        assert tr is not None
        assert tr.exit_reason == "target_gap"
        assert tr.exit_px > 50.0 * 0.99  # filled at the gapped open

    def test_costs_always_reduce_exit(self):
        D = 40
        C = np.full((D, 1), 10.0)
        C[5:, 0] = 30.0
        ep = _panel(C)
        tr = simulate_trade(ep, 0, 2, 2.0, None)
        raw_multiple = tr.exit_px / tr.entry_px
        assert raw_multiple < 2.0  # spreads + fees bite below the clean 2x


class TestPortfolio:
    def test_no_reentry_while_held(self):
        D = 300
        S = 3
        C = np.full((D, S), 10.0)
        ep = _panel(C)
        score = np.full((D, S), np.nan, dtype=np.float32)
        score[:, 0] = 1.0  # symbol 0 always screams "enter"
        score[:, 1:] = 0.1
        cfg = {"entry_pct": 0.5, "target_mult": 5.0, "stop_frac": None}
        res = run_event_strategy(ep, score, cfg, (0, 290))
        # entries only when symbol 0 is not already held: at most one open
        # position in it at any time
        intervals = [(t.entry_day, t.exit_day) for t in res["trades"] if t.symbol_idx == 0]
        for (e1, x1), (e2, x2) in zip(intervals, intervals[1:]):
            assert e2 > x1  # strictly after the previous exit

    def test_daily_returns_shape_and_flat_market(self):
        D = 60
        C = np.full((D, 2), 10.0)
        ep = _panel(C)
        score = np.full((D, 2), 0.9, dtype=np.float32)
        cfg = {"entry_pct": 0.5, "target_mult": 5.0, "stop_frac": None}
        res = run_event_strategy(ep, score, cfg, (0, 60))
        assert res["daily_net_returns"].size == 60
        # flat market: daily marks near zero except entry/exit cost days
        assert np.abs(res["daily_net_returns"]).max() < 0.01


class TestDirections:
    def _groups(self, high_hit: bool):
        rng = np.random.default_rng(0)
        groups = []
        for i in range(40):
            hit = np.full(len(FEATURE_NAMES), np.nan)
            ctrl = rng.normal(0, 1, (5, len(FEATURE_NAMES)))
            hit[0] = 3.0 if high_hit else -3.0  # first feature separates
            hit[1:] = rng.normal(0, 1, len(FEATURE_NAMES) - 1)
            groups.append((np.datetime64("2020-01-01") + np.timedelta64(i, "D"),
                           np.vstack([hit, ctrl])))
        return {5: groups}

    def test_direction_signs_learned(self):
        d = fold_directions(self._groups(True), 5, np.datetime64("2021-01-01"))
        assert d.get(FEATURE_NAMES[0]) == 1.0
        d = fold_directions(self._groups(False), 5, np.datetime64("2021-01-01"))
        assert d.get(FEATURE_NAMES[0]) == -1.0

    def test_no_future_hits_used(self):
        groups = self._groups(True)
        # cutoff before every hit date -> nothing to learn from
        d = fold_directions(groups, 5, np.datetime64("2019-01-01"))
        assert d == {}

    def test_composite_score_direction(self):
        C = np.full((10, 4), 10.0)
        ep = _panel(C)
        for f in FEATURE_NAMES:
            ep.feat_pctl[f][:, 0] = 0.99  # symbol 0 extreme-high in everything
            ep.feat_pctl[f][:, 1] = 0.01
        score = composite_score(ep, {FEATURE_NAMES[0]: 1.0, FEATURE_NAMES[1]: 1.0})
        assert score[5, 0] > score[5, 1]
        score_inv = composite_score(ep, {FEATURE_NAMES[0]: -1.0})
        assert score_inv[5, 1] > score_inv[5, 0]
