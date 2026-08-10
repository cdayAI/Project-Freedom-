import numpy as np
import pytest

from alpha_forge.research.traps import (
    Trade,
    check_cash_settlement,
    check_pdt,
    integer_lot_weights,
    stop_fill_price,
    stress_overnight_gaps,
)


class TestPDT:
    def test_violation_under_25k(self):
        trades = [Trade(entry_day=d, exit_day=d, notional=500) for d in (10, 11, 12, 13)]
        res = check_pdt(trades, account_equity=2_000)
        assert not res.feasible
        assert "flagged" in res.violations[0]

    def test_three_day_trades_ok(self):
        trades = [Trade(entry_day=d, exit_day=d, notional=500) for d in (10, 11, 12)]
        assert check_pdt(trades, account_equity=2_000).feasible

    def test_spread_out_day_trades_ok(self):
        trades = [Trade(entry_day=d, exit_day=d, notional=500) for d in (10, 20, 30, 40)]
        assert check_pdt(trades, account_equity=2_000).feasible

    def test_25k_exempt(self):
        trades = [Trade(entry_day=d, exit_day=d, notional=500) for d in range(10, 20)]
        assert check_pdt(trades, account_equity=25_000).feasible

    def test_swing_trades_never_pdt(self):
        trades = [Trade(entry_day=d, exit_day=d + 21, notional=500) for d in range(10, 20)]
        assert check_pdt(trades, account_equity=2_000).feasible


class TestCashSettlement:
    def test_gfv_detected(self):
        # sell A on day 5 (settles day 6), buy B day 5 with those unsettled
        # proceeds, sell B day 5 (before day-6 settlement) => GFV
        trades = [
            Trade(entry_day=0, exit_day=5, notional=2_000, symbol="A"),
            Trade(entry_day=5, exit_day=5, notional=2_000, symbol="B"),
        ]
        res = check_cash_settlement(trades, starting_cash=2_000)
        assert not res.feasible
        assert "good-faith" in res.violations[0]

    def test_waiting_for_settlement_ok(self):
        trades = [
            Trade(entry_day=0, exit_day=5, notional=2_000, symbol="A"),
            Trade(entry_day=5, exit_day=8, notional=2_000, symbol="B"),  # sells after T+1
        ]
        assert check_cash_settlement(trades, starting_cash=2_000).feasible

    def test_free_riding_detected(self):
        trades = [Trade(entry_day=0, exit_day=5, notional=5_000, symbol="A")]
        res = check_cash_settlement(trades, starting_cash=2_000)
        assert not res.feasible
        assert "free-riding" in res.violations[0]


class TestIntegerLots:
    def test_whole_share_friction_at_2k(self):
        # $2k across 10 names incl. a $500 stock: $200/name can't buy it
        prices = np.array([500.0, 50.0, 20.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0])
        w = np.full(10, 0.1)
        res = integer_lot_weights(2_000, prices, w, fractional_ok=False)
        assert res["infeasible_names"] == 1
        assert res["tracking_error_l1"] > 0
        assert res["unallocated_cash"] > 0

    def test_fractional_removes_friction(self):
        prices = np.array([500.0, 50.0])
        w = np.array([0.5, 0.5])
        res = integer_lot_weights(2_000, prices, w, fractional_ok=True)
        assert res["tracking_error_l1"] == 0.0
        assert res["infeasible_names"] == 0


class TestStressFills:
    def test_gap_through_stop_fills_at_open(self):
        assert stop_fill_price(stop=10.0, next_open=8.0, low_after=7.0) == 8.0

    def test_normal_stop_fills_at_stop(self):
        assert stop_fill_price(stop=10.0, next_open=10.5, low_after=9.8) == 10.0

    def test_untriggered_stop_is_nan(self):
        assert np.isnan(stop_fill_price(stop=10.0, next_open=10.5, low_after=10.2))

    def test_gap_injection_worsens_returns(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.01, 0.02, 200)
        gaps = rng.normal(-0.001, 0.05, 5000)
        stressed = stress_overnight_gaps(r, gaps, seed=1)
        assert stressed.mean() < r.mean()  # worst-tail gaps only hurt
        assert stressed.shape == r.shape
