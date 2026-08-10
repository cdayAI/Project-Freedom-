import numpy as np
import pytest

from alpha_forge.costs.engine import cost_drag_report
from alpha_forge.costs.fees import FeeSchedule, UnverifiedFeeError, equity_sell_fees
from alpha_forge.costs.slippage import corwin_schultz_spread, effective_half_spread


class TestDateAwareFees:
    def test_sec31_zero_window(self):
        sec = FeeSchedule.load("sec_section31")
        assert sec.rate_on("2025-06-01") == 0.0
        assert sec.rate_on("2026-04-03") == 0.0
        assert sec.rate_on("2026-04-04") == 20.60

    def test_sec31_historical_rates(self):
        sec = FeeSchedule.load("sec_section31")
        assert sec.rate_on("2014-06-02") == 22.10   # eff. 2014-03-18
        assert sec.rate_on("2018-06-01") == 13.00   # eff. 2018-05-22
        assert sec.rate_on("2021-03-01") == 5.10    # eff. 2021-02-25
        assert sec.rate_on("2024-12-31") == 27.80   # eff. 2024-05-22

    def test_predates_table_raises(self):
        sec = FeeSchedule.load("sec_section31")
        with pytest.raises(UnverifiedFeeError):
            sec.rate_on("2013-01-02")  # before earliest sourced entry

    def test_equity_sell_fee_composition(self):
        # 2024: SEC $27.80/M, TAF $0.000166/share cap $8.30
        fees = equity_sell_fees(notional_usd=10_000.0, shares=100, trade_date="2024-06-03")
        assert fees["sec_section31"] == pytest.approx(10_000 / 1e6 * 27.80)
        assert fees["finra_taf"] == pytest.approx(100 * 0.000166)
        assert fees["total"] == pytest.approx(fees["sec_section31"] + fees["finra_taf"])

    def test_taf_cap_binds_on_large_trades(self):
        fees = equity_sell_fees(notional_usd=5_000_000.0, shares=100_000, trade_date="2024-06-03")
        assert fees["finra_taf"] == pytest.approx(8.30)

    def test_unverified_table_blocks(self, tmp_path):
        with pytest.raises(UnverifiedFeeError):
            FeeSchedule.load("orf_by_exchange")  # not yet sourced -> no file -> blocked


class TestSpreadModel:
    def test_cs_estimator_recovers_synthetic_spread(self):
        # random walk observed through a known proportional spread s:
        # trade prices bounce between bid and ask around mid
        rng = np.random.default_rng(0)
        s = 0.02
        n = 4000
        mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
        half = s / 2
        high = mid * (1 + half)
        low = mid * (1 - half) * np.exp(-np.abs(rng.normal(0, 1e-4, n)))
        close = np.where(rng.random(n) > 0.5, high, low)
        est = corwin_schultz_spread(high, low, close)
        assert np.nanmedian(est) == pytest.approx(s, rel=0.5)  # right order of magnitude

    def test_half_spread_tick_floor(self):
        # a perfectly liquid synthetic series still pays half a tick
        n = 100
        c = np.full(n, 50.0)
        h = c * 1.0001
        l = c * 0.9999
        hs = effective_half_spread(h, l, c)
        assert np.all(hs[~np.isnan(hs)] >= 0.005 / 50.0 - 1e-15)


class TestCostDrag:
    def test_breakeven_math(self):
        # 10bp/day edge, 40bp round trip -> breakeven ~4 days
        g = np.full(50, 0.0101)  # ~1% per trade over 10 days
        h = np.full(50, 10.0)
        rep = cost_drag_report(g, h, cost_per_roundtrip=0.004)
        assert rep["net_total_return"] < rep["gross_total_return"]
        edge = rep["edge_log_per_holding_day"]
        assert rep["breakeven_holding_days"] == pytest.approx(
            -np.log1p(-0.004) / edge, rel=1e-9
        )
        assert rep["max_trades_per_year_before_edge_dies"] == pytest.approx(
            252 / rep["breakeven_holding_days"], rel=1e-9
        )

    def test_dead_edge_infinite_breakeven(self):
        g = np.full(30, -0.001)
        h = np.ones(30)
        rep = cost_drag_report(g, h, cost_per_roundtrip=0.002)
        assert rep["breakeven_holding_days"] == np.inf
