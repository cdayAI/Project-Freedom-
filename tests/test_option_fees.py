import pytest

from alpha_forge.costs.fees import (
    UnverifiedFeeError,
    option_buy_fees,
    option_sell_fees,
    orf_rate_on,
)


class TestOrf:
    def test_current_rates_by_exchange(self):
        assert orf_rate_on("CBOE", "2026-08-10") == pytest.approx(0.01248)
        assert orf_rate_on("NASDAQ_PHLX", "2026-07-15") == pytest.approx(0.008)
        assert orf_rate_on("BOX", "2026-07-01") == pytest.approx(0.022)

    def test_scheduled_change_applies_by_date(self):
        assert orf_rate_on("NYSE_AMERICAN", "2026-08-15") == pytest.approx(0.016)
        assert orf_rate_on("NYSE_AMERICAN", "2026-09-01") == pytest.approx(0.005)

    def test_pre_anchor_dates_raise(self):
        # as-of anchored rates carry no verified history; the ORF assessment
        # model itself changed 2026-07-01, so earlier dates must block
        with pytest.raises(UnverifiedFeeError):
            orf_rate_on("CBOE", "2026-06-01")
        with pytest.raises(UnverifiedFeeError):
            orf_rate_on("MIAX", "2025-01-15")

    def test_unknown_exchange_raises(self):
        with pytest.raises(UnverifiedFeeError):
            orf_rate_on("NOT_AN_EXCHANGE", "2026-08-10")


class TestOptionSellFees:
    def test_composition_2026(self):
        # 5 contracts, $2.50 premium each = $1,250 total premium, on CBOE
        fees = option_sell_fees(5, 1_250.0, "2026-08-10", exchange="CBOE")
        assert fees["finra_taf"] == pytest.approx(5 * 0.00329)   # 2026 rate, no cap
        assert fees["occ_clearing"] == pytest.approx(5 * 0.025)
        assert fees["orf"] == pytest.approx(5 * 0.01248)
        assert fees["sec_section31"] == pytest.approx(1_250 / 1e6 * 20.60)  # premium basis
        assert fees["total"] == pytest.approx(
            fees["finra_taf"] + fees["occ_clearing"] + fees["orf"] + fees["sec_section31"]
        )

    def test_occ_december_2025_fee_holiday(self):
        # the OCC schedule alone: December 2025 holiday, then reversion
        from alpha_forge.costs.fees import _schedule

        occ = _schedule("occ_clearing")
        assert occ.rate_on("2025-12-15") == 0.0
        assert occ.rate_on("2026-01-02") == pytest.approx(0.025)
        assert occ.rate_on("2025-11-28") == pytest.approx(0.025)

    def test_index_option_exemptions(self):
        fees = option_sell_fees(2, 10_000.0, "2026-08-10", exchange="CBOE",
                                is_index_option=True)
        assert fees["finra_taf"] == 0.0        # TAF-exempt (Schedule A Sec. 1)
        assert fees["sec_section31"] == 0.0    # 240.31(a)(11)(vi)
        assert fees["occ_clearing"] > 0.0      # clearing still applies
        assert fees["orf"] > 0.0

    def test_pre_2025_occ_blocks_historical_backtests(self):
        with pytest.raises(UnverifiedFeeError):
            option_sell_fees(1, 100.0, "2024-06-03", exchange="CBOE")

    def test_buy_side_has_no_taf_or_sec31(self):
        fees = option_buy_fees(3, "2026-08-10", exchange="CBOE_EDGX")
        assert set(fees) == {"occ_clearing", "orf", "total"}
        assert fees["orf"] == pytest.approx(3 * 0.00286)
