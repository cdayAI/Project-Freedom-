"""Offline tests for the SEC DERA insider-transactions feed.

Synthetic frames only — no network. Mirrors the raw DERA column names so
aggregate_form345 is exercised on the exact schema the ZIPs deliver
(including the DD-MON-YYYY filing-date strings)."""

from datetime import date

import polars as pl
import pytest

from alpha_forge.data.insider import (
    INSIDER_FEATURES,
    aggregate_form345,
    attach_insider_features,
)


def _features(sym="X", start=pl.date(2025, 1, 1), n=30) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [sym] * n,
            "date": pl.date_range(start, start + pl.duration(days=n - 1), "1d", eager=True),
            "valid": [True] * n,
        }
    )


class TestAggregation:
    def _sub(self):
        return pl.DataFrame(
            {
                "ACCESSION_NUMBER": ["A1", "A2", "A3", "A4", "A5", "A6"],
                "FILING_DATE": [
                    "15-JAN-2025",  # A1: clean form 4, lowercase ticker
                    "15-JAN-2025",  # A2: junk ticker -> dropped
                    "16-JAN-2025",  # A3: amendment -> dropped
                    "17-JAN-2025",  # A4: form 3, no transactions
                    "15-JAN-2025",  # A5: same symbol+date as A1, aggregates in
                    "15-JAN-2025",  # A6: "NONE" placeholder ticker -> dropped
                ],
                "DOCUMENT_TYPE": ["4", "4", "4/A", "3", "4", "4"],
                "ISSUERTRADINGSYMBOL": ["abc", "NYSE: KRC", "ABC", "ABC", "ABC", "none"],
            }
        )

    def _trans(self):
        rows = [
            # A1: two open-market buys, one open-market sale
            ("A1", "P", "A", "100", "10.0"),   # +1000
            ("A1", "P", "A", "50", "20.0"),    # +1000
            ("A1", "S", "D", "30", "10.0"),    # -300
            # A1: noise that must be ignored
            ("A1", "M", "A", "999", "1.0"),    # option exercise
            ("A1", "F", "D", "999", "1.0"),    # tax withholding
            ("A1", "P", "A", "77", None),      # null price -> skipped
            ("A1", "P", "D", "500", "10.0"),   # code/flag mismatch -> dropped
            ("A1", "S", "A", "500", "10.0"),   # code/flag mismatch -> dropped
            # A2: real P but junk symbol -> dropped via submission filter
            ("A2", "P", "A", "10", "10.0"),
            # A3: amendment restating a huge buy -> dropped
            ("A3", "P", "A", "100000", "10.0"),
            # A1: filer error — total value in the price field -> dropped
            ("A1", "P", "A", "100149060", "24035774.4"),
            # A5: one more sale for the same (ABC, 2025-01-15) bucket
            ("A5", "S", "D", "20", "5.0"),     # -100
            # A6: valid P under the placeholder ticker -> dropped
            ("A6", "P", "A", "10", "10.0"),
        ]
        return pl.DataFrame(
            {
                "ACCESSION_NUMBER": [r[0] for r in rows],
                "TRANS_CODE": [r[1] for r in rows],
                "TRANS_ACQUIRED_DISP_CD": [r[2] for r in rows],
                "TRANS_SHARES": [r[3] for r in rows],
                "TRANS_PRICEPERSHARE": [r[4] for r in rows],
            }
        )

    def test_net_buy_math(self):
        agg = aggregate_form345(self._sub(), self._trans())
        assert agg.height == 1  # everything else filtered away
        row = agg.row(0, named=True)
        assert row["symbol"] == "ABC"  # upper-cased
        assert row["filing_date"] == date(2025, 1, 15)
        assert row["buy_usd"] == pytest.approx(2000.0)
        assert row["sell_usd"] == pytest.approx(400.0)  # 300 (A1) + 100 (A5)
        assert row["net_buy_usd"] == pytest.approx(1600.0)
        assert row["n_buy_txns"] == 2
        assert row["n_sell_txns"] == 2

    def test_schema(self):
        agg = aggregate_form345(self._sub(), self._trans())
        assert agg.columns == [
            "symbol", "filing_date", "net_buy_usd", "buy_usd", "sell_usd",
            "n_buy_txns", "n_sell_txns",
        ]


class TestAttachInsiderFeatures:
    def _insider(self, rows):
        """rows: (symbol, 'YYYY-MM-DD', buy_usd, sell_usd)"""
        return pl.DataFrame(
            {
                "symbol": [r[0] for r in rows],
                "filing_date": [r[1] for r in rows],
                "buy_usd": [float(r[2]) for r in rows],
                "sell_usd": [float(r[3]) for r in rows],
            }
        ).with_columns(
            pl.col("filing_date").str.to_date(),
            (pl.col("buy_usd") - pl.col("sell_usd")).alias("net_buy_usd"),
        )

    def test_ex_ante_future_filing_excluded(self):
        ins = self._insider([("X", "2025-01-10", 1000, 0), ("X", "2025-02-15", 5000, 0)])
        feats = attach_insider_features(_features(start=pl.date(2025, 2, 10), n=10), ins)
        before = feats.filter(pl.col("date") == pl.date(2025, 2, 14))
        assert before["insider_net_buy_90d_usd"][0] == pytest.approx(1000.0)
        # filing dated D is knowable from D onward (knowledge date = filing date)
        on_day = feats.filter(pl.col("date") == pl.date(2025, 2, 15))
        assert on_day["insider_net_buy_90d_usd"][0] == pytest.approx(6000.0)

    def test_90d_window_boundary(self):
        ins = self._insider([("X", "2025-01-01", 1000, 0)])
        feats = attach_insider_features(_features(start=pl.date(2025, 3, 30), n=4), ins)
        # 2025-03-31 is 89 days after filing: still inside (D-90, D]
        inside = feats.filter(pl.col("date") == pl.date(2025, 3, 31))
        assert inside["insider_net_buy_90d_usd"][0] == pytest.approx(1000.0)
        # 2025-04-01 is exactly 90 days after: window has slid past it
        outside = feats.filter(pl.col("date") == pl.date(2025, 4, 1))
        assert outside["insider_net_buy_90d_usd"][0] == pytest.approx(0.0)

    def test_buy_ratio(self):
        ins = self._insider([("X", "2025-01-10", 3000, 1000)])
        feats = attach_insider_features(_features(start=pl.date(2025, 1, 10), n=5), ins)
        assert feats["insider_buy_ratio_90d"].to_list() == pytest.approx([0.75] * 5)

    def test_buy_ratio_nan_when_no_activity(self):
        # archive covers X from January; by June the 90d window is empty
        ins = self._insider([("X", "2025-01-10", 1000, 0)])
        feats = attach_insider_features(_features(start=pl.date(2025, 6, 1), n=5), ins)
        assert feats["insider_buy_ratio_90d"].fill_nan(None).drop_nulls().len() == 0
        # a covered symbol with an empty window is known-quiet: net buy is 0
        assert feats["insider_net_buy_90d_usd"].to_list() == pytest.approx([0.0] * 5)

    def test_missing_symbol_is_nan(self):
        ins = self._insider([("X", "2025-01-10", 1000, 0)])
        feats = attach_insider_features(_features(sym="ZZZZ", start=pl.date(2025, 2, 1)), ins)
        for c in INSIDER_FEATURES:
            assert feats[c].fill_nan(None).drop_nulls().len() == 0

    def test_pre_era_is_nan(self):
        # panel dates before the earliest filing in the archive: unknowable
        ins = self._insider([("X", "2025-01-10", 1000, 0)])
        feats = attach_insider_features(_features(start=pl.date(2024, 12, 1), n=10), ins)
        for c in INSIDER_FEATURES:
            assert feats[c].fill_nan(None).drop_nulls().len() == 0

    def test_none_archive_gives_null_columns(self):
        feats = attach_insider_features(_features(), None)
        for c in INSIDER_FEATURES:
            assert c in feats.columns
            assert feats[c].fill_nan(None).drop_nulls().len() == 0
