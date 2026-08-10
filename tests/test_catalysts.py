from datetime import date

import numpy as np
import polars as pl

from alpha_forge.data.catalysts import attach_catalyst_features, tag_hits


def _catalog():
    return pl.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "filing_date": [date(2024, 3, 1), date(2024, 6, 3), date(2024, 3, 15)],
            "items": ["2.02,9.01", "8.01", "2.02"],
            "is_earnings": [True, False, True],
        }
    )


class TestTagHits:
    def _hits(self, symbol, entry):
        return pl.DataFrame(
            {"symbol": [symbol], "entry_date": [entry], "n_multiple": [5],
             "catalyst_class": ["NONE"]}
        )

    def test_earnings_within_window(self):
        # entry 2024-03-05: AAA's Item-2.02 8-K on 03-01 is 4 days prior
        out = tag_hits(self._hits("AAA", "2024-03-05"), _catalog())
        assert out["catalyst_class"][0] == "EARNINGS_8K"

    def test_other_8k_within_window(self):
        # entry 2024-06-01: AAA's 8.01 filing lands 2 days later (in window)
        out = tag_hits(self._hits("AAA", "2024-06-01"), _catalog())
        assert out["catalyst_class"][0] == "OTHER_8K"

    def test_none_when_no_nearby_filing(self):
        out = tag_hits(self._hits("AAA", "2024-12-01"), _catalog())
        assert out["catalyst_class"][0] == "NONE"

    def test_unknown_symbol_none(self):
        out = tag_hits(self._hits("ZZZ", "2024-03-05"), _catalog())
        assert out["catalyst_class"][0] == "NONE"

    def test_empty_catalog_passthrough(self):
        hits = self._hits("AAA", "2024-03-05")
        assert tag_hits(hits, None)["catalyst_class"][0] == "NONE"


class TestCatalystFeatures:
    def _features(self, symbol="AAA"):
        days = pl.date_range(date(2024, 2, 25), date(2024, 6, 10), "1d", eager=True)
        return pl.DataFrame(
            {"symbol": [symbol] * len(days), "date": days,
             "valid": [True] * len(days), "starts_5x_fwd": [False] * len(days),
             "price": [10.0] * len(days)}
        )

    def test_days_since_earnings(self):
        out = attach_catalyst_features(self._features(), _catalog())
        row = out.filter(pl.col("date") == date(2024, 3, 11))
        assert row["days_since_earnings_8k"][0] == 10.0  # 03-01 filing
        assert row["days_since_any_8k"][0] == 10.0

    def test_any_8k_updates_but_earnings_does_not(self):
        out = attach_catalyst_features(self._features(), _catalog())
        row = out.filter(pl.col("date") == date(2024, 6, 5))
        assert row["days_since_any_8k"][0] == 2.0        # 06-03 (8.01)
        assert row["days_since_earnings_8k"][0] == 96.0  # still 03-01

    def test_no_lookahead_before_first_filing(self):
        out = attach_catalyst_features(self._features(), _catalog())
        row = out.filter(pl.col("date") == date(2024, 2, 27))
        assert row["days_since_earnings_8k"][0] is None or np.isnan(
            row["days_since_earnings_8k"][0]
        )

    def test_filing_day_counts_as_zero(self):
        out = attach_catalyst_features(self._features(), _catalog())
        row = out.filter(pl.col("date") == date(2024, 3, 1))
        assert row["days_since_earnings_8k"][0] == 0.0

    def test_missing_catalog_adds_null_columns(self):
        out = attach_catalyst_features(self._features(), None)
        assert "days_since_earnings_8k" in out.columns
        assert out["days_since_earnings_8k"].null_count() == out.height
