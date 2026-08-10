import numpy as np
import polars as pl
import pytest

from alpha_forge.data.catalysts import attach_catalyst_features
from alpha_forge.data.regsho import LAG_DAYS, attach_short_features


def _features(sym="X", start=pl.date(2025, 1, 1), n=30) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [sym] * n,
            "date": pl.date_range(start, start + pl.duration(days=n - 1), "1d", eager=True),
            "valid": [True] * n,
        }
    )


class TestShortFeatures:
    def _regsho(self, sym="X", n=120, ratio=0.4):
        return pl.DataFrame(
            {
                "date": pl.date_range(
                    pl.date(2024, 9, 1), pl.date(2024, 9, 1) + pl.duration(days=n - 1),
                    "1d", eager=True,
                ),
                "symbol": [sym] * n,
                "short_ratio": [ratio] * n,
                "total_volume": [1e6] * n,
            }
        )

    def test_flat_ratio_gives_flat_mean_and_zero_free_z(self):
        feats = attach_short_features(_features(start=pl.date(2024, 12, 1)), self._regsho())
        f5 = feats["short_ratio_5d"].fill_nan(None).drop_nulls()
        assert f5.len() > 0
        assert f5.to_numpy() == pytest.approx(0.4, abs=1e-12)
        # constant series has zero std -> z stays NaN rather than inf
        assert feats["short_ratio_z"].fill_nan(None).drop_nulls().len() == 0

    def test_lag_discipline_no_same_day_file(self):
        """A spike on day D must not appear in day-D features (LAG_DAYS=1)."""
        sho = self._regsho(n=100, ratio=0.3)
        # spike the LAST archive day
        sho = sho.with_columns(
            pl.when(pl.col("date") == sho["date"].max())
            .then(1.0)
            .otherwise(pl.col("short_ratio"))
            .alias("short_ratio")
        )
        spike_day = sho["date"].max()
        feats = attach_short_features(
            _features(start=spike_day, n=3), sho
        )
        row0 = feats.filter(pl.col("date") == spike_day)
        # day-D feature uses files through D-1 only: spike excluded
        assert row0["short_ratio_5d"][0] == pytest.approx(0.3, abs=1e-9)
        row1 = feats.filter(pl.col("date") == spike_day + pl.duration(days=LAG_DAYS))
        assert row1["short_ratio_5d"][0] > 0.3  # spike enters the next day

    def test_missing_symbol_is_nan(self):
        feats = attach_short_features(_features(sym="ZZZZ"), self._regsho(sym="X"))
        assert feats["short_ratio_5d"].fill_nan(None).drop_nulls().len() == 0


class TestCatalystFeaturesV2:
    def _catalog(self):
        rows = []
        # quarterly earnings 8-Ks: cadence 91 days
        for i, d in enumerate(["2024-01-10", "2024-04-10", "2024-07-10", "2024-10-09"]):
            rows.append({"symbol": "X", "form": "8-K", "filing_date": d,
                         "items": "2.02", "is_earnings": True})
        rows.append({"symbol": "X", "form": "4", "filing_date": "2024-12-01",
                     "items": "", "is_earnings": False})
        rows.append({"symbol": "X", "form": "424B5", "filing_date": "2024-11-15",
                     "items": "", "is_earnings": False})
        return pl.DataFrame(rows).with_columns(pl.col("filing_date").str.to_date())

    def test_expected_earnings_clock(self):
        feats = attach_catalyst_features(_features(start=pl.date(2025, 1, 1)), self._catalog())
        row = feats.filter(pl.col("date") == pl.date(2025, 1, 1))
        # last earnings 2024-10-09, median gap 91d -> expected 2025-01-08 -> +7
        assert row["days_until_expected_earnings"][0] == pytest.approx(7.0)
        late = feats.filter(pl.col("date") == pl.date(2025, 1, 20))
        assert late["days_until_expected_earnings"][0] == pytest.approx(-12.0)  # overdue

    def test_insider_and_dilution_recency(self):
        feats = attach_catalyst_features(_features(start=pl.date(2025, 1, 1)), self._catalog())
        row = feats.filter(pl.col("date") == pl.date(2025, 1, 1))
        assert row["days_since_form4"][0] == pytest.approx(31.0)
        assert row["days_since_dilution_filing"][0] == pytest.approx(47.0)
        assert row["n_form4_90d"][0] == pytest.approx(1.0)

    def test_cadence_requires_min_history(self):
        cat = self._catalog().filter(
            ~((pl.col("form") == "8-K") & (pl.col("filing_date") == pl.date(2024, 1, 10)))
        )  # only 3 earnings left
        feats = attach_catalyst_features(_features(start=pl.date(2025, 1, 1)), cat)
        assert feats["days_until_expected_earnings"].fill_nan(None).drop_nulls().len() == 0

    def test_legacy_catalog_without_form_column(self):
        cat = self._catalog().drop("form").filter(pl.col("is_earnings"))
        feats = attach_catalyst_features(_features(start=pl.date(2025, 1, 1)), cat)
        # legacy rows treated as 8-K: earnings features still work
        assert feats["days_since_earnings_8k"].fill_nan(None).drop_nulls().len() > 0
        assert feats["days_since_form4"].fill_nan(None).drop_nulls().len() == 0
