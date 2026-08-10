import numpy as np
import polars as pl
import pytest

from alpha_forge.ledger import Ledger
from alpha_forge.research.capacity import (
    edge_half_life,
    strategy_capacity,
    tier_report,
)
from alpha_forge.research.council import review_strategy


def _panel(adv_dollars: float, n_days: int = 100) -> pl.DataFrame:
    price = 10.0
    vol = adv_dollars / price
    return pl.DataFrame(
        {
            "symbol": ["X"] * n_days,
            "date": pl.date_range(
                pl.date(2025, 1, 1), pl.date(2025, 1, 1) + pl.duration(days=n_days - 1),
                interval="1d", eager=True,
            ),
            "open": [price] * n_days,
            "high": [price * 1.01] * n_days,
            "low": [price * 0.99] * n_days,
            "close": [price] * n_days,
            "volume": [vol] * n_days,
        }
    )


class TestCapacity:
    def test_capacity_from_adv(self):
        panel = _panel(adv_dollars=1_000_000)
        cap = strategy_capacity(panel, ["X"], n_concurrent_positions=10)
        # 1% of $1M ADV = $10k per name x 10 positions = $100k
        assert cap["capacity_usd"] == pytest.approx(100_000, rel=0.01)

    def test_thin_name_small_capacity(self):
        panel = _panel(adv_dollars=50_000)
        cap = strategy_capacity(panel, ["X"], n_concurrent_positions=5)
        assert cap["capacity_usd"] == pytest.approx(2_500, rel=0.01)
        # a $2k account fits; a $200k account does not — capacity expiration
        assert cap["capacity_usd"] > 2_000
        assert cap["capacity_usd"] < 200_000


class TestHalfLife:
    def test_insufficient_data_assumes_decay(self):
        out = edge_half_life(np.arange(3), np.array([0.1, 0.09, 0.08]))
        assert out["half_life_bars"] is None
        assert out["assumed_decaying"]

    def test_exponential_decay_recovered(self):
        t = np.arange(20)
        edge = 0.10 * np.exp(-0.0693 * t)  # half-life ~10 bars
        out = edge_half_life(t, edge)
        assert out["half_life_bars"] == pytest.approx(10.0, rel=0.05)


class TestTierMap:
    def test_empty_book_warns_all_tiers(self):
        panel = _panel(1_000_000)
        rep = tier_report(2_000, [], panel)
        assert rep["current_tier"] == "under_25k"
        assert rep["warning"] is not None

    def test_capacity_expiration_mapped(self):
        panel = _panel(1_000_000)
        book = [{"strategy_id": "s1", "capacity_usd": 20_000}]
        rep = tier_report(2_000, book, panel)
        assert rep["book_capacity"][0]["dies_at_tier"] == "under_25k"


class TestCouncil:
    def test_kill_on_pbo(self, tmp_path):
        ledger = Ledger(path=tmp_path / "l.jsonl")
        rng = np.random.default_rng(0)
        v = review_strategy(
            "s1", rng.normal(0.01, 0.02, 100), current_pbo=0.6, ledger=ledger,
            live_calibration_drift_days=0, live_win_flags=None,
        )
        assert v["kill"]
        assert any("PBO" in r for r in v["reasons"])
        assert any(e["kind"] == "KILL" for e in ledger.entries())

    def test_kill_on_calibration_drift(self, tmp_path):
        ledger = Ledger(path=tmp_path / "l.jsonl")
        rng = np.random.default_rng(1)
        # strong returns so DSR alone won't trigger with N=1 trials
        reg = ledger.preregister("h", "u", {})
        ledger.record_trial(reg, {}, 0.5)
        ledger.record_trial(reg, {}, 0.5)
        v = review_strategy(
            "s2", rng.normal(0.05, 0.02, 200), current_pbo=0.1, ledger=ledger,
            live_calibration_drift_days=35, live_win_flags=None,
        )
        assert v["kill"]
        assert any("calibration" in r for r in v["reasons"])

    def test_survivor_gets_live_only_projection(self, tmp_path):
        ledger = Ledger(path=tmp_path / "l.jsonl")
        reg = ledger.preregister("h", "u", {})
        ledger.record_trial(reg, {}, 0.5)
        ledger.record_trial(reg, {}, 0.5)
        rng = np.random.default_rng(2)
        wins = (rng.random(50) < 0.6).astype(float)
        v = review_strategy(
            "s3", rng.normal(0.08, 0.02, 300), current_pbo=0.1, ledger=ledger,
            live_calibration_drift_days=0, live_win_flags=wins,
        )
        assert not v["kill"]
        pw = v["projected_win_rate"]
        assert pw["source"] == "live reconciliation only"
        assert pw["ci95"][0] < pw["point"] < pw["ci95"][1]
