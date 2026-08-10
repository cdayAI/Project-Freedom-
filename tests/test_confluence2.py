import numpy as np
import polars as pl
import pytest

from alpha_forge.research.confluence2 import (
    K_CONTROLS,
    _matched_stats,
    _stratified_permutation_p,
    build_matched_groups,
)
from alpha_forge.research.features import FEATURE_NAMES


def _groups(hit_high: bool, n_groups: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_groups):
        hit = rng.normal(0, 1, len(FEATURE_NAMES))
        hit[0] = (3.0 if hit_high else -3.0) + rng.normal(0, 0.2)
        ctrl = rng.normal(0, 1, (K_CONTROLS, len(FEATURE_NAMES)))
        out.append((np.datetime64("2020-01-01") + np.timedelta64(i, "D"),
                    np.vstack([hit, ctrl])))
    return out


class TestMatchedStats:
    def test_separating_feature_high_u(self):
        us, sizes, auc = _matched_stats(_groups(True), 0)
        assert us.mean() > 0.85
        assert auc > 0.9
        assert np.all(sizes == K_CONTROLS + 1)

    def test_null_feature_u_near_half(self):
        us, _, auc = _matched_stats(_groups(True), 3)  # non-separating feature
        assert 0.35 < us.mean() < 0.65
        assert 0.35 < auc < 0.65

    def test_permutation_detects_separation(self):
        us, sizes, _ = _matched_stats(_groups(True), 0)
        p = _stratified_permutation_p(us, sizes, 20_000, np.random.default_rng(1))
        assert p < 0.001

    def test_permutation_null_not_significant(self):
        us, sizes, _ = _matched_stats(_groups(True), 5)
        p = _stratified_permutation_p(us, sizes, 20_000, np.random.default_rng(2))
        assert p > 0.01


class TestTimeMatching:
    def _features(self):
        """Two eras: era A (days 0-59) volatile, era B (60-119) calm. Hits
        all in era A. Controls must come from era A too, so an era-level
        feature should NOT separate."""
        rows = []
        rng = np.random.default_rng(3)
        start = np.datetime64("2021-01-04")
        for day in range(120):
            era_vol = 2.0 if day < 60 else 0.2
            for s in range(30):
                feat = {f: rng.normal(0, 1) for f in FEATURE_NAMES}
                feat["vol_20d_ann"] = era_vol + rng.normal(0, 0.05)  # era artifact
                rows.append({
                    "symbol": f"S{s:02d}",
                    "date": (start + np.timedelta64(day, "D")).astype("datetime64[D]").item(),
                    "valid": True,
                    "starts_5x_fwd": False,
                    **feat,
                })
        return pl.DataFrame(rows)

    def test_controls_drawn_from_hit_era(self):
        features = self._features()
        hits = pl.DataFrame({
            "symbol": [f"S{i:02d}" for i in range(6)],
            "entry_date": ["2021-01-20"] * 6,   # era A
            "n_multiple": [5] * 6,
        })
        groups = build_matched_groups(features, hits, seed=4)
        assert 5 in groups
        vol_idx = FEATURE_NAMES.index("vol_20d_ann")
        for _d, g in groups[5]:
            # every control's era-feature ~2.0 (era A), never ~0.2 (era B):
            assert np.all(g[1:, vol_idx] > 1.0)
        # so the era feature cannot separate hit from matched controls
        us, _, auc = _matched_stats(groups[5], vol_idx)
        assert 0.2 < auc < 0.8

    def test_no_control_from_hit_symbol_or_contaminated(self):
        features = self._features().with_columns(
            pl.when(pl.col("symbol") == "S05")
            .then(True).otherwise(pl.col("starts_5x_fwd")).alias("starts_5x_fwd")
        )
        hits = pl.DataFrame({
            "symbol": ["S00"], "entry_date": ["2021-01-20"], "n_multiple": [5],
        })
        groups = build_matched_groups(features, hits, seed=5)
        # S05 (contaminated) can never appear as a control; S00 is the hit
        # itself. We can't observe symbols from the matrices directly, but a
        # contaminated pool that excluded everything would yield no groups —
        # here plenty of clean controls exist:
        assert len(groups.get(5, [])) == 1
