"""Agent 7 — RECONCILER.

Grades every prediction against realized outcomes, at fixed horizons,
strictly against the immutable predictions file (sha256 re-verified against
the ledger before grading — a tampered file is never graded, it halts).

Outputs per (file, horizon): realized forward return per prediction from the
next session's open after the vintage date, whether the 5x target was
reached inside the horizon, and calibration rows (confidence bucket ->
realized rate). Grades are ledgered; calibration curves accumulate in
store/calibration.parquet and are the system's fitness function.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl

from alpha_forge.config import PREDICTIONS_DIR, STORE_DIR
from alpha_forge.ledger import Ledger

HORIZONS = (5, 21, 63, 126)
CALIBRATION_PATH = STORE_DIR / "calibration.parquet"
GRADES_PATH = STORE_DIR / "grades.parquet"


class TamperedPredictionsError(Exception):
    pass


def _ledgered_hash(ledger: Ledger, filename: str) -> str | None:
    for e in ledger.entries():
        if e["kind"] == "PREDICTION" and e["payload"].get("file") == filename:
            return e["payload"]["sha256"]
    return None


def _graded_already(ledger: Ledger, filename: str, horizon: int) -> bool:
    for e in ledger.entries():
        if (
            e["kind"] == "GRADE"
            and e["payload"].get("file") == filename
            and e["payload"].get("horizon") == horizon
        ):
            return True
    return False


def grade_all(panel: pl.DataFrame, ledger: Ledger) -> list[dict]:
    """Grade every prediction file at every horizon that has matured."""
    grades = []
    by_symbol = {str(s[0]): g.sort("date") for s, g in panel.group_by("symbol")}
    all_dates = np.sort(panel["date"].unique().to_numpy())

    for path in sorted(PREDICTIONS_DIR.glob("predictions_*.json")):
        expected = _ledgered_hash(ledger, path.name)
        if expected is None:
            continue  # unledgered file: not admissible for grading
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise TamperedPredictionsError(
                f"{path.name}: sha256 mismatch vs ledger — grading halted"
            )
        doc = json.loads(path.read_text())
        vintage = np.datetime64(doc["data_vintage"])
        preds = doc.get("predictions", [])
        if not preds:
            continue
        # entry bar: first trading date strictly after vintage
        later = all_dates[all_dates > vintage]
        if later.size == 0:
            continue
        for horizon in HORIZONS:
            if _graded_already(ledger, path.name, horizon):
                continue
            # matured only if >= horizon bars exist after entry
            if later.size < horizon + 1:
                continue
            rows = []
            for p in preds:
                g = by_symbol.get(p["instrument"])
                if g is None:
                    continue
                dates = g["date"].to_numpy()
                entry_candidates = np.nonzero(dates > vintage)[0]
                if entry_candidates.size == 0:
                    continue
                e = int(entry_candidates[0])
                if e + horizon >= g.height:
                    continue
                open_ = g["open"].to_numpy().astype(float)
                close = g["close"].to_numpy().astype(float)
                entry_px = open_[e]
                if not np.isfinite(entry_px) or entry_px <= 0:
                    continue
                fwd = close[e + 1 : e + 1 + horizon]
                rows.append(
                    {
                        "file": path.name,
                        "horizon": horizon,
                        "symbol": p["instrument"],
                        "confidence": p["confidence"]["value"],
                        "score": p.get("score"),
                        "entry_open": float(entry_px),
                        "fwd_return": float(close[e + horizon] / entry_px - 1.0),
                        "max_multiple": float(fwd.max() / entry_px) if fwd.size else np.nan,
                        "reached_5x": bool(fwd.size and fwd.max() / entry_px >= 5.0),
                    }
                )
            if not rows:
                continue
            gdf = pl.DataFrame(rows)
            summary = {
                "file": path.name,
                "horizon": horizon,
                "n_graded": gdf.height,
                "mean_fwd_return": float(gdf["fwd_return"].mean()),
                "median_fwd_return": float(gdf["fwd_return"].median()),
                "hit_rate_5x": float(gdf["reached_5x"].mean()),
            }
            ledger.append("GRADE", summary)
            grades.append(summary)
            _append_parquet(GRADES_PATH, gdf)
    _rebuild_calibration()
    return grades


def _append_parquet(path, df: pl.DataFrame) -> None:
    if path.exists():
        old = pl.read_parquet(path)
        df = pl.concat([old, df], how="diagonal")
    df.write_parquet(path)


def _rebuild_calibration() -> None:
    """Confidence bucket -> realized positive-return rate, per horizon."""
    if not GRADES_PATH.exists():
        return
    g = pl.read_parquet(GRADES_PATH)
    if g.height == 0:
        return
    cal = (
        g.with_columns((pl.col("confidence") * 5).floor().clip(0, 4).alias("bucket"))
        .group_by(["horizon", "bucket"])
        .agg(
            pl.len().alias("n"),
            (pl.col("fwd_return") > 0).mean().alias("realized_positive_rate"),
            pl.col("fwd_return").mean().alias("mean_fwd_return"),
            pl.col("reached_5x").mean().alias("realized_5x_rate"),
        )
        .sort(["horizon", "bucket"])
    )
    cal.write_parquet(CALIBRATION_PATH)


def calibration_summary() -> pl.DataFrame | None:
    if CALIBRATION_PATH.exists():
        return pl.read_parquet(CALIBRATION_PATH)
    return None
