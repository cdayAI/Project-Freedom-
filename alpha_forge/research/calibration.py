"""Isotonic calibration of scanner confidence from live reconciler grades.

The scanner's raw confidence is a similarity percentile — a rank, not a
probability. Once enough predictions have matured and been graded, this
module fits an isotonic (monotone nondecreasing) map from raw score to
P(positive forward return at horizon) via the pool-adjacent-violators
algorithm, and the scanner's confidence field flips from UNCALIBRATED to
CALIBRATED with the fit's sample size and Brier score attached.

Isotonic over Platt because the score-probability relation has no reason to
be sigmoid, and monotonicity is the only assumption a rank deserves.
Reference: Zadrozny & Elkan (2002), "Transforming classifier scores into
accurate multiclass probability estimates", KDD. See SOURCES.md.

Calibration data comes ONLY from the grades parquet (live outcomes graded
against ledger-verified prediction files) — never from backtests.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from alpha_forge.config import STORE_DIR

MIN_GRADED_FOR_CALIBRATION = 100  # below this, honesty beats a wobbly fit
CALIBRATOR_PATH = STORE_DIR / "score_calibrator.parquet"


def pav_isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators: weighted isotonic regression of y on x.
    Returns (breakpoint_x, fitted_y) suitable for stepwise interpolation."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order].astype(float), y[order].astype(float)
    # blocks: value, weight, left index
    vals = list(ys)
    wts = [1.0] * len(ys)
    lefts = list(range(len(ys)))
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-15:
            merged = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / (wts[i] + wts[i + 1])
            vals[i] = merged
            wts[i] += wts[i + 1]
            del vals[i + 1], wts[i + 1], lefts[i + 1]
            while i > 0 and vals[i - 1] > vals[i] + 1e-15:
                merged = (vals[i - 1] * wts[i - 1] + vals[i] * wts[i]) / (wts[i - 1] + wts[i])
                vals[i - 1] = merged
                wts[i - 1] += wts[i]
                del vals[i], wts[i], lefts[i]
                i -= 1
        else:
            i += 1
    bx = np.array([xs[l] for l in lefts])
    by = np.array(vals)
    return bx, by


def apply_isotonic(bx: np.ndarray, by: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Stepwise-constant application (right-continuous), clipped to [0,1]."""
    idx = np.searchsorted(bx, x, side="right") - 1
    idx = np.clip(idx, 0, by.size - 1)
    return np.clip(by[idx], 0.0, 1.0)


def fit_calibrator(horizon: int = 21) -> dict | None:
    """Fit score -> P(fwd_return > 0 at horizon) from the grades parquet.
    Persists breakpoints; returns fit summary or None if underpowered."""
    grades_path = STORE_DIR / "grades.parquet"
    if not grades_path.exists():
        return None
    g = pl.read_parquet(grades_path).filter(pl.col("horizon") == horizon)
    g = g.drop_nulls(["score", "fwd_return"])
    if g.height < MIN_GRADED_FOR_CALIBRATION:
        return None
    x = g["score"].to_numpy().astype(float)
    y = (g["fwd_return"].to_numpy() > 0).astype(float)
    bx, by = pav_isotonic(x, y)
    fitted = apply_isotonic(bx, by, x)
    brier = float(np.mean((fitted - y) ** 2))
    base_rate = float(y.mean())
    pl.DataFrame(
        {"breakpoint_score": bx, "calibrated_p": by}
    ).with_columns(
        pl.lit(horizon).alias("horizon"),
        pl.lit(int(g.height)).alias("n_graded"),
        pl.lit(brier).alias("brier"),
        pl.lit(base_rate).alias("base_rate"),
    ).write_parquet(CALIBRATOR_PATH)
    return {"horizon": horizon, "n_graded": int(g.height), "brier": brier,
            "base_rate": base_rate, "n_breakpoints": int(bx.size)}


def calibrate_score(raw_score: float) -> dict:
    """Confidence for a raw scanner score: calibrated when a fit exists,
    honestly UNCALIBRATED otherwise. The status always rides in-band."""
    if not CALIBRATOR_PATH.exists():
        return {"value": None, "status": "UNCALIBRATED",
                "note": "no calibrator fitted yet (needs "
                f">={MIN_GRADED_FOR_CALIBRATION} graded predictions)"}
    cal = pl.read_parquet(CALIBRATOR_PATH)
    bx = cal["breakpoint_score"].to_numpy()
    by = cal["calibrated_p"].to_numpy()
    p = float(apply_isotonic(bx, by, np.array([raw_score]))[0])
    return {
        "value": p,
        "status": "CALIBRATED",
        "horizon": int(cal["horizon"][0]),
        "n_graded": int(cal["n_graded"][0]),
        "brier": float(cal["brier"][0]),
        "base_rate": float(cal["base_rate"][0]),
        "note": "isotonic map from live graded outcomes; P(fwd_return>0 at horizon)",
    }
