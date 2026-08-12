"""Agent 6 — DAILY SCANNER.

Every run: compute the current feature state of every liquid ingested name
and rank it against the historical pre-move fingerprints that survived
Confluence. Output a dated, IMMUTABLE predictions file.

Honesty constraints, enforced here:
- Confidence is a raw similarity percentile stamped UNCALIBRATED until the
  Reconciler has produced calibration curves; the field says so in-band.
- size_usd is 0 and executable=false until a strategy that uses this screen
  has passed all Section-4 gates AND the paper-trading requirement. The
  scanner is a research instrument, not a signal service, until then.
- The file is write-once: an existing file for the same data vintage is
  never overwritten, and its sha256 is chained into the ledger, so grading
  is always against what was actually predicted.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import polars as pl

from alpha_forge.config import PREDICTIONS_DIR, WINDOW_TRADING_DAYS
from alpha_forge.data.regime import regime_on
from alpha_forge.ledger import Ledger
from alpha_forge.research.calibration import calibrate_score
from alpha_forge.research.features import FEATURE_NAMES, fingerprint_at

TOP_N_PREDICTIONS = 20
MIN_DOLLAR_VOL = 500_000.0
MIN_PRICE = 1.0


def _robust_z(x: float, med: float, mad: float) -> float:
    if mad <= 0 or np.isnan(x):
        return 0.0
    return float((x - med) / (1.4826 * mad))


def scan(
    panel: pl.DataFrame,
    confluence: pl.DataFrame | None,
    hits: pl.DataFrame | None,
    regime: pl.DataFrame | None,
    ledger: Ledger,
    data_vintage: str,
) -> dict | None:
    """Rank live names by similarity to identifiable hit fingerprints."""
    out_path = PREDICTIONS_DIR / f"predictions_{data_vintage}.json"
    if out_path.exists():
        # immutable: never regenerated — but a crash between file write and
        # ledger append would leave it unledgered and therefore ungradeable;
        # heal the ledger entry on cache hit if it is missing
        doc = json.loads(out_path.read_text(encoding="utf-8"))
        if not any(
            e["kind"] == "PREDICTION" and e["payload"].get("file") == out_path.name
            for e in ledger.entries()
        ):
            ledger.append(
                "PREDICTION",
                {"file": out_path.name, "sha256": _sha(out_path),
                 "n": len(doc.get("predictions", []))},
            )
        return doc

    if confluence is None or hits is None or hits.height == 0:
        return None
    survivors = confluence.filter(pl.col("verdict") == "IDENTIFIABLE")
    if survivors.height == 0:
        # no identifiable features -> an honest scanner emits nothing
        doc = {
            "data_vintage": data_vintage,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "predictions": [],
            "note": "no fingerprint feature survived BH correction; scanning "
            "without an identifiable signal would be noise dressed as research",
        }
        out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        ledger.append("PREDICTION", {"file": out_path.name, "sha256": _sha(out_path), "n": 0})
        return doc

    # direction per feature: AUC>0.5 means hits sit above baseline
    feat_dir = {}
    for row in survivors.iter_rows(named=True):
        key = row["feature"]
        d = 1.0 if row["auc"] > 0.5 else -1.0
        feat_dir.setdefault(key, []).append(d)
    feat_dir = {k: float(np.sign(np.mean(v))) for k, v in feat_dir.items()}

    # robust standardization stats from the hit catalog itself
    dedup_hits = hits.unique(subset=["symbol", "entry_date"])
    stats = {}
    for feat in feat_dir:
        vals = dedup_hits[f"fp_{feat}"].to_numpy().astype(float)
        vals = vals[~np.isnan(vals)]
        if vals.size < 10:
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        stats[feat] = (med, mad)
    if not stats:
        return None

    # today's feature state per symbol + similarity score
    rows = []
    hit_matrix = {
        feat: dedup_hits[f"fp_{feat}"].to_numpy().astype(float) for feat in stats
    }
    hit_ids = list(zip(dedup_hits["symbol"].to_list(), dedup_hits["entry_date"].to_list()))
    for (sym,), g in panel.group_by("symbol", maintain_order=True):
        g = g.sort("date")
        n = g.height
        if n < 200:
            continue
        close = g["close"].to_numpy().astype(float)
        vol = g["volume"].to_numpy().astype(float)
        if close[-1] < MIN_PRICE or np.nanmedian((close * vol)[-20:]) < MIN_DOLLAR_VOL:
            continue
        fp = fingerprint_at(
            n - 1,
            g["open"].to_numpy().astype(float),
            g["high"].to_numpy().astype(float),
            g["low"].to_numpy().astype(float),
            close,
            vol,
        )
        # similarity: how far today's state sits toward hit-typical values,
        # relative to hit-catalog dispersion, per surviving feature
        score = float(
            np.mean(
                [
                    feat_dir[f] * np.clip(_robust_z(fp[f], *stats[f]) * feat_dir[f], -3, 3)
                    for f in stats
                ]
            )
        )
        # nearest historical analogs in surviving-feature space
        dists = np.zeros(len(hit_ids))
        for f, (med, mad) in stats.items():
            hv = hit_matrix[f]
            z_h = (hv - med) / (1.4826 * mad + 1e-12)
            z_t = _robust_z(fp[f], med, mad)
            dists += np.where(np.isnan(z_h), 9.0, (z_h - z_t) ** 2)
        analog_idx = np.argsort(dists)[:3]
        rows.append(
            {
                "symbol": str(sym),
                "score": score,
                "features": {k: (None if np.isnan(v) else v) for k, v in fp.items()},
                "analogs": [
                    {"symbol": hit_ids[i][0], "entry_date": hit_ids[i][1]} for i in analog_idx
                ],
                "last_close": float(close[-1]),
            }
        )

    if not rows:
        return None
    rows.sort(key=lambda r: r["score"], reverse=True)
    scores = np.array([r["score"] for r in rows])
    reg = regime_on(regime, data_vintage) if regime is not None else {}

    preds = []
    for r in rows[:TOP_N_PREDICTIONS]:
        pct = float((scores < r["score"]).mean())
        # calibrated probability when the isotonic fit exists (live grades
        # only); until then the honest raw percentile with its status in-band
        cal = calibrate_score(r["score"])
        confidence = (
            cal
            if cal["status"] == "CALIBRATED"
            else {
                "value": pct,
                "status": "UNCALIBRATED",
                "note": "similarity percentile; becomes a probability only "
                "after reconciler calibration",
            }
        )
        preds.append(
            {
                "instrument": r["symbol"],
                "direction": "LONG",
                "entry": "next_session_open",
                "target_multiple": 5.0,
                "horizon_bars": WINDOW_TRADING_DAYS,
                "stop": None,
                "size_usd": 0.0,
                "executable": False,
                "confidence": confidence,
                "score": r["score"],
                "last_close": r["last_close"],
                "features": r["features"],
                "historical_analogs": r["analogs"],
                "regime": reg,
            }
        )

    doc = {
        "data_vintage": data_vintage,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "universe_scanned": len(rows),
        "surviving_features": sorted(stats.keys()),
        "predictions": preds,
    }
    out_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    ledger.append(
        "PREDICTION",
        {"file": out_path.name, "sha256": _sha(out_path), "n": len(preds)},
    )
    return doc


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
