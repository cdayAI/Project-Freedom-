"""Market Genome builder — assemble the point-in-time surface from the
stores that already exist, applying every field's declared knowability rule
mechanically (schema.py). The builder never computes a market quantity; it
re-times what the feature layer computed so that no decision row contains a
value a live system would not have had.

Output: data/store/genome/genome_v{N}.parquet + genome_v{N}_meta.json
(schema version, source panel hash, field registry with knowability rules,
survivorship flag). Old genome files are never rewritten — a schema change
bumps the version and writes a new pair.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date as _date
from pathlib import Path

import polars as pl

from alpha_forge.config import STORE_DIR
from alpha_forge.genome.schema import (
    FIELD_BY_NAME,
    GENOME_SCHEMA_VERSION,
    PUBLICATION_DELAY_DAYS,
    UNREGISTERED_PASSTHROUGH,
    knowability_table,
    outcome_fields,
)

GENOME_DIR = STORE_DIR / "genome"


class GenomeSchemaError(ValueError):
    pass


def _quarter_end(d: _date) -> _date:
    q_month = ((d.month - 1) // 3) * 3 + 3
    if q_month == 12:
        return _date(d.year, 12, 31)
    from calendar import monthrange

    return _date(d.year, q_month, monthrange(d.year, q_month)[1])


def _last_published_quarter_end(d: _date) -> _date:
    """The most recent quarter end whose DERA file is out by decision date d
    (conservative PUBLICATION_DELAY_DAYS after quarter end)."""
    import datetime as _dt

    q = _quarter_end(d)
    while q > d - _dt.timedelta(days=PUBLICATION_DELAY_DAYS):
        # step to the previous quarter end
        first_of_quarter = _date(q.year, q.month - 2, 1)
        q = _quarter_end(first_of_quarter - _dt.timedelta(days=1))
    return q


def build_genome(
    features: pl.DataFrame,
    regime: pl.DataFrame | None,
    out_dir: Path | None = None,
    panel_sha256: str | None = None,
) -> tuple[pl.DataFrame, dict]:
    """(genome, meta). Raises GenomeSchemaError when a registered field is
    missing from the inputs — silence is how leaks are born."""
    out_dir = out_dir or GENOME_DIR

    regime_cols: dict[str, str] = {}
    if regime is not None:
        rename = {"trend": "regime_trend", "vol_state": "regime_vol_state",
                  "chop": "regime_chop"}
        regime_cols = {v: k for k, v in rename.items()}
        features = features.join(
            regime.select(["date", *rename]).rename(rename), on="date", how="left"
        )

    missing = [
        name for name in FIELD_BY_NAME
        if name not in features.columns
    ]
    if missing:
        raise GenomeSchemaError(
            f"registered genome fields missing from inputs: {missing} — "
            "either wire the feed or remove the field with a schema bump"
        )

    features = features.sort(["symbol", "date"])

    exprs: list[pl.Expr] = [pl.col(c) for c in UNREGISTERED_PASSTHROUGH]
    for f in FIELD_BY_NAME.values():
        col = pl.col(f.name)
        if f.group == "outcome":
            # labels ride along unlagged and unmasked; the feature surface
            # excludes them by group, and event training purges them anyway
            exprs.append(col)
            continue
        if f.quarterly_published:
            # exposure is handled below via an as-of join; keep raw for now
            exprs.append(col)
            continue
        if f.lag_sessions:
            exprs.append(col.shift(f.lag_sessions).over("symbol").alias(f.name))
        else:
            exprs.append(col)
    genome = features.select(exprs)

    # ---- quarterly-published sources: re-time to the published quarter ---
    q_fields = [f for f in FIELD_BY_NAME.values() if f.quarterly_published]
    if q_fields:
        # decision date d may only see values as of the last PUBLISHED
        # quarter end; per (symbol, cutoff) take the newest row <= cutoff
        cutoff = (
            genome.select(pl.col("date").unique())
            .with_columns(
                pl.col("date")
                .map_elements(_last_published_quarter_end, return_dtype=pl.Date)
                .alias("published_cutoff")
            )
        )
        base = genome.select(
            ["symbol", "date", *[f.name for f in q_fields]]
        ).rename({"date": "source_date"})
        keyed = (
            genome.select(["symbol", "date"])
            .join(cutoff, on="date", how="left")
            .sort(["symbol", "published_cutoff"])
        )
        asof = keyed.join_asof(
            base.sort(["symbol", "source_date"]),
            left_on="published_cutoff",
            right_on="source_date",
            by="symbol",
            strategy="backward",
            check_sortedness=False,  # both sides sorted above; the by-group
            # check would warn unconditionally
        ).select(["symbol", "date", *[f.name for f in q_fields]])
        genome = genome.drop([f.name for f in q_fields]).join(
            asof, on=["symbol", "date"], how="left"
        )
        # the extra lag_sessions on top of publication timing
        lag_exprs = [
            pl.col(f.name).shift(f.lag_sessions).over("symbol").alias(f.name)
            for f in q_fields
            if f.lag_sessions
        ]
        if lag_exprs:
            genome = genome.sort(["symbol", "date"]).with_columns(lag_exprs)

    meta = {
        "schema_version": GENOME_SCHEMA_VERSION,
        "built_from": "features_panel + regime",
        "panel_sha256": panel_sha256,
        "universe_survivorship_biased": True,
        "publication_delay_days": PUBLICATION_DELAY_DAYS,
        "knowability": knowability_table(),
        "outcome_fields": [f.name for f in outcome_fields()],
        "regime_join": sorted(regime_cols) if regime_cols else [],
        "rows": genome.height,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    genome.write_parquet(out_dir / f"genome_v{GENOME_SCHEMA_VERSION}.parquet")
    (out_dir / f"genome_v{GENOME_SCHEMA_VERSION}_meta.json").write_text(
        json.dumps(meta, indent=1, default=str), encoding="utf-8"
    )
    return genome, meta


def audit_no_leak(features: pl.DataFrame, genome: pl.DataFrame,
                  sample: int = 2000, seed: int = 0) -> dict:
    """Mechanical knowability audit: for every lagged field, sampled genome
    rows must equal the SOURCE value lag sessions earlier — and must never
    equal a same-session source value that differs from it. Returns per-field
    counts; raises on any violation."""
    import numpy as np

    rng = np.random.default_rng(seed)
    features = features.sort(["symbol", "date"])
    genome = genome.sort(["symbol", "date"])
    out = {}
    for f in FIELD_BY_NAME.values():
        if f.group == "outcome" or f.quarterly_published or not f.lag_sessions:
            continue
        src = features.select(["symbol", "date", f.name]).rename({f.name: "src"})
        lagged_src = src.with_columns(
            pl.col("src").shift(f.lag_sessions).over("symbol").alias("expect")
        )
        joined = genome.select(["symbol", "date", f.name]).join(
            lagged_src.select(["symbol", "date", "expect"]),
            on=["symbol", "date"], how="inner",
        ).drop_nulls()
        if joined.height == 0:
            out[f.name] = {"checked": 0}
            continue
        take = joined if joined.height <= sample else joined.sample(
            sample, seed=int(rng.integers(0, 2**31))
        )
        both_nan = pl.col(f.name).is_nan() & pl.col("expect").is_nan()
        bad = take.filter(
            ~both_nan & ((pl.col(f.name) - pl.col("expect")).abs() > 1e-9)
        ).height
        if bad:
            raise GenomeSchemaError(
                f"knowability leak: {f.name} has {bad}/{take.height} sampled "
                f"rows not equal to the source value {f.lag_sessions} "
                "session(s) earlier"
            )
        out[f.name] = {"checked": take.height}
    return out
