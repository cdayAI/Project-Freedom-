"""State predicates — layer 1 of the mechanism grammar (BREAKTHROUGH_PHASE1 §3).

A predicate is a named, versioned boolean claim over Genome fields, stated
in one falsifiable English sentence, with its free thresholds drawn from a
small PREREGISTERED grid. The grid is part of the predicate's identity:
every grid point evaluated during discovery is a ledgered trial, so the
grids stay small on purpose. Changing a sentence, a field, or a grid is a
version bump — old ledger entries keep pointing at the exact claim that
was tested.

Only predicates whose Genome fields exist today are defined. The design
doc's SQUEEZE_FUEL days-to-cover clause and CONVEXITY_CHEAP (options state)
wait for the FINRA short-interest ingest and C9 respectively — a predicate
over fields that don't exist would be a claim nobody can evaluate.

Cross-sectional percentiles are computed per date over valid rows only, so
a predicate's meaning ("thin" / "surging") tracks the market of that day,
not the pooled history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import polars as pl


@dataclass(frozen=True)
class Predicate:
    name: str
    version: int
    sentence: str  # the one falsifiable English sentence
    fields: tuple[str, ...]  # genome fields read (must be registered)
    grid: dict  # param name -> tuple of preregistered values
    fn: Callable[[pl.DataFrame, dict], pl.Expr] = field(compare=False)

    def grid_points(self) -> list[dict]:
        keys = sorted(self.grid)
        return [dict(zip(keys, vals)) for vals in product(*(self.grid[k] for k in keys))]

    def evaluate(self, genome: pl.DataFrame, params: dict) -> pl.Series:
        """Boolean series aligned to genome rows; null-safe (null -> False:
        an unknown input is never evidence the state held)."""
        unknown = set(params) - set(self.grid)
        if unknown:
            raise ValueError(f"{self.name} v{self.version}: unknown params {unknown}")
        res = self.fn(genome, params)
        if isinstance(res, pl.Series):  # multi-step windows materialize
            return res.fill_null(False)
        return genome.with_columns(res.fill_null(False).alias("__p"))["__p"]


def _cs_pctl(col: str) -> pl.Expr:
    """Cross-sectional percentile of `col` per date over valid rows, in
    (0, 1]. Invalid and NaN rows get null — polars ranks NaN as the LARGEST
    value, so an unknown would otherwise masquerade as the top of the tape
    (predicates treat null as False)."""
    v = pl.when(pl.col("valid")).then(pl.col(col).fill_nan(None))
    return (v.rank("average").over("date") / v.count().over("date")).alias(f"__{col}_pctl")


PREDICATES: dict[str, Predicate] = {}


def _register(p: Predicate) -> None:
    PREDICATES[p.name] = p


_register(Predicate(
    name="ACCUMULATION", version=1,
    sentence=("Insiders were net buyers over the last published 90 days while "
              "short-volume pressure sat below its own trailing norm."),
    fields=("insider_net_buy_90d_usd", "short_ratio_z"),
    grid={"max_short_z": (0.0, -0.5)},
    fn=lambda g, p: (pl.col("insider_net_buy_90d_usd") > 0)
    & (pl.col("short_ratio_z") < p["max_short_z"]),
))

_register(Predicate(
    name="CATALYST_SHOCK", version=1,
    sentence="An 8-K was filed within the last k sessions (knowable next session).",
    fields=("days_since_any_8k",),
    grid={"k": (1, 3, 5)},
    fn=lambda g, p: pl.col("days_since_any_8k") <= p["k"],
))

def _volume_surge_thin(g: pl.DataFrame, p: dict) -> pl.Series:
    """Thinness is a property of the name BEFORE the surge: the same-day
    Corwin-Schultz estimate compresses mechanically on surge days (the
    same-day conjunction fired ~1 in 236k rows when measured), so the clause
    reads the cross-sectional percentile of the TRAILING 20-session median
    spread. Two materialized steps — polars silently nulls a rank window
    nested over an order_by window. min_samples keeps early sessions null
    (never evidence); order_by makes the rolling window row-order safe."""
    v = pl.when(pl.col("valid")).then(pl.col("cs_spread_est").fill_nan(None))
    t = g.with_columns(
        v.rolling_median(window_size=20, min_samples=10)
        .over("symbol", order_by="date").alias("__med"))
    t = t.with_columns(
        (pl.col("__med").rank("average").over("date")
         / pl.col("__med").count().over("date")).alias("__pctl"))
    return (t["volume_z_20v126"] >= p["min_vol_z"]) & (
        t["__pctl"] >= p["min_spread_pctl"])


_register(Predicate(
    name="VOLUME_SURGE_THIN", version=1,
    sentence=("Volume ran above its 126-session norm by z or more (the z is "
              "winsorized upstream; grid stays inside the reachable range) in "
              "a name whose trailing 20-session median spread sits in the "
              "thinnest q of that day's cross-section."),
    fields=("volume_z_20v126", "cs_spread_est"),
    grid={"min_vol_z": (1.5, 2.0), "min_spread_pctl": (0.80, 0.90)},
    fn=_volume_surge_thin,
))

_register(Predicate(
    name="THRESHOLD_BREAK", version=1,
    sentence="The close stands within b of its 252-session high.",
    fields=("dist_from_252d_high",),
    grid={"b": (0.02, 0.05, 0.10)},
    fn=lambda g, p: pl.col("dist_from_252d_high") >= -p["b"],
))

_register(Predicate(
    name="SQUEEZE_FUEL", version=1,
    sentence=("Short-volume ratio sits z or more above its own trailing norm "
              "(days-to-cover clause deferred until the FINRA short-interest "
              "ingest lands)."),
    fields=("short_ratio_z",),
    grid={"min_z": (1.5, 2.0)},
    fn=lambda g, p: pl.col("short_ratio_z") >= p["min_z"],
))

# outcome layer — labels, never usable inside a mechanism prefix
_register(Predicate(
    name="IGNITION", version=1,
    sentence="A >=5x forward path starts this session (label; matures 126 bars).",
    fields=("starts_5x_fwd",),
    grid={},
    fn=lambda g, p: pl.col("starts_5x_fwd").cast(pl.Boolean),
))

OUTCOME_PREDICATES = frozenset({"IGNITION"})


def validate_registry() -> None:
    """Every predicate field must be a registered genome field (or the valid
    flag); outcome predicates may read labels, state predicates may not."""
    from alpha_forge.genome.schema import FIELD_BY_NAME, outcome_fields

    labels = {f.name for f in outcome_fields()}
    for p in PREDICATES.values():
        for f in p.fields:
            if f not in FIELD_BY_NAME:
                raise ValueError(f"{p.name} reads unregistered field {f}")
            if f in labels and p.name not in OUTCOME_PREDICATES:
                raise ValueError(
                    f"{p.name} is a state predicate but reads label {f}"
                )
