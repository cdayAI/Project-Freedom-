"""Agent 2 — PATHFINDER.

Enumerates every N-x path (N in {5,10,20,50}) inside rolling 6-month
(126-trading-day) windows across the full history of every ingested symbol.

Path semantics (conservative, realizable):
  entry  = next session's OPEN after the signal bar (never the signal close)
  exit   = a later session's CLOSE inside the same window
  multiple(i) = max close over (i+1 .. window end] / open[i+1]

Two path types are counted separately:
  (a) single-instrument holds — enumerated exhaustively here;
  (b) SEQUENCES (chains of 2-8 trades compounding to N-x) — composed only
      from setups that individually passed the Section-4 gates. With an empty
      graduated book the sequence count is structurally zero; the composer
      below activates as the book fills. This is reported, not hidden.

Every hit carries its full pre-move fingerprint and a catalyst class tag —
NONE until the catalyst calendars are ingested, and that limitation is
stamped into the output.

Distinct-paths-per-window is a first-class output: per window we report the
number of distinct symbols with a qualifying path and, per symbol, the count
of non-overlapping qualifying entry->exit intervals (greedy disjoint cover).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from alpha_forge.config import PATH_MULTIPLES, WINDOW_TRADING_DAYS
from alpha_forge.research.features import fingerprint_at

WINDOW_STRIDE = 21  # windows step monthly; structural choice, see ARCHITECTURE.md


@dataclass
class PathHit:
    symbol: str
    window_start: str
    window_end: str
    n_multiple: int
    entry_date: str
    exit_date: str
    achieved_multiple: float
    disjoint_paths_in_window: int
    catalyst_class: str
    fingerprint: dict


def _window_hits_for_symbol(
    sym: str,
    dates: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> list[PathHit]:
    n = close.size
    hits: list[PathHit] = []
    if n < WINDOW_TRADING_DAYS + 2:
        return hits

    for w0 in range(0, n - WINDOW_TRADING_DAYS, WINDOW_STRIDE):
        w1 = w0 + WINDOW_TRADING_DAYS  # window rows [w0, w1)
        wclose = close[w0:w1]
        wopen = open_[w0:w1]

        # suffix max of close within window
        suffmax = np.maximum.accumulate(wclose[::-1])[::-1]
        # entry at open of local bar e (global w0+e), best exit close after e
        # multiple(e) = max close in [e..w1) / open[e]; signal bar is e-1
        with np.errstate(divide="ignore"):
            mult = suffmax / np.where(wopen > 0, wopen, np.inf)
        mult[0] = 0.0  # entry needs a prior signal bar inside the window

        best = float(np.nanmax(mult))
        for n_x in PATH_MULTIPLES:
            if best < n_x:
                continue
            # greedy disjoint cover: earliest qualifying entry, exit at first
            # bar achieving the multiple, then continue after that exit
            disjoint: list[tuple[int, int]] = []
            e = 1
            while e < WINDOW_TRADING_DAYS:
                if mult[e] >= n_x and wopen[e] > 0:
                    target = n_x * wopen[e]
                    exit_rel = e + int(np.argmax(wclose[e:] >= target))
                    disjoint.append((e, exit_rel))
                    e = exit_rel + 1
                else:
                    e += 1
            if not disjoint:
                continue
            e0, x0 = disjoint[0]
            signal_i = w0 + e0 - 1
            hits.append(
                PathHit(
                    symbol=sym,
                    window_start=str(dates[w0]),
                    window_end=str(dates[w1 - 1]),
                    n_multiple=n_x,
                    entry_date=str(dates[w0 + e0]),
                    exit_date=str(dates[w0 + x0]),
                    achieved_multiple=float(wclose[x0] / wopen[e0]),
                    disjoint_paths_in_window=len(disjoint),
                    catalyst_class="NONE",  # no catalyst calendars ingested yet
                    fingerprint=fingerprint_at(signal_i, open_, high, low, close, volume),
                )
            )
    return hits


def enumerate_paths(panel: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """panel: long format (symbol, date, open, high, low, close, volume).

    Returns (hits_df, window_counts_df). window_counts_df reports, per
    (window_start, n_multiple): distinct symbols with a qualifying path and
    total disjoint path count — the 'how many distinct ways existed' number.
    """
    all_hits: list[PathHit] = []
    for (sym,), g in panel.group_by("symbol", maintain_order=True):
        g = g.sort("date")
        all_hits.extend(
            _window_hits_for_symbol(
                str(sym),
                g["date"].to_numpy(),
                g["open"].to_numpy().astype(float),
                g["high"].to_numpy().astype(float),
                g["low"].to_numpy().astype(float),
                g["close"].to_numpy().astype(float),
                g["volume"].to_numpy().astype(float),
            )
        )

    if not all_hits:
        empty = pl.DataFrame()
        return empty, empty

    rows = []
    for h in all_hits:
        row = {
            "symbol": h.symbol,
            "window_start": h.window_start,
            "window_end": h.window_end,
            "n_multiple": h.n_multiple,
            "entry_date": h.entry_date,
            "exit_date": h.exit_date,
            "achieved_multiple": h.achieved_multiple,
            "disjoint_paths_in_window": h.disjoint_paths_in_window,
            "catalyst_class": h.catalyst_class,
        }
        row.update({f"fp_{k}": v for k, v in h.fingerprint.items()})
        rows.append(row)
    hits_df = pl.DataFrame(rows)

    window_counts = (
        hits_df.group_by(["window_start", "window_end", "n_multiple"])
        .agg(
            pl.col("symbol").n_unique().alias("distinct_symbols_with_path"),
            pl.col("disjoint_paths_in_window").sum().alias("total_disjoint_paths"),
        )
        .sort(["window_start", "n_multiple"])
    )
    return hits_df, window_counts


def compose_sequences(graduated_setups: list[dict]) -> dict:
    """Sequence paths (2-8 legs compounding to N-x) built ONLY from setups
    that passed all gates. Empty book => structurally zero sequences."""
    if not graduated_setups:
        return {
            "sequences_found": 0,
            "reason": "no gate-passing setups in the book yet; sequences are "
            "composed only from graduated setups by design",
        }
    raise NotImplementedError(
        "sequence composition activates when the first setups graduate; "
        "design: DAG over setup occurrences within a window, best-product "
        "path search with per-leg costs and non-overlap constraints"
    )
