"""Cross-sectional monthly-rebalance backtest lab with realistic fills.

Fill convention (non-negotiable): signals are computed on a month-end CLOSE;
entries and exits execute at the NEXT session's OPEN. Gap-throughs are
inherent to open fills. Costs per name-month round trip:
  half-spread at entry + half-spread at exit (Corwin-Schultz estimate,
  tick-floored) + SEC Section 31 + FINRA TAF on the sell side, at the rates
  in effect on the SELL DATE (date-aware; unverified rates raise).

Cost conservatism: every held name pays a full round trip every month even if
it would be re-selected (real turnover is lower). Conservative bias is
acceptable; optimistic bias is not.

The demo hypothesis run by the first cycle is classical cross-sectional
momentum (Jegadeesh & Titman 1993) — chosen because it is pre-registrable
from the literature rather than mined from this dataset. Every config in the
sweep is one ledgered TRIAL; the full config-return matrix feeds CSCV/PBO.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl

from alpha_forge.costs.fees import FeeSchedule
from alpha_forge.costs.slippage import effective_half_spread

MIN_PRICE = 1.0            # sub-$1 names: spreads and halts dominate; structural filter
MIN_DOLLAR_VOL = 500_000.0  # median 20d dollar volume floor for tradability


@dataclass
class MonthPanel:
    dates: np.ndarray          # all trading dates (D,)
    symbols: list[str]
    open_: np.ndarray          # (D, S)
    close: np.ndarray          # (D, S)
    signal_idx: np.ndarray     # month-end bar indices (M,)
    entry_idx: np.ndarray      # next-open bar indices (M,)
    r_gross: np.ndarray        # (M-1, S) open->open gross simple returns
    r_net: np.ndarray          # (M-1, S) net of full round-trip costs
    eligible: np.ndarray       # (M-1, S) bool at signal time
    close_hist: np.ndarray     # (D, S) close, for score computation


def build_month_panel(panel: pl.DataFrame) -> MonthPanel:
    wide_close = panel.pivot(index="date", on="symbol", values="close").sort("date")
    wide_open = panel.pivot(index="date", on="symbol", values="open").sort("date")
    wide_high = panel.pivot(index="date", on="symbol", values="high").sort("date")
    wide_low = panel.pivot(index="date", on="symbol", values="low").sort("date")
    wide_vol = panel.pivot(index="date", on="symbol", values="volume").sort("date")

    dates = wide_close["date"].to_numpy()
    symbols = [c for c in wide_close.columns if c != "date"]
    C = wide_close.select(symbols).to_numpy().astype(float)
    O = wide_open.select(symbols).to_numpy().astype(float)
    H = wide_high.select(symbols).to_numpy().astype(float)
    L = wide_low.select(symbols).to_numpy().astype(float)
    V = wide_vol.select(symbols).to_numpy().astype(float)
    D, S = C.shape

    # month-end bars: last bar of each (year, month)
    months = dates.astype("datetime64[M]")
    is_month_end = np.append(months[:-1] != months[1:], True)
    signal_idx = np.nonzero(is_month_end)[0]
    signal_idx = signal_idx[signal_idx + 1 < D]  # need a next open
    entry_idx = signal_idx + 1
    M = signal_idx.size

    # per-symbol half-spread series (rolling CS estimate, tick-floored)
    half_spread = np.full((D, S), np.nan)
    for s in range(S):
        col_ok = ~np.isnan(C[:, s])
        if col_ok.sum() > 30:
            hs = effective_half_spread(H[col_ok, s], L[col_ok, s], C[col_ok, s])
            half_spread[col_ok, s] = hs

    sec = FeeSchedule.load("sec_section31")
    taf = FeeSchedule.load("finra_taf_equity")

    r_gross = np.full((M - 1, S), np.nan)
    r_net = np.full((M - 1, S), np.nan)
    eligible = np.zeros((M - 1, S), dtype=bool)

    for m in range(M - 1):
        e_in, e_out = entry_idx[m], entry_idx[m + 1]
        sell_date = str(dates[e_out])[:10]
        sec_prop = sec.rate_on(sell_date) / 1_000_000.0
        taf_entry = taf.entry_on(sell_date)
        po, px = O[e_in], O[e_out]
        with np.errstate(invalid="ignore", divide="ignore"):
            g = px / po - 1.0
            taf_prop = float(taf_entry["rate"]) / np.where(px > 0, px, np.nan)
            cost = half_spread[e_in] + half_spread[e_out] + sec_prop + taf_prop
        r_gross[m] = g
        r_net[m] = g - cost

        si = signal_idx[m]
        dv_win = C[max(0, si - 19) : si + 1] * V[max(0, si - 19) : si + 1]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice")  # unlisted symbols
            dv_med = np.nanmedian(dv_win, axis=0)
        eligible[m] = (
            ~np.isnan(C[si])
            & ~np.isnan(po)
            & ~np.isnan(px)
            & ~np.isnan(g)
            & ~np.isnan(cost)
            & (C[si] >= MIN_PRICE)
            & (dv_med >= MIN_DOLLAR_VOL)
        )

    return MonthPanel(
        dates=dates, symbols=symbols, open_=O, close=C,
        signal_idx=signal_idx, entry_idx=entry_idx,
        r_gross=r_gross, r_net=r_net, eligible=eligible, close_hist=C,
    )


def momentum_scores(mp: MonthPanel, formation_months: int, skip_months: int) -> np.ndarray:
    """Score at each signal bar: close[t-skip]/close[t-skip-formation] - 1,
    in bars of 21 trading days per month. NaN where history is missing."""
    f, s = formation_months * 21, skip_months * 21
    M = mp.signal_idx.size
    S = len(mp.symbols)
    scores = np.full((M - 1, S), np.nan)
    for m in range(M - 1):
        si = mp.signal_idx[m]
        i_end = si - s
        i_start = i_end - f
        if i_start < 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            scores[m] = mp.close_hist[i_end] / mp.close_hist[i_start] - 1.0
    return scores


def run_config(mp: MonthPanel, scores: np.ndarray, top_k: int) -> dict:
    """Equal-weight long top_k by score among eligible names, monthly.

    Only months with a rankable cross-section (>= 2*top_k scored eligible
    names) are traded; the traded month indices are returned so permutation
    and null-baseline draws run on EXACTLY the same months.
    """
    M = scores.shape[0]
    net, gross, n_held, months = [], [], [], []
    for m in range(M):
        ok = mp.eligible[m] & ~np.isnan(scores[m])
        idx = np.nonzero(ok)[0]
        if idx.size < top_k * 2:
            continue
        sel = idx[np.argsort(scores[m][idx])[::-1][:top_k]]
        net.append(float(np.mean(mp.r_net[m][sel])))
        gross.append(float(np.mean(mp.r_gross[m][sel])))
        n_held.append(int(sel.size))
        months.append(m)
    return {
        "net_returns": np.asarray(net),
        "gross_returns": np.asarray(gross),
        "n_trade_events": int(np.sum(n_held)),
        "month_indices": np.asarray(months, dtype=int),
        "avg_holding_days": 21.0,
    }


def random_selection_draws(
    mp: MonthPanel,
    valid_months: np.ndarray,
    top_k: int,
    n_draws: int,
    seed: int,
    metric: str = "log_growth",
) -> np.ndarray:
    """Matched null: same months, same k, same holding period, same universe,
    same cost model — selection replaced by uniform random draw.

    Vectorized per month: k-of-n sampling without replacement for all draws
    at once via random-key top-k (argpartition of uniform keys).
    """
    rng = np.random.default_rng(seed)
    month_returns = np.empty((len(valid_months), n_draws))
    for mi, m in enumerate(valid_months):
        pool = np.nonzero(mp.eligible[m])[0]
        k = min(top_k, pool.size)
        keys = rng.random((n_draws, pool.size))
        sel = np.argpartition(keys, k - 1, axis=1)[:, :k]
        month_returns[mi] = mp.r_net[m][pool[sel]].mean(axis=1)
    if metric == "log_growth":
        return np.log1p(np.maximum(month_returns, -0.9999)).sum(axis=0)
    return month_returns.mean(axis=0)
