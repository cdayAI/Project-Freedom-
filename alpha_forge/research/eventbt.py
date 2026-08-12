"""Event-driven backtester with walk-forward TRAINING.

This is the vehicle for the system's actual thesis: enter a name at the
NEXT session's open when its fingerprint state crosses an extremity
threshold, ride toward a target multiple with a stop and a time limit,
under full costs and slot-limited capital.

Training is honest walk-forward:
  - the signal's feature DIRECTIONS are re-derived per fold from matched
    confluence groups whose hits occurred STRICTLY BEFORE the fold's train
    end (no future hits inform past signals);
  - the config grid (entry percentile x target x stop) is evaluated on the
    train segment only; the best train config runs once on the test segment;
  - folds are purged by the full 126-bar holding period;
  - the final 15% of trading days is holdout, untouched here.

Fill realism: entry open*(1+half_spread); target is a limit sell (gap-open
above target fills at open, else at target); stop fills at open when gapped
through, else at stop; time exit at the open after HOLD_MAX bars; sells pay
date-aware SEC 31 + TAF. Missing bars (halts) simply extend the hold — no
fill happens on a day with no trading.

Every config evaluated anywhere becomes a ledgered trial. The matched null
re-simulates each realized entry day across EVERY eligible symbol with the
same exit rules and costs, so gate 2/9 nulls share the strategy's exact
mechanics and differ only in selection skill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from alpha_forge.config import WINDOW_TRADING_DAYS
from alpha_forge.costs.fees import FeeSchedule
from alpha_forge.costs.slippage import effective_half_spread
from alpha_forge.research.features import FEATURE_NAMES

HOLD_MAX = WINDOW_TRADING_DAYS  # 126 bars, the research window
MAX_CONCURRENT = 10             # structural: slots of 1/10th capital each
MIN_PRICE = 1.0
MIN_DOLLAR_VOL = 500_000.0
DIRECTION_MARGIN = 0.08         # |mean_u - 0.5| needed to admit a feature

CONFIG_GRID = [
    {"entry_pct": ep, "target_mult": t, "stop_frac": s}
    for ep in (0.995, 0.99, 0.98)
    for t in (2.0, 5.0)
    for s in (0.5, None)
]


@dataclass
class EventPanel:
    """Wide arrays + per-day feature percentiles, built once per night."""

    dates: np.ndarray            # (D,)
    symbols: list[str]
    open_: np.ndarray            # (D, S)
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    half_spread: np.ndarray      # (D, S)
    eligible: np.ndarray         # (D, S) bool — price/liquidity/warmup
    feat_pctl: dict[str, np.ndarray]  # feature -> (D, S) cross-sectional pctl, float32
    sell_fee_prop: np.ndarray    # (D, S) proportional sell-side fees by date
    overnight_gaps: np.ndarray   # pooled distribution for stress tests
    feat_cols: list[str] | None = None  # dynamic feature list (None -> FEATURE_NAMES)


def build_event_panel(panel: pl.DataFrame, features: pl.DataFrame) -> EventPanel:
    from alpha_forge.research.features_panel import feature_columns

    feat_cols = feature_columns(features)
    wide = {}
    for col in ("open", "high", "low", "close"):
        w = panel.pivot(index="date", on="symbol", values=col).sort("date")
        wide[col] = w
    dates = wide["close"]["date"].to_numpy()
    symbols = [c for c in wide["close"].columns if c != "date"]
    O = wide["open"].select(symbols).to_numpy().astype(float)
    H = wide["high"].select(symbols).to_numpy().astype(float)
    L = wide["low"].select(symbols).to_numpy().astype(float)
    C = wide["close"].select(symbols).to_numpy().astype(float)
    D, S = C.shape

    hs = np.full((D, S), np.nan)
    for s in range(S):
        ok = ~np.isnan(C[:, s])
        if ok.sum() > 30:
            hs[ok, s] = effective_half_spread(H[ok, s], L[ok, s], C[ok, s])

    # per-day cross-sectional percentile of each feature (float32 to keep
    # the D x S matrices affordable)
    fwide = {}
    for feat in feat_cols:
        w = (
            features.pivot(index="date", on="symbol", values=feat)
            .sort("date")
            .select(symbols)
            .to_numpy()
            .astype(float)
        )
        fwide[feat] = w
    valid = (
        features.pivot(index="date", on="symbol", values="valid")
        .sort("date")
        .select(symbols)
        .to_numpy()
    )
    valid = np.where(valid == None, False, valid).astype(bool)  # noqa: E711

    dv = fwide["dollar_vol_med_20d"]
    eligible = (
        valid
        & ~np.isnan(C)
        & ~np.isnan(O)
        & (C >= MIN_PRICE)
        & (dv >= MIN_DOLLAR_VOL)
        & ~np.isnan(hs)
    )

    feat_pctl = {}
    for feat in feat_cols:
        m = np.where(eligible, fwide[feat], np.nan)
        counts = np.sum(~np.isnan(m), axis=1, keepdims=True).astype(float)
        # rank via double argsort per row (NaN sorts last, then masked out)
        rank_of = np.argsort(np.argsort(m, axis=1), axis=1).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.where(np.isnan(m) | (counts < 10), np.nan, (rank_of + 0.5) / counts)
        feat_pctl[feat] = pct.astype(np.float32)

    # date-aware proportional sell-side fees
    sec = FeeSchedule.load("sec_section31")
    taf = FeeSchedule.load("finra_taf_equity")
    sell_fee = np.zeros((D, S))
    for d in range(D):
        ds = str(dates[d])[:10]
        sec_prop = sec.rate_on(ds) / 1_000_000.0
        taf_rate = float(taf.entry_on(ds)["rate"])
        with np.errstate(invalid="ignore", divide="ignore"):
            sell_fee[d] = sec_prop + taf_rate / np.where(C[d] > 0, C[d], np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        gaps = (O[1:] / C[:-1] - 1.0).ravel()
    gaps = gaps[~np.isnan(gaps)]

    return EventPanel(
        dates=dates, symbols=symbols, open_=O, high=H, low=L, close=C,
        half_spread=hs, eligible=eligible, feat_pctl=feat_pctl,
        sell_fee_prop=sell_fee, overnight_gaps=gaps, feat_cols=feat_cols,
    )


# ---------------------------------------------------------------- signal


def fold_directions(
    groups_with_dates: dict[int, list[tuple[object, np.ndarray]]],
    n_class: int,
    cutoff_date,
    feat_cols: list[str] | None = None,
) -> dict[str, float]:
    """Feature -> direction (+1/-1) learned ONLY from matched groups whose
    hit predates cutoff_date. Features inside the margin are dropped."""
    feat_cols = feat_cols or list(FEATURE_NAMES)
    groups = [g for d, g in groups_with_dates.get(n_class, []) if d < cutoff_date]
    out: dict[str, float] = {}
    if len(groups) < 20:
        return out
    for fi, feat in enumerate(feat_cols):
        us = []
        for g in groups:
            vals = g[:, fi]
            if np.isnan(vals[0]):
                continue
            ctrl = vals[1:]
            ctrl = ctrl[~np.isnan(ctrl)]
            if ctrl.size < 2:
                continue
            below = (ctrl < vals[0]).sum() + 0.5 * (ctrl == vals[0]).sum()
            us.append((below + 0.5) / (ctrl.size + 1))
        if len(us) >= 20:
            mu = float(np.mean(us))
            if abs(mu - 0.5) >= DIRECTION_MARGIN:
                out[feat] = 1.0 if mu > 0.5 else -1.0
    return out


def ridge_logistic(X: np.ndarray, y: np.ndarray, lam: float = 1.0, iters: int = 30) -> np.ndarray:
    """IRLS ridge logistic. X: (n, p) centered inputs; returns (p+1,) with
    intercept last. lam is fixed (structural) — no tuning inside folds."""
    n, p = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(p + 1)
    for _ in range(iters):
        z = Xb @ w
        mu = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        s = np.maximum(mu * (1 - mu), 1e-6)
        # ridge on feature weights only, not the intercept
        reg = lam * np.eye(p + 1)
        reg[p, p] = 0.0
        H = Xb.T @ (Xb * s[:, None]) + reg
        g = Xb.T @ (y - mu) - reg @ w
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if float(np.abs(step).max()) < 1e-10:
            break
    return w


TRAIN_CLASSES_POOLED = (3, 5)   # v7: auxiliary 3x episodes pool with 5x —
                                # same phenomenon family, ~5-10x the examples
MAX_ROUNDTRIP_SPREAD = 0.10     # v7 tradability floor: skip entries whose
                                # measured roundtrip spread exceeds 10% —
                                # tonight's kill showed spread, not selection,
                                # was the cause of death (structural constant,
                                # pre-registered with the candidate)


def _adaptive_lam(n_groups: int, n_features: int) -> float:
    """Pre-registered regularization formula — NEVER hand-tuned: shrink
    harder when groups are scarce relative to features. lam = max(1, 25p/G):
    at 300 groups x 15 features -> lam ~ 1.25; at 60 groups -> lam ~ 6."""
    return max(1.0, 25.0 * n_features / max(n_groups, 1))


def fold_weights_pooled(
    groups_with_dates: dict[int, list[tuple[object, np.ndarray]]],
    classes: tuple[int, ...],
    cutoff_date,
    feat_cols: list[str] | None = None,
) -> dict[str, float]:
    """v7 signal: pool matured groups across the training classes, then fit
    with the adaptive lambda. Pooling note (honest dilution): controls are
    contamination-filtered against 5x starts; a 3x-group control may itself
    start a 3x, which DILUTES the 3x signal toward zero — conservative, never
    inflationary."""
    feat_cols = feat_cols or list(FEATURE_NAMES)
    merged: list[tuple[object, np.ndarray]] = []
    for c in classes:
        merged.extend(groups_with_dates.get(c, []))
    pooled = {0: merged}
    matured = [g for d, g in merged if d < cutoff_date]
    lam = _adaptive_lam(len(matured), len(feat_cols))
    return fold_weights(pooled, 0, cutoff_date, feat_cols, lam=lam)


def fold_weights(
    groups_with_dates: dict[int, list[tuple[object, np.ndarray]]],
    n_class: int,
    cutoff_date,
    feat_cols: list[str] | None = None,
    lam: float = 1.0,
) -> dict[str, float]:
    """v6 signal: per-feature WEIGHTS learned by ridge logistic on matured
    matched groups (hit=1 vs controls=0), inputs = within-group normalized
    ranks centered at 0 (scale-free, NaN -> neutral 0). Same maturation rule
    as fold_directions; same 20-group floor. A feature that separates at
    matched-AUC 0.9 now outvotes one at 0.6 instead of tying it."""
    feat_cols = feat_cols or list(FEATURE_NAMES)
    groups = [g for d, g in groups_with_dates.get(n_class, []) if d < cutoff_date]
    if len(groups) < 20:
        return {}
    rows, ys = [], []
    for g in groups:
        m, p = g.shape
        ranks = np.full((m, p), 0.0)
        for fi in range(p):
            col = g[:, fi]
            ok = ~np.isnan(col)
            if ok.sum() >= 2:
                order = col[ok].argsort(kind="mergesort").argsort().astype(float)
                ranks[ok, fi] = (order + 0.5) / ok.sum() - 0.5  # centered (−.5,.5)
        rows.append(ranks)
        ys.append(np.concatenate([[1.0], np.zeros(m - 1)]))
    X = np.vstack(rows)
    y = np.concatenate(ys)
    w = ridge_logistic(X, y, lam=lam)
    return {f: float(w[i]) for i, f in enumerate(feat_cols) if abs(w[i]) > 1e-9}


NEG_PER_POS = 20  # hazard training: negatives subsampled per positive
                  # (structural; the intercept absorbs the induced base-rate
                  # shift and the RANKING the engine uses is unaffected)


def fold_hazard_weights(
    ep: EventPanel,
    label_mat: np.ndarray,
    train_end_idx: int,
    purge: int,
    rng: np.random.Generator,
    lam: float | None = None,
) -> dict[str, float]:
    """v8 — discrete-time hazard formulation: instead of one snapshot per
    episode (matched groups), train on EVERY eligible stock-day with the
    label 'does a >=5x path start here?' (starts_5x_fwd). Hundreds of
    thousands of rows from the same data, and the training question is
    EXACTLY the deployment question the scanner answers each day.

    Label maturation: a day's label needs HOLD_MAX bars to resolve, so
    training rows stop at train_end_idx - purge. Inputs are centered
    cross-sectional percentiles (identical to scoring). Negatives are
    subsampled at NEG_PER_POS per positive; lambda from the adaptive
    formula on the positive count unless given.

    Overlap control: two positive start-days on the SAME symbol closer than
    the label horizon describe the same forward episode — counting both
    manufactures pseudo-independent observations. Only the first start of
    each cluster survives as a positive (the later days are dropped from
    training entirely rather than kept as negatives, which would be a lie)."""
    feat_cols = ep.feat_cols or list(FEATURE_NAMES)
    if label_mat.shape != ep.eligible.shape:
        raise ValueError(
            f"label_mat shape {label_mat.shape} does not match the panel's "
            f"(dates, symbols) grid {ep.eligible.shape} — a misaligned pivot "
            "would silently train on the wrong symbols' labels"
        )
    d_max = max(0, train_end_idx - purge)
    if d_max < 50:
        return {}
    valid = ep.eligible[:d_max]
    y_all = label_mat[:d_max]
    pos_idx = np.argwhere(valid & y_all)
    if pos_idx.shape[0]:
        keep = []
        last_kept: dict[int, int] = {}  # symbol -> day of last kept positive
        for d, s in pos_idx[np.lexsort((pos_idx[:, 0], pos_idx[:, 1]))]:
            if int(s) in last_kept and int(d) - last_kept[int(s)] < HOLD_MAX:
                continue  # same forward episode, not a new observation
            last_kept[int(s)] = int(d)
            keep.append((int(d), int(s)))
        pos_idx = np.array(keep, dtype=np.int64).reshape(-1, 2)
    # negatives exclude EVERY originally-labeled day: an overlap-dropped
    # positive is not a counter-example, it is a duplicate observation
    neg_idx = np.argwhere(valid & ~y_all)
    if pos_idx.shape[0] < 30 or neg_idx.shape[0] < pos_idx.shape[0]:
        return {}
    take_neg = min(neg_idx.shape[0], pos_idx.shape[0] * NEG_PER_POS)
    neg_idx = neg_idx[rng.choice(neg_idx.shape[0], size=take_neg, replace=False)]
    rows = np.vstack([pos_idx, neg_idx])
    y = np.concatenate([np.ones(pos_idx.shape[0]), np.zeros(neg_idx.shape[0])])

    X = np.zeros((rows.shape[0], len(feat_cols)))
    for fi, feat in enumerate(feat_cols):
        p = ep.feat_pctl[feat][rows[:, 0], rows[:, 1]].astype(float)
        X[:, fi] = np.where(np.isnan(p), 0.0, p - 0.5)  # missing -> neutral
    lam = lam if lam is not None else _adaptive_lam(int(pos_idx.shape[0]), len(feat_cols))
    w = ridge_logistic(X, y, lam=lam)
    return {f: float(w[i]) for i, f in enumerate(feat_cols) if abs(w[i]) > 1e-9}


def composite_score_weighted(ep: EventPanel, weights: dict[str, float]) -> np.ndarray:
    """(D, S) weighted sum of centered cross-sectional percentiles — the
    logit of the trained model up to the intercept, monotone in P(hit)."""
    if not weights:
        return np.full(ep.close.shape, np.nan, dtype=np.float32)
    acc = np.zeros(ep.close.shape, dtype=np.float32)
    seen = np.zeros(ep.close.shape, dtype=bool)
    for feat, w in weights.items():
        p = ep.feat_pctl.get(feat)
        if p is None:
            continue
        v = (p - 0.5) * np.float32(w)
        ok = ~np.isnan(v)
        acc[ok] += v[ok]
        seen |= ok
    return np.where(seen, acc, np.nan)


def composite_score(ep: EventPanel, directions: dict[str, float]) -> np.ndarray:
    """(D, S) mean directional percentile over the admitted features."""
    if not directions:
        return np.full(ep.close.shape, np.nan, dtype=np.float32)
    acc = np.zeros(ep.close.shape, dtype=np.float32)
    cnt = np.zeros(ep.close.shape, dtype=np.float32)
    for feat, d in directions.items():
        p = ep.feat_pctl[feat]
        v = p if d > 0 else (1.0 - p)
        ok = ~np.isnan(v)
        acc[ok] += v[ok]
        cnt[ok] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, acc / cnt, np.nan)


# ---------------------------------------------------------------- trades


@dataclass
class EventTrade:
    symbol_idx: int
    signal_day: int
    entry_day: int
    exit_day: int
    entry_px: float
    exit_px: float
    net_log_ret: float
    gross_log_ret: float  # raw fills, no spread/fees — gate 6 pairs it with net
    holding_days: int
    exit_reason: str


def simulate_trade(
    ep: EventPanel,
    s: int,
    signal_day: int,
    target_mult: float,
    stop_frac: float | None,
    hard_end: int | None = None,
) -> EventTrade | None:
    """One trade from a signal at close of signal_day. Entry next open.
    hard_end caps the exit day (exclusive fold/holdout boundary): the trade
    is force-exited at the last tradable bar at or before it, so no fold's
    trade ever marks against data outside its block."""
    e = signal_day + 1
    D = ep.dates.size
    if e >= D:
        return None
    entry_raw = ep.open_[e, s]
    if not np.isfinite(entry_raw) or entry_raw <= 0:
        return None
    hs_in = ep.half_spread[e, s]
    entry_px = entry_raw * (1.0 + (hs_in if np.isfinite(hs_in) else 0.005 / entry_raw))
    target_px = entry_px * target_mult
    stop_px = entry_px * (1.0 - stop_frac) if stop_frac is not None else -np.inf

    last = min(D - 1, e + HOLD_MAX)
    if hard_end is not None:
        last = min(last, max(e, hard_end))
    exit_day, exit_raw, reason = None, None, None
    for tau in range(e, last + 1):
        o, h, l = ep.open_[tau, s], ep.high[tau, s], ep.low[tau, s]
        if not np.isfinite(o):
            continue  # halt/missing bar: hold
        if tau == last and tau > e:
            exit_day, exit_raw, reason = tau, o, "time"  # sell at the open
            break
        # OPEN-based exits first — the open happens before any intraday path,
        # so a bar that gaps through the target fills the resting limit at
        # the open even if it later collapses through the stop. Then intraday:
        # stop (low) before target (high), conservatively.
        if stop_frac is not None and o <= stop_px:
            exit_day, exit_raw, reason = tau, o, "stop_gap"
            break
        if tau > e and o >= target_px:
            exit_day, exit_raw, reason = tau, o, "target_gap"
            break
        if stop_frac is not None and np.isfinite(l) and l <= stop_px:
            exit_day, exit_raw, reason = tau, stop_px, "stop"
            break
        # entry-bar intrabar target touches are NOT filled: a resting limit's
        # queue position on the very bar we market-bought is unknowable, and
        # granting the touch overstates the edge exactly where it matters
        if tau > e and np.isfinite(h) and h >= target_px:
            exit_day, exit_raw, reason = tau, target_px, "target"
            break
    if exit_day is None:
        # no tradable bar before the boundary: walk BACK to the last finite
        # close after entry — a name that stops printing (delisting-in-hold)
        # must exit at its last known mark, never silently vanish from the
        # record (vanishing winners is bias; vanishing losers is worse)
        for tau in range(last, e - 1, -1):
            c = ep.close[tau, s]
            if np.isfinite(c):
                exit_day, exit_raw, reason = tau, c, "data_end"
                break
        if exit_day is None:
            return None  # no post-entry print at all: entry itself was unfillable
    hs_out = ep.half_spread[exit_day, s]
    fee = ep.sell_fee_prop[exit_day, s]
    exit_px = exit_raw * (1.0 - (hs_out if np.isfinite(hs_out) else 0.005 / exit_raw))
    exit_px *= (1.0 - (fee if np.isfinite(fee) else 0.0))
    if exit_px <= 0 or exit_raw <= 0:
        return None
    return EventTrade(
        symbol_idx=s, signal_day=signal_day, entry_day=e, exit_day=exit_day,
        entry_px=float(entry_px), exit_px=float(exit_px),
        net_log_ret=float(np.log(exit_px / entry_px)),
        gross_log_ret=float(np.log(exit_raw / entry_raw)),
        holding_days=int(exit_day - e),
        exit_reason=reason,
    )


def run_event_strategy(
    ep: EventPanel,
    score: np.ndarray,
    config: dict,
    day_range: tuple[int, int],
) -> dict:
    """Slot-limited portfolio simulation over [day_range). Returns trades,
    daily net portfolio returns, and summary stats."""
    d0, d1 = day_range
    entry_pct = config["entry_pct"]
    trades: list[EventTrade] = []
    open_positions: dict[int, int] = {}  # symbol_idx -> exit_day

    for d in range(d0, min(d1, ep.dates.size - 1)):
        # release positions that exited on or before today: capital from a
        # day-d exit is available for a day-d signal (fills at d+1's open)
        open_positions = {s: x for s, x in open_positions.items() if x > d}
        slots_free = MAX_CONCURRENT - len(open_positions)
        if slots_free <= 0:
            continue
        row = score[d]
        held = np.zeros(len(ep.symbols), dtype=bool)
        if open_positions:
            held[list(open_positions.keys())] = True
        # v7 tradability floor: a name whose roundtrip spread exceeds the cap
        # cannot pay its own toll — no signal strength overrides this
        hs_row = ep.half_spread[d]
        tradable = np.isfinite(hs_row) & (2.0 * hs_row <= MAX_ROUNDTRIP_SPREAD)
        elig = ep.eligible[d] & ~np.isnan(row) & ~held & tradable
        if not elig.any():
            continue
        vals = row[elig]
        thresh = np.quantile(vals, entry_pct) if vals.size >= 50 else np.inf
        cand = np.nonzero(elig & (row >= thresh))[0]
        if cand.size == 0:
            continue
        cand = cand[np.argsort(row[cand])[::-1]][:slots_free]
        for s in cand:
            tr = simulate_trade(
                ep, int(s), d, config["target_mult"], config["stop_frac"],
                hard_end=d1 - 1,
            )
            if tr is None:
                continue
            trades.append(tr)
            open_positions[int(s)] = tr.exit_day

    # daily net portfolio returns (per-slot weight, idle capital flat)
    daily = np.zeros(max(0, d1 - d0))
    w = 1.0 / MAX_CONCURRENT
    for tr in trades:
        e, x = tr.entry_day, tr.exit_day
        if x == e:
            # same-day round trip: whole net return lands on the entry day
            if d0 <= e < d1:
                daily[e - d0] += w * float(np.expm1(tr.net_log_ret))
            continue
        px_path = ep.close[e:x + 1, tr.symbol_idx].copy()
        px_path[0] = tr.entry_px
        px_path[-1] = tr.exit_px
        # halted bars: carry the last mark forward so a single NaN doesn't
        # zero out both adjacent daily returns
        for k in range(1, px_path.size):
            if not np.isfinite(px_path[k]):
                px_path[k] = px_path[k - 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.diff(px_path) / px_path[:-1]
        r = np.nan_to_num(r, nan=0.0)
        for k, tau in enumerate(range(e + 1, x + 1)):
            if d0 <= tau < d1:
                daily[tau - d0] += w * r[k]

    logs = np.array([t.net_log_ret for t in trades])
    return {
        "config": config,
        "trades": trades,
        "n_trades": len(trades),
        "daily_net_returns": daily,
        "mean_net_log_per_trade": float(logs.mean()) if logs.size else 0.0,
        "total_net_log": float(logs.sum()) if logs.size else 0.0,
        "annualized_log_growth": float(daily.sum() / max(1, daily.size) * 252.0),
        "holding_days": [t.holding_days for t in trades],
        "exit_reasons": {r: sum(1 for t in trades if t.exit_reason == r)
                         for r in ("target", "target_gap", "stop", "stop_gap", "time", "data_end")},
    }


# ---------------------------------------------------------------- nulls


def matched_null_matrix(
    ep: EventPanel,
    trades: list[EventTrade],
    trade_configs: list[dict],
    trade_block_ends: list[int],
    score: np.ndarray | None,
    n_draws: int,
    seed: int,
    max_pool: int = 400,
) -> np.ndarray:
    """(n_draws, n_trades) matrix of null net log returns: for each realized
    trade's SIGNAL DAY, the same exit rules — that trade's OWN fold config —
    applied to a random symbol from the strategy's actual selection support
    (eligible AND scoreable that day). Per-day pools are simulated once and
    sampled."""
    rng = np.random.default_rng(seed)
    cols = []
    for tr, cfg, block_end in zip(trades, trade_configs, trade_block_ends):
        d = tr.signal_day
        support = ep.eligible[d]
        if score is not None:
            support = support & ~np.isnan(score[d])
        pool = np.nonzero(support)[0]
        if pool.size > max_pool:
            pool = rng.choice(pool, size=max_pool, replace=False)
        outs = []
        for s in pool:
            # identical block cap as the real trade: matched means matched
            t = simulate_trade(ep, int(s), d, cfg["target_mult"], cfg["stop_frac"],
                               hard_end=block_end - 1)
            if t is not None:
                outs.append(t.net_log_ret)
        cols.append(np.asarray(outs) if outs else np.asarray([0.0]))
    mat = np.empty((n_draws, len(trades)))
    for j, vec in enumerate(cols):
        mat[:, j] = vec[rng.integers(0, vec.size, size=n_draws)]
    return mat


# ------------------------------------------------------------ walk-forward


def walk_forward_train(
    ep: EventPanel,
    groups_with_dates: dict[int, list[tuple[object, np.ndarray]]],
    n_class: int,
    research_end: int,
    n_folds: int = 4,
    min_train_trades: int = 20,
    signal_mode: str = "directions",
    label_mat: np.ndarray | None = None,
    seed: int = 41,
) -> dict:
    """Expanding-window training on [0, research_end).

    Per fold: directions are learned only from matched groups whose hit
    signal is old enough to have MATURED before the train cutoff (cutoff
    minus HOLD_MAX+2 bars — a hit whose 126-bar outcome resolves after the
    cutoff has leaked its label otherwise). The config grid is scored on the
    train block; the winner runs on the purged test block; and EVERY config
    also runs on the test block, so the PBO matrix is stitched, fully
    out-of-sample daily P&L per config — no in-sample selection inside it.

    Returns concatenated OOS results (trades annotated with their fold's
    config), the stitched per-config OOS matrix, per-(fold, config) trial
    records, and the final trained spec.
    """
    fold_edges = np.linspace(int(research_end * 0.4), research_end, n_folds + 1, dtype=int)
    purge = HOLD_MAX + 2  # signal -> entry (+1) -> exit (+HOLD_MAX) full span
    oos_trades: list[EventTrade] = []
    oos_trade_configs: list[dict] = []
    oos_trade_block_ends: list[int] = []
    oos_daily = []
    fold_summaries = []
    trial_records: list[dict] = []
    config_oos_cols: dict[str, list[np.ndarray]] = {json_key(c): [] for c in CONFIG_GRID}
    last_spec = None

    for k in range(n_folds):
        train_end = fold_edges[k]
        test_start = min(research_end, train_end + purge)
        test_end = fold_edges[k + 1]
        if test_start >= test_end:
            continue
        # label-maturation cutoff: hits must have fully resolved pre-cutoff
        matured_idx = max(0, train_end - purge)
        cutoff = ep.dates[matured_idx]
        if signal_mode == "logistic":
            dirs = fold_weights(groups_with_dates, n_class, cutoff, ep.feat_cols)
        elif signal_mode == "logistic_pooled":
            dirs = fold_weights_pooled(
                groups_with_dates, TRAIN_CLASSES_POOLED, cutoff, ep.feat_cols
            )
        elif signal_mode == "hazard":
            if label_mat is None:
                raise ValueError("hazard mode requires label_mat")
            dirs = fold_hazard_weights(
                ep, label_mat, int(train_end), purge,
                np.random.default_rng(seed + k),
            )
        else:
            dirs = fold_directions(groups_with_dates, n_class, cutoff, ep.feat_cols)
        if not dirs:
            fold_summaries.append({"fold": k, "skipped": "no matured signal pre-cutoff"})
            for c in CONFIG_GRID:
                config_oos_cols[json_key(c)].append(np.zeros(test_end - test_start))
            continue
        score = (
            composite_score_weighted(ep, dirs)
            if signal_mode in ("logistic", "logistic_pooled", "hazard")
            else composite_score(ep, dirs)
        )
        best, best_cfg = None, None
        fold_tests = {}
        for cfg in CONFIG_GRID:
            train_res = run_event_strategy(ep, score, cfg, (0, train_end))
            test_res = run_event_strategy(ep, score, cfg, (test_start, test_end))
            fold_tests[json_key(cfg)] = test_res
            config_oos_cols[json_key(cfg)].append(test_res["daily_net_returns"])
            trial_records.append(
                {
                    "fold": k,
                    "config": cfg,
                    "train_n_trades": train_res["n_trades"],
                    "train_mean_net_log": train_res["mean_net_log_per_trade"],
                    "test_n_trades": test_res["n_trades"],
                    "test_mean_net_log": test_res["mean_net_log_per_trade"],
                    "test_daily_sharpe": _daily_sharpe(test_res["daily_net_returns"]),
                }
            )
            if train_res["n_trades"] >= min_train_trades:
                metric = train_res["mean_net_log_per_trade"]
                if best is None or metric > best:
                    best, best_cfg = metric, cfg
        if best_cfg is None:
            fold_summaries.append({"fold": k, "skipped": "no config with enough train trades"})
            continue
        chosen = fold_tests[json_key(best_cfg)]
        oos_trades.extend(chosen["trades"])
        oos_trade_configs.extend([best_cfg] * len(chosen["trades"]))
        oos_trade_block_ends.extend([int(test_end)] * len(chosen["trades"]))
        oos_daily.append(chosen["daily_net_returns"])
        last_spec = {"directions": dirs, "config": best_cfg}
        fold_summaries.append(
            {
                "fold": k,
                "train_days": int(train_end),
                "test_days": [int(test_start), int(test_end)],
                "chosen_config": best_cfg,
                "directions": dirs,
                "train_mean_net_log": best,
                "test_n_trades": chosen["n_trades"],
                "test_mean_net_log": chosen["mean_net_log_per_trade"],
            }
        )

    # stitched fully-OOS per-config matrix for CSCV/PBO
    config_daily = None
    config_summaries = []
    if any(cols for cols in config_oos_cols.values()):
        cols = []
        for cfg in CONFIG_GRID:
            key = json_key(cfg)
            stitched = (
                np.concatenate(config_oos_cols[key]) if config_oos_cols[key] else np.array([])
            )
            cols.append(stitched)
            config_summaries.append(
                {"config": cfg,
                 "oos_daily_sharpe": _daily_sharpe(stitched),
                 "n_trades": int(sum(t["test_n_trades"] for t in trial_records
                                     if json_key(t["config"]) == key))}
            )
        min_len = min(c.size for c in cols)
        config_daily = np.column_stack([c[:min_len] for c in cols]) if min_len > 0 else None

    return {
        "oos_trades": oos_trades,
        "oos_trade_configs": oos_trade_configs,
        "oos_trade_block_ends": oos_trade_block_ends,
        "oos_daily": np.concatenate(oos_daily) if oos_daily else np.array([]),
        "fold_summaries": fold_summaries,
        "final_spec": last_spec,
        "config_daily_matrix": config_daily,
        "config_summaries": config_summaries,
        "trial_records": trial_records,
    }


def json_key(cfg: dict) -> str:
    import json as _json

    return _json.dumps(cfg, sort_keys=True)


def _daily_sharpe(daily: np.ndarray) -> float | None:
    d = np.asarray(daily, dtype=float)
    if d.size < 20 or d.std(ddof=1) == 0:
        return None
    return float(d.mean() / d.std(ddof=1))
