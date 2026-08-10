"""Nightly loop — the full agent chain, idempotent and halt-on-anomaly.

Order of operations each night (each stage feeds the next; a validation
failure halts everything downstream rather than feeding it silently):

  1. verify ledger chain                  (broken chain => full stop)
  2. ingest equities + regime series      (anomalous pull => halt)
  3. pathfinder over full history         (Agent 2)
  4. reconciler grades matured predictions(Agent 7 — before anything new)
  5. confluence identifiability          (Agent 3, one family per vintage)
  6. scanner -> immutable predictions    (Agent 6)
  7. gates 1-11: event candidate (trained walk-forward) + demo + generated
  8. replacement rate, weekly memo, FINDINGS.md, dashboard export
     (council keep/kill activates when the book is non-empty)
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone

import duckdb
import numpy as np
import polars as pl

from alpha_forge.config import (
    DUCKDB_PATH,
    NULL_BASELINE_MIN_DRAWS,
    PERMUTATION_MIN_SHUFFLES,
    PREDICTIONS_DIR,
    RAW_DIR,
    STORE_DIR,
)
from alpha_forge.costs.fees import earliest_verified_equity_fee_date
from alpha_forge.data.equities import YahooDailyAdapter, load_equities_panel
from alpha_forge.data.manifest import record_dataset
from alpha_forge.data.regime import build_regime_series, load_regime
from alpha_forge.data.universe import eligible_common_stocks, fetch_symbol_directory, research_sample
from alpha_forge.gates.dsr import sharpe_ratio
from alpha_forge.gates.gatekeeper import run_gates
from alpha_forge.gates.walkforward import HoldoutRegistry, purged_walk_forward_splits
from alpha_forge.ledger import Ledger
from alpha_forge.reporting.export import build_dashboard_data, build_static_snapshot
from alpha_forge.reporting.findings import render_findings, write_findings
from alpha_forge.research.backtest import (
    MonthPanel,
    build_month_panel,
    momentum_scores,
    random_selection_draws,
    run_config,
)
from alpha_forge.research.confluence2 import build_matched_groups, run_confluence_v2
from alpha_forge.research.eventbt import (
    CONFIG_GRID as EVENT_CONFIG_GRID,
    build_event_panel,
    matched_null_matrix,
    walk_forward_train,
)
from alpha_forge.research.features import fingerprint_at
from alpha_forge.research.features_panel import build_features_panel
from alpha_forge.research.loop import (
    generate_hypotheses,
    generate_hypotheses_from_calibration,
    generator_hit_rate,
    replacement_rate,
    write_weekly_memo,
)
from alpha_forge.research.pathfinder import enumerate_paths
from alpha_forge.research.reconciler import GRADES_PATH, calibration_summary, grade_all
from alpha_forge.research.scanner import scan

SAMPLE_SIZE = 1500  # research sample of the eligible universe
HOLDOUT_FRACTION = 0.15
# v2: signal-day features (v1 had a one-day lookahead via entry-day join),
# label-matured direction cutoffs, block-capped trades, fully-OOS PBO matrix
EVENT_STRATEGY_ID = "evt_fp5x_v2"
EVENT_CLASS = 5  # pre-registered primary class: 5x paths (largest sample)
# The night gates at most this many candidates; each permutation p-value is
# Bonferroni-corrected against the whole family, not tested alone.
NIGHTLY_CANDIDATE_BUDGET = 4

# The standing demo hypothesis (cycle 1's pre-registered momentum, kept as a
# permanent null-hypothesis exercise for pipeline regression testing).
PRIMARY_CONFIG = {"formation_months": 6, "skip_months": 1, "top_k": 10}
CONFIG_GRID = [
    {"formation_months": f, "skip_months": s, "top_k": k}
    for f in (3, 6, 9, 12)
    for s in (0, 1)
    for k in (5, 10)
]
STRATEGY_ID = "xsmom_demo_v1"
GENERATED_TOP_K_GRID = (5, 10, 20)


def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- ingest


def ingest_equities(force: bool = False) -> dict:
    marker = RAW_DIR / f"ingest_{date.today().isoformat()}.json"
    if marker.exists() and not force and (STORE_DIR / "equities_daily.parquet").exists():
        _log("ingest: today's pull already on disk (idempotent skip)")
        return json.loads(marker.read_text())

    _log("ingest: fetching NASDAQ Trader symbol directory")
    directory = fetch_symbol_directory()
    eligible = eligible_common_stocks(directory)
    sample = research_sample(eligible, SAMPLE_SIZE)
    _log(f"ingest: universe {len(eligible)} eligible common stocks; "
         f"seeded sample of {len(sample)} for daily pulls")

    adapter = YahooDailyAdapter()
    summary = adapter.ingest(sample)
    summary["universe_size"] = len(eligible)
    summary["sample_size"] = len(sample)

    if summary["ingested"] < summary["sample_size"] * 0.5:
        raise RuntimeError(
            f"ingest anomaly: only {summary['ingested']}/{summary['sample_size']} "
            "series passed validation — halting downstream agents"
        )

    _log("ingest: building SPY/^VIX regime series")
    build_regime_series(adapter)

    record_dataset(
        dataset_id="equities_daily_yahoo",
        vendor=adapter.vendor,
        description=f"daily OHLCV, uniform random sample n={len(sample)} of "
        f"{len(eligible)} eligible current US common-stock listings + SPY/^VIX regime",
        known_biases=adapter.known_biases,
        validation_summary={
            "requested": summary["requested"],
            "ingested": summary["ingested"],
            "quarantined": len(summary["quarantined"]),
        },
        survivorship_free=adapter.survivorship_free,
    )
    marker.write_text(json.dumps({k: v for k, v in summary.items() if k != "validation_reports"}))
    _log(f"ingest: {summary['ingested']} series in store, "
         f"{len(summary['quarantined'])} quarantined")
    return summary


# ------------------------------------------------------------- pathfinder


def run_pathfinder(panel: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    _log("pathfinder: enumerating N-x paths in rolling 6-month windows")
    hits, window_counts = enumerate_paths(panel)
    if hits.height:
        hits.write_parquet(STORE_DIR / "path_hits.parquet")
        window_counts.write_parquet(STORE_DIR / "path_window_counts.parquet")
        con = duckdb.connect(str(DUCKDB_PATH))
        hits_pq = str(STORE_DIR / "path_hits.parquet").replace("'", "''")
        counts_pq = str(STORE_DIR / "path_window_counts.parquet").replace("'", "''")
        con.execute(f"CREATE OR REPLACE VIEW path_hits AS SELECT * FROM read_parquet('{hits_pq}')")
        con.execute(
            f"CREATE OR REPLACE VIEW path_window_counts AS SELECT * FROM read_parquet('{counts_pq}')"
        )
        con.close()
    _log(f"pathfinder: {hits.height if hits.height else 0} hits recorded")
    return hits, window_counts


# ------------------------------------------------------------------ gates


def _feature_scores(mp: MonthPanel, panel: pl.DataFrame, feature: str, ascending: bool) -> np.ndarray:
    """Score matrix (months x symbols): fingerprint feature at each signal
    bar, sign-flipped so 'descending rank' always means 'select the max'."""
    M = mp.signal_idx.size - 1
    S = len(mp.symbols)
    scores = np.full((M, S), np.nan)
    by_symbol = {}
    for (sym,), g in panel.group_by("symbol"):
        g = g.sort("date")
        by_symbol[str(sym)] = (
            g["date"].to_numpy(),
            g["open"].to_numpy().astype(float),
            g["high"].to_numpy().astype(float),
            g["low"].to_numpy().astype(float),
            g["close"].to_numpy().astype(float),
            g["volume"].to_numpy().astype(float),
        )
    for s, sym in enumerate(mp.symbols):
        if sym not in by_symbol:
            continue
        dates, o, h, l, c, v = by_symbol[sym]
        date_to_idx = {d: i for i, d in enumerate(dates)}
        for m in range(M):
            sig_date = mp.dates[mp.signal_idx[m]]
            i = date_to_idx.get(sig_date)
            if i is None or i < 60:
                continue
            val = fingerprint_at(i, o, h, l, c, v).get(feature, float("nan"))
            if not np.isnan(val):
                scores[m, s] = -val if ascending else val
    return scores


def _gate_candidate(
    ledger: Ledger,
    mp: MonthPanel,
    strategy_id: str,
    reg_id: str,
    grid_results: dict[str, dict],
    primary_key: str,
    data_vintage: str,
) -> dict | None:
    """Shared Section-4 gate run for any candidate with a config grid."""
    primary = grid_results[primary_key]
    net_r, gross_r = primary["net_returns"], primary["gross_returns"]
    if net_r.size < 24:
        _log(f"gates: {strategy_id}: insufficient history — abort")
        return None

    min_len = min(r["net_returns"].size for r in grid_results.values())
    if min_len < 32:
        _log(f"gates: {strategy_id}: config matrix too short for CSCV — abort")
        return None
    config_matrix = np.column_stack([r["net_returns"][-min_len:] for r in grid_results.values()])

    valid_months = primary["month_indices"]
    _log(f"gates: {strategy_id}: permutation test, {PERMUTATION_MIN_SHUFFLES} matched draws")
    perm_draws = random_selection_draws(
        mp, valid_months, primary["top_k"], PERMUTATION_MIN_SHUFFLES, seed=1, metric="mean"
    )
    observed = float(net_r.mean())
    p_perm = float((1 + np.sum(perm_draws >= observed)) / (1 + perm_draws.size))
    rejected = p_perm <= 0.05 / NIGHTLY_CANDIDATE_BUDGET
    permutation_result = {
        "p_value": p_perm,
        "observed": observed,
        "null_mean": float(perm_draws.mean()),
        "n_permutations": int(perm_draws.size),
        "rejected_after_correction": bool(rejected),
        "correction": f"bonferroni alpha=0.05/{NIGHTLY_CANDIDATE_BUDGET} across "
        "the night's candidate family",
    }

    _log(f"gates: {strategy_id}: null baseline, {NULL_BASELINE_MIN_DRAWS} full-path draws")
    null_growth = random_selection_draws(
        mp, valid_months, primary["top_k"], NULL_BASELINE_MIN_DRAWS, seed=2, metric="log_growth"
    )
    strat_growth = float(np.sum(np.log1p(np.maximum(net_r, -0.9999))))

    folds = purged_walk_forward_splits(net_r.size, n_folds=5, purge=1)
    fold_srs = []
    for _, test_idx in folds:
        seg = net_r[test_idx]
        if seg.size >= 6 and seg.std() > 0:
            fold_srs.append(round(sharpe_ratio(seg), 3))
    walkforward_summary = {
        "all_folds_evaluated": len(fold_srs) == len(folds),
        "fold_net_sharpes": fold_srs,
        "purge_months": 1,
    }

    report = run_gates(
        strategy_id=strategy_id,
        reg_id=reg_id,
        ledger=ledger,
        net_returns=net_r,
        gross_returns=gross_r,
        config_returns_matrix=config_matrix,
        n_trade_events=primary["n_trade_events"],
        permutation_result=permutation_result,
        null_result_inputs=(strat_growth, null_growth),
        survivorship_free=False,
        walkforward_summary=walkforward_summary,
        extra_checks={"data_vintage": data_vintage},
    )
    _log(f"gates: {strategy_id}: verdict {report.verdict} — "
         f"{'; '.join(report.reasons) or 'clean pass'}")
    return report.to_payload()


def _already_gated(ledger: Ledger, strategy_id: str, data_vintage: str) -> dict | None:
    for e in ledger.entries():
        if (
            e["kind"] == "GATE_REPORT"
            and e["payload"].get("strategy_id") == strategy_id
            and e["payload"].get("checks", {}).get("data_vintage") == data_vintage
        ):
            return e["payload"]
    return None


def gate_demo_hypothesis(ledger: Ledger, panel: pl.DataFrame, data_vintage: str) -> dict | None:
    existing = _already_gated(ledger, STRATEGY_ID, data_vintage)
    if existing:
        _log(f"gates: {STRATEGY_ID} already gated on vintage {data_vintage} (skip)")
        return existing

    fee_floor = earliest_verified_equity_fee_date()
    panel = panel.filter(pl.col("date") >= fee_floor)
    reg_id = ledger.preregister(
        hypothesis="Cross-sectional 6-1 momentum (Jegadeesh-Titman 1993): top-10 "
        "prior-6-month winners (1-month skip) among eligible names, monthly "
        "rebalance, next-open fills, beat a matched random-selection null net "
        "of full costs.",
        universe="uniform random sample (seed 20260810) of current US common-stock "
        "listings, price>=$1, median 20d dollar volume>=$500k — SURVIVORSHIP-BIASED",
        parameters={"primary": PRIMARY_CONFIG, "grid": CONFIG_GRID,
                    "fills": "next_session_open",
                    "costs": "CS half-spread x2 + SEC31 + TAF",
                    "fee_verified_window_start": str(fee_floor)},
    )
    mp = build_month_panel(panel)
    n_months = mp.r_net.shape[0]
    holdout = HoldoutRegistry(n_obs=n_months, holdout_fraction=0.15)
    research_n = holdout.research_indices().size
    _log(f"gates: {n_months} month-periods, research {research_n}, "
         f"holdout {n_months - research_n} (untouched)")

    grid_results = {}
    for cfg in CONFIG_GRID:
        scores = momentum_scores(mp, cfg["formation_months"], cfg["skip_months"])
        res = run_config(mp, scores[:research_n], cfg["top_k"])
        res["top_k"] = cfg["top_k"]
        sr = None
        if res["net_returns"].size >= 12 and res["net_returns"].std() > 0:
            sr = sharpe_ratio(res["net_returns"])
        ledger.record_trial(reg_id, cfg, sr)
        grid_results[json.dumps(cfg, sort_keys=True)] = res

    primary_key = json.dumps(PRIMARY_CONFIG, sort_keys=True)
    report = _gate_candidate(ledger, mp, STRATEGY_ID, reg_id, grid_results, primary_key, data_vintage)

    # persist net/gross cumulative curves for the dashboard
    primary = grid_results[primary_key]
    if primary["net_returns"].size:
        pl.DataFrame(
            {
                "month": [str(m) for m in range(primary["net_returns"].size)],
                "gross": np.cumprod(1 + primary["gross_returns"]),
                "net": np.cumprod(1 + primary["net_returns"]),
            }
        ).write_parquet(STORE_DIR / "backtest_curves.parquet")
    return report


def gate_generated_hypotheses(
    ledger: Ledger, panel: pl.DataFrame, hypotheses: list[dict], data_vintage: str
) -> list[dict]:
    """Test the night's generated hypotheses under the compute budget.
    Every config is a ledgered trial; the generator is recorded in the
    preregistration so meta-learning can grade it."""
    if not hypotheses:
        return []
    fee_floor = earliest_verified_equity_fee_date()
    panel = panel.filter(pl.col("date") >= fee_floor)
    mp = build_month_panel(panel)
    n_months = mp.r_net.shape[0]
    research_n = HoldoutRegistry(n_obs=n_months, holdout_fraction=0.15).research_indices().size

    reports = []
    for hyp in hypotheses:
        gen_tag = "cal" if hyp["generator"].startswith("calibration") else "conf"
        sid = f"gen_{gen_tag}_{hyp['feature']}_{hyp['rank_direction']}_v1"
        existing = _already_gated(ledger, sid, data_vintage)
        if existing:
            _log(f"gates: {sid} already gated on vintage {data_vintage} (skip)")
            reports.append(existing)
            continue
        reg_id = ledger.preregister(
            hypothesis=hyp["hypothesis"],
            universe="same eligible sample as the panel (SURVIVORSHIP-BIASED)",
            parameters={
                "generator": hyp["generator"],
                "feature": hyp["feature"],
                "rank_direction": hyp["rank_direction"],
                "source_auc": hyp.get("auc", hyp.get("auc_win_vs_loss")),
                "top_k_grid": list(GENERATED_TOP_K_GRID),
                "fills": "next_session_open",
                "fee_verified_window_start": str(fee_floor),
            },
        )
        scores = _feature_scores(mp, panel, hyp["feature"], hyp["rank_direction"] == "asc")
        grid_results = {}
        for k in GENERATED_TOP_K_GRID:
            res = run_config(mp, scores[:research_n], k)
            res["top_k"] = k
            sr = None
            if res["net_returns"].size >= 12 and res["net_returns"].std() > 0:
                sr = sharpe_ratio(res["net_returns"])
            ledger.record_trial(reg_id, {"feature": hyp["feature"], "top_k": k}, sr)
            grid_results[str(k)] = res
        primary_key = str(GENERATED_TOP_K_GRID[1])  # pre-registered primary k=10
        report = _gate_candidate(ledger, mp, sid, reg_id, grid_results, primary_key, data_vintage)
        if report:
            reports.append(report)
    return reports


# ------------------------------------------------------- event candidate


def gate_event_candidate(
    ledger: Ledger,
    panel: pl.DataFrame,
    features: pl.DataFrame,
    hits: pl.DataFrame,
    groups_by_class: dict,
    data_vintage: str,
) -> dict | None:
    """The system's primary thesis, trained on past data: fingerprint-scored
    entries toward 5x paths, walk-forward-trained, all twelve configs
    ledgered, gates 1-11 applied to the OOS record."""
    existing = _already_gated(ledger, EVENT_STRATEGY_ID, data_vintage)
    if existing:
        _log(f"gates: {EVENT_STRATEGY_ID} already gated on vintage {data_vintage} (skip)")
        return existing

    reg_id = ledger.preregister(
        hypothesis="Event strategy: enter names whose fingerprint state is in "
        "the cross-sectional extreme (directions learned per walk-forward fold "
        "from matched confluence groups predating the fold), next-open fills, "
        "exit at target multiple / stop / 126-bar time limit under full costs "
        "and 10-slot capital. OOS record beats matched same-mechanics nulls.",
        universe="uniform random sample of current listings (SURVIVORSHIP-"
        "BIASED); holdout = final 15% of trading days, untouched",
        parameters={
            "generator": "event_fingerprint_v1",
            "class": EVENT_CLASS,
            "config_grid": EVENT_CONFIG_GRID,
            "max_concurrent": 10,
            "hold_max_bars": 126,
            "fills": "next_open + half-spread; stop/target gap-aware; "
            "date-aware sell fees",
        },
    )

    _log("event: building event panel (wide arrays + feature percentiles)")
    # the event engine pays sell-side fees on every exit: bound it to the
    # fee-verified window (features stay full-history for warmup correctness)
    fee_floor = earliest_verified_equity_fee_date()
    panel_bt = panel.filter(pl.col("date") >= fee_floor)
    features_bt = features.filter(pl.col("date") >= fee_floor)
    ep = build_event_panel(panel_bt, features_bt)
    research_end = int(ep.dates.size * (1 - HOLDOUT_FRACTION))

    _log("event: walk-forward training")
    wf = walk_forward_train(ep, groups_by_class, EVENT_CLASS, research_end)
    # EVERY (fold, config) evaluation is a trial; the recorded statistic is
    # the config's OOS daily Sharpe on that fold's test block (a genuine
    # per-period Sharpe, comparable across trials; None when degenerate)
    for tr_rec in wf["trial_records"]:
        ledger.record_trial(
            reg_id,
            {"fold": tr_rec["fold"], **tr_rec["config"]},
            tr_rec["test_daily_sharpe"],
        )
    oos_trades = wf["oos_trades"]
    oos_daily = wf["oos_daily"]
    _log(f"event: {len(oos_trades)} OOS trades across folds; "
         f"{sum(1 for f in wf['fold_summaries'] if 'skipped' not in f)} live folds")
    if len(oos_trades) < 10 or oos_daily.size < 100 or wf["final_spec"] is None:
        result = {
            "strategy_id": EVENT_STRATEGY_ID,
            "verdict": "KILL",
            "reasons": [f"insufficient OOS record: {len(oos_trades)} trades"],
            "checks": {"data_vintage": data_vintage,
                       "fold_summaries": wf["fold_summaries"]},
            "reg_id": reg_id,
        }
        ledger.append("GATE_REPORT", result)
        return result

    final_cfg = wf["final_spec"]["config"]
    trade_logs = np.array([t.net_log_ret for t in oos_trades])
    gross_logs = np.array([t.gross_log_ret for t in oos_trades])
    observed_mean = float(trade_logs.mean())
    strat_total = float(trade_logs.sum())

    if oos_daily.std() == 0:
        result = {
            "strategy_id": EVENT_STRATEGY_ID,
            "verdict": "KILL",
            "reasons": ["degenerate OOS daily series (zero variance) — nothing "
                        "statistically evaluable was traded"],
            "checks": {"data_vintage": data_vintage,
                       "fold_summaries": wf["fold_summaries"]},
            "reg_id": reg_id,
        }
        ledger.append("GATE_REPORT", result)
        return result

    _log(f"event: matched nulls ({PERMUTATION_MIN_SHUFFLES} draws, same mechanics)")
    from alpha_forge.research.eventbt import composite_score

    final_score = composite_score(ep, wf["final_spec"]["directions"])
    null_mat = matched_null_matrix(
        ep, oos_trades, wf["oos_trade_configs"], wf["oos_trade_block_ends"],
        final_score, PERMUTATION_MIN_SHUFFLES, seed=3,
    )
    null_means = null_mat.mean(axis=1)
    null_totals = null_mat.sum(axis=1)
    p_perm = float((1 + np.sum(null_means >= observed_mean)) / (1 + null_means.size))
    rejected = p_perm <= 0.05 / NIGHTLY_CANDIDATE_BUDGET
    permutation_result = {
        "p_value": p_perm,
        "observed": observed_mean,
        "null_mean": float(null_means.mean()),
        "n_permutations": int(null_means.size),
        "rejected_after_correction": bool(rejected),
        "correction": f"bonferroni alpha=0.05/{NIGHTLY_CANDIDATE_BUDGET} across "
        "the night's candidate family",
    }
    null_p95_total = float(np.percentile(null_totals, 95))

    # gate-10 stress uses the RESEARCH era's gap distribution only — holdout
    # gap statistics must not leak into a research-segment verdict
    research_gaps = None
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (ep.open_[1:research_end] / ep.close[: research_end - 1] - 1.0).ravel()
    research_gaps = g[~np.isnan(g)]

    fold_srs = [
        round(f["test_mean_net_log"], 4)
        for f in wf["fold_summaries"] if "skipped" not in f
    ]
    walkforward_summary = {
        "all_folds_evaluated": all("skipped" not in f for f in wf["fold_summaries"]),
        "fold_summaries": wf["fold_summaries"],
        "fold_test_mean_net_logs": fold_srs,
        "purge_bars": 126,
        "holdout_days_untouched": int(ep.dates.size - research_end),
    }

    report = run_gates(
        strategy_id=EVENT_STRATEGY_ID,
        reg_id=reg_id,
        ledger=ledger,
        net_returns=oos_daily,
        gross_returns=oos_daily,  # per-day series is already net; gross-vs-net
        config_returns_matrix=wf["config_daily_matrix"],
        n_trade_events=len(oos_trades),
        permutation_result=permutation_result,
        null_result_inputs=(strat_total, null_totals),
        survivorship_free=False,
        walkforward_summary=walkforward_summary,
        extra_checks={
            "data_vintage": data_vintage,
            "final_spec": {
                "directions": wf["final_spec"]["directions"],
                "config": final_cfg,
            },
            "exit_reasons": {
                r: sum(1 for t in oos_trades if t.exit_reason == r)
                for r in ("target", "target_gap", "stop", "stop_gap", "time", "data_end")
            },
        },
        net_vs_gross_override={
            "gross_total_return": float(np.expm1(gross_logs.sum())),
            "net_total_return": float(np.expm1(trade_logs.sum())),
            "basis": "compounded per-trade OOS returns (raw fills vs full costs)",
        },
        stress_sizing_inputs={
            "trade_net_returns": np.expm1(trade_logs),
            "trade_entry_days": [t.entry_day for t in oos_trades],
            "trade_exit_days": [t.exit_day for t in oos_trades],
            "overnight_gaps": research_gaps,
            "null_p95_total_log": null_p95_total,
            "account_equity": 2000.0,
            "per_trade_notional": 200.0,
        },
    )
    _log(f"event: verdict {report.verdict} — {'; '.join(report.reasons) or 'clean pass'}")
    return report.to_payload()


# ------------------------------------------------------------------- main


def main() -> int:
    ledger = Ledger()
    n_entries = ledger.verify_chain()
    _log(f"ledger: chain verified, {n_entries} entries, "
         f"cumulative trials={ledger.trial_count()}")

    ingest_equities()
    panel = load_equities_panel()
    data_vintage = str(panel["date"].max())
    try:
        regime = load_regime()
    except FileNotFoundError:
        regime = build_regime_series()

    hits, window_counts = run_pathfinder(panel)

    _log("reconciler: grading matured predictions")
    grades = grade_all(panel, ledger)
    if grades:
        _log(f"reconciler: {len(grades)} (file, horizon) grades recorded")

    _log("features: building vectorized panel (1e-9-verified vs fingerprint_at)")
    features = build_features_panel(panel)
    features.write_parquet(STORE_DIR / "features_panel.parquet")

    # holdout boundary on trading days: hits whose LABELS mature inside the
    # holdout era never inform confluence, training, or the scanner — the
    # filter is on window_end (label resolution), not entry (label creation)
    all_days = np.sort(panel["date"].unique().to_numpy())
    boundary_date = all_days[int(all_days.size * (1 - HOLDOUT_FRACTION))]
    hits_research = (
        hits.filter(pl.col("window_end").str.slice(0, 10).str.to_date() < boundary_date)
        if isinstance(hits, pl.DataFrame) and hits.height
        else hits
    )

    _log("confluence v2: time-matched identifiability (supersedes v1)")
    groups_by_class = build_matched_groups(features, hits_research, seed=17) \
        if isinstance(hits_research, pl.DataFrame) and hits_research.height else {}
    confluence = run_confluence_v2(
        features, hits_research, ledger, data_vintage, groups_by_class=groups_by_class
    )
    if confluence is not None:
        n_ident = confluence.filter(pl.col("verdict") == "IDENTIFIABLE").height
        _log(f"confluence v2: {n_ident} identifiable (feature x class) cells "
             "(era confound removed)")

    _log("scanner: emitting immutable predictions")
    conf_for_scanner = (
        confluence.rename({"matched_auc": "auc"}) if confluence is not None else None
    )
    pred_doc = scan(panel, conf_for_scanner, hits_research, regime, ledger, data_vintage)
    if pred_doc:
        _log(f"scanner: {len(pred_doc.get('predictions', []))} predictions "
             f"({pred_doc.get('note', 'ok')})")

    gate_reports = []
    event_report = gate_event_candidate(
        ledger, panel, features, hits_research, groups_by_class, data_vintage
    )
    if event_report:
        gate_reports.append(event_report)

    demo = gate_demo_hypothesis(ledger, panel, data_vintage)
    if demo:
        gate_reports.append(demo)

    # calibration-driven hypotheses take priority in the budget: learning
    # from being wrong beats re-mining the same identifiability signal
    cal_grades = pl.read_parquet(GRADES_PATH) if GRADES_PATH.exists() else None
    hyps = generate_hypotheses_from_calibration(cal_grades, PREDICTIONS_DIR)
    hyps += generate_hypotheses(conf_for_scanner, ledger)
    hyps = hyps[:2]
    _log(f"loop: {len(hyps)} generated hypotheses under tonight's budget")
    gate_reports.extend(gate_generated_hypotheses(ledger, panel, hyps, data_vintage))

    rep = replacement_rate(ledger)
    gen_stats = generator_hit_rate(ledger)
    calibration = calibration_summary()

    memo = write_weekly_memo(
        ledger,
        rep,
        calibration,
        best_validated="N/A — no strategy has passed all gates plus live requirements",
        open_questions=[
            "options/futures data purchase (Polygon/ORATS/databento keys) to leave DESIGN mode",
            "survivorship-free equities vendor (Norgate/Sharadar) to clear the gate-7 flag",
            "catalyst calendars (earnings/FDA/short interest) — hits are still tagged NONE",
        ],
    )
    if memo:
        _log(f"memo: wrote {memo}")

    rate = rep["trailing_3m_rate"]
    rate_str = ("N/A (pipeline age < 1 month)" if rate is None
                else "inf (no deaths)" if rate == float("inf") else f"{rate:.2f}")
    graduated = sum(1 for g in gate_reports if g["verdict"] == "PASS")
    killed = sum(1 for g in gate_reports if g["verdict"] == "KILL")

    extra = []
    if confluence is not None:
        surv = confluence.filter(pl.col("verdict") == "IDENTIFIABLE")
        extra.append(
            "## Confluence v2 (Agent 3) — TIME-MATCHED controls\n\n"
            "Each hit is ranked only against same-era controls (other symbols, "
            "±10 trading days), so era effects cancel; v1's unmatched design is "
            "superseded and its numbers should not be quoted.\n\n"
            + (
                f"{surv.height} of {confluence.filter(pl.col('verdict') != 'UNDERPOWERED').height} "
                "tested (feature x class) cells are IDENTIFIABLE after BH correction: "
                + ", ".join(
                    f"{r['feature']}@{r['n_multiple']}x (matched AUC {r['matched_auc']:.2f})"
                    for r in surv.iter_rows(named=True)
                )
                if surv.height
                else "NO fingerprint feature survives once controls are drawn from "
                "the hit's own era — v1's identifiability was largely time-"
                "confounded, which is exactly what this correction exists to expose"
            )
        )
    if event_report is not None:
        checks = event_report.get("checks", {})
        spec = checks.get("final_spec", {})
        sizing = checks.get("sizing", {})
        fold_sums = checks.get("fold_summaries") or checks.get("walkforward", {}).get(
            "fold_summaries", []
        )
        extra.append(
            "## Event strategy (trained on past data)\n\n"
            f"**{EVENT_STRATEGY_ID}** — verdict **{event_report['verdict']}**. "
            f"Walk-forward folds: {len([f for f in fold_sums if 'skipped' not in f]) or 'see report'}; "
            f"final spec: {spec.get('config')}; directions: {spec.get('directions')}. "
            + (
                f"Ruin-constrained size: {sizing['constrained_optimum']['fraction']:.2f} "
                f"of equity per position (Kelly {sizing.get('kelly_fraction', 0):.2f})."
                if sizing.get("constrained_optimum")
                else "No ruin-safe sizing (or killed before sizing)."
            )
        )
    if gen_stats:
        extra.append(
            "## Generator meta-learning\n\n"
            + "\n".join(
                f"- {g}: {s['passed']}/{s['tested']} survived gates"
                for g, s in gen_stats.items()
            )
        )

    findings = render_findings(
        ledger,
        window_counts if isinstance(window_counts, pl.DataFrame) and window_counts.height else None,
        hits if isinstance(hits, pl.DataFrame) and hits.height else None,
        gate_reports,
        dataset_note=f"Yahoo chart-API daily OHLCV through {data_vintage}, "
        "survivorship-biased current-listing sample (see data/manifest.jsonl)",
        replacement_rate_note=f"trailing 3m: {rate_str} "
        f"({graduated} graduated / {killed} killed this cycle)",
        extra_sections=extra,
    )
    write_findings(findings)

    build_dashboard_data(ledger, rep)
    snap = build_static_snapshot()
    _log(f"dashboard: data exported{' + static snapshot ' + snap if snap else ''}")
    _log(f"FINDINGS.md regenerated. Cumulative ledgered trials: {ledger.trial_count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
