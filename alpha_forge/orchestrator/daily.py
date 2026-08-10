"""Nightly loop — the full agent chain, idempotent and halt-on-anomaly.

Order of operations each night (each stage feeds the next; a validation
failure halts everything downstream rather than feeding it silently):

  1. verify ledger chain                  (broken chain => full stop)
  2. ingest equities + regime series      (anomalous pull => halt)
  3. pathfinder over full history         (Agent 2)
  4. reconciler grades matured predictions(Agent 7 — before anything new)
  5. confluence identifiability          (Agent 3, one family per vintage)
  6. scanner -> immutable predictions    (Agent 6)
  7. gates: pre-registered hypotheses    (demo + generated, budgeted)
  8. council over the book               (Agent 8b; empty book = no-op)
  9. replacement rate, weekly memo, FINDINGS.md, dashboard export
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
from alpha_forge.gates.permutation import benjamini_hochberg
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
from alpha_forge.research.confluence import run_confluence
from alpha_forge.research.features import fingerprint_at
from alpha_forge.research.loop import (
    generate_hypotheses,
    generator_hit_rate,
    replacement_rate,
    write_weekly_memo,
)
from alpha_forge.research.pathfinder import enumerate_paths
from alpha_forge.research.reconciler import calibration_summary, grade_all
from alpha_forge.research.scanner import scan

SAMPLE_SIZE = 200  # research sample of the eligible universe (rate-limit bound)

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
    rejected = benjamini_hochberg([p_perm], q=0.05)[0]
    permutation_result = {
        "p_value": p_perm,
        "observed": observed,
        "null_mean": float(perm_draws.mean()),
        "n_permutations": int(perm_draws.size),
        "rejected_after_correction": bool(rejected),
        "correction": "benjamini_hochberg q=0.05 over today's hypothesis family",
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
        sid = f"gen_{hyp['feature']}_{hyp['rank_direction']}_v1"
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
                "source_auc": hyp["auc"],
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

    _log("confluence: identifiability testing (one family per vintage)")
    confluence = run_confluence(panel, hits, ledger, data_vintage)
    if confluence is not None:
        n_ident = confluence.filter(pl.col("verdict") == "IDENTIFIABLE").height
        _log(f"confluence: {n_ident} identifiable (feature x class) cells")

    _log("scanner: emitting immutable predictions")
    pred_doc = scan(panel, confluence, hits, regime, ledger, data_vintage)
    if pred_doc:
        _log(f"scanner: {len(pred_doc.get('predictions', []))} predictions "
             f"({pred_doc.get('note', 'ok')})")

    gate_reports = []
    demo = gate_demo_hypothesis(ledger, panel, data_vintage)
    if demo:
        gate_reports.append(demo)
    hyps = generate_hypotheses(confluence, ledger)
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
            "## Confluence (Agent 3)\n\n"
            + (
                f"{surv.height} of {confluence.filter(pl.col('verdict') != 'UNDERPOWERED').height} "
                "tested (feature x class) cells are IDENTIFIABLE after BH correction: "
                + ", ".join(
                    f"{r['feature']}@{r['n_multiple']}x (AUC {r['auc']:.2f})"
                    for r in surv.iter_rows(named=True)
                )
                if surv.height
                else "no fingerprint feature survived BH correction on this vintage"
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
