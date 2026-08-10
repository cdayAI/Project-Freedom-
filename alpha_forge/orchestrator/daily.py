"""Nightly loop — Agents 0 (data), 2 (pathfinder), Section-4 gates, findings.

Idempotent and safe to rerun: ingestion skips fresh pulls, the demo
hypothesis is not re-gated for the same data vintage (reruns would inflate
the trial count for no informational gain), and every stage that fails
validation halts downstream agents loudly instead of feeding them silently.

Order of operations each night:
  1. verify ledger chain            (broken chain => full stop)
  2. ingest + validate data         (anomalous pull => halt downstream)
  3. pathfinder over full history
  4. pre-register -> test -> gate the day's hypothesis set
  5. regenerate FINDINGS.md; print summary with cumulative trial count
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
from alpha_forge.data.universe import eligible_common_stocks, fetch_symbol_directory, research_sample
from alpha_forge.gates.dsr import sharpe_ratio
from alpha_forge.gates.gatekeeper import run_gates
from alpha_forge.gates.permutation import benjamini_hochberg
from alpha_forge.gates.walkforward import HoldoutRegistry, purged_walk_forward_splits
from alpha_forge.ledger import Ledger
from alpha_forge.reporting.findings import render_findings, write_findings
from alpha_forge.research.backtest import (
    build_month_panel,
    momentum_scores,
    random_selection_draws,
    run_config,
)
from alpha_forge.research.pathfinder import enumerate_paths

SAMPLE_SIZE = 200  # research sample of the eligible universe (rate-limit bound)

# The first cycle's pre-registered hypothesis: classical cross-sectional
# momentum (Jegadeesh & Titman 1993) — chosen from literature, not mined here.
PRIMARY_CONFIG = {"formation_months": 6, "skip_months": 1, "top_k": 10}
CONFIG_GRID = [
    {"formation_months": f, "skip_months": s, "top_k": k}
    for f in (3, 6, 9, 12)
    for s in (0, 1)
    for k in (5, 10)
]
STRATEGY_ID = "xsmom_demo_v1"


def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


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

    record_dataset(
        dataset_id="equities_daily_yahoo",
        vendor=adapter.vendor,
        description=f"daily OHLCV, uniform random sample n={len(sample)} of "
        f"{len(eligible)} eligible current US common-stock listings",
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


def gate_demo_hypothesis(ledger: Ledger, panel: pl.DataFrame, data_vintage: str) -> dict | None:
    # idempotency: one gate run per strategy per data vintage
    for e in ledger.entries():
        if (
            e["kind"] == "GATE_REPORT"
            and e["payload"].get("strategy_id") == STRATEGY_ID
            and e["payload"].get("checks", {}).get("data_vintage") == data_vintage
        ):
            _log(f"gates: {STRATEGY_ID} already gated on vintage {data_vintage} (skip)")
            return e["payload"]

    # The backtest may only touch dates whose sell-side fees are verified;
    # the bound comes from the fee tables themselves, not a hardcoded date.
    fee_floor = earliest_verified_equity_fee_date()
    panel = panel.filter(pl.col("date") >= fee_floor)
    _log(f"gates: backtest bounded to fee-verified window (>= {fee_floor})")

    # Gate 1: pre-registration BEFORE anything is computed.
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
    _log(f"gates: pre-registered {STRATEGY_ID} as {reg_id}")

    mp = build_month_panel(panel)
    n_months = mp.r_net.shape[0]
    holdout = HoldoutRegistry(n_obs=n_months, holdout_fraction=0.15)
    research_months = holdout.research_indices()
    _log(f"gates: {n_months} month-periods, research segment {research_months.size}, "
         f"holdout {n_months - research_months.size} (untouched)")

    # Config sweep — every config is one ledgered trial.
    grid_results = {}
    for cfg in CONFIG_GRID:
        scores = momentum_scores(mp, cfg["formation_months"], cfg["skip_months"])
        res = run_config(mp, scores[: research_months.size], cfg["top_k"])
        sr = None
        if res["net_returns"].size >= 12 and res["net_returns"].std() > 0:
            sr = sharpe_ratio(res["net_returns"])
        ledger.record_trial(reg_id, cfg, sr)
        grid_results[json.dumps(cfg, sort_keys=True)] = res

    primary = grid_results[json.dumps(PRIMARY_CONFIG, sort_keys=True)]
    net_r, gross_r = primary["net_returns"], primary["gross_returns"]
    if net_r.size < 24:
        _log("gates: insufficient history for the demo hypothesis — abort")
        return None

    # config matrix for PBO: align on common tail length
    min_len = min(r["net_returns"].size for r in grid_results.values())
    config_matrix = np.column_stack([r["net_returns"][-min_len:] for r in grid_results.values()])

    # permutation test: mean net monthly return vs random-selection null,
    # drawn on exactly the months the primary config traded
    valid_months = primary["month_indices"]
    _log(f"gates: permutation test, {PERMUTATION_MIN_SHUFFLES} matched draws")
    perm_draws = random_selection_draws(
        mp, valid_months, PRIMARY_CONFIG["top_k"], PERMUTATION_MIN_SHUFFLES,
        seed=1, metric="mean",
    )
    observed = float(net_r.mean())
    p_perm = float((1 + np.sum(perm_draws >= observed)) / (1 + perm_draws.size))
    # BH across today's hypothesis family (one primary hypothesis today)
    rejected = benjamini_hochberg([p_perm], q=0.05)[0]
    permutation_result = {
        "p_value": p_perm,
        "observed": observed,
        "null_mean": float(perm_draws.mean()),
        "n_permutations": int(perm_draws.size),
        "rejected_after_correction": bool(rejected),
        "correction": "benjamini_hochberg q=0.05 over today's 1 hypothesis",
    }

    _log(f"gates: null baseline, {NULL_BASELINE_MIN_DRAWS} full-path matched draws")
    null_growth = random_selection_draws(
        mp, valid_months, PRIMARY_CONFIG["top_k"], NULL_BASELINE_MIN_DRAWS,
        seed=2, metric="log_growth",
    )
    strat_growth = float(np.sum(np.log1p(np.maximum(net_r, -0.9999))))

    # walk-forward on the research segment (no fitting: stability check)
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
        "holdout_months_untouched": int(n_months - research_months.size),
    }

    report = run_gates(
        strategy_id=STRATEGY_ID,
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
    _log(f"gates: verdict {report.verdict} — {'; '.join(report.reasons) or 'clean pass'}")
    return report.to_payload()


def main() -> int:
    ledger = Ledger()
    n_entries = ledger.verify_chain()
    _log(f"ledger: chain verified, {n_entries} entries, "
         f"cumulative trials={ledger.trial_count()}")

    ingest_equities()
    panel = load_equities_panel()
    data_vintage = str(panel["date"].max())

    hits, window_counts = run_pathfinder(panel)
    gate_report = gate_demo_hypothesis(ledger, panel, data_vintage)

    gate_reports = [gate_report] if gate_report else []
    graduated = sum(1 for g in gate_reports if g["verdict"] == "PASS")
    killed = sum(1 for g in gate_reports if g["verdict"] == "KILL")
    findings = render_findings(
        ledger,
        window_counts if isinstance(window_counts, pl.DataFrame) and window_counts.height else None,
        hits if isinstance(hits, pl.DataFrame) and hits.height else None,
        gate_reports,
        dataset_note=f"Yahoo chart-API daily OHLCV through {data_vintage}, "
        "survivorship-biased current-listing sample (see data/manifest.jsonl)",
        replacement_rate_note=(
            f"pipeline age < 1 month: {graduated} graduated / {killed} killed this "
            "cycle; a meaningful monthly rate needs a running book (tracked from now on)"
        ),
    )
    write_findings(findings)
    _log(f"FINDINGS.md regenerated. Cumulative ledgered trials: {ledger.trial_count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
