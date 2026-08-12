"""Alpha-v0 — the acceptance test for the loop itself, not for an edge.

One command replays an existing KILLED candidate (the standing demo
hypothesis, xsmom_demo_v1) through the complete cycle — ingest verification,
preregistration, trials, backtest with verified costs, the Section-4 gate
battery, matched null — on an ISOLATED acceptance ledger, and emits one
deterministic evidence-chain report. Run twice, it must append nothing the
second time and reproduce a byte-identical evidence hash.

A KILL verdict is a PASS here: the test proves the loop tells the same truth
twice, with every claim hash-chained to its ledger entry. It manufactures no
performance numbers — the replayed candidate was killed, and the report says
so under a PIPELINE_PROOF_ONLY watermark.

Ledger discipline (standing decision): the research ledger is opened
READ-ONLY for statistics (DSR deflation uses its audited trial history) and
its entry count is asserted unchanged. Replay trials land in the acceptance
ledger only — a trial invented to test software must never inflate the
research record.

Usage:
    python -m alpha_forge.alpha_v0          # run twice + verdict (the test)
    python -m alpha_forge.alpha_v0 --once   # single pass (idempotent)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from alpha_forge.config import (
    LEDGER_DIR,
    MANIFEST_PATH,
    NULL_BASELINE_MIN_DRAWS,
    PERMUTATION_MIN_SHUFFLES,
    REPORTS_DIR,
    STATUS_CEILING,
    STORE_DIR,
)
from alpha_forge.costs.fees import earliest_verified_equity_fee_date
from alpha_forge.gates.dsr import sharpe_ratio
from alpha_forge.gates.gatekeeper import run_gates
from alpha_forge.gates.walkforward import HoldoutRegistry, purged_walk_forward_splits
from alpha_forge.ledger import Ledger
from alpha_forge.research.backtest import (
    build_month_panel,
    momentum_scores,
    random_selection_draws,
    run_config,
)

REPLAY_OF = "xsmom_demo_v1"
STRATEGY_ID = f"alpha_v0_replay_{REPLAY_OF}"
ACCEPTANCE_LEDGER = "acceptance_v0.jsonl"

# identical mechanics to the nightly demo path — imported constants would
# drag the whole orchestrator module in; these are pinned copies, and the
# preregistration records them so any drift is visible in the evidence
PRIMARY_CONFIG = {"formation_months": 6, "skip_months": 1, "top_k": 10}
CONFIG_GRID = [
    {"formation_months": f, "skip_months": s, "top_k": k}
    for f in (3, 6, 9, 12)
    for s in (0, 1)
    for k in (5, 10)
]
CANDIDATE_FAMILY = 4  # Bonferroni divisor used by the nightly family


def _canonical_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                   default=str).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_manifest_record(manifest_path: Path, dataset_id: str) -> dict | None:
    if not manifest_path.exists():
        return None
    rec = None
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if doc.get("dataset_id") == dataset_id:
                rec = doc
    return rec


def _original_verdicts(research: Ledger, strategy_id: str) -> list[dict]:
    return [
        {"vintage": e["payload"].get("checks", {}).get("data_vintage"),
         "verdict": e["payload"].get("verdict"),
         "entry_hash": e["hash"]}
        for e in research.entries()
        if e["kind"] == "GATE_REPORT" and e["payload"].get("strategy_id") == strategy_id
    ]


def _round_floats(obj, ndigits: int = 12):
    """Stabilize float reprs inside the evidence body. Determinism on one
    machine is exact; rounding guards against repr jitter across BLAS
    builds when the evidence is compared cross-machine."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def run_once(
    store_dir: Path | None = None,
    research_ledger_path: Path | None = None,
    acceptance_ledger_path: Path | None = None,
    reports_dir: Path | None = None,
    manifest_path: Path | None = None,
    n_permutations: int = PERMUTATION_MIN_SHUFFLES,
    n_null_draws: int = NULL_BASELINE_MIN_DRAWS,
    log=print,
) -> dict:
    """One idempotent pass. First invocation on a vintage appends the full
    chain to the acceptance ledger; every later invocation recomputes in
    memory, appends NOTHING, and must reproduce the stored evidence hash."""
    store = store_dir or STORE_DIR
    reports = reports_dir or REPORTS_DIR
    manifest = manifest_path or MANIFEST_PATH
    research = Ledger(path=research_ledger_path)  # default: the real ledger
    research_entries_before = sum(1 for _ in research.entries())
    research.verify_chain()
    audit = research.trial_audit()

    # ---- step 1: data, verified not refetched ----------------------------
    panel_path = store / "equities_daily.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"no equities panel at {panel_path} — Alpha-v0 verifies the stored "
            "vintage; run the nightly ingest first"
        )
    panel = pl.read_parquet(panel_path)
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(panel.columns)
    if missing or panel.height == 0:
        raise ValueError(f"panel validation failed: missing={sorted(missing)}, rows={panel.height}")
    vintage = str(panel["date"].max())[:10]
    panel_hash = _file_sha256(panel_path)
    provenance = _latest_manifest_record(manifest, "equities_daily_yahoo")

    acceptance = Ledger(
        path=acceptance_ledger_path or (LEDGER_DIR / ACCEPTANCE_LEDGER)
    )

    # ---- idempotence: an existing result for this vintage means run 2+ ---
    stored_result = None
    stored_entries: dict[str, str] = {}
    for e in acceptance.entries():
        p = e["payload"]
        vint = (
            p.get("vintage")
            or p.get("checks", {}).get("vintage")
            or p.get("checks", {}).get("data_vintage")
        )
        if p.get("strategy_id") == STRATEGY_ID and vint == vintage:
            stored_entries[e["kind"]] = e["hash"]
            if e["kind"] == "RESULT":
                stored_result = p
    first_run = stored_result is None

    # ---- step 2: preregistration (before any price is read for scoring) --
    originals = _original_verdicts(research, REPLAY_OF)
    if not originals:
        raise RuntimeError(
            f"no research-ledger GATE_REPORT found for {REPLAY_OF} — Alpha-v0 "
            "replays a KILLED candidate; it does not invent one"
        )
    original = originals[-1]
    fee_floor = earliest_verified_equity_fee_date()
    prereg_payload = {
        "strategy_id": STRATEGY_ID,
        "vintage": vintage,
        "mode": "ACCEPTANCE_REPLAY",
        "status_ceiling": STATUS_CEILING,
        "replay_of": REPLAY_OF,
        "original_gate_report_hash": original["entry_hash"],
        "original_verdict": original["verdict"],
        "hypothesis": "REPLAY of the standing cross-sectional 6-1 momentum "
        "demo (Jegadeesh-Titman 1993): top-10 prior-6-month winners (1-month "
        "skip), monthly rebalance, next-open fills, vs matched random null, "
        "net of full costs. Expected outcome: the original KILL, reproduced.",
        "universe": "the stored research panel (SURVIVORSHIP-BIASED current "
        "listings) at the pinned vintage",
        "parameters": {
            "primary": PRIMARY_CONFIG,
            "grid": CONFIG_GRID,
            "fills": "next_session_open",
            "costs": "max(CS,AR) half-spread x2 + SEC31 + TAF",
            "fee_verified_window_start": str(fee_floor),
            "n_permutations": int(n_permutations),
            "n_null_draws": int(n_null_draws),
            "bonferroni_family": CANDIDATE_FAMILY,
        },
        "panel_sha256": panel_hash,
    }
    if first_run:
        acceptance.append("DATA_PULL", {
            "strategy_id": STRATEGY_ID, "vintage": vintage,
            "mode": "ACCEPTANCE_REPLAY", "panel_sha256": panel_hash,
            "rows": panel.height, "provenance": provenance,
            "survivorship_free": False,
        })
        prereg_entry = acceptance.append("PREREGISTRATION", prereg_payload)
        stored_entries["DATA_PULL"] = _last_hash(acceptance, "DATA_PULL")
        stored_entries["PREREGISTRATION"] = prereg_entry["hash"]
        reg_id = prereg_entry["hash"][:16]
    else:
        reg_id = stored_entries.get("PREREGISTRATION", "")[:16]

    # ---- steps 3-6: trials, backtest, gates, matched null ----------------
    log(f"alpha_v0: vintage {vintage}, replaying {REPLAY_OF} "
        f"({'first run' if first_run else 'rerun — memory only'})")
    mp = build_month_panel(panel.filter(pl.col("date") >= fee_floor))
    n_months = mp.r_net.shape[0]
    holdout = HoldoutRegistry(n_obs=n_months, holdout_fraction=0.15)
    research_n = holdout.research_indices().size

    grid_results = {}
    for cfg in CONFIG_GRID:
        scores = momentum_scores(mp, cfg["formation_months"], cfg["skip_months"])
        res = run_config(mp, scores[:research_n], cfg["top_k"])
        res["top_k"] = cfg["top_k"]
        sr = None
        if res["net_returns"].size >= 12 and res["net_returns"].std() > 0:
            sr = sharpe_ratio(res["net_returns"])
        if first_run:
            acceptance.record_trial(reg_id, cfg, sr)
        grid_results[json.dumps(cfg, sort_keys=True)] = res

    primary = grid_results[json.dumps(PRIMARY_CONFIG, sort_keys=True)]
    net_r, gross_r = primary["net_returns"], primary["gross_returns"]
    if net_r.size < 24:
        raise RuntimeError(f"insufficient history: {net_r.size} months < 24")
    min_len = min(r["net_returns"].size for r in grid_results.values())
    config_matrix = np.column_stack(
        [r["net_returns"][-min_len:] for r in grid_results.values()]
    )

    valid_months = primary["month_indices"]
    log(f"alpha_v0: permutation test, {n_permutations} matched draws")
    perm = random_selection_draws(
        mp, valid_months, primary["top_k"], n_permutations, seed=1, metric="mean"
    )
    observed = float(net_r.mean())
    p_perm = float((1 + np.sum(perm >= observed)) / (1 + perm.size))
    permutation_result = {
        "p_value": p_perm,
        "observed": observed,
        "null_mean": float(perm.mean()),
        "n_permutations": int(perm.size),
        "rejected_after_correction": bool(p_perm <= 0.05 / CANDIDATE_FAMILY),
        "correction": f"bonferroni alpha=0.05/{CANDIDATE_FAMILY}",
    }
    log(f"alpha_v0: null baseline, {n_null_draws} full-path draws")
    null_growth = random_selection_draws(
        mp, valid_months, primary["top_k"], n_null_draws, seed=2, metric="log_growth"
    )
    strat_growth = float(np.sum(np.log1p(np.maximum(net_r, -0.9999))))

    folds = purged_walk_forward_splits(net_r.size, n_folds=5, purge=1)
    fold_srs = [
        round(sharpe_ratio(net_r[t]), 3)
        for _, t in folds
        if net_r[t].size >= 6 and net_r[t].std() > 0
    ]
    walkforward_summary = {
        "all_folds_evaluated": len(fold_srs) == len(folds),
        "fold_net_sharpes": fold_srs,
        "purge_months": 1,
    }

    if first_run:
        report = run_gates(
            strategy_id=STRATEGY_ID,
            reg_id=reg_id,
            ledger=acceptance,
            net_returns=net_r,
            gross_returns=gross_r,
            config_returns_matrix=config_matrix,
            n_trade_events=primary["n_trade_events"],
            permutation_result=permutation_result,
            null_result_inputs=(strat_growth, null_growth),
            survivorship_free=False,
            walkforward_summary=walkforward_summary,
            extra_checks={"data_vintage": vintage, "mode": "ACCEPTANCE_REPLAY",
                          "vintage": vintage},
            stats_ledger=research,  # deflation from the REAL trial history
        )
        verdict, checks, reasons = report.verdict, report.checks, report.reasons
        stored_entries["GATE_REPORT"] = _last_hash(acceptance, "GATE_REPORT")
    else:
        # memory-only recompute: same battery, nothing appended
        scratch = Ledger(path=acceptance.path.parent / "_alpha_v0_scratch.jsonl")
        if scratch.path.exists():
            scratch.path.unlink()
        scratch_reg = scratch.append("PREREGISTRATION", prereg_payload)
        report = run_gates(
            strategy_id=STRATEGY_ID,
            reg_id=scratch_reg["hash"][:16],
            ledger=scratch,
            net_returns=net_r,
            gross_returns=gross_r,
            config_returns_matrix=config_matrix,
            n_trade_events=primary["n_trade_events"],
            permutation_result=permutation_result,
            null_result_inputs=(strat_growth, null_growth),
            survivorship_free=False,
            walkforward_summary=walkforward_summary,
            extra_checks={"data_vintage": vintage, "mode": "ACCEPTANCE_REPLAY",
                          "vintage": vintage},
            stats_ledger=research,
        )
        verdict, checks, reasons = report.verdict, report.checks, report.reasons
        scratch.path.unlink()  # scratch chain discarded — it was never the record

    # ---- step 7: deterministic evidence ----------------------------------
    checks_clean = {k: v for k, v in checks.items() if k != "preregistration"}
    evidence_body = _round_floats({
        "alpha_v0": 1,
        "strategy_id": STRATEGY_ID,
        "replay_of": REPLAY_OF,
        "vintage": vintage,
        "panel_sha256": panel_hash,
        "preregistration": prereg_payload,
        "research_trial_audit": audit,
        "gate_checks": checks_clean,
        "verdict": verdict,
        "reasons": reasons,
        "original_verdict": original["verdict"],
        "original_gate_report_hash": original["entry_hash"],
    })
    evidence_hash = _canonical_hash(evidence_body)

    if first_run:
        acceptance.append("RESULT", {
            "strategy_id": STRATEGY_ID, "vintage": vintage,
            "mode": "ACCEPTANCE_REPLAY", "reg_id": reg_id,
            "verdict": verdict, "evidence_hash": evidence_hash,
        })
        stored_entries["RESULT"] = _last_hash(acceptance, "RESULT")
        stored_hash = evidence_hash
    else:
        stored_hash = stored_result["evidence_hash"]

    research_entries_after = sum(1 for _ in Ledger(path=research.path).entries())

    outcome = {
        "vintage": vintage,
        "verdict": verdict,
        "original_verdict": original["verdict"],
        "evidence_hash": evidence_hash,
        "stored_evidence_hash": stored_hash,
        "evidence_reproduced": evidence_hash == stored_hash,
        "first_run": first_run,
        "acceptance_entries": {k: v for k, v in stored_entries.items()},
        "research_ledger_untouched": research_entries_before == research_entries_after,
        "research_entries": research_entries_after,
    }

    # ---- step 8: the evidence-chain report (deterministic) ---------------
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"alpha_v0_{vintage}.md"
    report_path.write_text(
        _render_report(outcome, evidence_body, audit), encoding="utf-8"
    )
    outcome["report_path"] = str(report_path)
    log(f"alpha_v0: verdict {verdict} (original {original['verdict']}); "
        f"evidence {evidence_hash[:16]}… reproduced={outcome['evidence_reproduced']}")
    return outcome


def _last_hash(ledger: Ledger, kind: str) -> str:
    h = ""
    for e in ledger.entries():
        if e["kind"] == kind and e["payload"].get("strategy_id") == STRATEGY_ID:
            h = e["hash"]
    return h


def _render_report(outcome: dict, evidence: dict, audit: dict) -> str:
    e = evidence
    lines = [
        f"# ALPHA-V0 EVIDENCE CHAIN — {e['vintage']}",
        "",
        f"**WATERMARK: {STATUS_CEILING}.** This report proves the loop, not an "
        "edge. The replayed candidate was killed, and this replay exists to "
        "reproduce that verdict deterministically. No number below is a "
        "performance claim.",
        "",
        f"- Replay of: **{e['replay_of']}** (original verdict "
        f"**{e['original_verdict']}**, research-ledger entry "
        f"`{e['original_gate_report_hash']}`)",
        f"- Data vintage: {e['vintage']} — panel sha256 `{e['panel_sha256']}`",
        "- Universe: survivorship-biased current listings (disclosed on every "
        "surface; the anti-selection finding remains a HYPOTHESIS until "
        "point-in-time data with delisted securities is installed)",
        f"- Verdict this replay: **{outcome['verdict']}**",
        "",
        "## Deflation inputs (research ledger, read-only)",
        "",
        f"- Raw cumulative trials: {audit['raw_cumulative_trials']}",
        f"- Sharpe-bearing trials: {audit['sharpe_bearing_trials']}",
        f"- Distinct registrations: {audit['distinct_registrations']}",
        f"- Effective independent trials: {audit['effective_independent_trials']} "
        f"(bounds {audit['effective_independent_bounds']})",
        "",
        "## Not estimable (and therefore not printed)",
        "",
        "- Replacement-rate health: NOT YET ESTIMABLE",
        "- Book growth rate / CAGR / confidence intervals: NOT YET ESTIMABLE",
        "- Capacity: NOT YET ESTIMABLE",
        "",
        "## Evidence chain",
        "",
        "Acceptance-ledger entries this report derives from (hash-chained; "
        "verify with `python -m alpha_forge.ledger.verify`):",
        "",
    ]
    for kind in ("DATA_PULL", "PREREGISTRATION", "GATE_REPORT", "RESULT"):
        h = outcome["acceptance_entries"].get(kind, "")
        if h:
            lines.append(f"- {kind}: `{h}`")
    lines += [
        "",
        f"**Evidence hash (deterministic, over the full evidence body):** "
        f"`{outcome['evidence_hash']}`",
        "",
        "A rerun on this vintage must append nothing and reproduce this hash "
        "byte-identically. Gate reasons:",
        "",
    ]
    lines += [f"- {r}" for r in e["reasons"]] or ["- clean pass"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    once = "--once" in argv
    r1 = run_once()
    if once:
        return 0
    r2 = run_once()
    checks = {
        "verdict_matches_original": r2["verdict"] == r2["original_verdict"],
        "evidence_hash_identical": r1["evidence_hash"] == r2["evidence_hash"]
        and r2["evidence_reproduced"],
        "second_run_appended_nothing": not r2["first_run"],
        "research_ledger_untouched": r1["research_ledger_untouched"]
        and r2["research_ledger_untouched"],
    }
    passed = all(checks.values())
    print()
    print(f"ALPHA-V0: {'PASS' if passed else 'FAIL'}")
    for k, v in checks.items():
        print(f"  {'ok' if v else 'FAIL'}  {k}")
    print(f"  report: {r2['report_path']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
