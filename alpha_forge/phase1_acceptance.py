"""Phase-1 acceptance — prove the discovery -> compile -> gate pipeline
end-to-end on real data, deterministically (mirrors the Alpha-v0 harness:
run twice, the evidence hash must reproduce byte-identically).

Acceptance is about CONSISTENCY, not performance. The pipeline passes when:

  A1  the Genome exists, its schema version matches, and its knowability
      meta is present (the audit ran at build time);
  A2  a bounded, deterministic discovery sweep runs and EVERY candidate
      lands in the acceptance ledger as a trial (count delta == sweep size);
  A3  the sweep's top-lift candidate compiles through the FULL Section-4
      battery without error, deflating by the RESEARCH ledger's audited
      trial history (acceptance replays must never shrink deflation), and
      the verdicts are consistent: a discovery-KILLED mechanism must not
      produce a compiled PASS;
  A4  transition statistics run for the canonical 3-stage mechanism and
      every field test is ledgered;
  A5  the evidence hash is stable: a second run reproduces it exactly.

Status ceiling PIPELINE_PROOF_ONLY: a PASS here says the machinery is
sound and honest, never that an edge exists."""

from __future__ import annotations

import json
from datetime import date

import polars as pl

from alpha_forge.alpha_v0 import _canonical_hash, _file_sha256, _round_floats
from alpha_forge.config import LEDGER_DIR, REPORTS_DIR, STORE_DIR
from alpha_forge.ledger.ledger import Ledger
from alpha_forge.mechanism.compiler import compile_and_gate
from alpha_forge.mechanism.discovery import run_discovery
from alpha_forge.mechanism.episodes import Mechanism
from alpha_forge.mechanism.transitions import transition_stats

ACCEPTANCE_LEDGER = "acceptance_phase1.jsonl"
SWEEP_STAGES = 2
SWEEP_MAX_CANDIDATES = 12
COMPILE_SLICE_START = date(2018, 1, 1)
TRANSITION_SLICE_START = date(2024, 1, 1)
CANONICAL_3STAGE = Mechanism(
    (("CATALYST_SHOCK", (("k", 3),)),
     ("VOLUME_SURGE_THIN", (("min_vol_z", 1.5), ("min_spread_pctl", 0.80))),
     ("THRESHOLD_BREAK", (("b", 0.05),))),
    (5, 10),
)


def run_once(genome_path=None, acceptance_ledger_path=None,
             research_ledger_path=None, verbose=True) -> dict:
    def log(msg):
        if verbose:
            print(f"[phase1-acceptance] {msg}")

    genome_path = genome_path or (STORE_DIR / "genome" / "genome_v1.parquet")
    meta_path = genome_path.with_name(genome_path.stem + "_meta.json")
    failures: list[str] = []

    # ---- A1: genome present with knowability meta ------------------------
    genome = pl.read_parquet(genome_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1 or not meta.get("knowability"):
        failures.append("A1: genome meta missing or schema mismatch")
    genome_hash = _file_sha256(genome_path)
    log(f"A1 genome: {genome.height} rows, sha {genome_hash[:12]}")

    acc = Ledger(path=acceptance_ledger_path
                 or (LEDGER_DIR / ACCEPTANCE_LEDGER))
    research = Ledger(path=research_ledger_path) if research_ledger_path \
        else Ledger()

    # ---- A2: bounded deterministic sweep, fully ledgered -----------------
    n0 = acc.trial_count()
    results = run_discovery(genome, acc, n_stages=SWEEP_STAGES,
                            max_candidates=SWEEP_MAX_CANDIDATES)
    if acc.trial_count() - n0 != results.height:
        failures.append("A2: sweep trials not fully ledgered")
    log(f"A2 sweep: {results.height} candidates, statuses "
        f"{results['status'].value_counts().to_dicts()}")

    # ---- A3: compile top-lift candidate through the full battery ---------
    ranked = results.filter(pl.col("lift").is_not_null()).sort(
        "lift", descending=True)
    compile_payload = {"verdict": "SKIPPED", "reason": "no scored candidate"}
    if ranked.height:
        import ast

        stages, windows = ast.literal_eval(ranked["mechanism"][0])
        mech = Mechanism(stages, windows)
        disc_status = ranked["status"][0]
        from alpha_forge.research.eventbt import build_event_panel

        panel = pl.read_parquet(STORE_DIR / "equities_daily.parquet").filter(
            pl.col("date") >= COMPILE_SLICE_START)
        feats = pl.read_parquet(STORE_DIR / "features_panel.parquet").filter(
            pl.col("date") >= COMPILE_SLICE_START)
        ep = build_event_panel(panel, feats)
        compile_payload = compile_and_gate(
            ep, genome.filter(pl.col("date") >= COMPILE_SLICE_START),
            mech, acc, data_vintage="phase1-acceptance",
            stats_ledger=research)
        if disc_status == "KILLED" and compile_payload["verdict"] == "PASS":
            failures.append(
                "A3: discovery-KILLED mechanism produced a compiled PASS — "
                "layers disagree")
        log(f"A3 compile: discovery={disc_status} -> "
            f"battery={compile_payload['verdict']}")

    # ---- A4: transition statistics, ledgered -----------------------------
    t0 = acc.trial_count()
    transitions = transition_stats(
        genome.filter(pl.col("date") >= TRANSITION_SLICE_START),
        CANONICAL_3STAGE, ledger=acc)
    n_field_tests = sum(len(t.get("fields", [])) for t in transitions)
    if acc.trial_count() - t0 != n_field_tests:
        failures.append("A4: transition field tests not fully ledgered")
    log(f"A4 transitions: {[(t['transition'], t['n']) for t in transitions]}")

    # ---- A5: deterministic evidence --------------------------------------
    evidence_body = _round_floats({
        "harness": "phase1_acceptance_v1",
        "genome_sha256": genome_hash,
        "genome_rows": genome.height,
        "schema_version": meta.get("schema_version"),
        "sweep": results.select(
            ["mechanism", "n", "hits", "lift", "status", "kill_reason"]
        ).to_dicts(),
        "compile": {k: compile_payload.get(k) for k in
                    ("verdict", "reasons", "n_signals", "p_matched_null",
                     "primary", "reason")},
        "transitions": [{k: v for k, v in t.items() if k != "fields"}
                        for t in transitions],
        "transition_significant_fields": [
            [f["field"] for f in t.get("fields", [])
             if f.get("bh_significant")] for t in transitions],
        "constants": {"sweep_stages": SWEEP_STAGES,
                      "sweep_max": SWEEP_MAX_CANDIDATES,
                      "compile_slice": str(COMPILE_SLICE_START),
                      "transition_slice": str(TRANSITION_SLICE_START)},
    })
    evidence_hash = _canonical_hash(evidence_body)

    stored = None
    for e in acc.entries():
        if (e["kind"] == "RESULT"
                and e["payload"].get("mode") == "PHASE1_ACCEPTANCE"):
            stored = e["payload"]["evidence_hash"]
    if stored is not None and stored != evidence_hash:
        failures.append(f"A5: evidence hash drifted (stored {stored[:12]}, "
                        f"got {evidence_hash[:12]})")
    verdict = "PASS" if not failures else "FAIL"
    acc.append("RESULT", {
        "mode": "PHASE1_ACCEPTANCE",
        "verdict": verdict, "evidence_hash": evidence_hash,
        "failures": failures, "first_run": stored is None})

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "phase1_acceptance_evidence.json").write_text(
        json.dumps({"evidence_hash": evidence_hash, "verdict": verdict,
                    "failures": failures, "body": evidence_body},
                   indent=1, default=str), encoding="utf-8")
    log(f"A5 evidence {evidence_hash[:16]} verdict {verdict} "
        f"(prior hash: {'match' if stored == evidence_hash else stored is None and 'none' or 'DRIFT'})")
    return {"verdict": verdict, "evidence_hash": evidence_hash,
            "failures": failures}


if __name__ == "__main__":
    out = run_once()
    raise SystemExit(0 if out["verdict"] == "PASS" else 1)
