# Phase-1 Acceptance — discovery → compile → gate, end to end

**Verdict: PASS** (two consecutive runs, byte-identical evidence hash)
**Evidence hash:** `1c9ac41a5b8ff0d2…` (full value in
`reports/phase1_acceptance_evidence.json`; both runs ledgered in
`ledger/acceptance_phase1.jsonl`)
**Date:** 2026-08-13 · **Harness:** `alpha_forge/phase1_acceptance.py`
(`python -m alpha_forge.phase1_acceptance`)

Acceptance proves **consistency, not performance**. The status ceiling
remains **PIPELINE_PROOF_ONLY**: this PASS says the machinery is sound and
honest — it does not say an edge exists, and nothing here changes sizing
or graduation.

## What was proven

| Criterion | Result |
|---|---|
| A1 Genome integrity | 5,146,787 rows, schema v1, knowability meta present; parquet sha `b31d411448a5…` |
| A2 Sweep fully ledgered | 12/12 deterministic candidates became acceptance-ledger TRIALs; all 12 **KILLED** (lift 0.83–0.89 < 1.5), matching the research-ledger C8 run exactly |
| A3 Cross-layer consistency | Top-lift candidate compiled through the full Section-4 battery (2018+ slice, 34k+ trades): discovery **KILLED** → battery **KILL** (permutation, DSR, PBO, matched-null gates all concur); DSR deflated by the **research** ledger's audited trial history via `stats_ledger` — acceptance replays never shrink deflation |
| A4 Transitions ledgered | Canonical 3-stage mechanism, 2024+ slice: transition 1 n=36,186 (hazard 1.13× base; volume_z and realized-vol BH-significant separators), transition 2 n=57 (hazard 0.51× base; distance-from-52w-high separator); one TRIAL per field test |
| A5 Determinism | Run 1 hash == Run 2 hash, verified against the stored ledger copy (`prior hash: match`) |

## Notes for the record

- The harness caught a real nondeterminism bug before it could taint
  evidence: compiled strategy ids used Python's per-process randomized
  `hash()`; they are now sha256-derived.
- Options-state fields ride in the genome (REAL_DELAYED, coverage from
  2026-08-10) but no options-state predicate is sweep-eligible yet —
  with two snapshot days, any such trial is UNDECIDABLE by construction.
- The survivorship-biased universe remains marked in every gate report
  (gate 7) and in the genome meta.

## What Phase 1 did NOT do

No survivor exists. No policy passed the battery. No sizing was bound, no
capital is implied, and no schedule runs without explicit approval. The
next honest steps are wider sweeps (3-stage, more grid points) under the
same ledger discipline, options-state predicates once the chain archive
deepens, and the shadow-scoring loop (§11) — each a preregistered,
ledgered exercise like everything above.
