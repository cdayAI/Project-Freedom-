# ALPHA FORGE — Architecture

**Objective function (explicit, everywhere):** maximize expected log-wealth
growth per year, net of all costs, subject to
P(drawdown below 50% of high-water mark) ≤ human-set limit.
Log-wealth is the objective because geometric compounding over decades makes
drawdown control part of the return, not a preference.

**North-star health metric:** pipeline replacement rate — strategies
graduating all gates per month vs strategies dying. Reported in every
summary; a rate < 1.0 leads the weekly memo.

## Status: cycle 2 (full agent chain on equities)

Built and running nightly: Agents 0-8b plus the loop (9) and execution
readiness (10) on equities. In one pass: ingest + regime series →
pathfinder → reconciler grading → confluence identifiability (every cell a
ledgered trial, BH-corrected) → immutable predictions → gated hypotheses
(standing demo + up to 2 generated from confluence survivors per night) →
replacement rate → weekly memo → FINDINGS.md → dashboard export + static
snapshot. Options/futures remain DESIGN-mode (no keys); catalyst calendars
are not yet ingested (hits tagged NONE); sequences remain structurally zero
until first graduation.

Agent map: 0 data (`data/`), 1 costs (`costs/`), 2 pathfinder, 3 confluence,
4 sizing, 5 traps, 6 scanner, 7 reconciler, 8a capacity, 8b council, 9 loop
(all `research/`), 10 execution (`execution/`).

## Language choices (justified per Operating Rules)

| Component | Language | Why |
|---|---|---|
| Statistics/gates (DSR, PBO, permutation) | Python (numpy/scipy) | The validated reference implementations and their test suite live here. Ports must match to 1e-9 (`tests/test_dsr.py`, `tests/test_pbo.py` pin golden values). |
| Data layer | Python (Polars/Parquet/DuckDB) | Columnar, lazy, zero-server; Parquet readable from any future language. |
| Pathfinder | Python (numpy inner loops) | Full run over 200 symbols × 10y × monthly-strided windows takes seconds. A Rust port is justified only after profiling shows >5x need on the full-universe enumeration (Operating Rule); today it would be a build dependency bought for nothing. |
| Orchestrator | Python stdlib | Simplest thing that survives a nightly cron. |
| Dashboard | TypeScript/React | Roadmap; reads the same DuckDB/Parquet store. |

One toolchain today. Every added toolchain is a nightly-loop failure mode.

## Data flow

```
NASDAQ Trader symbol directory ──► eligible universe ──► seeded uniform sample
                                                              │
Yahoo daily OHLCV ──► validation (rows/continuity/OHLC/split) ─┤ quarantine on fail
                                                              ▼
                              data/store/equities_daily.parquet + manifest.jsonl
                                                              │
              ┌───────────────────────────────┬───────────────┤
              ▼                               ▼               ▼
      Pathfinder (Agent 2)            Backtest lab      cost engine (Agent 1)
      N-x paths + fingerprints        next-open fills   date-aware fees + CS spread
              │                               │
              ▼                               ▼
      path_hits.parquet              Section-4 gatekeeper ──► ledger (hash chain)
      path_window_counts.parquet              │
              └───────────────┬───────────────┘
                              ▼
                        FINDINGS.md (four numbers + trial count)
```

## Key design decisions

- **Ledger is the statistical ground truth.** Append-only JSONL, each entry
  sha256-chained to the previous. Pre-registration entries must exist before
  results reference them (enforced in code). The cumulative TRIAL count and
  the cross-trial Sharpe variance feed the DSR — deleting the ledger would
  falsify the math, so the chain verifier runs first every night and a broken
  chain halts everything.
- **Fill convention:** signals on close, fills at next session's OPEN, no
  exceptions. The pathfinder's entry basis is the next open too; a test
  (`test_entry_basis_is_next_open_never_signal_close`) exists specifically to
  make close-basis regressions fail.
- **Costs are date-aware and sourced.** Fee tables carry effective dates and
  primary-source URLs (SEC advisories, FINRA/SEC rule filings, NFA Bylaw
  1301); an unverified rate raises and blocks the backtest that touched it.
  Spread is estimated from observed OHLC via Corwin-Schultz, floored at half
  a tick, applied per side. Cost conservatism is deliberately asymmetric:
  overstating costs is acceptable, understating never.
- **Universe sampling.** Cycle 1 uses a seeded uniform random sample
  (n=200) of eligible current listings, not a hand-picked list: hand-picking
  adds selection bias on top of survivorship bias, and uniform sampling keeps
  per-window path counts extrapolable to the full universe. Both biases are
  recorded in the manifest and flagged by gate 7 in every result.
- **Sequences are gated-composable only.** Sequence paths (2-8 legs) are
  built exclusively from setups that individually passed all gates. Empty
  book ⇒ zero sequences, reported as such. The composer's design (DAG over
  setup occurrences, best-product path with per-leg costs) activates on first
  graduation.
- **Structural constants** (126-day window, monthly window stride, monthly
  rebalance, $1/$500k eligibility floors, 15% holdout, 5 walk-forward folds,
  1-month purge) are research-design choices, not fitted parameters. They are
  defined once in `config.py`/module constants and changing them is a
  pre-registered act like any other trial.

## Reliability model

- `make daily` is idempotent: same-day reruns skip the pull and skip
  re-gating the same strategy on the same data vintage (rerunning would
  inflate the trial count with no information).
- Every pull is validated before it touches the store; a >50% failure rate
  aborts the run before any downstream agent sees the data.
- The ledger chain is verified at the top of every run.

## Additional cycle-2 design decisions

- **Confluence statistic is rank-based** (Mann-Whitney AUC): fingerprint
  distributions are fat-tailed, and mean-based statistics would be driven by
  single outliers. Permutations are exact label shuffles vectorized via the
  rank-invariance trick (ranks computed once; 20k label draws are index
  gathers). Underpowered cells (<20 hits) are reported, never tested.
- **The scanner refuses to predict without an identifiable signal.** If no
  fingerprint feature survives BH correction, the predictions file is empty
  with the reason in-band. Confidence is stamped UNCALIBRATED until the
  reconciler produces calibration curves; size is 0 and executable=false
  until gates + 60 paper-trading days are done.
- **Predictions are graded against their sha256.** The scanner ledgeres the
  file hash at write time; the reconciler re-hashes before grading and halts
  on mismatch. Editing a prediction after the fact is detected, not silently
  graded.
- **Generated hypotheses are budgeted** (2/night) and carry their generator
  name in the preregistration, so meta-learning can compute survivor rate
  per generator and force a generation-strategy change when it stagnates.
- **The council's DSR check uses the CURRENT trial count** — graduation is
  retroactively revocable as N grows. That is the point of deflation.
- **Live trading is doubly locked**: env flag set by a human AND a ledgered
  HUMAN_DECISION; a grep-guard test asserts no source file sets the flag.
- **Broker connectivity is paper-first**: `AlpacaClient` refuses the live
  endpoint except through the double-locked `place_order` path;
  `paper_order` works only against the paper endpoint. The command center
  (`make command-center`) serves the dashboard plus sanitized live account
  state; secrets stay server-side in `.env` (gitignored). The test suite
  strips broker credentials from every test and hard-fails any test that
  attempts a broker HTTP call — a guard added after a legacy test placed a
  real paper order (caught same-session, position closed, cost $0.13).

## Cycle-3 design decisions (training on past data)

- **The event-driven backtester is the thesis vehicle.** Enter at the next
  open when the fingerprint composite is in the cross-sectional extreme;
  exit at target multiple / stop / 126-bar time limit; 10 slots of 1/10th
  capital; stop and target fills are gap-aware; sells pay date-aware fees.
  The monthly top-k lab remains only as a pipeline regression exercise.
- **Training is walk-forward, and the SIGNAL is trained, not just tuned:**
  feature directions are re-derived per fold from matched confluence groups
  whose hits predate the fold's cutoff; the config grid is scored on train
  only; folds are purged by the full holding period; the last 15% of trading
  days is holdout that no stage (confluence groups, scanner, training) may
  read — holdout-era hits are filtered out before anything sees them.
- **Confluence v2 replaces v1**: matched groups (one hit + 5 same-era
  controls within ±10 trading days) with a within-group rank statistic, so
  era confounds cancel by construction. v1's unmatched AUCs are superseded
  and not to be quoted.
- **Gates 10-11 are now formal**: 10 = stressed fills (worst-tail overnight
  gap injection) must still beat the null p95, and the trade plan must be
  feasible in cash or margin at the target equity; 11 = a ruin-constrained
  bet size must exist AND grow wealth (median terminal > 1 at the best
  P(-90%)<5% fraction). Both run inside the gatekeeper for any candidate
  with a real trade ledger.
- **Position sizes are outputs, not choices**: every surviving candidate
  carries its constrained Kelly fraction, the unconstrained optimum beside
  it, and the $2k integer-lot feasibility from the trap detector.
- **The vectorized features panel** is bound to `fingerprint_at` by a
  1e-9 equivalence test — two implementations, one definition.

## Roadmap (activation conditions, in order)

1. **Catalyst data** (earnings calendar, FINRA short interest, halts) — every
   pathfinder hit is currently tagged catalyst_class=NONE; identifiability
   per catalyst class starts when these feeds land.
2. **Survivorship-free equities** (Norgate/Sharadar) on credential presence —
   flips `survivorship_free` and removes the standing gate-7 flag.
3. **Options (DESIGN→LIVE)** on API key presence; futures likewise; both
   blocked from backtests by unverified fee tables until their fee schedules
   are sourced (ORF, OCC, CME member/non-member).
4. **Sequence composer** activates on first gate-PASS + paper-trading
   graduation (DAG over setup occurrences, best-product path, per-leg costs).
5. **Sizing Lab + Trap Detector wired into the gate pipeline** as gates 10-11
   for the first candidate that passes gates 1-9 (both are built and tested;
   they have nothing real to size yet).
