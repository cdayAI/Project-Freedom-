# ALPHA FORGE — Architecture

**Objective function (explicit, everywhere):** maximize expected log-wealth
growth per year, net of all costs, subject to
P(drawdown below 50% of high-water mark) ≤ human-set limit.
Log-wealth is the objective because geometric compounding over decades makes
drawdown control part of the return, not a preference.

**North-star health metric:** pipeline replacement rate — strategies
graduating all gates per month vs strategies dying. Reported in every
summary; a rate < 1.0 leads the weekly memo.

## Status: cycle 1 (equities end-to-end)

Built and running: Agent 0 (data), Agent 1 (costs), Agent 2 (pathfinder),
the full Section-4 gate library, the hash-chained ledger, and the nightly
orchestrator that takes one pre-registered hypothesis through every gate on
real equities data. Everything else exists as enforced interfaces (DESIGN
mode) and is listed under Roadmap with its activation condition.

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

## Roadmap (activation conditions, in order)

1. **Agent 7 (Reconciler)** — activates with the first dated predictions
   file; calibration curves become the system's fitness function.
2. **Agent 3 (Confluence)** + fingerprint identifiability testing on the
   pathfinder hit catalog (every combination tried = ledgered trial).
3. **Agents 4/5 (Sizing Lab, Trap Detector)** — activate on first gate-PASS
   candidate; PDT/settlement/integer-lot simulator specified in Section 3 of
   the mission is the acceptance test list.
4. **Catalyst data** (earnings calendar, FINRA short interest, halts) — every
   pathfinder hit is currently tagged catalyst_class=NONE; identifiability
   per catalyst class starts when these feeds land.
5. **Options (DESIGN→LIVE)** on API key presence; futures likewise; both
   blocked from backtests by unverified fee tables until their fee schedules
   are sourced (ORF, OCC, CME member/non-member).
6. **Survivorship-free equities** (Norgate/Sharadar) on credential presence —
   flips `survivorship_free` and removes the standing gate-7 flag.
7. **Dashboard + static HTML snapshot**, **Agent 8a/8b**, **Agent 10
   (paper-trading harness)** — in that order, each reading the same store.
