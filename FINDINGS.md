# ALPHA FORGE — FINDINGS (regenerated 2026-08-10 19:49 UTC)

## The four numbers

**1. Sustainable net compounded growth rate of the book:** N/A — the strategy book is EMPTY. No strategy has yet completed all Section-4 gates plus live reconciliation; a growth rate quoted from backtest data alone would be exactly the kind of number this system exists to refuse to print.

**2. Pipeline replacement rate:** trailing 3m: 0.00 (0 graduated / 3 killed this cycle)

**3. Capacity ceiling of current book:** N/A — empty book; capacity estimation (Agent 8a) activates on first graduation.

**4. Projected wealth fan chart:** deferred — a fan chart requires a validated growth-rate distribution from live calibration data, which does not exist yet. No single-point projection is printed in its place.

**Cumulative ledgered trial count: 452** (feeds every DSR computation; persists across sessions and reruns)

_Data: Yahoo chart-API daily OHLCV through 2026-08-07, survivorship-biased current-listing sample (see data/manifest.jsonl)_

## Section-4 gate outcomes

### evt_fp5x_v5 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0000 < 0.95 at N=428 ledgered trials
- gate5: walk-forward incomplete
- gate8: 75 trade events < 100 minimum
- gate9: net metric -16.5462 does not beat null 95th percentile -0.7387
- gate10: stressed net log -16.8990 no longer beats the null p95 -0.7387 — the edge is a clean-fill artifact
- gate11: the best ruin-safe fraction (0.01) still has median terminal wealth 0.910 <= 1 under parameter uncertainty — no sizing grows this edge inside the ruin constraint
- gate4: PBO 0.355 > 0.35 (flag)
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -100.0% / **net -100.0%** (gate 6: never shown apart)
- DSR probability 0.0000 at N=428 trials
- PBO 0.355 over 12870 CSCV splits
- null baseline: net metric -16.5462 vs null p95 -0.7387 (20000 matched draws)

### evt_fp5x_logit_v6 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0000 < 0.95 at N=452 ledgered trials
- gate4: PBO 0.738 > 0.5
- gate5: walk-forward incomplete
- gate8: 65 trade events < 100 minimum
- gate9: net metric -16.6973 does not beat null 95th percentile -0.7496
- gate10: stressed net log -17.0500 no longer beats the null p95 -0.7496 — the edge is a clean-fill artifact
- gate11: the best ruin-safe fraction (0.01) still has median terminal wealth 0.924 <= 1 under parameter uncertainty — no sizing grows this edge inside the ruin constraint
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -100.0% / **net -100.0%** (gate 6: never shown apart)
- DSR probability 0.0000 at N=452 trials
- PBO 0.738 over 12870 CSCV splits
- null baseline: net metric -16.6973 vs null p95 -0.7496 (20000 matched draws)

### xsmom_demo_v1 — verdict: **KILL**
- gate2: permutation p=0.9039 not significant after multiple-testing correction
- gate3: DSR probability 0.0799 < 0.95 at N=16 ledgered trials
- gate9: net metric -0.7153 does not beat null 95th percentile 0.7115
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return +25.6% / **net -51.1%** (gate 6: never shown apart)
- DSR probability 0.0799 at N=16 trials
- PBO 0.205 over 12870 CSCV splits
- null baseline: net metric -0.7153 vs null p95 0.7115 (1000 matched draws)

## Historical N-x path catalog (Pathfinder, Agent 2)

| N-x | symbol-window hits | windows with ≥1 path |
|-----|-------------------:|---------------------:|
| 5x | 310 | 223 |
| 10x | 125 | 113 |
| 20x | 63 | 61 |
| 50x | 16 | 16 |

Full catalog with ex-ante fingerprints: `data/store/path_hits.parquet` (514 hits). Sequences: 0 — composed only from gate-passing setups, and the graduated book is empty (by design, not omission).

## Event strategy (trained on past data)

**evt_fp5x_v5** — verdict **KILL**. Walk-forward folds: 2; final spec: {'entry_pct': 0.98, 'target_mult': 2.0, 'stop_frac': 0.5}; directions: {'ret_21d': -1.0, 'ret_126d': -1.0, 'ret_252d': -1.0, 'vol_20d_ann': 1.0, 'dollar_vol_med_20d': -1.0, 'dist_from_252d_high': -1.0, 'days_since_form4': 1.0, 'n_form4_90d': -1.0}. Ruin-constrained size: 0.01 of equity per position (Kelly 0.00).

## Generator meta-learning

- confluence_survivor_screen_v1: 0/6 survived gates
- event_fingerprint_v1: 0/6 survived gates

## Standing disclosures

- Universe is built from CURRENT listings (survivorship-biased); every result above carries gate-7 flag until a survivorship-free vendor is wired.
- Path counts from the research sample extrapolate to the full universe only under the documented uniform-sampling assumption.
- Options/futures agents are in DESIGN mode (no keys present): they emit queries, never synthetic backtests.
