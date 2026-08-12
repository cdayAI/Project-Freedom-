# ALPHA FORGE — FINDINGS (regenerated 2026-08-12 17:58 UTC)

## The four numbers

**1. Sustainable net compounded growth rate of the book:** N/A — the strategy book is EMPTY. No strategy has yet completed all Section-4 gates plus live reconciliation; a growth rate quoted from backtest data alone would be exactly the kind of number this system exists to refuse to print.

**2. Pipeline replacement rate:** trailing 3m: 0.00 (0 graduated / 5 killed this cycle)

**3. Capacity ceiling of current book:** N/A — empty book; capacity estimation (Agent 8a) activates on first graduation.

**4. Projected wealth fan chart:** deferred — a fan chart requires a validated growth-rate distribution from live calibration data, which does not exist yet. No single-point projection is printed in its place.

**Cumulative ledgered trial count: 900** (feeds every DSR computation; persists across sessions and reruns)

_Data: Yahoo chart-API daily OHLCV through 2026-08-11, survivorship-biased current-listing sample (see data/manifest.jsonl)_

## Section-4 gate outcomes

### evt_fp5x_v5_1 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0000 < 0.95 at N=740 ledgered trials
- gate4: PBO 0.742 > 0.5
- gate9: net metric -38.1925 does not beat null 95th percentile 3.2539
- gate10: stressed net log -39.1901 no longer beats the null p95 3.2539 — the edge is a clean-fill artifact
- gate11: the best ruin-safe fraction (0.01) still has median terminal wealth 0.891 <= 1 under parameter uncertainty — no sizing grows this edge inside the ruin constraint
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -100.0% / **net -100.0%** (gate 6: never shown apart)
- DSR probability 0.0000 at N=740 trials
- PBO 0.742 over 12870 CSCV splits
- null baseline: net metric -38.1925 vs null p95 3.2539 (20000 matched draws)

### evt_fp5x_logit_v6_1 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0000 < 0.95 at N=788 ledgered trials
- gate4: PBO 0.521 > 0.5
- gate9: net metric -32.6849 does not beat null 95th percentile 0.9385
- gate10: stressed net log -33.6990 no longer beats the null p95 0.9385 — the edge is a clean-fill artifact
- gate11: the best ruin-safe fraction (0.01) still has median terminal wealth 0.908 <= 1 under parameter uncertainty — no sizing grows this edge inside the ruin constraint
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -100.0% / **net -100.0%** (gate 6: never shown apart)
- DSR probability 0.0000 at N=788 trials
- PBO 0.521 over 12870 CSCV splits
- null baseline: net metric -32.6849 vs null p95 0.9385 (20000 matched draws)

### evt_fp5x_pooled_v7 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0000 < 0.95 at N=836 ledgered trials
- gate9: net metric -28.5597 does not beat null 95th percentile 1.7150
- gate10: stressed net log -29.4993 no longer beats the null p95 1.7150 — the edge is a clean-fill artifact
- gate11: the best ruin-safe fraction (0.01) still has median terminal wealth 0.931 <= 1 under parameter uncertainty — no sizing grows this edge inside the ruin constraint
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -100.0% / **net -100.0%** (gate 6: never shown apart)
- DSR probability 0.0000 at N=836 trials
- PBO 0.145 over 12870 CSCV splits
- null baseline: net metric -28.5597 vs null p95 1.7150 (20000 matched draws)

### evt_fp5x_hazard_v8 — verdict: **KILL**
- gate2: permutation p=0.9975 not significant after multiple-testing correction
- gate3: DSR probability 0.0000 < 0.95 at N=884 ledgered trials
- gate4: PBO 0.624 > 0.5
- gate9: net metric -11.4531 does not beat null 95th percentile 5.3462
- gate10: stressed net log -12.4672 no longer beats the null p95 5.3462 — the edge is a clean-fill artifact
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -99.9% / **net -100.0%** (gate 6: never shown apart)
- DSR probability 0.0000 at N=884 trials
- PBO 0.624 over 12870 CSCV splits
- null baseline: net metric -11.4531 vs null p95 5.3462 (20000 matched draws)

### xsmom_demo_v1 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0000 < 0.95 at N=900 ledgered trials
- gate9: net metric -4.5533 does not beat null 95th percentile 0.4184
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -90.3% / **net -98.9%** (gate 6: never shown apart)
- DSR probability 0.0000 at N=900 trials
- PBO 0.030 over 12870 CSCV splits
- null baseline: net metric -4.5533 vs null p95 0.4184 (1000 matched draws)

## Historical N-x path catalog (Pathfinder, Agent 2)

| N-x | symbol-window hits | windows with ≥1 path |
|-----|-------------------:|---------------------:|
| 3x | 9806 | 2182 |
| 5x | 3143 | 1374 |
| 10x | 923 | 627 |
| 20x | 340 | 279 |
| 50x | 114 | 101 |

Full catalog with ex-ante fingerprints: `data/store/path_hits.parquet` (14326 hits). Sequences: 0 — composed only from gate-passing setups, and the graduated book is empty (by design, not omission).

## Event strategy (trained on past data)

**evt_fp5x_v5_1** — verdict **KILL**. Walk-forward folds: 4; final spec: {'entry_pct': 0.995, 'target_mult': 2.0, 'stop_frac': 0.5}; directions: {'ret_21d': -1.0, 'ret_63d': -1.0, 'ret_126d': -1.0, 'ret_252d': -1.0, 'vol_20d_ann': 1.0, 'dollar_vol_med_20d': -1.0, 'dist_from_252d_high': -1.0, 'cs_spread_est': 1.0, 'days_since_form4': 1.0, 'n_form4_90d': -1.0}. Ruin-constrained size: 0.01 of equity per position (Kelly 0.00).

## Generator meta-learning

- confluence_survivor_screen_v1: 0/6 survived gates
- event_fingerprint_v1: 0/15 survived gates

## Standing disclosures

- Universe is built from CURRENT listings (survivorship-biased); every result above carries gate-7 flag until a survivorship-free vendor is wired.
- Path counts from the research sample extrapolate to the full universe only under the documented uniform-sampling assumption.
- Options/futures agents are in DESIGN mode (no keys present): they emit queries, never synthetic backtests.
