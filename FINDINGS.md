# ALPHA FORGE — FINDINGS (regenerated 2026-08-10 14:48 UTC)

## The four numbers

**1. Sustainable net compounded growth rate of the book:** N/A — the strategy book is EMPTY. No strategy has yet completed all Section-4 gates plus live reconciliation; a growth rate quoted from backtest data alone would be exactly the kind of number this system exists to refuse to print.

**2. Pipeline replacement rate:** trailing 3m: 0.00 (0 graduated / 3 killed this cycle)

**3. Capacity ceiling of current book:** N/A — empty book; capacity estimation (Agent 8a) activates on first graduation.

**4. Projected wealth fan chart:** deferred — a fan chart requires a validated growth-rate distribution from live calibration data, which does not exist yet. No single-point projection is printed in its place.

**Cumulative ledgered trial count: 52** (feeds every DSR computation; persists across sessions and reruns)

_Data: Yahoo chart-API daily OHLCV through 2026-08-07, survivorship-biased current-listing sample (see data/manifest.jsonl)_

## Section-4 gate outcomes

### xsmom_demo_v1 — verdict: **KILL**
- gate2: permutation p=0.9039 not significant after multiple-testing correction
- gate3: DSR probability 0.0799 < 0.95 at N=16 ledgered trials
- gate9: net metric -0.7153 does not beat null 95th percentile 0.7115
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return +25.6% / **net -51.1%** (gate 6: never shown apart)
- DSR probability 0.0799 at N=16 trials
- PBO 0.205 over 12870 CSCV splits
- null baseline: net metric -0.7153 vs null p95 0.7115 (1000 matched draws)

### gen_vol_20d_ann_desc_v1 — verdict: **KILL**
- gate2: permutation p=0.8689 not significant after multiple-testing correction
- gate3: DSR probability 0.0804 < 0.95 at N=49 ledgered trials
- gate4: PBO 0.646 > 0.5
- gate9: net metric -1.1395 does not beat null 95th percentile 0.8305
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return +42.7% / **net -68.0%** (gate 6: never shown apart)
- DSR probability 0.0804 at N=49 trials
- PBO 0.646 over 12870 CSCV splits
- null baseline: net metric -1.1395 vs null p95 0.8305 (1000 matched draws)

### gen_dist_from_252d_high_asc_v1 — verdict: **KILL**
- gate2: permutation p=0.9794 not significant after multiple-testing correction
- gate3: DSR probability 0.0493 < 0.95 at N=52 ledgered trials
- gate9: net metric -1.4746 does not beat null 95th percentile 0.8305
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -11.3% / **net -77.1%** (gate 6: never shown apart)
- DSR probability 0.0493 at N=52 trials
- PBO 0.241 over 12870 CSCV splits
- null baseline: net metric -1.4746 vs null p95 0.8305 (1000 matched draws)

## Historical N-x path catalog (Pathfinder, Agent 2)

| N-x | symbol-window hits | windows with ≥1 path |
|-----|-------------------:|---------------------:|
| 5x | 310 | 223 |
| 10x | 125 | 113 |
| 20x | 63 | 61 |
| 50x | 16 | 16 |

Full catalog with ex-ante fingerprints: `data/store/path_hits.parquet` (514 hits). Sequences: 0 — composed only from gate-passing setups, and the graduated book is empty (by design, not omission).

## Confluence (Agent 3)

28 of 30 tested (feature x class) cells are IDENTIFIABLE after BH correction: ret_21d@5x (AUC 0.35), ret_63d@5x (AUC 0.33), ret_126d@5x (AUC 0.37), ret_252d@5x (AUC 0.32), vol_20d_ann@5x (AUC 0.84), volume_z_20v126@5x (AUC 0.55), dollar_vol_med_20d@5x (AUC 0.22), dist_from_252d_high@5x (AUC 0.20), price@5x (AUC 0.34), cs_spread_est@5x (AUC 0.62), ret_21d@10x (AUC 0.34), ret_63d@10x (AUC 0.32), ret_126d@10x (AUC 0.33), ret_252d@10x (AUC 0.28), vol_20d_ann@10x (AUC 0.88), dollar_vol_med_20d@10x (AUC 0.21), dist_from_252d_high@10x (AUC 0.18), price@10x (AUC 0.28), cs_spread_est@10x (AUC 0.67), ret_21d@20x (AUC 0.28), ret_63d@20x (AUC 0.18), ret_126d@20x (AUC 0.21), ret_252d@20x (AUC 0.14), vol_20d_ann@20x (AUC 0.93), dollar_vol_med_20d@20x (AUC 0.21), dist_from_252d_high@20x (AUC 0.09), price@20x (AUC 0.25), cs_spread_est@20x (AUC 0.62)

## Generator meta-learning

- confluence_survivor_screen_v1: 0/2 survived gates

## Standing disclosures

- Universe is built from CURRENT listings (survivorship-biased); every result above carries gate-7 flag until a survivorship-free vendor is wired.
- Path counts from the research sample extrapolate to the full universe only under the documented uniform-sampling assumption.
- Options/futures agents are in DESIGN mode (no keys present): they emit queries, never synthetic backtests.
