# ALPHA FORGE — FINDINGS (regenerated 2026-08-10 17:07 UTC)

## The four numbers

**1. Sustainable net compounded growth rate of the book:** N/A — the strategy book is EMPTY. No strategy has yet completed all Section-4 gates plus live reconciliation; a growth rate quoted from backtest data alone would be exactly the kind of number this system exists to refuse to print.

**2. Pipeline replacement rate:** trailing 3m: 0.00 (0 graduated / 4 killed this cycle)

**3. Capacity ceiling of current book:** N/A — empty book; capacity estimation (Agent 8a) activates on first graduation.

**4. Projected wealth fan chart:** deferred — a fan chart requires a validated growth-rate distribution from live calibration data, which does not exist yet. No single-point projection is printed in its place.

**Cumulative ledgered trial count: 311** (feeds every DSR computation; persists across sessions and reruns)

_Data: Yahoo chart-API daily OHLCV through 2026-08-10, survivorship-biased current-listing sample (see data/manifest.jsonl)_

## Section-4 gate outcomes

### evt_fp5x_v3 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0000 < 0.95 at N=311 ledgered trials
- gate4: PBO 0.661 > 0.5
- gate9: net metric -18.1898 does not beat null 95th percentile 3.5237
- gate10: stressed net log -19.0470 no longer beats the null p95 3.5237 — the edge is a clean-fill artifact
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -100.0% / **net -100.0%** (gate 6: never shown apart)
- DSR probability 0.0000 at N=311 trials
- PBO 0.661 over 12870 CSCV splits
- null baseline: net metric -18.1898 vs null p95 3.5237 (20000 matched draws)

### xsmom_demo_v1 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0010 < 0.95 at N=119 ledgered trials
- gate9: net metric -2.5717 does not beat null 95th percentile 0.7366
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -68.7% / **net -92.4%** (gate 6: never shown apart)
- DSR probability 0.0010 at N=119 trials
- PBO 0.275 over 12870 CSCV splits
- null baseline: net metric -2.5717 vs null p95 0.7366 (1000 matched draws)

### gen_conf_dollar_vol_med_20d_asc_v1 — verdict: **KILL**
- gate3: DSR probability 0.0813 < 0.95 at N=215 ledgered trials
- gate4: PBO 0.832 > 0.5
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return +641.8% / **net +146.6%** (gate 6: never shown apart)
- DSR probability 0.0813 at N=215 trials
- PBO 0.832 over 12870 CSCV splits
- null baseline: net metric 0.9028 vs null p95 0.8608 (1000 matched draws)

### gen_conf_dist_from_252d_high_asc_v1 — verdict: **KILL**
- gate2: permutation p=0.9997 not significant after multiple-testing correction
- gate3: DSR probability 0.0015 < 0.95 at N=218 ledgered trials
- gate4: PBO 0.970 > 0.5
- gate9: net metric -2.5596 does not beat null 95th percentile 0.8608
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -46.0% / **net -92.3%** (gate 6: never shown apart)
- DSR probability 0.0015 at N=218 trials
- PBO 0.970 over 12870 CSCV splits
- null baseline: net metric -2.5596 vs null p95 0.8608 (1000 matched draws)

## Historical N-x path catalog (Pathfinder, Agent 2)

| N-x | symbol-window hits | windows with ≥1 path |
|-----|-------------------:|---------------------:|
| 5x | 1622 | 903 |
| 10x | 483 | 387 |
| 20x | 182 | 162 |
| 50x | 57 | 55 |

Full catalog with ex-ante fingerprints: `data/store/path_hits.parquet` (2344 hits). Sequences: 0 — composed only from gate-passing setups, and the graduated book is empty (by design, not omission).

## Confluence v2 (Agent 3) — TIME-MATCHED controls

Each hit is ranked only against same-era controls (other symbols, ±10 trading days), so era effects cancel; v1's unmatched design is superseded and its numbers should not be quoted.

31 of 45 tested (feature x class) cells are IDENTIFIABLE after BH correction: ret_21d@5x (matched AUC 0.36), ret_63d@5x (matched AUC 0.34), ret_126d@5x (matched AUC 0.35), ret_252d@5x (matched AUC 0.27), vol_20d_ann@5x (matched AUC 0.81), volume_z_20v126@5x (matched AUC 0.55), dollar_vol_med_20d@5x (matched AUC 0.21), dist_from_252d_high@5x (matched AUC 0.17), price@5x (matched AUC 0.36), cs_spread_est@5x (matched AUC 0.66), ret_21d@10x (matched AUC 0.36), ret_63d@10x (matched AUC 0.34), ret_126d@10x (matched AUC 0.38), ret_252d@10x (matched AUC 0.32), vol_20d_ann@10x (matched AUC 0.73), dollar_vol_med_20d@10x (matched AUC 0.17), dist_from_252d_high@10x (matched AUC 0.23), price@10x (matched AUC 0.35), cs_spread_est@10x (matched AUC 0.63), ret_63d@20x (matched AUC 0.37), ret_252d@20x (matched AUC 0.40), vol_20d_ann@20x (matched AUC 0.63), dollar_vol_med_20d@20x (matched AUC 0.09), dist_from_252d_high@20x (matched AUC 0.36), price@20x (matched AUC 0.37), days_since_earnings_8k@20x (matched AUC 0.64), ret_21d@50x (matched AUC 0.34), ret_63d@50x (matched AUC 0.29), ret_126d@50x (matched AUC 0.27), ret_252d@50x (matched AUC 0.27), dollar_vol_med_20d@50x (matched AUC 0.15)

## Event strategy (trained on past data)

**evt_fp5x_v3** — verdict **KILL**. Walk-forward folds: 4; final spec: {'entry_pct': 0.995, 'target_mult': 5.0, 'stop_frac': None}; directions: {'ret_21d': -1.0, 'ret_63d': -1.0, 'ret_126d': -1.0, 'ret_252d': -1.0, 'vol_20d_ann': 1.0, 'dollar_vol_med_20d': -1.0, 'dist_from_252d_high': -1.0, 'price': -1.0, 'cs_spread_est': 1.0}. Ruin-constrained size: 0.07 of equity per position (Kelly 0.07).

## Generator meta-learning

- confluence_survivor_screen_v1: 0/6 survived gates
- event_fingerprint_v1: 0/3 survived gates

## Standing disclosures

- Universe is built from CURRENT listings (survivorship-biased); every result above carries gate-7 flag until a survivorship-free vendor is wired.
- Path counts from the research sample extrapolate to the full universe only under the documented uniform-sampling assumption.
- Options/futures agents are in DESIGN mode (no keys present): they emit queries, never synthetic backtests.
