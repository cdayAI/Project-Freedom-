# ALPHA FORGE — FINDINGS (regenerated 2026-08-10 21:03 UTC)

## The four numbers

**1. Sustainable net compounded growth rate of the book:** N/A — the strategy book is EMPTY. No strategy has yet completed all Section-4 gates plus live reconciliation; a growth rate quoted from backtest data alone would be exactly the kind of number this system exists to refuse to print.

**2. Pipeline replacement rate:** trailing 3m: 0.00 (0 graduated / 3 killed this cycle)

**3. Capacity ceiling of current book:** N/A — empty book; capacity estimation (Agent 8a) activates on first graduation.

**4. Projected wealth fan chart:** deferred — a fan chart requires a validated growth-rate distribution from live calibration data, which does not exist yet. No single-point projection is printed in its place.

**Cumulative ledgered trial count: 548** (feeds every DSR computation; persists across sessions and reruns)

_Data: Yahoo chart-API daily OHLCV through 2026-08-10, survivorship-biased current-listing sample (see data/manifest.jsonl)_

## Section-4 gate outcomes

### evt_fp5x_v5 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0000 < 0.95 at N=500 ledgered trials
- gate4: PBO 0.582 > 0.5
- gate9: net metric -37.9509 does not beat null 95th percentile 1.8957
- gate10: stressed net log -38.8823 no longer beats the null p95 1.8957 — the edge is a clean-fill artifact
- gate11: the best ruin-safe fraction (0.01) still has median terminal wealth 0.866 <= 1 under parameter uncertainty — no sizing grows this edge inside the ruin constraint
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -100.0% / **net -100.0%** (gate 6: never shown apart)
- DSR probability 0.0000 at N=500 trials
- PBO 0.582 over 12870 CSCV splits
- null baseline: net metric -37.9509 vs null p95 1.8957 (20000 matched draws)

### evt_fp5x_logit_v6 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0000 < 0.95 at N=548 ledgered trials
- gate4: PBO 0.889 > 0.5
- gate9: net metric -40.5620 does not beat null 95th percentile 2.7785
- gate10: stressed net log -41.5696 no longer beats the null p95 2.7785 — the edge is a clean-fill artifact
- gate11: the best ruin-safe fraction (0.01) still has median terminal wealth 0.901 <= 1 under parameter uncertainty — no sizing grows this edge inside the ruin constraint
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -100.0% / **net -100.0%** (gate 6: never shown apart)
- DSR probability 0.0000 at N=548 trials
- PBO 0.889 over 12870 CSCV splits
- null baseline: net metric -40.5620 vs null p95 2.7785 (20000 matched draws)

### xsmom_demo_v1 — verdict: **KILL**
- gate2: permutation p=1 not significant after multiple-testing correction
- gate3: DSR probability 0.0010 < 0.95 at N=119 ledgered trials
- gate9: net metric -2.5717 does not beat null 95th percentile 0.7366
- gate7: universe includes only surviving listings — results are biased and marked as such in every display
- gross total return -68.7% / **net -92.4%** (gate 6: never shown apart)
- DSR probability 0.0010 at N=119 trials
- PBO 0.275 over 12870 CSCV splits
- null baseline: net metric -2.5717 vs null p95 0.7366 (1000 matched draws)

## Historical N-x path catalog (Pathfinder, Agent 2)

| N-x | symbol-window hits | windows with ≥1 path |
|-----|-------------------:|---------------------:|
| 5x | 2938 | 1362 |
| 10x | 832 | 598 |
| 20x | 306 | 254 |
| 50x | 95 | 85 |

Full catalog with ex-ante fingerprints: `data/store/path_hits.parquet` (4171 hits). Sequences: 0 — composed only from gate-passing setups, and the graduated book is empty (by design, not omission).

## Event strategy (trained on past data)

**evt_fp5x_v5** — verdict **KILL**. Walk-forward folds: 4; final spec: {'entry_pct': 0.995, 'target_mult': 2.0, 'stop_frac': 0.5}; directions: {'ret_21d': -1.0, 'ret_63d': -1.0, 'ret_126d': -1.0, 'ret_252d': -1.0, 'vol_20d_ann': 1.0, 'dollar_vol_med_20d': -1.0, 'dist_from_252d_high': -1.0, 'price': -1.0, 'days_since_form4': 1.0, 'n_form4_90d': -1.0}. Ruin-constrained size: 0.01 of equity per position (Kelly 0.00).

## Generator meta-learning

- confluence_survivor_screen_v1: 0/6 survived gates
- event_fingerprint_v1: 0/8 survived gates

## Standing disclosures

- Universe is built from CURRENT listings (survivorship-biased); every result above carries gate-7 flag until a survivorship-free vendor is wired.
- Path counts from the research sample extrapolate to the full universe only under the documented uniform-sampling assumption.
- Options/futures agents are in DESIGN mode (no keys present): they emit queries, never synthetic backtests.
