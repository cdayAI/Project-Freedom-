# ALPHA FORGE — FINDINGS (regenerated 2026-08-10 02:37 UTC)

## The four numbers

**1. Sustainable net compounded growth rate of the book:** N/A — the strategy book is EMPTY. No strategy has yet completed all Section-4 gates plus live reconciliation; a growth rate quoted from backtest data alone would be exactly the kind of number this system exists to refuse to print.

**2. Pipeline replacement rate:** pipeline age < 1 month: 0 graduated / 1 killed this cycle; a meaningful monthly rate needs a running book (tracked from now on)

**3. Capacity ceiling of current book:** N/A — empty book; capacity estimation (Agent 8a) activates on first graduation.

**4. Projected wealth fan chart:** deferred — a fan chart requires a validated growth-rate distribution from live calibration data, which does not exist yet. No single-point projection is printed in its place.

**Cumulative ledgered trial count: 16** (feeds every DSR computation; persists across sessions and reruns)

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

## Historical N-x path catalog (Pathfinder, Agent 2)

| N-x | symbol-window hits | windows with ≥1 path |
|-----|-------------------:|---------------------:|
| 5x | 310 | 223 |
| 10x | 125 | 113 |
| 20x | 63 | 61 |
| 50x | 16 | 16 |

Full catalog with ex-ante fingerprints: `data/store/path_hits.parquet` (514 hits). Sequences: 0 — composed only from gate-passing setups, and the graduated book is empty (by design, not omission).

## Standing disclosures

- Universe is built from CURRENT listings (survivorship-biased); every result above carries gate-7 flag until a survivorship-free vendor is wired.
- Path counts from the research sample extrapolate to the full universe only under the documented uniform-sampling assumption.
- Options/futures agents are in DESIGN mode (no keys present): they emit queries, never synthetic backtests.
