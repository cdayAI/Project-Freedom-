# Wide discovery sweep — 2026-08-14

First post-Phase-1 discovery run. Full 2-stage enumeration (450 candidates
after skipping 12 previously killed) + the deterministic first 600 of the
3-stage family. **1,050 new ledgered trials** (research ledger now at 2,012);
every candidate preregistered, scored on matched lift, and pushed through
the falsification battery.

## Sweep verdicts

| Family | Candidates | KILLED | UNDECIDABLE | SURVIVOR |
|---|---|---|---|---|
| 2-stage (full) | 450 | 409 | 38 | 3 |
| 3-stage (first 600) | 600 | 322 | 275 | 3 |

Six mechanisms survived discovery (min support, lift ≥ 1.5, era stability,
regime stability, order necessity, ablation necessity where applicable):

1. surge-in-thin-name → insider accumulation (3 sessions) — n=275, lift 2.96
2. surge-in-thin-name → accumulation (5 sessions, tighter thinness) — n=94, lift 4.62
3. surge-in-thin-name → accumulation (10 sessions) — n=107, lift 4.92
4. accumulation → 8-K (10) → surge-in-thin-name (10) — n=73, lift 8.89
5. accumulation → 8-K within 5 sessions (k=5 variant) → surge — n=76, lift 8.82
6. accumulation → squeeze-fuel (5) → 8-K (5) — n=117, lift 2.19

The highest raw lifts in the sweep (≈15.6×) sat on 5–14 completions and were
held as UNDECIDABLE — support below 30 is never a survivor.

## Compiled battery: all six KILLED

Survivors of discovery are not strategies. Each was compiled through the C10
shares policy family (next-open entry; 3×/5× targets × 35%/50% stops;
126-bar time stop; matched same-session nulls) and the full Section-4
battery on the 2018+ event panel. **Every one was killed** — full reasons in
`survivor_battery_2026-08-14.json`:

- matched-null permutation p between 0.045 and 1.0 — none significant after
  correction;
- DSR ≈ 0 after deflating by the ledger's 2,000+ audited trials;
- net log-growth negative or below the null 95th percentile in 5 of 6;
- 44–95 post-collapse trades, all under the 100-trade gate-8 minimum;
- PBO > 0.5 on the three-stage variants.

## Reading the disconnect honestly

Ignition-start lift measures whether a ≥5x forward path *begins* more often
after the pattern. The compiled policy must *monetize* that with defined
exits net of thin-name spreads — and most ignition starts never reach a 3×
target cleanly inside the hold window. Two structural caveats compound it:

- **Era confinement.** ACCUMULATION requires positive published insider
  flow, and the DERA archive spans ~1 year — every completion lives in
  2025–2026, so "era stability" could only test two halves of one regime
  era. A longer insider history (more DERA quarters are freely fetchable)
  is the single highest-value data extension this sweep points to.
- **Survivorship bias** (gate 7) marks every result; the universe contains
  only surviving listings.

## Status

No survivor exists at the policy level; no edge is claimed; nothing touches
sizing. Status ceiling **PIPELINE_PROOF_ONLY** unchanged. The pipeline did
its job twice over: it generated and killed 1,044 candidates mechanically,
and it killed the 6 that looked best before they could become positions.

Next candidates for the queue (each preregistered, none started): deepen the
DERA insider history to widen the testable era; the remaining 3,144
three-stage candidates; options-state predicates once the chain archive has
enough days; the §11 shadow-scoring loop for anything that ever survives.
