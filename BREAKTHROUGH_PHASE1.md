# BREAKTHROUGH PHASE 1 — The Market Mechanism Discovery Engine

Status: DESIGN — approved corrections are being implemented (commits C1-C5);
the discovery engine itself (C6+) builds after the Alpha-v0 acceptance test
passes twice with an identical evidence hash.

Mission: build the most capable information-to-trade discovery system we can,
engineered to maximize the probability of turning $7,000 into $1,000,000
within 48 months. The required trajectory — roughly 0.493% net per trading
day, or fewer, more convex opportunities compounding to the same path — is
the **scoreboard, not a promised outcome**. Nothing in this document asserts
the target will be reached; everything in it is judged by whether it raises
the probability that it can be, without hidden ruin.

The validation machinery (ledger, gates, costs, account mechanics) is the
immune system. This document designs the brain.

---

## 0. Corrections register (accepted and verified against the code)

Every item below was independently claimed in external review, then verified
here against the actual code before acceptance. Each becomes a commit in
Section 12.

| # | Correction | Verified location | Fix |
|---|-----------|-------------------|-----|
| 1 | 900 trials are **not** 900 kills. They are ledgered configuration/test trials; the branch contains **5 killed candidate strategies** (xsmom_demo_v1, evt_fp5x v5_1/v6_1/v7/v8). Reporting must never conflate the two. | FINDINGS generator, weekly memo | C1 |
| 2 | Survivorship-driven anti-selection is a **hypothesis**, not a demonstrated cause. Candidates lose in a survivorship-biased universe; causation requires the same research on point-in-time data containing delisted securities. All language downgraded accordingly. | FINDINGS.md, memo language | C1 |
| 3 | The implemented ruin gate binds on `P(losing 90%) < 5%`; the 50%-drawdown probability is computed but **never binds** (`sizing2.py` — `feasible` filters on `p_ruin_worst` with `ruin_loss=0.90`). The binding constraint becomes `P(maxDD >= 50% over 20 years) <= 5%`, with 1/5/10/20-year breach probabilities reported and the 90%-loss probability kept as a secondary measure. | `alpha_forge/research/sizing2.py:250` | C2 |
| 4 | DSR trial semantics are an unexplained mixture: `n_trials = ledger.trial_count()` (all TRIAL entries, including Sharpe-less confluence trials) while variance comes only from Sharpe-bearing trials (`gatekeeper.py:92-93`). The audit reports raw cumulative trials, Sharpe-bearing trials, estimated effective independent trials, the preregistered method, and a conservative raw-count sensitivity result. All 900 trials are preserved. | `alpha_forge/gates/gatekeeper.py:91-99` | C1 |
| 5 | Alpha-v0 must not pollute the research ledger with a trial invented to test software. It **replays an existing killed candidate** on an **isolated acceptance ledger**. | new `alpha_forge/alpha_v0.py` | C5 |
| 6 | Replacement rate calls an empty pipeline healthy (`healthy: (rate is None) or (rate >= 1.0)`, `loop.py:41`) and counts raw gate reports. It now reports observed counts (e.g. `0 graduations / 5 strategy kills`) and returns `NOT YET ESTIMABLE` for the health metric until a minimum observation period and strategy lifecycle exist. | `alpha_forge/research/loop.py:19-43` | C1 |
| 7 | Repository files are valid UTF-8; the mojibake was Windows PowerShell's default decoding. **No artifact rewriting.** Fix: explicit `encoding="utf-8"` on all text I/O, a cross-platform setup path, and documented Windows console configuration. | `Makefile`, `costs/build_tables.py:67`, README | C3 |
| 8 | PR #12's evidence was overstated: 204 collected tests pass (not the claimed 214), the hazard "maturation" test used an all-zero label matrix (it proves refuse-on-no-positives, not cutoff exclusion), and date/symbol alignment plus overlapping-label dependence are untested. PR #12 is already merged, so these land as follow-up commits with a corrected record. | `tests/test_hazard.py:39-44` | C4 |

Standing decisions adopted:

- Research constraint: **P(maxDD >= 50% over 20 years) <= 5%**, with 1-, 5-,
  10-, and 20-year breach probabilities reported. (A 5% *annual* limit would
  compound to roughly a 64% chance of at least one breach over 20 years under
  independence — inconsistent with a multi-decade hard constraint.)
- Alpha-v0 runs now on free data, labeled **`PIPELINE_PROOF_ONLY`**.
- **No candidate may hold `VALIDATED`, `GRADUATED`, or `EXECUTABLE` status
  until point-in-time, survivorship-free equities data is installed.**
  Enforced in code, not by convention.
- The complete 900-trial ledger is kept permanently.
- v9 conventional feature/model iteration is **frozen**. No new options
  research beyond the sensor layer, no dashboard expansion, no new agents.
- No data purchases without explicit human approval.
- Options remain **defined-risk only** throughout this program: long calls or
  debit spreads; no uncovered writing, ever.

---

## 1. Revised Alpha-v0 acceptance contract

Alpha-v0 proves the loop, not an edge. One command:

```
python -m alpha_forge.alpha_v0
```

**Setup.** The run uses an **isolated acceptance ledger**
(`ledger/acceptance_v0.jsonl`, hash-chained with the same code path as the
research ledger, clearly marked `ACCEPTANCE` in every entry). The research
ledger is read-only to this process. The replayed candidate is an existing
killed strategy (`evt_fp5x_v5_1` — the simplest event strategy with a
complete preregistration and a KILL verdict on record).

**The ten steps, each emitting a ledgered artifact:**

1. **Ingest + validate** the equities panel for the pinned vintage; provenance
   (source, fetch time, sha256, row count) recorded; validation failures
   abort; universe survivorship status recorded as a first-class field.
2. **Re-register** the replayed candidate's exact preregistration (mechanics,
   universe, window, costs, thresholds, null construction) in the acceptance
   ledger, before any price is read, with a pointer to the original research
   ledger entry hash.
3. **Trial accounting**: the acceptance ledger's DSR uses the research
   ledger's audited trial counts (raw + Sharpe-bearing + effective), read-only.
4. **Backtest**: close-signal, next-open fill, verified date-aware fees,
   blended half-spread, window bounded by `earliest_verified_equity_fee_date()`.
5. **Gates**: the full Section-4 battery; every gate's number in the report.
6. **Matched null**: >= 1,000 random-entry draws, identical mechanics.
7. **Kill path**: expected outcome is KILL (it was killed before); the reason
   is ledgered; no forward-looking performance numbers are emitted.
8. **Survive path** (not expected here): sizing under the C2 constraint +
   stress, only if all gates pass.
9. **Evidence-chain report**: one immutable file; every claim carries the
   sha256 of the ledger entries it derives from; the whole report carries a
   single top-level evidence hash; `PIPELINE_PROOF_ONLY` watermark on every
   page; replacement-rate health prints `NOT YET ESTIMABLE`.
10. **Idempotence**: a second invocation on the same vintage performs zero new
    trials, appends zero duplicate entries, and reproduces the **identical
    evidence hash**.

**PASS =** both runs complete, verdicts match the original KILL, the two
evidence hashes are byte-identical, and the research ledger's entry count is
unchanged after both runs. A KILL verdict is a **pass** — the test proves the
loop tells the truth twice, not that an edge exists.

---

## 2. Market Genome — the point-in-time data schema

One row per (symbol, session). Every field is a triple: **value**,
**asof_ts** (the moment the value became knowable), and **source**. A field
whose asof_ts is after the session's decision time (close) is unusable for
that session by construction — the knowability audit (Section 11) enforces
this mechanically, not by discipline.

Stored as `data/store/genome/*.parquet`, one file per schema version. The
schema itself is a committed registry (`alpha_forge/genome/schema.py`) where
every field declares: name, dtype, source feed, publication lag, and the
first date the feed is trustworthy. Adding a field is a schema version bump;
old genome files are never rewritten.

**Field groups** (⊕ already ingested, ○ derivable from existing feeds, ✗ gap):

| Group | Fields | Status |
|-------|--------|--------|
| Price/liquidity | OHLCV, dollar volume, blended half-spread (CS/AR), overnight gap, distance to 52w high/low, consecutive-up-days | ⊕ Yahoo + costs engine |
| Volatility | realized vol (5/21d), vol-of-vol, gap frequency | ○ from prices |
| Capital structure | recent S-1/S-3/F-1/F-3 shelf, 424B pricing events, days since last offering, dilution-risk flag | ⊕ EDGAR catalysts |
| Insider | net open-market buy 90d (USD), buy ratio 90d | ⊕ DERA Form 3/4/5 |
| Short pressure | daily short-volume ratio, its 5/21d trend, FINRA short interest + days-to-cover (bi-monthly, lagged as published) | ⊕ Reg SHO daily; ✗ SI file (free, needs ingest) |
| Options state | ATM IV proxy, IV vs realized spread, put/call skew asymmetry, call OI concentration near spot, premium-per-gap-dollar (convexity price), term slope into next catalyst | ○ from archived Cboe chains (REAL_DELAYED) |
| Catalysts | 8-K item flags, days-until-expected-earnings, catalyst-type one-hots | ⊕ EDGAR |
| Attention | filing-view/news acceleration | ✗ candidate free sources (EDGAR full-text search counts, Wikipedia pageviews); needs a source decision |
| Regime | SPY trend state, VIX level/percentile | ⊕ regime feed |
| Participation constraints | price bands (institutional floors, penny thresholds), listing venue | ○ from universe file |
| Outcome (labels only) | forward path multiples (3/5/10/20/50x), fast-mover tiers (+30%/10d, +15%/5d), maturation dates | ⊕ pathfinder |

The known structural defect — **no delisted securities** — is carried as a
schema-level flag (`universe_survivorship_biased: true`) that propagates into
every report until point-in-time data is installed.

---

## 3. State-transition representation

A mechanism is not a feature vector; it is a **sequence claim**. The
representation has three layers:

**Layer 1 — state predicates.** A small library of named, versioned boolean
predicates over Genome fields, each computable point-in-time, e.g.:

```
ACCUMULATION      insider_net_buy_90d > 0 AND short_vol_ratio_trend < 0
CATALYST_SHOCK    8-K item in {1.01, 2.02, 5.02, 8.01} within k days
VOLUME_SURGE_THIN dollar_vol_pctl > 95 AND spread_pctl > 80
THRESHOLD_BREAK   close > 52w_high * (1 - b)
SQUEEZE_FUEL      short_vol_ratio_pctl > 90 AND days_to_cover > d
CONVEXITY_CHEAP   premium_per_gap_dollar_pctl < 20
IGNITION          (outcome layer) fast-mover tier reached
FIZZLE            matched pre-state, tier not reached within window
```

Predicates carry free threshold parameters drawn from small preregistered
grids — every grid point is a ledgered trial (multiplicity is paid, as
always).

**Layer 2 — mechanism grammar.** A mechanism is an ordered sequence of 2-5
predicates with timing windows:

```
M := P1 --(within w1 days)--> P2 --(within w2)--> ... --> IGNITION?
```

This grammar is deliberately restrictive: small enough to search and to
falsify, expressive enough to encode the reflexive-ignition chain (catalyst →
expectation shift → thin-liquidity volume → threshold break → forced buying →
attention). Every mechanism is expressible in one falsifiable English
sentence and is stored that way in its preregistration.

**Layer 3 — transition statistics.** For each mechanism prefix, the engine
estimates the conditional hazard of advancing to the next state vs dying,
against matched controls (Section 4). The learned object is the set of
transition conditions — *what separates sequences that advance from sequences
that stall* — not a black-box score.

---

## 4. Ignition-versus-failure matching

For every ignition (outcome tier reached), the engine finds the k nearest
non-ignitions in **pre-window Genome space** (same era, same regime stratum,
nearest by normalized rank distance on the mechanism's own predicate margins
— the confluence2 episode-matching machinery generalized). This yields
matched (ignition, failure) sets on which the discovery questions become
concrete statistics:

- What was different immediately before ignition? → per-field rank gaps with
  stratified permutation tests, BH-corrected per family.
- Did event **ordering** matter? → order-permuted sequence lift (Section 5).
- Was there a critical threshold? → hazard discontinuity across the
  predicate's grid.
- Which component was necessary but insufficient? → ablation lift (Section 5).
- Was it predictable or only explainable afterward? → prospective shadow
  scoring (Section 11); anything failing this stays a specimen.

---

## 5. Discovery algorithm — generate, then try to kill

A bounded generate-and-falsify loop, run in the cold layer:

1. **Generate.** Beam search over the mechanism grammar seeded two ways:
   (a) mechanically, from predicate combinations with the highest matched
   lift; (b) from the reasoning layer, which reads filings/catalyst context
   around ignitions and proposes *candidate predicates or orderings* — but
   its proposals only enter as grammar-conforming, preregistered mechanisms.
   The LLM proposes; the ledger disposes.
2. **Score.** Matched ignition-vs-failure lift on training eras only.
3. **Falsification battery** (all must survive):
   - **Era stability**: lift holds in >= 2 disjoint eras.
   - **Regime stability**: lift is not confined to one regime stratum.
   - **Ablation necessity**: removing any step must materially reduce lift —
     otherwise the shorter mechanism replaces it (Occam by construction).
   - **Order necessity**: shuffling step order must reduce lift — otherwise
     it is a static screen wearing a sequence costume, and is demoted to one.
   - **Knowability audit**: every predicate evaluated strictly on
     asof-knowable values; one violation kills the mechanism.
   - **Anti-overlap**: episodes sharing (symbol, overlapping windows) are
     collapsed to one observation before any p-value is computed.
4. **Compile and gate.** Survivors go to the strategy compiler (Section 7);
   every compiled policy runs the full Section-4 battery plus matched nulls.
   The discovery engine gets no exemptions from the immune system.
5. **Learn from failure.** Every killed mechanism's failure mode is ledgered
   (which battery item killed it); generation round N+1 is constrained away
   from the killed region. Prediction errors in shadow operation re-enter
   here: identify which assumed transition failed → find the observation that
   would distinguish competing explanations → add that field to the Genome
   (or mark it unobtainable) → regenerate mutually exclusive candidates.

---

## 6. Options as sensors

Archived Cboe delayed chains become Genome fields before options are ever
traded (Section 2, options-state group): implied-vs-realized spread, skew
asymmetry, OI concentration, term slope into catalysts, and premium-per-unit
of historical gap exposure ("convexity price"). Two uses:

- **Sensor**: a mechanism may include options-state predicates (e.g. "IV term
  structure ignores a filed catalyst date") — information invisible in OHLCV.
- **Disagreement flag**: when the options surface contradicts the equity-side
  mechanism state, the disagreement itself is a field.

Delayed data is watermarked `REAL_DELAYED` and its lag is part of the
knowability audit. Options as *instruments* appear only in the strategy
compiler, only defined-risk.

---

## 7. Strategy compiler

A forecast is not a strategy. Each surviving mechanism is compiled into a
small preregistered policy family:

```
instrument   ∈ {shares, long call, call debit spread}   (defined-risk only)
entry        ∈ {immediate next-open, confirmation (next state advances)}
exit         ∈ {fixed target/stop, state-invalidation (mechanism dies)}
staging      ∈ {single entry, two-stage}
account tier ∈ {$7k, $25k, $85k+}  — min ticket, max concurrent,
                                     settlement/PDT constraints simulated
```

Every policy is backtested by the same event engine with the same costs and
account mechanics; the family is one multiplicity-corrected unit. Selection
maximizes net log-wealth growth subject to the C2 drawdown constraint. Option
policies price from archived chains with spread-crossing costs and are only
eligible when the chain history covers the era; otherwise the mechanism
trades shares or waits.

---

## 8. Target-Path Controller

State: (equity, tier), tiers $7K → $25K → $85K → $290K → $1M. For the
current book of validated policies (today: empty), the controller uses the
sizing engine's non-i.i.d. path generators to estimate, per candidate action:

```
ΔP(reach next tier before breaching the drawdown constraint)
```

and sizes within — never instead of — the binding constraint
`P(maxDD >= 50% over 20y) <= 5%`. Milestone probabilities at 1/5/10/20-year
horizons are reported with uncertainty bands from the Bayesian generator.
With an empty validated book the controller reports
**P(reach $25K) : NOT YET ESTIMABLE** — the honest zero. The controller can
change policy mix and aggressiveness per tier; it cannot override a gate,
touch status ceilings, or relax ε. ε and the tier ladder are human-owned
constants in config.

---

## 9. Two-speed online architecture and execution flow

**Cold layer (nightly, minutes-to-hours, LLM-permitted):** ingest feeds →
build Genome vintage → run discovery loop → falsification battery → compile
policies → gates → update ledger, FINDINGS, predictions. Output: a small
**signed policy table** (pure functions of Genome fields, with sizes) written
to disk with its evidence hash.

**Hot layer (session-time, milliseconds-to-seconds, no LLM anywhere):**
deterministic code loads the signed policy table; for each symbol event:
evaluate predicates → check spread/liquidity floors → size via precomputed
tier rules → construct order (shares or defined-risk option) → route to
paper broker (later: live behind the existing double-lock) → log the decision
prospectively and immutably *before* the outcome is knowable. Stops,
invalidation, and containment are hot-layer responsibilities with hard
per-day and per-position loss bounds read from config.

An LLM never sits between market data and an order. The hot layer refuses to
run if the policy table's evidence hash fails verification or its ledger
lineage is missing.

---

## 10. Historical replay + prospective shadow operation

- **Replay**: any policy table can be replayed over any historical vintage
  range; replay uses only asof-knowable fields (the knowability audit runs in
  replay too) and must reproduce backtest results within tolerance — a
  disagreement between the event engine and the replay engine is a bug, and
  the replay is authoritative.
- **Shadow**: every night, current policies emit immutable, sha256-ledgered
  predictions (machinery exists); sessions grade them; calibration (isotonic,
  live grades only) accumulates. Shadow operation is the only source of
  "prospective information value" — the quantity that decides promotion.

---

## 11. Test protocol and activation conditions

**Phase gate A — Alpha-v0** (Section 1): passes twice, identical evidence
hash. Nothing in C6+ builds until this holds.

**Phase gate B — discovery engine credibility.** The engine is
breakthrough-capable only when it can:

1. Discover a multistep mechanism that was not manually encoded (arrives via
   grammar search, not a hand-seeded template).
2. State it in one falsifiable sentence, preregistered.
3. Exhibit matched ignition and failure sets.
4. Predict future ignitions prospectively (shadow), before outcomes.
5. Reproduce across >= 2 eras and >= 2 regimes.
6. Survive point-in-time data, full costs, and account mechanics.
7. Compile into an executable policy that survives all gates.
8. Show prospective calibration in >= 60 shadow sessions.
9. Show learning: a documented case where a shadow failure changed the
   Genome or the grammar (not just added a feature until something passed).
10. Produce additional validated mechanisms at a measurable, reported rate
    (until then: NOT YET ESTIMABLE).

**Live-capital activation (all required, in order):**

1. Point-in-time survivorship-free data installed (human purchase approval).
2. Status ceiling lifted only by that installation (code-enforced).
3. >= 1 mechanism at `VALIDATED` via the full battery on point-in-time data.
4. >= 60 shadow sessions with positive net prospective information value and
   calibration inside its own confidence bands.
5. Written human sign-off recorded as HUMAN_DECISION in the ledger.
6. Initial live exposure bounded (proposal: <= 10% of equity at risk across
   all mechanism trades; human sets the number), scaling only on live
   evidence, never on backtest evidence.

**Standing safety rails:** defined-risk options only; the existing
double-lock on live trading stays; per-day loss containment in the hot
layer; no uncovered writing; no new leverage instruments in Phase 1.

---

## 12. Ordered commit plan

Each commit lands green (full test suite) and is pushed to the designated
branch; one draft PR carries C1-C5, a second carries C6+ after Phase gate A.

| # | Commit | Content |
|---|--------|---------|
| C1 | integrity semantics | Trials-vs-kills separation in FINDINGS/memo; replacement-rate lifecycle (`NOT YET ESTIMABLE` + observed counts); DSR audit block (raw / Sharpe-bearing / effective independent / preregistered method / raw-count sensitivity); survivorship language downgraded to hypothesis; status ceiling `PIPELINE_PROOF_ONLY` with code-enforced prohibition of VALIDATED/GRADUATED/EXECUTABLE while `universe_survivorship_biased` |
| C2 | binding constraint | `sizing_frontier_v2` binds on `P(maxDD >= 50% over horizon) <= ε` (ε=0.05, horizon=20y via trades/year mapping); 1/5/10/20-year breach probabilities reported; `P(lose 90%)` demoted to secondary; `STARTING_CAPITAL_USD = 7000`; ε and tier ladder as human-owned config |
| C3 | portability | Explicit `encoding="utf-8"` on all text I/O; cross-platform `python -m alpha_forge.setup`; README Windows console note (PowerShell UTF-8); Makefile kept as Unix convenience; **no artifact rewriting** |
| C4 | PR#12 follow-ups | Real maturation regression (labels planted after cutoff must not influence weights); date/symbol alignment assertions in the event panel; overlapping-label dependence test; NumPy deprecation warnings resolved; corrected test-count record |
| C5 | Alpha-v0 harness | `alpha_forge/alpha_v0.py` per Section 1: isolated acceptance ledger, killed-candidate replay, evidence-chain report + top-level hash, run-twice identity check |
| C6 | Genome schema + builder | Field registry with asof/source/lag; builder over existing feeds; FINRA short-interest ingest; knowability audit |
| C7 | states + matching | Predicate library, episode extraction, ignition/failure matched sets (fast-mover tiers added to pathfinder as outcome fields, not new models) |
| C8 | mechanism engine | Grammar, beam generation, falsification battery, failure-mode ledger |
| C9 | options sensors | Chain-derived Genome fields, REAL_DELAYED lag handling |
| C10 | strategy compiler | Policy families, defined-risk options pricing from archived chains, tier constraints |
| C11 | target-path controller | Milestone probability estimation under C2 constraint; NOT YET ESTIMABLE when the book is empty |
| C12 | two-speed + shadow | Signed policy table, hot-layer evaluator (no LLM), prospective shadow protocol, Phase-gate-B checklist wired into reporting |

**Frozen for Phase 1:** v9-style model/feature iteration, options *trading*
research beyond the compiler's defined-risk set, dashboard expansion, new
agents, any data purchase (requires approval), any scheduled/nightly Routine
(requires approval), live capital (Section 11 conditions).

---

## 13. What this is not

This design does not promise 0.493%/day, does not manufacture milestone
probabilities from an empty book, and does not treat the five killed
strategies as evidence the approach works — they are evidence the immune
system works. The bet of Phase 1 is specific: that mechanisms — ordered,
falsifiable, point-in-time-knowable sequences of market forces — carry
information that static fingerprints provably (five kills) did not, and that
a compiler from mechanisms to account-constrained policies converts whatever
information survives into growth. If that bet is wrong, this system will say
so, in the ledger, with the same discipline that killed everything before it.
