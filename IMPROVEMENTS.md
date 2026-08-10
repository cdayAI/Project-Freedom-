# ACCURACY IMPROVEMENT AUDIT — stocks & options

Scope decision (owner, 2026-08-10): the system concentrates on **US equities
and equity options**. Futures adapters stay dormant (interfaces retained,
no research compute spent).

Accuracy means three different things here, and each item below names which
one it moves:
- **[D] Data fidelity** — is the input true?
- **[S] Statistical validity** — is the inference sound at the stated error rate?
- **[C] Calibration** — do stated probabilities match realized frequencies?

## Implemented in this change

| Item | Type | What it fixes |
|---|---|---|
| Conservative spread blend: per-side cost = max(Corwin-Schultz, Abdi-Ranaldo)/2, tick-floored | D | One estimator can understate spread exactly where estimation is hardest; the max may overstate cost (kills marginal edges) but never quietly understates it (graduates fictions). AR reference: Abdi & Ranaldo (2017), RFS 30(12). |
| Isotonic confidence calibration (PAV) from live grades | C | The scanner's similarity percentile is a rank, not a probability. Once ≥100 predictions are graded at the 21-bar horizon, an isotonic map converts score → P(fwd>0), Brier-scored, and the confidence field flips to CALIBRATED. Until then it honestly says UNCALIBRATED. |
| Catalyst-conditional identifiability family | S | Pooling EARNINGS_8K / OTHER_8K / NONE hits averages away class-specific signatures. New pre-registered family measures each class separately (mission Section 1 requirement). |
| Two-family stability rule for generated hypotheses | S | A feature that clears BH once and never again is likely a vintage artifact; gate compute now goes only to features identifiable in BOTH of the last two family runs. |
| Universe sample 1,500 → 2,500 | D/S | +67% expected hit episodes → more matched groups → tighter permutation nulls and more identifiable-cell power per family. |
| Real options chains, archived nightly (Cboe delayed feed) | D | No free source sells chain history, so the system builds its own: daily snapshots of real bids/asks/IV/OI for the prediction names + 30 most liquid names. Watermarked REAL_DELAYED; nothing is ever model-priced. Powers measured spread-of-premium stats (the dominant options cost) today, and becomes a genuine young backtest dataset over months. |
| OCC/ORF options fee tables | D | Sourced from primary schedules (see data/fees/); completes the options cost model to the same date-aware, verified-or-raise standard as equities. |
| Sizing engine v2: the i.i.d. Monte Carlo is demoted to a measured baseline | S | v1 resampled trades independently — no loss clustering, no regimes, no parameter uncertainty; every assumption biased risk DOWN. v2 sizes under the WORST of: stationary block bootstrap (Politis-Romano 1994, preserves serial dependence), Bayesian bootstrap (Rubin 1981, propagates parameter uncertainty; yields a derived Kelly posterior whose 5th percentile replaces hand-waved fractional Kelly), and a regime-conditional transition chain when per-trade regime labels exist. Gate 11 now uses it; the i.i.d. number is reported beside the honest ones so the understatement is measured, never assumed. Verified on constructed clustered data: i.i.d. understates 50%-drawdown probability measurably. ≥20k paths per generator. |

## Implemented in the follow-up "go get the data" change

| Item | Type | What it adds |
|---|---|---|
| FINRA Reg SHO daily short-sale volume feed | D | CAUSE-side data OHLCV cannot see: per-symbol daily short-sale pressure (12k+ symbols/day, free, primary). Features `short_ratio_5d` / `short_ratio_z`, lagged one session by construction (files publish after the close). Resumable backfill toward ~8y of files, extended nightly. |
| Full EDGAR filing-type catalog | D | The submissions responses already fetched for 8-Ks also carry Form 4 (insider transactions), S-1/S-3/F-1/F-3/424B (dilution pipeline). New ex-ante features: `days_since_form4`, `n_form4_90d`, `days_since_dilution_filing`. |
| Forward earnings clock | D | `days_until_expected_earnings` from each symbol's OWN filing cadence (median inter-earnings gap, ≥4 past reports) — the forward-looking half of catalyst timing, no calendar purchase needed. Negative = overdue. |

All three flow through the features panel automatically (the feature list is
data, not configuration), so the next confluence family measures their
identifiability under the same BH discipline, and the generator can screen
on any that survive twice.

## Specced, next in line (no purchases needed)

1. **Weighted multivariate signal** [S]: replace the equal-weight directional
   composite in `eventbt.fold_directions` with per-fold ridge-logistic
   weights fit on matured matched groups (IRLS, numpy-only). Same walk-forward
   discipline, same leakage guards; becomes `evt_fp5x_v6` with its own
   preregistration. Expected gain: features with matched AUC 0.9 should not
   count the same as 0.6 ones. Risk: more capacity to overfit — that is what
   the fold structure and gates are for.
2. **Per-catalyst-class event strategies** [S]: if the catalyst-conditional
   family shows stable class-specific signatures, train separate event
   candidates per class (EARNINGS_8K entries behave differently from
   squeeze-type NONE entries).
3. **Expected-earnings-window feature** [D]: quarterly 8-K cadence per symbol
   predicts the next earnings window (~91-day rhythm from own filing
   history, strictly ex-ante). `days_since_earnings_8k` captures half of
   this; `days_until_expected_earnings` is the forward-looking half.
4. **Options paper structures** [C]: with chains archiving and fees verified,
   Agent 10 can paper-trade defined-risk call spreads on scanner candidates
   against real quotes — building the options calibration record the same
   way equities predictions build theirs.
5. **8-K item-code taxonomy** [S]: the catalog already stores full item
   strings; classes beyond 2.02 (1.01 material agreements, 5.02 officer
   changes, 3.01 delisting notices, 8.01 other) are computable tonight from
   data on disk and may split OTHER_8K into informative subclasses.

## Purchase-gated (the honest ceiling of free data)

Ranked by expected accuracy per dollar; all are standing items in the weekly
memo until decided:

1. **Survivorship-free daily equities** (Norgate ~$30-40/mo or Sharadar SEP).
   THE dominant remaining bias: every fingerprint is learned from names that
   survived to today's listing file. Path counts are a lower bound, but
   fingerprints are distorted in unknown directions — reverse-loser effects
   are inflated (dead losers are missing), squeeze signatures deflated.
   Every gate-7 flag in every report traces to this. Nothing statistical can
   substitute; the adapter interface is already built.
2. **Historical options chains** (ORATS/Polygon options, ~$30-100/mo): turns
   the options program from forward-collection into immediate backtesting.
   The self-built archive shortens the wait but cannot recreate the past.
3. **Intraday bars/quotes** (Polygon/databento): replaces the estimated
   spread blend with measured spreads, models open-auction fills properly,
   and tightens gap-through-stop realism. Third priority because the current
   model errs conservative by construction.

## Rejected after consideration

- **Scraping Stooq/Yahoo-crumb workarounds for more history** — anti-bot
  walls are a no, and 15y depth is not the binding constraint; survivorship
  is.
- **Synthetic option pricing to "backtest" options now** — banned by the
  operating rules for good reason; a Black-Scholes backtest of a convexity
  strategy is a fiction with error bars.
- **More features per family without more episodes** — power comes from
  groups, not columns; BH across a wider family with the same 30-ish
  episodes per class just raises the bar each cell must clear.
