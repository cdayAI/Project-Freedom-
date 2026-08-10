# SIGNALS & INPUTS — the complete inventory

> Update (cycle 3): the universe is now a 1,500-name uniform sample (1,308
> ingested), the confluence layer uses TIME-MATCHED controls (36 identifiable
> feature x class cells survive era-matching), and the event-driven engine
> trains walk-forward on past data with gates 10-11 (stress fills + ruin-
> constrained sizing) formalized. First trained candidate was killed by the
> matched null: the fingerprint identifies volatility, not direction —
> catalyst data and survivorship-free history remain the binding inputs.

Every input the system consumes now, and every input it still needs, with
source, status, and what it unlocks. LIVE = flowing tonight. DESIGN = adapter
built, waiting on a key/purchase. MISSING = not yet built. This file is the
answer to "what does the machine need to get smarter."

## 1. Flowing tonight (LIVE)

| Input | Source | Feeds | Notes |
|---|---|---|---|
| Daily OHLCV, 15y, 178-name uniform sample | Yahoo chart API (keyless) | Pathfinder, backtests, fingerprints, deep charts | survivorship-biased (gate-7 flag on everything) |
| Universe directory (6,954 common stocks) | NASDAQ Trader SymDir | sampling, eligibility | current listings only |
| Regime series (trend / vol / chop) | SPY + ^VIX, same API | hit tagging, prediction stamps | 200d MA, VIX terciles, realized-vol median |
| Regulatory fee tables (SEC 31, TAF, NFA) | SEC/FINRA/NFA primary docs | cost engine, every net figure | 14 SEC rate changes 2013→2026, all cited |
| Spread estimates | Corwin-Schultz from own OHLC | cost engine, half-spread per side | tick-floored; real NBBO replaces it when options data lands |
| Pre-move fingerprints (10 features) | computed, no lookahead | confluence, scanner, hypothesis generator | momentum stack, vol, volume-z, float-proxy via $vol, 52w-high distance, price, spread |
| Intraday bars 1m/5m/15m/1h + live trades/quotes/bid-ask | Alpaca Market Data (IEX feed) | terminal charts, quote stream, paper fills | free tier = IEX slice (~2-3% of consolidated volume) |
| Paper account state (equity, BP, PDT counter, positions) | Alpaca Trading API (paper) | command center, trap-detector reality checks | sanitized before any export |
| Prediction outcomes (5/21/63/126-bar grades) | own reconciler | calibration curves — the loop's fitness function | first grades mature 2026-08-14 |

## 2. Adapter built, waiting on a key (DESIGN)

| Input | Best sources (cost) | Unlocks | Priority |
|---|---|---|---|
| Survivorship-free daily equities incl. delisted | Norgate (~$30-40/mo) · Sharadar SEP (~$39/mo) | clears the gate-7 flag; unbiased fingerprints — delisted moonshots AND delisted disasters | **#1 — highest value per dollar** |
| Options chains w/ NBBO, IV, greeks, OI | Polygon options ($29-199/mo) · ORATS ($99/mo) · databento OPRA | options strategies; real spread costs; IV-crush simulation in traps | #3 |
| Consolidated SIP quotes (full-market, low-latency) | Alpaca Algo Trader Plus ($99/mo) | terminal latency + accurate NBBO for paper fills | #4 — nice, not blocking |
| Futures continuous (ES/MES/NQ/MNQ/CL/GC) | databento GLBX (~usage-priced) | futures agents; PDT-exempt strategies at small equity | #5 |

## 3. Not yet built (MISSING) — catalyst layer, priority #2 overall

Most extreme 6-month moves are catalyst-driven; every path hit is currently
tagged `catalyst_class: NONE`, which the mission correctly calls
aftermath-fingerprinting. Needed feeds, cheapest-first:

| Signal | Source candidates | Feeds |
|---|---|---|
| Earnings calendar + surprise history | Alpaca corporate actions (free w/ account) · Finnhub free tier · Nasdaq API | catalyst class EARNINGS; gap-risk conditioning in traps |
| FDA/PDUFA calendar | openFDA (free) + curated PDUFA lists (BiopharmCatalyst, paid) | catalyst class FDA — where 10x+ biotech paths live |
| Short interest (bi-monthly) | FINRA equity SI files (free download) | squeeze-fingerprint features; borrow-availability proxy |
| Shares float / outstanding | Sharadar fundamentals (bundled) · SEC EDGAR facts API (free) | float-rotation feature — small-float is the 50x substrate |
| Trading halts (LULD, T1/T2) | NASDAQ Trader halts feed (free) | halt-frequency trap simulation per name |
| Corporate actions (splits/mergers/spinoffs) | Alpaca corporate actions API (free) | data validation; delisting forensics |
| SEC filings stream (8-K, S-1, 13D) | EDGAR full-text (free) | catalyst classes DILUTION / ACTIVIST / GOING-CONCERN |
| Index/sector membership + breadth | free ETF holdings scrapes · Sharadar | regime refinement; sector-relative fingerprints |
| Borrow fee / hard-to-borrow | IBKR API (with account) · paid (S3/Ortex) | short-side feasibility in traps |
| Macro calendar (FOMC/CPI/NFP) | free (BLS/Fed schedules are published) | event-day risk flags on entries |

## 4. Signals the machine GENERATES (its own outputs as inputs)

- Calibration error by confidence bucket → hypothesis generator's steering
- Generator survivor rate per generator → meta-learning (change strategy when flat)
- Replacement rate → research-compute allocation
- Fingerprint identifiability per class (28 cells currently) → scanner features
- Kill reasons distribution → which gate kills most = where edges die

## 5. Latency budget (honest numbers)

| Hop | Today | Best available |
|---|---|---|
| Feed → Alpaca | IEX slice, ~ms at source | SIP consolidated ($99/mo) |
| Alpaca → command center | 1s server-side poll (batched) | WebSocket stream (~50-200ms) — planned upgrade, `websocket-client` |
| Server → browser | SSE push, ~0.5s cadence, loopback | same (loopback is not the bottleneck) |
| Render | canvas, 60fps | — |

The system trades daily bars at next-open; quote latency affects the
terminal experience and paper-fill realism, not the research edge. Spending
on the catalyst layer beats spending on latency until the book holds an
intraday strategy.
