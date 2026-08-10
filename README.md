# ALPHA FORGE

A perpetual compounding engine: an autonomous research platform that
discovers, validates, sizes, retires, and replaces trading strategies, with
one objective — long-run net log-wealth growth under a hard ruin constraint.

Cycle 2: the full agent chain runs nightly on equities — data + regime
(Agent 0), date-aware costs (1), pathfinder (2), confluence identifiability
(3), sizing lab (4), trap detector (5), scanner predictions (6), reconciler
calibration (7), capacity/tiers (8a), council (8b), the self-improvement
loop with replacement-rate tracking and weekly memos (9), and execution
readiness with a doubly-locked broker interface (10). See `ARCHITECTURE.md`
for the system map, `SOURCES.md` for where every number comes from, and
`FINDINGS.md` (regenerated nightly) for current results.

## Run it

```bash
make setup           # venv + deps + editable install
make daily           # full nightly loop, end to end
make test            # reference test suite (101 tests incl. golden values for DSR/PBO)
make dashboard       # build TS/React dashboard + single-file HTML snapshot
make command-center  # local server: dashboard + live paper-account state
make verify-ledger
```

## Command center / terminal

`make command-center` serves the full trading terminal at
http://127.0.0.1:8321 (run it on YOUR machine — 127.0.0.1 is wherever the
server runs):

- **Terminal tab**: liquid-glass UI; canvas candlestick charts (wheel zoom,
  drag pan, crosshair OHLCV) with 1m/5m/15m/1h intraday (Alpaca IEX feed)
  and 15-year daily history from our own store; live quote stream (SSE,
  server-multiplexed); watchlist seeded from positions + predictions; paper
  order ticket (market/limit — PAPER endpoint only, journaled).
- **Research tab**: replacement rate, gate outcomes, path catalog,
  confluence, prediction feed, calibration.
- Top strip: live equity, buying power, PDT day-trade counter, market clock.

Credentials live in `.env` (gitignored, see `.env.example`) and never reach
the browser — the server proxies sanitized data only. See `SIGNALS.md` for
the full inventory of inputs the system uses and still needs, including the
honest latency budget (free tier = IEX feed; SIP is a paid upgrade).

Paper only, by construction: the client refuses the live endpoint outside
`place_order`'s human double-lock, the test suite strips broker credentials
and fails any test that attempts a broker API call, and live order flow
additionally requires a human-set env flag plus a ledgered HUMAN_DECISION.

No cloud dependencies. Real data (NASDAQ Trader + Stooq) is pulled on first
run; everything lands in `data/` (Parquet + DuckDB) with a manifest.

## What is enforced, not promised

- Pre-registration before testing (results without a registration are
  inadmissible — the ledger API refuses them).
- Hash-chained append-only ledger; the chain is verified before every run.
- The cumulative trial count feeds the Deflated Sharpe Ratio.
- Date-aware fees from primary sources; an unverified rate blocks the
  backtest that touches it.
- Signals fill at the next session's open, never the signal close.
- Gross is never displayed without net beside it.
- Survivorship bias is flagged on every result until a survivorship-free
  vendor is wired.
