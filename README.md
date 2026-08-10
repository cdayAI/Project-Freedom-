# ALPHA FORGE

A perpetual compounding engine: an autonomous research platform that
discovers, validates, sizes, retires, and replaces trading strategies, with
one objective — long-run net log-wealth growth under a hard ruin constraint.

This is cycle 1: equities end-to-end. See `ARCHITECTURE.md` for the system
map and roadmap, `SOURCES.md` for where every number comes from, and
`FINDINGS.md` (regenerated nightly) for current results.

## Run it

```bash
make setup   # venv + deps + editable install
make daily   # full nightly loop: ingest -> validate -> pathfinder -> gates -> FINDINGS.md
make test    # reference test suite (45 tests incl. golden values for DSR/PBO)
make verify-ledger
```

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
