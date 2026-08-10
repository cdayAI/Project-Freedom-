"""Central paths and hard constants.

Every numeric constant here is either a structural choice (documented in
ARCHITECTURE.md) or cited in SOURCES.md. Nothing here is an estimated
market quantity — those live in data files with verification status.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("ALPHA_FORGE_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # vendor pulls, gitignored, refetchable
STORE_DIR = DATA_DIR / "store"             # parquet datasets + duckdb pattern store
FEES_DIR = DATA_DIR / "fees"               # sourced fee tables (committed)
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
LEDGER_DIR = REPO_ROOT / "ledger"          # append-only, hash-chained, committed
REPORTS_DIR = REPO_ROOT / "reports"
PREDICTIONS_DIR = REPO_ROOT / "predictions"  # dated immutable prediction files

DUCKDB_PATH = STORE_DIR / "alpha_forge.duckdb"

# --- Research-design constants (structural choices, see ARCHITECTURE.md) ---
TRADING_DAYS_PER_YEAR = 252          # NYSE calendar convention
WINDOW_TRADING_DAYS = 126            # 6-month rolling research window
PATH_MULTIPLES = (5, 10, 20, 50)     # N-x path targets
SEQUENCE_MIN_LEGS = 2
SEQUENCE_MAX_LEGS = 8

# --- Section 4 gate thresholds (non-negotiable, from the mission spec) ---
DSR_MIN_PROBABILITY = 0.95
PBO_KILL = 0.50
PBO_FLAG = 0.35
PERMUTATION_MIN_SHUFFLES = 20_000
CSCV_BLOCKS = 16
NULL_BASELINE_MIN_DRAWS = 1_000
NULL_BASELINE_PERCENTILE = 95.0
MIN_INDEPENDENT_TRADES = 100
MONTE_CARLO_MIN_PATHS = 20_000

# --- Ruin constraint defaults (human-settable; see ARCHITECTURE.md) ---
RUIN_DRAWDOWN_LEVEL = 0.50           # drawdown below 50% of high-water mark
STARTING_CAPITAL_USD = 2_000.0

for _d in (RAW_DIR, STORE_DIR, FEES_DIR, LEDGER_DIR, REPORTS_DIR, PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
