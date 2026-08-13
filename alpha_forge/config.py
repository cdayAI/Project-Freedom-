"""Central paths and hard constants.

Every numeric constant here is either a structural choice (documented in
ARCHITECTURE.md) or cited in SOURCES.md. Nothing here is an estimated
market quantity — those live in data files with verification status.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("ALPHA_FORGE_ROOT", Path(__file__).resolve().parent.parent))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE lines, '#' comments, no expansion.
    Existing environment variables always win over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(REPO_ROOT / ".env")

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
PATH_MULTIPLES = (3, 5, 10, 20, 50)  # N-x path targets; 3x is an AUXILIARY
                                     # training class (many more episodes for
                                     # the learned signal), 5x+ are the
                                     # reported research targets
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

# --- Ruin constraint (human-owned; see BREAKTHROUGH_PHASE1.md §0) ---
# Binding research constraint: P(maxDD >= RUIN_DRAWDOWN_LEVEL over
# RUIN_HORIZON_YEARS) <= RUIN_EPSILON. The 90%-loss probability is reported
# as a secondary measure, never as the binding constraint.
RUIN_DRAWDOWN_LEVEL = 0.50           # drawdown below 50% of high-water mark
RUIN_EPSILON = 0.05                  # human-set breach probability budget
RUIN_HORIZON_YEARS = 20.0            # multi-decade horizon (a 5% ANNUAL limit
                                     # would compound to ~64% over 20y)
STARTING_CAPITAL_USD = 7_000.0
MILESTONE_LADDER_USD = (7_000.0, 25_000.0, 85_000.0, 290_000.0, 1_000_000.0)

# --- Status ceiling (standing decision; see BREAKTHROUGH_PHASE1.md §0) ---
# Until point-in-time survivorship-free equities data is installed, every
# result is at most pipeline proof. The prohibition is enforced in code
# (ledger refuses GRADUATION entries), not by convention. Flipping the flag
# is a human act that must be accompanied by a HUMAN_DECISION ledger entry.
SURVIVORSHIP_FREE_DATA_INSTALLED = False
STATUS_CEILING = "PIPELINE_PROOF_ONLY"
PROHIBITED_STATUSES_UNTIL_PIT_DATA = ("VALIDATED", "GRADUATED", "EXECUTABLE")

# --- Replacement-rate lifecycle (see loop.replacement_rate) ---
MIN_REPLACEMENT_OBS_MONTHS = 6

for _d in (RAW_DIR, STORE_DIR, FEES_DIR, LEDGER_DIR, REPORTS_DIR, PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
