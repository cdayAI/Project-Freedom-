"""CLI: verify the ledger hash chain. Exit nonzero if broken.

Run nightly before anything else; a broken chain halts the loop.
"""

from __future__ import annotations

import sys

from alpha_forge.ledger.ledger import ChainBrokenError, Ledger


def main() -> int:
    ledger = Ledger()
    try:
        n = ledger.verify_chain()
    except ChainBrokenError as exc:
        print(f"LEDGER CHAIN BROKEN: {exc}", file=sys.stderr)
        return 1
    print(f"ledger ok: {n} entries, cumulative trials={ledger.trial_count()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
