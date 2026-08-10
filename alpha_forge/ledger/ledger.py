"""Append-only, tamper-evident research ledger.

Every hypothesis, configuration, and result is recorded here BEFORE and AFTER
each test. Entries are hash-chained: entry[i].hash = sha256(canonical(entry[i]
minus hash field)), and entry[i].prev_hash = entry[i-1].hash. Rewriting any
historical entry breaks every hash after it, so silent edits are detectable.

The cumulative TRIAL count in this ledger is an input to the Deflated Sharpe
Ratio. Wiping or editing the ledger is a statistical act, not housekeeping:
the only sanctioned reset is a HUMAN_DECISION entry that explicitly says so.

Entry kinds:
  PREREGISTRATION  hypothesis + universe + parameters, written before the test
  TRIAL            one configuration evaluated (feeds the DSR trial count)
  RESULT           outcome of a preregistered test (must reference a reg_id)
  GATE_REPORT      full Section-4 gate output for a candidate
  DATA_PULL        dataset ingestion event with validation summary
  HOLDOUT_TOUCH    a strategy touched the final holdout (once per strategy, ever)
  HUMAN_DECISION   explicitly logged human call (resets, purchases, overrides)
  KILL / GRADUATION strategy lifecycle events
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from alpha_forge.config import LEDGER_DIR

VALID_KINDS = {
    "PREREGISTRATION",
    "TRIAL",
    "RESULT",
    "GATE_REPORT",
    "DATA_PULL",
    "HOLDOUT_TOUCH",
    "HUMAN_DECISION",
    "KILL",
    "GRADUATION",
    "PREDICTION",   # content hash of a dated immutable predictions file
    "GRADE",        # reconciler's grading of a prediction file at a horizon
}

GENESIS_HASH = "0" * 64


class LedgerError(Exception):
    pass


class ChainBrokenError(LedgerError):
    pass


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _entry_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(_canonical(body)).hexdigest()


class Ledger:
    """Single append-only JSONL chain. Not safe for concurrent writers;
    the nightly orchestrator is the only writer by design."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else LEDGER_DIR / "ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tail_hash: str | None = None
        self._tail_seq: int | None = None

    # ---------- reading ----------

    def entries(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _load_tail(self) -> tuple[int, str]:
        """(last_seq, last_hash), scanning the file. Cached after first append."""
        if self._tail_seq is not None:
            return self._tail_seq, self._tail_hash  # type: ignore[return-value]
        last_seq, last_hash = -1, GENESIS_HASH
        for e in self.entries():
            last_seq, last_hash = e["seq"], e["hash"]
        self._tail_seq, self._tail_hash = last_seq, last_hash
        return last_seq, last_hash

    def verify_chain(self) -> int:
        """Recompute every hash and check linkage. Returns entry count."""
        prev = GENESIS_HASH
        count = 0
        for e in self.entries():
            if e.get("prev_hash") != prev:
                raise ChainBrokenError(
                    f"entry seq={e.get('seq')} prev_hash mismatch: chain edited or truncated"
                )
            if _entry_hash(e) != e.get("hash"):
                raise ChainBrokenError(f"entry seq={e.get('seq')} content hash mismatch")
            if e.get("seq") != count:
                raise ChainBrokenError(f"entry seq={e.get('seq')} out of order (expected {count})")
            prev = e["hash"]
            count += 1
        return count

    # ---------- writing ----------

    def append(self, kind: str, payload: dict) -> dict:
        if kind not in VALID_KINDS:
            raise LedgerError(f"unknown ledger entry kind: {kind}")
        last_seq, last_hash = self._load_tail()
        entry = {
            "seq": last_seq + 1,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "payload": payload,
            "prev_hash": last_hash,
        }
        entry["hash"] = _entry_hash(entry)
        with self.path.open("a") as f:
            f.write(json.dumps(entry, sort_keys=True, ensure_ascii=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._tail_seq, self._tail_hash = entry["seq"], entry["hash"]
        return entry

    # ---------- research API ----------

    def preregister(
        self,
        hypothesis: str,
        universe: str,
        parameters: dict,
        author: str = "loop",
    ) -> str:
        """Write the hypothesis BEFORE the test runs. Returns reg_id."""
        entry = self.append(
            "PREREGISTRATION",
            {
                "hypothesis": hypothesis,
                "universe": universe,
                "parameters": parameters,
                "author": author,
            },
        )
        return entry["hash"][:16]

    def _find_registration(self, reg_id: str) -> dict | None:
        for e in self.entries():
            if e["kind"] == "PREREGISTRATION" and e["hash"].startswith(reg_id):
                return e
        return None

    def record_trial(self, reg_id: str, config: dict, sharpe: float | None) -> None:
        """One evaluated configuration = one trial. sharpe may be None for
        trials where no Sharpe was computed (still counts toward N)."""
        if self._find_registration(reg_id) is None:
            raise LedgerError(f"trial references unknown preregistration {reg_id}")
        self.append("TRIAL", {"reg_id": reg_id, "config": config, "sharpe": sharpe})

    def record_result(self, reg_id: str, result: dict) -> None:
        reg = self._find_registration(reg_id)
        if reg is None:
            raise LedgerError(
                f"result references unknown preregistration {reg_id}: "
                "results without preregistration are inadmissible"
            )
        self.append("RESULT", {"reg_id": reg_id, "result": result})

    # ---------- statistics feeding the gates ----------

    def trial_count(self) -> int:
        return sum(1 for e in self.entries() if e["kind"] == "TRIAL")

    def trial_sharpes(self) -> list[float]:
        return [
            e["payload"]["sharpe"]
            for e in self.entries()
            if e["kind"] == "TRIAL" and e["payload"].get("sharpe") is not None
        ]

    def holdout_touches(self, strategy_id: str | None = None) -> list[dict]:
        touches = [e for e in self.entries() if e["kind"] == "HOLDOUT_TOUCH"]
        if strategy_id is not None:
            touches = [t for t in touches if t["payload"].get("strategy_id") == strategy_id]
        return touches
