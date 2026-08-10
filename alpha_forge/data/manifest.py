"""Data manifest: every dataset records vendor, pull time, and known biases.

Append-only JSONL beside the data store. A dataset without a manifest entry
does not exist as far as the research agents are concerned.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from alpha_forge.config import MANIFEST_PATH


def record_dataset(
    dataset_id: str,
    vendor: str,
    description: str,
    known_biases: list[str],
    validation_summary: dict,
    survivorship_free: bool,
) -> dict:
    entry = {
        "dataset_id": dataset_id,
        "vendor": vendor,
        "pulled_utc": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "known_biases": known_biases,
        "survivorship_free": survivorship_free,
        "validation": validation_summary,
    }
    with MANIFEST_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def latest_entry(dataset_id: str) -> dict | None:
    if not MANIFEST_PATH.exists():
        return None
    found = None
    with MANIFEST_PATH.open() as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e["dataset_id"] == dataset_id:
                    found = e
    return found
