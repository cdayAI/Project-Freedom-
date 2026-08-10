"""Build data/fees/*.json from a compiled, source-cited rate history.

Input: a JSON document with sec_section31 / finra_taf_equity /
finra_taf_options arrays, each entry carrying effective_date, the rate,
source_url, source_title, and verified. Entries with verified=false are kept
(they document what is still missing) but FeeSchedule refuses to serve them.

Usage: python -m alpha_forge.costs.build_tables <compiled.json>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from alpha_forge.config import FEES_DIR

TABLE_SPECS = {
    "sec_section31": {
        "unit": "usd_per_million_usd_covered_sales",
        "rate_key": "rate_per_million_usd",
    },
    "finra_taf_equity": {
        "unit": "usd_per_share_sold_capped",
        "rate_key": "rate_per_share_usd",
    },
    "finra_taf_options": {
        "unit": "usd_per_contract_sold",
        "rate_key": "rate_per_contract_usd",
    },
}


def build(compiled: dict) -> list[str]:
    written = []
    for name, spec in TABLE_SPECS.items():
        rows = compiled.get(name, [])
        if not rows:
            continue
        entries = []
        for r in rows:
            entry = {
                "effective_date": r["effective_date"],
                "rate": r[spec["rate_key"]],
                "verified": bool(r.get("verified", False)),
                "source_url": r.get("source_url", ""),
                "source_title": r.get("source_title", ""),
            }
            if "cap_per_trade_usd" in r:
                entry["cap_per_trade_usd"] = r["cap_per_trade_usd"]
            entries.append(entry)
        doc = {
            "fee": name,
            "unit": spec["unit"],
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "entries": sorted(entries, key=lambda e: e["effective_date"]),
            "notes": compiled.get("notes", []),
        }
        path = FEES_DIR / f"{name}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        written.append(str(path))
    return written


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        compiled = json.load(f)
    for p in build(compiled):
        print(f"wrote {p}")
