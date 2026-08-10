"""US equity universe from the NASDAQ Trader symbol directory.

Source: https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
Files: nasdaqlisted.txt (NASDAQ) and otherlisted.txt (NYSE/AMEX/etc), pipe-
delimited, refreshed nightly. These are CURRENT listings only — delisted
names are absent, so any universe built from them is SURVIVORSHIP-BIASED.
That flag propagates into every result via the data manifest and gate 7.
If Norgate/Sharadar/CRSP credentials appear in .env, the ingestion switches
to those and marks the dataset survivorship-free.

The research sample is a deterministic (seeded) uniform random sample of the
eligible universe rather than a hand-picked list: hand-picking injects
selection bias on top of survivorship bias; a uniform sample at least makes
per-window path counts extrapolable (sample_hits x universe/sample).
"""

from __future__ import annotations

import random

import requests

from alpha_forge.config import RAW_DIR

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

UNIVERSE_SAMPLE_SEED = 20260810  # pinned: sample must be identical across reruns


def fetch_symbol_directory(timeout: int = 60) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for name, url in (("nasdaqlisted", NASDAQ_LISTED_URL), ("otherlisted", OTHER_LISTED_URL)):
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        (RAW_DIR / f"{name}.txt").write_text(resp.text)
        lines = resp.text.strip().splitlines()
        header = lines[0].split("|")
        rows = []
        for line in lines[1:]:
            if line.startswith("File Creation Time"):
                continue
            parts = line.split("|")
            if len(parts) == len(header):
                rows.append(dict(zip(header, parts)))
        out[name] = rows
    return out


def eligible_common_stocks(directory: dict[str, list[dict]]) -> list[str]:
    """Common-stock-like issues: drop ETFs, test issues, and symbols with
    share-class/unit/warrant suffixes (heuristic: symbols with $ or . or
    length > 4 on the other-listed file's ACT symbol)."""
    symbols: set[str] = set()
    for row in directory.get("nasdaqlisted", []):
        sym = row.get("Symbol", "")
        if row.get("Test Issue") == "N" and row.get("ETF") == "N" and sym.isalpha():
            symbols.add(sym)
    for row in directory.get("otherlisted", []):
        sym = row.get("ACT Symbol", "")
        if row.get("Test Issue") == "N" and row.get("ETF") == "N" and sym.isalpha() and len(sym) <= 4:
            symbols.add(sym)
    return sorted(symbols)


def research_sample(symbols: list[str], n: int, seed: int = UNIVERSE_SAMPLE_SEED) -> list[str]:
    rng = random.Random(seed)
    if n >= len(symbols):
        return list(symbols)
    return sorted(rng.sample(symbols, n))
