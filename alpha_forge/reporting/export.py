"""Nightly export: one data.json consumed by the TS/React dashboard and the
static HTML snapshot. Everything comes from the same DuckDB/Parquet store
the agents write — the dashboard never computes research numbers itself.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpha_forge.config import PREDICTIONS_DIR, REPO_ROOT, STORE_DIR
from alpha_forge.execution.alpaca import account_snapshot_public
from alpha_forge.ledger import Ledger

DASHBOARD_DATA = REPO_ROOT / "dashboard" / "public" / "data.json"


def _maybe(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


def build_dashboard_data(ledger: Ledger, replacement: dict) -> dict:
    window_counts = _maybe(STORE_DIR / "path_window_counts.parquet")
    # v2 (time-matched) supersedes v1; the dashboard shows matched AUCs only
    confluence = _maybe(STORE_DIR / "confluence_v2_results.parquet")
    if confluence is not None and "matched_auc" in confluence.columns:
        confluence = confluence.rename({"matched_auc": "auc"})
    calibration = _maybe(STORE_DIR / "calibration.parquet")
    curves = _maybe(STORE_DIR / "backtest_curves.parquet")

    gate_reports = [
        e["payload"] for e in ledger.entries() if e["kind"] == "GATE_REPORT"
    ][-10:]

    latest_pred = None
    pred_files = sorted(PREDICTIONS_DIR.glob("predictions_*.json"))
    if pred_files:
        latest_pred = json.loads(pred_files[-1].read_text(encoding="utf-8"))

    data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ledger": {
            "entries": sum(1 for _ in ledger.entries()),
            "cumulative_trials": ledger.trial_count(),
        },
        "replacement_rate": replacement,
        "book": [],  # populated on first graduation
        "gate_reports": [
            {
                "strategy_id": g.get("strategy_id"),
                "verdict": g.get("verdict"),
                "reasons": g.get("reasons", []),
                "net_vs_gross": g.get("checks", {}).get("net_vs_gross"),
                "dsr": {
                    k: g.get("checks", {}).get("dsr", {}).get(k)
                    for k in ("dsr_probability", "n_trials")
                },
                "pbo": g.get("checks", {}).get("pbo", {}).get("pbo"),
            }
            for g in gate_reports
        ],
        "path_catalog": window_counts.group_by("n_multiple")
        .agg(
            pl.col("distinct_symbols_with_path").sum().alias("symbol_window_hits"),
            pl.len().alias("windows_with_any_path"),
        )
        .sort("n_multiple")
        .to_dicts()
        if window_counts is not None
        else [],
        "confluence": confluence.to_dicts() if confluence is not None else [],
        "calibration": calibration.to_dicts() if calibration is not None else [],
        "equity_curves": curves.to_dicts() if curves is not None else [],
        "predictions": latest_pred,
        # sanitized paper-account snapshot (balances + flags only, no
        # identifiers); None when no credentials are configured
        "account": account_snapshot_public(),
    }
    DASHBOARD_DATA.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DATA.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def build_static_snapshot() -> str | None:
    """Single self-contained HTML: built dashboard bundle + inlined data."""
    dist = REPO_ROOT / "dashboard" / "dist"
    index = dist / "index.html"
    if not index.exists() or not DASHBOARD_DATA.exists():
        return None
    html = index.read_text(encoding="utf-8")
    # inline the built assets referenced from index.html
    import re

    def inline_asset(m):
        rel = m.group(1).lstrip("/")
        asset = dist / rel
        if not asset.exists():
            return m.group(0)
        if rel.endswith(".js"):
            return f"<script type=\"module\">{asset.read_text(encoding="utf-8")}</script>"
        if rel.endswith(".css"):
            return f"<style>{asset.read_text(encoding="utf-8")}</style>"
        return m.group(0)

    html = re.sub(r'<script type="module"[^>]*src="([^"]+)"></script>', inline_asset, html)
    html = re.sub(r'<link rel="stylesheet"[^>]*href="([^"]+)"[^>]*>', inline_asset, html)
    data = DASHBOARD_DATA.read_text(encoding="utf-8")
    html = html.replace(
        "<head>", f"<head><script>window.__DATA__ = {data};</script>", 1
    )
    out = REPO_ROOT / "reports" / "dashboard_snapshot.html"
    out.write_text(html, encoding="utf-8")
    return str(out)
