"""Agent 9 support: replacement rate, hypothesis generation, weekly memo,
meta-learning bookkeeping. The loop's fitness function is out-of-sample
calibration accuracy — never in-sample Sharpe.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone

import polars as pl

from alpha_forge.config import REPORTS_DIR
from alpha_forge.ledger import Ledger

MAX_GENERATED_HYPOTHESES_PER_NIGHT = 2  # research-compute budget, structural


def replacement_rate(ledger: Ledger) -> dict:
    """Monthly graduations vs deaths. The system's single most important
    health metric; a trailing-3-month rate < 1.0 leads the weekly memo."""
    by_month: dict[str, dict] = defaultdict(lambda: {"graduated": 0, "killed": 0})
    for e in ledger.entries():
        month = e["ts_utc"][:7]
        if e["kind"] == "GRADUATION":
            by_month[month]["graduated"] += 1
        elif e["kind"] == "KILL":
            by_month[month]["killed"] += 1
        elif e["kind"] == "GATE_REPORT" and e["payload"].get("verdict") == "KILL":
            by_month[month]["killed"] += 1
    months = sorted(by_month)
    recent = months[-3:]
    grad = sum(by_month[m]["graduated"] for m in recent)
    dead = sum(by_month[m]["killed"] for m in recent)
    rate = grad / dead if dead else (float("inf") if grad else None)
    return {
        "by_month": {m: by_month[m] for m in months},
        "trailing_3m_graduated": grad,
        "trailing_3m_killed": dead,
        "trailing_3m_rate": rate,
        "healthy": (rate is None) or (rate >= 1.0),
        "note": "rate is graduations/deaths; None = no deaths yet AND no graduations",
    }


def generate_hypotheses(confluence: pl.DataFrame | None, ledger: Ledger) -> list[dict]:
    """Turn Confluence survivors into pre-registrable screening hypotheses.

    v1 generation strategy (ledgered so meta-learning can grade it): each
    IDENTIFIABLE (feature, class) cell becomes 'rank the cross-section by
    this feature in the hit-typical direction; hold 1 month; top-k'. The
    hypothesis is only EMITTED here — preregistration + testing happen in
    the nightly loop under the compute budget, and every config becomes a
    ledgered trial like any other.
    """
    if confluence is None:
        return []
    survivors = confluence.filter(pl.col("verdict") == "IDENTIFIABLE")
    if survivors.height == 0:
        return []
    # strongest effects first: |AUC - 0.5|, dedup by feature
    seen = set()
    out = []
    rows = sorted(
        survivors.iter_rows(named=True), key=lambda r: abs(r["auc"] - 0.5), reverse=True
    )
    for r in rows:
        if r["feature"] in seen:
            continue
        seen.add(r["feature"])
        direction = "desc" if r["auc"] > 0.5 else "asc"
        out.append(
            {
                "generator": "confluence_survivor_screen_v1",
                "feature": r["feature"],
                "source_class": r["n_multiple"],
                "auc": r["auc"],
                "rank_direction": direction,
                "hypothesis": (
                    f"Ranking the eligible cross-section by {r['feature']} "
                    f"({direction}ending, the hit-typical direction, AUC={r['auc']:.3f} "
                    f"for {r['n_multiple']}x paths) and holding the top-k for one month "
                    "beats the matched random-selection null net of full costs."
                ),
            }
        )
        if len(out) >= MAX_GENERATED_HYPOTHESES_PER_NIGHT:
            break
    return out


def generator_hit_rate(ledger: Ledger) -> dict:
    """Meta-learning: survivor rate of generated hypotheses, per generator.
    If it is not improving, the loop must change generation strategy and log
    the change as a HUMAN_DECISION-visible RESULT."""
    stats: dict[str, dict] = defaultdict(lambda: {"tested": 0, "passed": 0})
    reg_generator: dict[str, str] = {}
    for e in ledger.entries():
        if e["kind"] == "PREREGISTRATION":
            gen = e["payload"].get("parameters", {}).get("generator")
            if gen:
                reg_generator[e["hash"][:16]] = gen
        elif e["kind"] == "GATE_REPORT":
            gen = reg_generator.get(e["payload"].get("reg_id", ""))
            if gen:
                stats[gen]["tested"] += 1
                if e["payload"].get("verdict") == "PASS":
                    stats[gen]["passed"] += 1
    return {
        g: {**s, "survivor_rate": (s["passed"] / s["tested"]) if s["tested"] else None}
        for g, s in stats.items()
    }


def write_weekly_memo(
    ledger: Ledger,
    rep_rate: dict,
    calibration: pl.DataFrame | None,
    best_validated: str,
    open_questions: list[str],
) -> str | None:
    """One page for the human owner, Sundays (or when missing this week)."""
    today = date.today()
    iso = today.isocalendar()
    path = REPORTS_DIR / f"memo-{iso.year}-W{iso.week:02d}.md"
    if path.exists():
        return None
    rate = rep_rate["trailing_3m_rate"]
    rate_str = "N/A (no completed lifecycle events yet)" if rate is None else (
        "inf (deaths=0)" if rate == float("inf") else f"{rate:.2f}"
    )
    lines = [
        f"# ALPHA FORGE weekly decision memo — {today.isoformat()}",
        "",
    ]
    if rep_rate["trailing_3m_killed"] > 0 and (rate is not None and rate < 1.0):
        lines += [
            f"**REPLACEMENT RATE {rate_str} < 1.0 — the graveyard is outrunning the "
            "pipeline. Research compute is reallocated to hypothesis generation "
            "until this recovers. This is the lead item by rule.**",
            "",
        ]
    lines += [
        f"- Pipeline replacement rate (trailing 3m): **{rate_str}** "
        f"({rep_rate['trailing_3m_graduated']} graduated / {rep_rate['trailing_3m_killed']} killed)",
        f"- Cumulative ledgered trial count: **{ledger.trial_count()}**",
        f"- Best validated 6-month projected multiple: {best_validated}",
        "- Calibration drift: "
        + (
            "no live calibration data yet (first prediction files are maturing)"
            if calibration is None or calibration.height == 0
            else f"{calibration.height} calibration cells accumulated; see dashboard"
        ),
        "",
        "## What graduated / died this week",
    ]
    week_start = f"{today.year}-{today.month:02d}"
    events = [
        e for e in ledger.entries()
        if e["kind"] in ("GRADUATION", "KILL", "GATE_REPORT") and e["ts_utc"][:7] == week_start
    ]
    if events:
        for e in events[-10:]:
            p = e["payload"]
            what = p.get("strategy_id", "?")
            verdict = p.get("verdict", e["kind"])
            reasons = "; ".join(p.get("reasons", [])[:2])
            lines.append(f"- {e['ts_utc'][:10]} **{what}** → {verdict}" + (f" ({reasons})" if reasons else ""))
    else:
        lines.append("- nothing this week")
    lines += ["", "## Open questions needing a human call"]
    lines += [f"- {q}" for q in open_questions] or ["- none"]
    lines += [
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()} — the dashboard is "
        "for browsing; this memo is for deciding._",
    ]
    path.write_text("\n".join(lines) + "\n")
    return str(path)
