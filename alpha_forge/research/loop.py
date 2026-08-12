"""Agent 9 support: replacement rate, hypothesis generation, weekly memo,
meta-learning bookkeeping. The loop's fitness function is out-of-sample
calibration accuracy — never in-sample Sharpe.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone

import polars as pl

from alpha_forge.config import MIN_REPLACEMENT_OBS_MONTHS, REPORTS_DIR
from alpha_forge.ledger import Ledger

MAX_GENERATED_HYPOTHESES_PER_NIGHT = 2  # research-compute budget, structural


def replacement_rate(ledger: Ledger) -> dict:
    """Observed pipeline lifecycle counts, plus a health metric that refuses
    to exist before it is measurable.

    A KILL gate report is an event; a STRATEGY kill is a distinct strategy
    identity with a KILL verdict — the two are reported separately, and
    neither is a trial count. The replacement-rate HEALTH metric (are
    graduations replacing deaths?) is meaningful only once at least one
    strategy has ever graduated AND the pipeline has a minimum observation
    window — an empty pipeline is not 'healthy', it is unmeasured, so until
    then health is NOT YET ESTIMABLE and only the raw counts are reported."""
    by_month: dict[str, dict] = defaultdict(lambda: {"graduated": 0, "killed": 0})
    graduated_ever = 0
    months_seen: set[str] = set()
    for e in ledger.entries():
        month = e["ts_utc"][:7]
        months_seen.add(month)
        if e["kind"] == "GRADUATION":
            by_month[month]["graduated"] += 1
            graduated_ever += 1
        elif e["kind"] == "KILL":
            by_month[month]["killed"] += 1
        elif e["kind"] == "GATE_REPORT" and e["payload"].get("verdict") == "KILL":
            by_month[month]["killed"] += 1
    months = sorted(by_month)
    recent = months[-3:]
    grad = sum(by_month[m]["graduated"] for m in recent)
    dead = sum(by_month[m]["killed"] for m in recent)
    estimable = len(months_seen) >= MIN_REPLACEMENT_OBS_MONTHS and graduated_ever >= 1
    rate = (grad / dead if dead else (float("inf") if grad else None)) if estimable else None
    if estimable:
        health = "healthy" if (rate is None or rate >= 1.0) else "unhealthy"
    else:
        health = (
            "NOT YET ESTIMABLE — requires >= "
            f"{MIN_REPLACEMENT_OBS_MONTHS} observed months and >= 1 graduation "
            "before a replacement rate exists"
        )
    return {
        "by_month": {m: by_month[m] for m in months},
        "observed": {
            "graduations_ever": graduated_ever,
            "distinct_strategies_killed": len(ledger.killed_strategies()),
            "kill_gate_reports_trailing_3m": dead,
            "graduations_trailing_3m": grad,
            "months_observed": len(months_seen),
        },
        "trailing_3m_graduated": grad,
        "trailing_3m_killed": dead,
        "trailing_3m_rate": rate,
        "health": health,
        "note": "kill gate reports are events, not distinct strategies; "
        "trials are a third, separate population (see ledger.trial_audit)",
    }


def stable_survivor_features(ledger: Ledger, family_prefix: str = "confluence_identifiability") -> set[str] | None:
    """Features IDENTIFIABLE in BOTH of the two most recent runs of the main
    confluence family. One-vintage artifacts (a feature that clears BH once
    and never again) don't deserve a night of gate compute; stability across
    re-tests is the cheapest replication test available. Returns None while
    fewer than two families exist (bootstrapping: no filter yet)."""
    survivor_sets: list[set[str]] = []
    for e in ledger.entries():
        if e["kind"] != "RESULT":
            continue
        res = e["payload"].get("result", {})
        fam = res.get("family", "")
        if fam.startswith(family_prefix) and "catalyst" not in fam:
            cells = res.get("survivor_cells", [])
            survivor_sets.append({c["feature"] for c in cells})
    if len(survivor_sets) < 2:
        return None
    return survivor_sets[-1] & survivor_sets[-2]


def generate_hypotheses(confluence: pl.DataFrame | None, ledger: Ledger) -> list[dict]:
    """Turn Confluence survivors into pre-registrable screening hypotheses.

    v1 generation strategy (ledgered so meta-learning can grade it): each
    IDENTIFIABLE (feature, class) cell becomes 'rank the cross-section by
    this feature in the hit-typical direction; hold 1 month; top-k'. The
    hypothesis is only EMITTED here — preregistration + testing happen in
    the nightly loop under the compute budget, and every config becomes a
    ledgered trial like any other.

    Stability rule: once two family runs exist, only features identifiable
    in BOTH of the last two runs are eligible.
    """
    if confluence is None:
        return []
    survivors = confluence.filter(pl.col("verdict") == "IDENTIFIABLE")
    stable = stable_survivor_features(ledger)
    if stable is not None:
        survivors = survivors.filter(pl.col("feature").is_in(sorted(stable)))
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


def generate_hypotheses_from_calibration(
    grades: pl.DataFrame | None,
    predictions_dir,
    horizon: int = 21,
    min_graded: int = 30,
    auc_margin: float = 0.15,
) -> list[dict]:
    """Mission 5.2 — learn from being wrong: among GRADED predictions, which
    feature best separates the winners from the losers? A separating feature
    the scanner is not already exploiting becomes the next pre-registrable
    hypothesis. Activates automatically once enough grades exist; returns []
    until then (never fabricates a learning signal).
    """
    import json as _json

    from alpha_forge.research.confluence import _auc  # rank-based, reused

    if grades is None or grades.height == 0:
        return []
    g = grades.filter(pl.col("horizon") == horizon)
    if g.height < min_graded:
        return []

    # join each graded row to the features recorded in its immutable file
    feat_rows = []
    import pathlib

    for path in sorted(pathlib.Path(predictions_dir).glob("predictions_*.json")):
        doc = _json.loads(path.read_text(encoding="utf-8"))
        for p in doc.get("predictions", []):
            feat_rows.append(
                {"file": path.name, "symbol": p["instrument"], **{
                    f"feat_{k}": v for k, v in (p.get("features") or {}).items()
                }}
            )
    if not feat_rows:
        return []
    feats = pl.DataFrame(feat_rows)
    joined = g.join(feats, on=["file", "symbol"], how="inner")
    if joined.height < min_graded:
        return []

    win = joined.filter(pl.col("fwd_return") > 0)
    loss = joined.filter(pl.col("fwd_return") <= 0)
    if min(win.height, loss.height) < 10:
        return []

    import numpy as np

    best = None
    for col in [c for c in joined.columns if c.startswith("feat_")]:
        w = win[col].drop_nulls().to_numpy().astype(float)
        l = loss[col].drop_nulls().to_numpy().astype(float)
        if w.size < 10 or l.size < 10:
            continue
        auc = _auc(w, l)
        if best is None or abs(auc - 0.5) > abs(best[1] - 0.5):
            best = (col.removeprefix("feat_"), auc)
    if best is None or abs(best[1] - 0.5) < auc_margin:
        return []
    feature, auc = best
    direction = "desc" if auc > 0.5 else "asc"
    return [
        {
            "generator": "calibration_error_v2",
            "feature": feature,
            "auc_win_vs_loss": auc,
            "rank_direction": direction,
            "horizon": horizon,
            "n_graded": joined.height,
            "hypothesis": (
                f"Among graded predictions at the {horizon}-bar horizon, "
                f"{feature} separates winners from losers (AUC {auc:.3f}, "
                f"n={joined.height}). Adding it ({direction}) to the entry "
                "score improves the event strategy net of costs."
            ),
        }
    ]


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
    obs = rep_rate.get("observed", {})
    rate_str = rep_rate["health"] if rate is None else (
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
        f"- Pipeline replacement rate (trailing 3m): **{rate_str}** — observed: "
        f"{obs.get('graduations_ever', 0)} graduations ever / "
        f"{obs.get('distinct_strategies_killed', 0)} distinct strategies killed "
        f"({rep_rate['trailing_3m_graduated']} graduated / "
        f"{rep_rate['trailing_3m_killed']} kill reports, trailing 3m)",
        f"- Cumulative ledgered trial count: **{ledger.trial_count()}** "
        "(trials are evaluated configurations — NOT strategy kills; see "
        "dsr_audit in any gate report for the population breakdown)",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
