"""Market Genome schema v1 — the point-in-time field registry.

Every field the discovery engine may read is registered here with the three
things a backtest cannot be trusted without: where the value comes from,
WHEN it became knowable at a close-of-session decision, and the group it
belongs to. A field that is not registered does not exist to the engine;
adding one is a schema version bump, never a rewrite of old genome files.

Knowability is mechanical, not disciplinary:

  lag_sessions          the value for event-date d is exposed to decisions
                        at session d + lag. 0 = knowable at that session's
                        close (same-session closes, prints). EDGAR-derived
                        fields carry lag 1 by policy — a filing stamped
                        during or after the close is not tradable that
                        close, and conservatism is the house style.
  quarterly_published   DERA-style sources drop one bulk file per quarter,
                        ~PUBLICATION_DELAY_DAYS after quarter end. A live
                        system inside the quarter does NOT have the file, so
                        the genome exposes the newest value from the last
                        PUBLISHED quarter, not the filing-date value the
                        archive retroactively contains. This closes a real
                        optimism gap: the nightly features panel attaches
                        insider flow by filing date, which is honest for the
                        live scanner but optimistic in backtests.

Groups follow BREAKTHROUGH_PHASE1.md §2. Fields whose feeds are not wired
yet (options state — C9; attention — needs a source decision; FINRA
bi-monthly short interest — needs ingest) are NOT registered: the registry
describes what exists, the design doc describes what is planned.

outcome-group fields are labels. They are registered so provenance covers
them, but the builder refuses to lag them into the feature surface — a
label is never a feature.
"""

from __future__ import annotations

from dataclasses import dataclass

GENOME_SCHEMA_VERSION = 1
PUBLICATION_DELAY_DAYS = 35  # DERA quarterly drop: conservative upper bound


@dataclass(frozen=True)
class GenomeField:
    name: str
    group: str  # price_liquidity | volatility | catalysts | insider |
    #             short_pressure | regime | options_state | outcome
    source: str
    lag_sessions: int = 0
    quarterly_published: bool = False
    description: str = ""
    source_frame: str = "panel"  # panel | regime | options — which input
    #             frame carries it; the refuse-on-missing check guards the
    #             panel, joined frames have their own presence semantics


FIELDS: tuple[GenomeField, ...] = (
    # ---- price / liquidity (same-session closes: knowable at the close) --
    GenomeField("ret_21d", "price_liquidity", "yahoo_daily", 0, False, "21-session return"),
    GenomeField("ret_63d", "price_liquidity", "yahoo_daily", 0, False, "63-session return"),
    GenomeField("ret_126d", "price_liquidity", "yahoo_daily", 0, False, "126-session return"),
    GenomeField("ret_252d", "price_liquidity", "yahoo_daily", 0, False, "252-session return"),
    GenomeField("volume_z_20v126", "price_liquidity", "yahoo_daily", 0, False,
                "20d volume z vs 126d base"),
    GenomeField("dollar_vol_med_20d", "price_liquidity", "yahoo_daily", 0, False,
                "median 20d dollar volume"),
    GenomeField("dist_from_252d_high", "price_liquidity", "yahoo_daily", 0, False,
                "distance below 52w high"),
    GenomeField("price", "price_liquidity", "yahoo_daily", 0, False, "close"),
    GenomeField("cs_spread_est", "price_liquidity", "derived_corwin_schultz", 0, False,
                "estimated half-spread"),
    # ---- volatility ------------------------------------------------------
    GenomeField("vol_20d_ann", "volatility", "yahoo_daily", 0, False,
                "20d realized vol, annualized"),
    # ---- catalysts (EDGAR real-time; lag 1 by conservative policy) -------
    GenomeField("days_since_earnings_8k", "catalysts", "edgar_submissions", 1, False,
                "sessions since last 2.02 8-K"),
    GenomeField("days_since_any_8k", "catalysts", "edgar_submissions", 1, False,
                "sessions since last 8-K"),
    GenomeField("days_until_expected_earnings", "catalysts", "edgar_submissions", 1, False,
                "median-cadence earnings countdown"),
    GenomeField("days_since_form4", "catalysts", "edgar_submissions", 1, False,
                "sessions since last Form 4"),
    GenomeField("n_form4_90d", "catalysts", "edgar_submissions", 1, False,
                "Form 4 count, trailing 90d"),
    GenomeField("days_since_dilution_filing", "capital_structure", "edgar_submissions", 1,
                False, "sessions since S-1/S-3/F-1/F-3/424B"),
    # ---- insider flow (DERA quarterly: published, not filed) -------------
    GenomeField("insider_net_buy_90d_usd", "insider", "sec_dera_form345", 1, True,
                "net open-market buying, trailing 90d, USD"),
    GenomeField("insider_buy_ratio_90d", "insider", "sec_dera_form345", 1, True,
                "buys / (buys + sells), trailing 90d"),
    # ---- short pressure (Reg SHO daily files; T+1 lag applied at attach) -
    GenomeField("short_ratio_5d", "short_pressure", "finra_regsho_daily", 0, False,
                "short volume ratio, 5d mean (source lag already applied)"),
    GenomeField("short_ratio_z", "short_pressure", "finra_regsho_daily", 0, False,
                "short ratio z-score vs trailing base (source lag already applied)"),
    # ---- regime (index closes) -------------------------------------------
    GenomeField("regime_trend", "regime", "yahoo_spy_vix", 0, False, "SPY trend state",
                source_frame="regime"),
    GenomeField("regime_vol_state", "regime", "yahoo_spy_vix", 0, False, "VIX state",
                source_frame="regime"),
    GenomeField("regime_chop", "regime", "yahoo_spy_vix", 0, False, "chop flag",
                source_frame="regime"),
    # ---- options state (Cboe delayed chains, REAL_DELAYED watermark;
    #      intraday capture => public before the close, lag 0; coverage
    #      begins with the archive 2026-08-10, null before) ---------------
    GenomeField("iv_atm_near", "options_state", "cboe_delayed", 0, False,
                "near-expiry ATM vendor IV", source_frame="options"),
    GenomeField("iv_term_slope", "options_state", "cboe_delayed", 0, False,
                "far minus near ATM IV", source_frame="options"),
    GenomeField("iv_skew_asym", "options_state", "cboe_delayed", 0, False,
                "OTM put IV minus OTM call IV, near expiry", source_frame="options"),
    GenomeField("oi_conc_top_strike", "options_state", "cboe_delayed", 0, False,
                "share of chain OI at the most-loaded strike", source_frame="options"),
    # ---- outcome labels (NEVER features) ---------------------------------
    GenomeField("starts_5x_fwd", "outcome", "pathfinder", 0, False,
                "a >=5x forward path starts this session (label maturation: "
                "126 bars; usable only behind a purged cutoff)"),
)

FIELD_BY_NAME: dict[str, GenomeField] = {f.name: f for f in FIELDS}

# features_panel columns the genome intentionally does NOT carry:
#   symbol/date/valid  keys + eligibility, passed through unchanged
UNREGISTERED_PASSTHROUGH = ("symbol", "date", "valid")


def feature_fields() -> list[GenomeField]:
    return [f for f in FIELDS if f.group != "outcome"]


def outcome_fields() -> list[GenomeField]:
    return [f for f in FIELDS if f.group == "outcome"]


def knowability_table() -> list[dict]:
    """The audit surface: one row per field stating exactly when a value
    dated d becomes decidable. Rendered into genome metadata so every
    consumer (and every reviewer) sees the rules beside the data."""
    rows = []
    for f in FIELDS:
        if f.quarterly_published:
            rule = (f"value dated d is exposed from the last quarter published "
                    f">= {PUBLICATION_DELAY_DAYS} calendar days before the "
                    f"decision session, then lagged {f.lag_sessions} session(s)")
        elif f.lag_sessions:
            rule = f"value dated d exposed at session d + {f.lag_sessions}"
        else:
            rule = "knowable at session d close"
        if f.group == "outcome":
            rule = "LABEL — excluded from the feature surface; " + rule
        rows.append({"field": f.name, "group": f.group, "source": f.source,
                     "rule": rule})
    return rows
