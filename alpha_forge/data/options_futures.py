"""Options chains and futures continuous contracts — DESIGN mode.

No options/futures API keys are present, so these adapters emit the exact
queries they would run instead of data. Nothing here ever synthesizes option
prices from a model into a backtest: if synthetic pricing is ever used for
prototyping, every output row carries watermark SYNTHETIC and the gatekeeper
refuses SYNTHETIC inputs for graduation.
"""

from __future__ import annotations

import os


class OptionsChainAdapter:
    """Adapter interface for Polygon / ORATS / CBOE DataShop / databento."""

    PROVIDERS = {
        "polygon": "POLYGON_API_KEY",
        "orats": "ORATS_API_KEY",
        "cboe_datashop": "CBOE_DATASHOP_TOKEN",
        "databento": "DATABENTO_API_KEY",
    }

    @classmethod
    def mode(cls) -> str:
        for env in cls.PROVIDERS.values():
            if os.environ.get(env):
                return "LIVE"
        return "DESIGN"

    @classmethod
    def design_queries(cls, underlying: str, start: str, end: str) -> list[str]:
        return [
            f"polygon: GET /v3/snapshot/options/{underlying} + "
            f"/v2/aggs historical NBBO for chains {start}..{end} (needs POLYGON_API_KEY, options tier)",
            f"orats: GET /datav2/hist/strikes?ticker={underlying}&tradeDate={start}..{end} "
            "(smoothed IV surface + NBBO; needs ORATS_API_KEY)",
            f"databento: OPRA MBP-1 {underlying} {start}..{end} schema=mbp-1 (true NBBO snapshots)",
        ]


class FuturesContinuousAdapter:
    """ES/MES/NQ/MNQ/CL/GC continuous series.

    Roll methodology is recorded with the data, not assumed: default spec is
    volume-triggered roll (roll when next contract's volume exceeds front's),
    back-adjusted by ratio. DESIGN mode without a futures data key.
    """

    ROLL_SPEC = {
        "trigger": "volume_crossover",
        "adjustment": "ratio_back_adjusted",
        "note": "spec recorded per-dataset in the manifest at ingest time",
    }

    @classmethod
    def mode(cls) -> str:
        return "LIVE" if os.environ.get("DATABENTO_API_KEY") else "DESIGN"

    @classmethod
    def design_queries(cls) -> list[str]:
        return [
            "databento: GLBX.MDP3 ES/MES/NQ/MNQ/CL/GC definition + ohlcv-1d, "
            "2010-present, with per-contract volume for roll detection",
        ]
