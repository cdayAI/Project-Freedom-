"""Widen the research sample to WIDE_N names and (re)build the panel.

Resumable: per-symbol parquet pulled today is reused, so interruptions and
re-runs cost nothing. Rewrites the daily ingest marker so `make daily`
sees the widened panel as today's pull.
"""

import json
import sys
from datetime import date

from alpha_forge.config import RAW_DIR
from alpha_forge.data.equities import YahooDailyAdapter
from alpha_forge.data.manifest import record_dataset
from alpha_forge.data.regime import build_regime_series
from alpha_forge.data.universe import eligible_common_stocks, fetch_symbol_directory, research_sample

WIDE_N = 1500


def main() -> int:
    directory = fetch_symbol_directory()
    eligible = eligible_common_stocks(directory)
    sample = research_sample(eligible, WIDE_N)
    print(f"universe {len(eligible)} eligible; sample {len(sample)}", flush=True)

    adapter = YahooDailyAdapter(pause_s=0.20)
    summary = adapter.ingest(sample, resume=True)
    print(f"ingested {summary['ingested']}, quarantined {len(summary['quarantined'])}", flush=True)
    if summary["ingested"] < len(sample) * 0.5:
        print("INGEST ANOMALY — halting without touching marker", flush=True)
        return 1

    build_regime_series(adapter)
    record_dataset(
        dataset_id="equities_daily_yahoo",
        vendor=adapter.vendor,
        description=f"daily OHLCV, uniform random sample n={len(sample)} of "
        f"{len(eligible)} eligible current US common-stock listings + SPY/^VIX regime",
        known_biases=adapter.known_biases,
        validation_summary={
            "requested": summary["requested"],
            "ingested": summary["ingested"],
            "quarantined": len(summary["quarantined"]),
        },
        survivorship_free=adapter.survivorship_free,
    )
    marker = RAW_DIR / f"ingest_{date.today().isoformat()}.json"
    marker.write_text(json.dumps({
        "requested": summary["requested"],
        "ingested": summary["ingested"],
        "quarantined": [q["symbol"] for q in summary["quarantined"]],
        "universe_size": len(eligible),
        "sample_size": len(sample),
    }))
    print("marker + manifest updated", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
