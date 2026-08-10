import polars as pl

from alpha_forge.ledger import Ledger
from alpha_forge.research.confluence import should_retest


def _panel(n_days: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": pl.date_range(
                pl.date(2026, 8, 1),
                pl.date(2026, 8, 1) + pl.duration(days=n_days - 1),
                "1d",
                eager=True,
            )
        }
    )


def _ledger_with_family(tmp_path, vintage: str) -> Ledger:
    ledger = Ledger(path=tmp_path / "l.jsonl")
    reg = ledger.preregister("confluence family", "u", {})
    ledger.record_result(
        reg, {"family": "confluence_identifiability", "data_vintage": vintage}
    )
    return ledger


def test_first_family_always_runs(tmp_path):
    ledger = Ledger(path=tmp_path / "l.jsonl")
    assert should_retest(ledger, _panel(30), "2026-08-30")


def test_same_vintage_never_reruns(tmp_path):
    ledger = _ledger_with_family(tmp_path, "2026-08-30")
    assert not should_retest(ledger, _panel(30), "2026-08-30")


def test_too_few_new_days_serves_cache(tmp_path):
    # family at Aug 26; panel through Aug 30 has 4 newer days -> no re-test
    ledger = _ledger_with_family(tmp_path, "2026-08-26")
    assert not should_retest(ledger, _panel(30), "2026-08-30")


def test_enough_new_days_retests(tmp_path):
    # family at Aug 24; panel through Aug 30 has 6 newer days -> re-test
    ledger = _ledger_with_family(tmp_path, "2026-08-24")
    assert should_retest(ledger, _panel(30), "2026-08-30")
