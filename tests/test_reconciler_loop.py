import json

import numpy as np
import polars as pl
import pytest

import alpha_forge.research.reconciler as rec
from alpha_forge.ledger import Ledger
from alpha_forge.research.loop import generate_hypotheses, replacement_rate


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    preds = tmp_path / "predictions"
    store = tmp_path / "store"
    preds.mkdir()
    store.mkdir()
    monkeypatch.setattr(rec, "PREDICTIONS_DIR", preds)
    monkeypatch.setattr(rec, "CALIBRATION_PATH", store / "calibration.parquet")
    monkeypatch.setattr(rec, "GRADES_PATH", store / "grades.parquet")
    return preds, store


def _panel(symbol: str, n: int, daily_ret: float, start=pl.date(2025, 1, 1)) -> pl.DataFrame:
    close = 10.0 * np.cumprod(np.full(n, 1 + daily_ret))
    open_ = np.concatenate([[10.0], close[:-1]])
    return pl.DataFrame(
        {
            "symbol": [symbol] * n,
            "date": pl.date_range(start, start + pl.duration(days=n - 1), "1d", eager=True),
            "open": open_,
            "high": close * 1.01,
            "low": open_ * 0.99,
            "close": close,
            "volume": np.full(n, 1e6),
        }
    )


def _write_prediction(preds_dir, ledger, vintage: str, symbols: list[str]):
    doc = {
        "data_vintage": vintage,
        "predictions": [
            {"instrument": s, "confidence": {"value": 0.9}, "score": 1.0} for s in symbols
        ],
    }
    path = preds_dir / f"predictions_{vintage}.json"
    path.write_text(json.dumps(doc))
    import hashlib

    ledger.append("PREDICTION", {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "n": len(symbols)})
    return path


class TestReconciler:
    def test_grades_matured_horizons(self, sandbox, tmp_path):
        preds_dir, _ = sandbox
        ledger = Ledger(path=tmp_path / "l.jsonl")
        panel = _panel("UP", 200, 0.01)
        _write_prediction(preds_dir, ledger, "2025-01-10", ["UP"])
        grades = rec.grade_all(panel, ledger)
        horizons = {g["horizon"] for g in grades}
        assert {5, 21, 63, 126} == horizons
        g5 = next(g for g in grades if g["horizon"] == 5)
        assert g5["mean_fwd_return"] > 0
        # grading is idempotent
        assert rec.grade_all(panel, ledger) == []

    def test_tampered_file_halts(self, sandbox, tmp_path):
        preds_dir, _ = sandbox
        ledger = Ledger(path=tmp_path / "l.jsonl")
        panel = _panel("UP", 200, 0.01)
        path = _write_prediction(preds_dir, ledger, "2025-01-10", ["UP"])
        doc = json.loads(path.read_text())
        doc["predictions"][0]["confidence"]["value"] = 0.1  # rewrite after the fact
        path.write_text(json.dumps(doc))
        with pytest.raises(rec.TamperedPredictionsError):
            rec.grade_all(panel, ledger)

    def test_unledgered_file_ignored(self, sandbox, tmp_path):
        preds_dir, _ = sandbox
        ledger = Ledger(path=tmp_path / "l.jsonl")
        panel = _panel("UP", 200, 0.01)
        (preds_dir / "predictions_2025-01-10.json").write_text(
            json.dumps({"data_vintage": "2025-01-10", "predictions": [{"instrument": "UP", "confidence": {"value": 0.9}}]})
        )
        assert rec.grade_all(panel, ledger) == []


class TestLoop:
    def test_replacement_rate_counts_gate_kills(self, tmp_path):
        ledger = Ledger(path=tmp_path / "l.jsonl")
        reg = ledger.preregister("h", "u", {})
        ledger.append("GATE_REPORT", {"strategy_id": "a", "verdict": "KILL", "reg_id": reg})
        ledger.append("GRADUATION", {"strategy_id": "b"})
        rep = replacement_rate(ledger)
        assert rep["trailing_3m_killed"] == 1
        assert rep["trailing_3m_graduated"] == 1
        assert rep["trailing_3m_rate"] == 1.0
        assert rep["healthy"]

    def test_hypotheses_from_survivors_only(self, tmp_path):
        ledger = Ledger(path=tmp_path / "l.jsonl")
        conf = pl.DataFrame(
            {
                "n_multiple": [5, 5, 10],
                "feature": ["vol_20d_ann", "ret_252d", "vol_20d_ann"],
                "auc": [0.75, 0.4, 0.72],
                "p_value": [0.001, 0.2, 0.002],
                "verdict": ["IDENTIFIABLE", "NOT_SIGNIFICANT", "IDENTIFIABLE"],
            }
        )
        hyps = generate_hypotheses(conf, ledger)
        assert len(hyps) == 1  # deduped by feature; budget respected
        assert hyps[0]["feature"] == "vol_20d_ann"
        assert hyps[0]["rank_direction"] == "desc"  # AUC > 0.5 => hit-typical is high

    def test_no_survivors_no_hypotheses(self, tmp_path):
        ledger = Ledger(path=tmp_path / "l.jsonl")
        conf = pl.DataFrame(
            {"n_multiple": [5], "feature": ["price"], "auc": [0.51],
             "p_value": [0.4], "verdict": ["NOT_SIGNIFICANT"]}
        )
        assert generate_hypotheses(conf, ledger) == []
