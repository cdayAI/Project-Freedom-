import numpy as np
import pytest

from alpha_forge.research.calibration import apply_isotonic, pav_isotonic


def test_pav_recovers_monotone_signal():
    rng = np.random.default_rng(0)
    x = rng.random(2000)
    p_true = 0.2 + 0.6 * x  # monotone in x
    y = (rng.random(2000) < p_true).astype(float)
    bx, by = pav_isotonic(x, y)
    fitted = apply_isotonic(bx, by, np.array([0.1, 0.5, 0.9]))
    assert fitted[0] < fitted[1] < fitted[2]
    assert fitted[0] == pytest.approx(0.26, abs=0.08)
    assert fitted[2] == pytest.approx(0.74, abs=0.08)


def test_pav_output_is_nondecreasing():
    rng = np.random.default_rng(1)
    x = rng.random(500)
    y = (rng.random(500) < 0.5).astype(float)  # pure noise
    bx, by = pav_isotonic(x, y)
    assert np.all(np.diff(by) >= -1e-12)
    # noise should collapse to few blocks near the base rate
    assert abs(by.mean() - 0.5) < 0.1


def test_pav_perfect_separation():
    x = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    bx, by = pav_isotonic(x, y)
    out = apply_isotonic(bx, by, np.array([0.15, 0.85]))
    assert out[0] == 0.0 and out[1] == 1.0


def test_apply_clips_below_first_breakpoint():
    bx, by = np.array([0.5, 0.8]), np.array([0.3, 0.9])
    out = apply_isotonic(bx, by, np.array([0.1, 0.6, 0.95]))
    assert out[0] == pytest.approx(0.3)   # below range -> first block
    assert out[1] == pytest.approx(0.3)
    assert out[2] == pytest.approx(0.9)


def test_pav_antitonic_data_collapses_to_mean():
    # y strictly DECREASING in x: isotonic fit must pool everything
    x = np.linspace(0, 1, 100)
    y = 1.0 - x
    bx, by = pav_isotonic(x, y)
    assert by.size == 1
    assert by[0] == pytest.approx(0.5, abs=1e-9)
