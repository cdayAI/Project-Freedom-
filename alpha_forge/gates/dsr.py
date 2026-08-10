"""Deflated Sharpe Ratio.

Reference: Bailey, D. H. & Lopez de Prado, M. (2014), "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
Journal of Portfolio Management 40(5). See SOURCES.md.

This module is the reference implementation. Any port for speed must
reproduce these outputs to 1e-9 on the fixed cases in tests/test_dsr.py.

All Sharpe ratios here are per-period (not annualized); T is the number of
return observations. The trial count N and the variance of trial Sharpes come
from the ledger — never from a guess.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray) -> float:
    """Per-period Sharpe, ddof=1. Excess returns are the caller's job."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        raise ValueError("need at least 2 returns")
    sd = r.std(ddof=1)
    if sd == 0:
        raise ValueError("zero-variance returns")
    return float(r.mean() / sd)


def probabilistic_sharpe_ratio(
    sr_observed: float,
    sr_benchmark: float,
    n_obs: int,
    skew: float,
    kurt: float,
) -> float:
    """PSR = P(true SR > sr_benchmark | observed SR, non-normality).

    kurt is Pearson kurtosis (normal = 3), per the paper's Eq. for sigma(SR):
      sigma^2(SR) = (1 - skew*SR + (kurt-1)/4 * SR^2) / (T - 1)
    """
    if n_obs < 3:
        raise ValueError("need at least 3 observations")
    var_sr = (1.0 - skew * sr_observed + (kurt - 1.0) / 4.0 * sr_observed**2) / (n_obs - 1)
    if var_sr <= 0:
        # Extreme skew/kurt combinations can push the plug-in variance negative;
        # that means the normal approximation has broken down, not that we are
        # infinitely confident. Fail loudly.
        raise ValueError(f"non-positive SR variance ({var_sr:.3g}); PSR approximation invalid")
    z = (sr_observed - sr_benchmark) / np.sqrt(var_sr)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_trial_sr: float) -> float:
    """E[max SR] across n_trials of zero-true-SR strategies whose SR estimates
    have cross-trial variance var_trial_sr (Bailey & LdP 2014, Eq. via
    Mertens/extreme-value approximation):

      SR0 = sqrt(V) * [ (1-gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N e)) ]
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if var_trial_sr < 0:
        raise ValueError("variance must be >= 0")
    if n_trials == 1 or var_trial_sr == 0:
        return 0.0
    g = EULER_MASCHERONI
    q1 = norm.ppf(1.0 - 1.0 / n_trials)
    q2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_trial_sr) * ((1.0 - g) * q1 + g * q2))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    var_trial_sr: float,
) -> dict:
    """DSR = PSR evaluated at the expected-max-SR benchmark.

    n_trials: TRUE cumulative trial count from the ledger.
    var_trial_sr: variance of per-period SR estimates across ledger trials.

    Returns dict with sr, sr0 (deflation benchmark), dsr_probability, and the
    moments used, so gate reports can show their work.
    """
    r = np.asarray(returns, dtype=float)
    sr = sharpe_ratio(r)
    n = r.size
    mean = r.mean()
    sd = r.std(ddof=1)
    skew = float(((r - mean) ** 3).mean() / sd**3)
    kurt = float(((r - mean) ** 4).mean() / sd**4)  # Pearson (normal = 3)
    sr0 = expected_max_sharpe(n_trials, var_trial_sr)
    p = probabilistic_sharpe_ratio(sr, sr0, n, skew, kurt)
    return {
        "sr_per_period": sr,
        "sr0_benchmark": sr0,
        "dsr_probability": p,
        "n_obs": int(n),
        "n_trials": int(n_trials),
        "var_trial_sr": float(var_trial_sr),
        "skew": skew,
        "kurt_pearson": kurt,
    }
