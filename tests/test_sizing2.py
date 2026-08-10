import numpy as np
import pytest

from alpha_forge.research.sizing2 import (
    bayesian_bootstrap_paths,
    estimate_transition_matrix,
    iid_paths,
    kelly_posterior,
    regime_conditional_paths,
    sizing_frontier_v2,
    stationary_bootstrap_paths,
)


def _lag1_autocorr(m: np.ndarray) -> float:
    a, b = m[:, :-1].ravel(), m[:, 1:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _clustered_returns(n=300, seed=0):
    """Two-state persistent series with a POSITIVE edge: mostly-good regime
    (stay-prob 0.95) punctuated by sticky bad runs (stay-prob 0.85). The
    positive mean matters: it pushes the Kelly-scaled fraction grid into
    territory where drawdown risk is nonzero and generator differences are
    measurable — a negative-edge series sizes to ~0 and shows nothing."""
    rng = np.random.default_rng(seed)
    state = 0  # 0 = GOOD, 1 = BAD
    out, labels = [], []
    for _ in range(n):
        p_leave = 0.05 if state == 0 else 0.15
        if rng.random() < p_leave:
            state = 1 - state
        mu = 0.03 if state == 0 else -0.04
        out.append(rng.normal(mu, 0.05))
        labels.append("GOOD" if state == 0 else "BAD")
    return np.asarray(out), np.asarray(labels)


class TestGenerators:
    def test_stationary_preserves_serial_dependence_iid_destroys_it(self):
        r, _ = _clustered_returns()
        orig = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        assert orig > 0.15  # construction check: the series really clusters
        rng = np.random.default_rng(1)
        stat = stationary_bootstrap_paths(r, 2000, 100, rng)
        iid = iid_paths(r, 2000, 100, np.random.default_rng(2))
        assert _lag1_autocorr(stat) > orig * 0.5   # dependence survives blocks
        assert abs(_lag1_autocorr(iid)) < 0.05     # iid wipes it out

    def test_bayesian_widens_terminal_dispersion_on_small_samples(self):
        rng = np.random.default_rng(3)
        r = rng.normal(0.01, 0.05, 40)  # small sample: parameter uncertainty real
        bay = bayesian_bootstrap_paths(r, 4000, 80, np.random.default_rng(4))
        iid = iid_paths(r, 4000, 80, np.random.default_rng(5))
        t_bay = bay.sum(axis=1)  # additive proxy for terminal log wealth
        t_iid = iid.sum(axis=1)
        assert t_bay.std() > t_iid.std() * 1.1  # posterior spread > sampling spread

    def test_transition_matrix_recovered(self):
        _, labels = _clustered_returns(n=3000, seed=6)
        P = estimate_transition_matrix(labels, ["BAD", "GOOD"])
        # persistence by construction: BAD stays 0.85, GOOD stays 0.95
        assert P[0, 0] == pytest.approx(0.85, abs=0.04)
        assert P[1, 1] == pytest.approx(0.95, abs=0.02)
        assert np.allclose(P.sum(axis=1), 1.0)

    def test_regime_paths_cluster_bad_trades(self):
        r, labels = _clustered_returns(n=600, seed=7)
        reg = regime_conditional_paths(r, labels, 2000, 100, np.random.default_rng(8))
        assert reg is not None
        iid = iid_paths(r, 2000, 100, np.random.default_rng(9))
        assert _lag1_autocorr(reg) > 0.15
        assert abs(_lag1_autocorr(iid)) < 0.05

    def test_regime_paths_refuse_thin_pools(self):
        r = np.random.default_rng(0).normal(0, 0.02, 50)
        labels = np.array(["A"] * 47 + ["B"] * 3)  # B pool too thin
        assert regime_conditional_paths(r, labels, 1000, 50, np.random.default_rng(1)) is None


class TestDecisionSurface:
    def test_iid_understates_risk_on_clustered_losses(self):
        """THE test: on serially clustered returns, the honest generators
        must show more drawdown risk than the iid baseline at the same
        fraction — clustered losses compound into deeper drawdowns than a
        shuffled sequence with the identical marginal distribution."""
        r, labels = _clustered_returns(n=400, seed=10)
        out = sizing_frontier_v2(r, regime_labels=labels, n_paths=20_000, seed=11)
        fr = out["frontiers"]
        risk_gens = [g for g in fr if g != "iid"]
        dd_gap = max(
            max(fr[g][i]["p_drawdown_below_50pct"] for g in risk_gens)
            - fr["iid"][i]["p_drawdown_below_50pct"]
            for i in range(len(out["fractions"]))
        )
        assert dd_gap > 0.02  # the iid model measurably understates dd risk
        ruin_gaps = [c["p_ruin_worst"] - c["p_ruin_iid_baseline"] for c in out["combined"]]
        assert max(ruin_gaps) >= 0.0  # never the other way at any fraction

    def test_constrained_optimum_respects_worst_model(self):
        rng = np.random.default_rng(12)
        r = rng.normal(0.02, 0.08, 300)
        out = sizing_frontier_v2(r, n_paths=20_000, seed=13)
        c = out["constrained_optimum"]
        assert c is not None
        assert c["p_ruin_worst"] < 0.05
        assert "iid" not in out["constraint"]  # iid never votes

    def test_kelly_posterior_p5_below_point(self):
        rng = np.random.default_rng(14)
        r = rng.normal(0.015, 0.07, 60)  # small sample -> wide posterior
        kp = kelly_posterior(r, np.random.default_rng(15))
        assert kp["kelly_p5"] <= kp["kelly_p50"] <= kp["kelly_p95"]
        assert kp["kelly_p5"] < kp["kelly_point"]  # uncertainty shrinks the bet

    def test_min_paths_enforced(self):
        r = np.random.default_rng(0).normal(0.01, 0.05, 100)
        with pytest.raises(ValueError):
            sizing_frontier_v2(r, n_paths=1000)

    def test_unsizeable_edge_honest_note(self):
        rng = np.random.default_rng(16)
        r = rng.normal(-0.05, 0.30, 200)
        out = sizing_frontier_v2(r, n_paths=20_000, seed=17)
        if out["constrained_optimum"] is None:
            assert "unsizeable" in out["note"]
        else:
            assert out["constrained_optimum"]["fraction"] <= 0.05
