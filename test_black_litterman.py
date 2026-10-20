"""Tests for black litterman."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from black_litterman import (
    ALPHA,
    LAMBDA,
    N,
    T_MAX,
    TAU,
    TICKERS,
    BlackLittermanModel,
    BLInputs,
)


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def dummy_returns_df() -> pd.DataFrame:
    """
    A returns_df that satisfies _validate_inputs' column check.
    Content is irrelevant for tests that monkeypatch _rolling_covariance,
    since the real method is never called in those tests.
    """
    idx = pd.date_range("2016-01-31", "2019-01-31", freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 0.01, size=(len(idx), N)),
                         index=idx, columns=TICKERS)


@pytest.fixture
def dummy_w_mkt_df() -> pd.DataFrame:
    idx = pd.date_range("2016-01-31", "2019-01-31", freq="ME")
    w = np.ones(N) / N
    return pd.DataFrame(np.tile(w, (len(idx), 1)), index=idx, columns=TICKERS)


@pytest.fixture
def model() -> BlackLittermanModel:
    return BlackLittermanModel()


def diagonal_sigma(variances: np.ndarray) -> np.ndarray:
    """Build an (N,N) diagonal covariance matrix from an (N,) variance vector."""
    return np.diag(variances)


def analytic_diagonal_posterior(
    Pi: np.ndarray, Q: np.ndarray, sigma2: np.ndarray,
    c_final: np.ndarray, d_adjusted: np.ndarray,
    tau: float = TAU, alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Independent, from-scratch reference implementation of the BL posterior
    for the diagonal-Sigma / diagonal-Omega / P=I special case.

    Deliberately written as plain per-asset scalar arithmetic, not vector
    matrix algebra copied from black_litterman.py, so this is a genuine
    second, independent derivation to compare against.

    Returns (mu_star, m_diag) where m_diag is the diagonal of M
    (equivalently, the diagonal of Sigma_star - Sigma).
    """
    mu_star = np.zeros(N)
    m_diag = np.zeros(N)
    for i in range(N):
        omega_i = max((1 - c_final[i]) ** 2 * sigma2[i] * (1 + alpha * d_adjusted[i]), 1e-12)
        p0 = 1.0 / (tau * sigma2[i])
        p1 = 1.0 / omega_i
        p = p0 + p1
        m_diag[i] = 1.0 / p
        mu_star[i] = (p0 * Pi[i] + p1 * Q[i]) / p
    return mu_star, m_diag


# --------------------------------------------------------------------------- #
# 1. BLInputs validation
# --------------------------------------------------------------------------- #
class TestBLInputsValidation:

    def test_accepts_valid_inputs(self):
        BLInputs(
            Q=np.zeros(N), c_final=np.full(N, 0.5), d_adjusted=np.zeros(N),
        )  # must not raise

    def test_rejects_wrong_shape_Q(self):
        with pytest.raises(ValueError, match="Q"):
            BLInputs(Q=np.zeros(N - 1), c_final=np.full(N, 0.5), d_adjusted=np.zeros(N))

    def test_rejects_c_final_out_of_range_high(self):
        c = np.full(N, 0.5)
        c[0] = 1.5
        with pytest.raises(ValueError, match="c_final"):
            BLInputs(Q=np.zeros(N), c_final=c, d_adjusted=np.zeros(N))

    def test_rejects_c_final_out_of_range_low(self):
        c = np.full(N, 0.5)
        c[0] = -0.01
        with pytest.raises(ValueError, match="c_final"):
            BLInputs(Q=np.zeros(N), c_final=c, d_adjusted=np.zeros(N))

    def test_rejects_d_adjusted_out_of_range(self):
        d = np.zeros(N)
        d[0] = 1.01
        with pytest.raises(ValueError, match="d_adjusted"):
            BLInputs(Q=np.zeros(N), c_final=np.full(N, 0.5), d_adjusted=d)

    def test_accepts_boundary_values(self):
        """c_final and d_adjusted exactly 0 or 1 are valid, not off-by-one errors."""
        BLInputs(Q=np.zeros(N), c_final=np.zeros(N), d_adjusted=np.zeros(N))
        BLInputs(Q=np.zeros(N), c_final=np.ones(N), d_adjusted=np.ones(N))


# --------------------------------------------------------------------------- #
# 2. Worked example — posterior formula verified against an independent
#    closed-form reference, across a range of confidence/disagreement values
# --------------------------------------------------------------------------- #
class TestWorkedExamplePosterior:

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_mu_star_matches_analytic_precision_weighted_average(
        self, model, dummy_returns_df, dummy_w_mkt_df, monkeypatch, seed
    ):
        rng = np.random.default_rng(seed)

        # Hand-chosen diagonal Sigma (annualised-looking monthly variances,
        # deliberately varied per sector so this isn't a degenerate all-equal
        # test case).
        sigma2 = rng.uniform(0.0008, 0.006, size=N)   # monthly variance range
        Sigma = diagonal_sigma(sigma2)
        w_mkt = rng.dirichlet(np.ones(N))              # random valid weights, sum to 1

        Q = rng.uniform(-0.03, 0.03, size=N)
        c_final = rng.uniform(0.1, 0.9, size=N)
        d_adjusted = rng.uniform(0.0, 0.8, size=N)

        monkeypatch.setattr(model, "_rolling_covariance", lambda *a, **kw: Sigma)
        monkeypatch.setattr(model, "_market_weights", lambda *a, **kw: w_mkt)

        inputs = BLInputs(Q=Q, c_final=c_final, d_adjusted=d_adjusted)
        out = model.run(
            date=pd.Timestamp("2019-01-31"),
            returns_df=dummy_returns_df,
            w_mkt_df=dummy_w_mkt_df,
            w_previous=np.ones(N) / N,
            bl_inputs=inputs,
        )

        # Pi should match lambda * Sigma @ w_mkt exactly (Sigma diagonal ->
        # Pi_i = lambda * sigma2_i * w_mkt_i, a trivial elementwise check too)
        expected_Pi = LAMBDA * Sigma @ w_mkt
        assert np.allclose(out.Pi, expected_Pi, atol=1e-10)

        expected_mu_star, expected_m_diag = analytic_diagonal_posterior(
            expected_Pi, Q, sigma2, c_final, d_adjusted
        )

        assert np.allclose(out.mu_star, expected_mu_star, atol=1e-8), (
            f"mu_star does not match independent Bayesian conjugate-update "
            f"reference.\n  model:    {out.mu_star}\n  expected: {expected_mu_star}"
        )

        # Sigma_star = Sigma + M; M should be diagonal with entries m_diag,
        # and off-diagonal entries should be exactly 0 (Sigma diagonal + M
        # diagonal in this special case).
        M_actual = out.Sigma_star - Sigma
        assert np.allclose(np.diag(M_actual), expected_m_diag, atol=1e-8)
        off_diag_mask = ~np.eye(N, dtype=bool)
        assert np.allclose(M_actual[off_diag_mask], 0.0, atol=1e-8)

    def test_full_confidence_pulls_mu_star_to_the_view(
        self, model, dummy_returns_df, dummy_w_mkt_df, monkeypatch
    ):
        """
        c_final = 1, d_adjusted = 0 -> Omega floored at 1e-12 -> view precision
        dominates prior precision -> mu_star should converge to Q, which is
        the qualitative claim in the research project ("higher confidence pulls
        the posterior toward the agent view") stated as an exact numeric
        result rather than an assertion.
        """
        sigma2 = np.full(N, 0.002)
        Sigma = diagonal_sigma(sigma2)
        w_mkt = np.ones(N) / N
        monkeypatch.setattr(model, "_rolling_covariance", lambda *a, **kw: Sigma)
        monkeypatch.setattr(model, "_market_weights", lambda *a, **kw: w_mkt)

        Q = np.linspace(-0.02, 0.02, N)
        inputs = BLInputs(Q=Q, c_final=np.ones(N), d_adjusted=np.zeros(N))
        out = model.run(
            date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns_df,
            w_mkt_df=dummy_w_mkt_df, w_previous=np.ones(N) / N, bl_inputs=inputs,
        )
        assert np.allclose(out.mu_star, Q, atol=1e-6)

    def test_zero_confidence_still_uses_precision_weighted_average(
        self, model, dummy_returns_df, dummy_w_mkt_df, monkeypatch
    ):
        """
        c_final = 0 does NOT mean Omega -> infinity (a common misreading of
        the formula) — it means Omega = sigma^2 * (1 + alpha*d), which is
        finite. This test guards against ever "fixing" this into an
        infinite-Omega interpretation, since that would silently change
        every zero-confidence month's posterior.
        """
        sigma2 = np.full(N, 0.002)
        Sigma = diagonal_sigma(sigma2)
        w_mkt = np.ones(N) / N
        monkeypatch.setattr(model, "_rolling_covariance", lambda *a, **kw: Sigma)
        monkeypatch.setattr(model, "_market_weights", lambda *a, **kw: w_mkt)

        Q = np.full(N, 0.05)   # a large view value
        inputs = BLInputs(Q=Q, c_final=np.zeros(N), d_adjusted=np.zeros(N))
        out = model.run(
            date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns_df,
            w_mkt_df=dummy_w_mkt_df, w_previous=np.ones(N) / N, bl_inputs=inputs,
        )
        expected_Pi = LAMBDA * Sigma @ w_mkt
        expected_mu_star, _ = analytic_diagonal_posterior(
            expected_Pi, Q, sigma2, np.zeros(N), np.zeros(N)
        )
        assert np.allclose(out.mu_star, expected_mu_star, atol=1e-8)
        # And explicitly: mu_star must NOT equal Pi (which is what it would
        # be if Omega had gone to infinity and the view were fully ignored).
        assert not np.allclose(out.mu_star, expected_Pi, atol=1e-4)


# --------------------------------------------------------------------------- #
# 3. Prior-only path (bl_inputs=None)
# --------------------------------------------------------------------------- #
class TestPriorOnlyPath:

    def test_mu_star_equals_pi_exactly(
        self, model, dummy_returns_df, dummy_w_mkt_df, monkeypatch
    ):
        sigma2 = np.full(N, 0.0015)
        Sigma = diagonal_sigma(sigma2)
        w_mkt = np.ones(N) / N
        monkeypatch.setattr(model, "_rolling_covariance", lambda *a, **kw: Sigma)
        monkeypatch.setattr(model, "_market_weights", lambda *a, **kw: w_mkt)

        out = model.run(
            date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns_df,
            w_mkt_df=dummy_w_mkt_df, w_previous=np.ones(N) / N, bl_inputs=None,
        )
        assert out.used_prior_only is True
        expected_Pi = LAMBDA * Sigma @ w_mkt
        assert np.array_equal(out.mu_star, expected_Pi)

    def test_sigma_star_equals_sigma_plus_tau_sigma(
        self, model, dummy_returns_df, dummy_w_mkt_df, monkeypatch
    ):
        sigma2 = np.full(N, 0.0015)
        Sigma = diagonal_sigma(sigma2)
        w_mkt = np.ones(N) / N
        monkeypatch.setattr(model, "_rolling_covariance", lambda *a, **kw: Sigma)
        monkeypatch.setattr(model, "_market_weights", lambda *a, **kw: w_mkt)

        out = model.run(
            date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns_df,
            w_mkt_df=dummy_w_mkt_df, w_previous=np.ones(N) / N, bl_inputs=None,
        )
        expected = Sigma + TAU * Sigma
        assert np.allclose(out.Sigma_star, expected, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4. Long-only clipping (a real, code-level deviation from the textbook BL
#    formula: raw negative weights are clipped to zero before normalising)
# --------------------------------------------------------------------------- #
class TestLongOnlyConstraint:

    def test_negative_raw_weight_is_clipped_to_zero(
        self, model, dummy_returns_df, dummy_w_mkt_df, monkeypatch
    ):
        # Strongly negative, highly confident view on sector 0, positive
        # elsewhere -> unconstrained mean-variance weights would go short
        # on sector 0. Confirm the production long-only clip catches this.
        sigma2 = np.full(N, 0.002)
        Sigma = diagonal_sigma(sigma2)
        w_mkt = np.ones(N) / N
        monkeypatch.setattr(model, "_rolling_covariance", lambda *a, **kw: Sigma)
        monkeypatch.setattr(model, "_market_weights", lambda *a, **kw: w_mkt)

        Q = np.full(N, 0.01)
        Q[0] = -0.15
        c_final = np.full(N, 0.5)
        c_final[0] = 0.95
        inputs = BLInputs(Q=Q, c_final=c_final, d_adjusted=np.zeros(N))
        out = model.run(
            date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns_df,
            w_mkt_df=dummy_w_mkt_df, w_previous=np.ones(N) / N, bl_inputs=inputs,
        )
        assert np.all(out.w_final >= -1e-9)
        assert np.all(out.w_uncapped >= -1e-9)
        assert out.w_final[0] == pytest.approx(0.0, abs=1e-9) or out.w_uncapped[0] == pytest.approx(0.0, abs=1e-9)
        assert out.w_final.sum() == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 5. Turnover cap
# --------------------------------------------------------------------------- #
class TestTurnoverCap:

    def test_cap_binds_and_turnover_does_not_exceed_t_max(
        self, model, dummy_returns_df, dummy_w_mkt_df, monkeypatch
    ):
        sigma2 = np.full(N, 0.002)
        Sigma = diagonal_sigma(sigma2)
        w_mkt = np.ones(N) / N
        monkeypatch.setattr(model, "_rolling_covariance", lambda *a, **kw: Sigma)
        monkeypatch.setattr(model, "_market_weights", lambda *a, **kw: w_mkt)

        # Extreme, highly confident, one-sided views -> large target shift
        Q = np.full(N, -0.03)
        Q[0] = 0.05
        c_final = np.full(N, 0.9)
        inputs = BLInputs(Q=Q, c_final=c_final, d_adjusted=np.zeros(N))

        w_previous = np.ones(N) / N
        out = model.run(
            date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns_df,
            w_mkt_df=dummy_w_mkt_df, w_previous=w_previous, bl_inputs=inputs,
        )
        assert bool(out.cap_applied) is True
        assert out.turnover <= T_MAX + 1e-6
        assert out.w_final.sum() == pytest.approx(1.0, abs=1e-9)

    def test_cap_not_applied_when_shift_is_small(
        self, model, dummy_returns_df, dummy_w_mkt_df, monkeypatch
    ):
        sigma2 = np.full(N, 0.002)
        Sigma = diagonal_sigma(sigma2)
        w_mkt = np.ones(N) / N
        monkeypatch.setattr(model, "_rolling_covariance", lambda *a, **kw: Sigma)
        monkeypatch.setattr(model, "_market_weights", lambda *a, **kw: w_mkt)

        # Start already at the equilibrium weights with no agent views ->
        # target should equal (or be very close to) w_previous -> no cap.
        w_previous = w_mkt.copy()
        out = model.run(
            date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns_df,
            w_mkt_df=dummy_w_mkt_df, w_previous=w_previous, bl_inputs=None,
        )
        assert bool(out.cap_applied) is False
        assert np.allclose(out.w_final, out.w_uncapped)


# --------------------------------------------------------------------------- #
# 6. Look-ahead bias guards inside the model itself
# --------------------------------------------------------------------------- #
class TestLookAheadGuards:

    def test_rolling_covariance_excludes_data_on_or_after_date(self, model):
        """
        Sigma for month `date` must be estimated ONLY from returns strictly
        before `date`. This plants a huge, easily-detected variance spike in
        the months >= date and confirms it does NOT leak into Sigma.
        """
        idx = pd.date_range("2016-01-31", "2019-06-30", freq="ME")
        rng = np.random.default_rng(1)
        data = rng.normal(0, 0.005, size=(len(idx), N))
        returns_df = pd.DataFrame(data, index=idx, columns=TICKERS)

        target_date = pd.Timestamp("2019-01-31")
        # Inject an obvious anomaly into data on/after target_date only
        returns_df.loc[returns_df.index >= target_date, :] = 5.0  # absurd 500% return

        Sigma = model._rolling_covariance(target_date, returns_df)
        # If the anomaly had leaked in, variances would be enormous (>> 1).
        assert np.all(np.diag(Sigma) < 1.0), (
            "Sigma appears to include data on/after the rebalance date — "
            "look-ahead bias in _rolling_covariance."
        )

    def test_rolling_covariance_raises_on_insufficient_history(self, model):
        idx = pd.date_range("2018-06-30", "2019-01-31", freq="ME")  # only 8 months
        rng = np.random.default_rng(2)
        returns_df = pd.DataFrame(rng.normal(0, 0.01, size=(len(idx), N)),
                                   index=idx, columns=TICKERS)
        with pytest.raises(ValueError, match="need 36"):
            model._rolling_covariance(pd.Timestamp("2019-01-31"), returns_df)

    def test_market_weights_uses_lag_one_not_current_month(self, model):
        """
        w_mkt for month `date` must come from the row dated strictly before
        `date` (brief item 9.3). A weight vector dated exactly on `date`
        must NOT be used even though the label matches.
        """
        idx = pd.date_range("2018-11-30", "2019-02-28", freq="ME")
        w_mkt_df = pd.DataFrame(index=idx, columns=TICKERS, dtype=float)
        # Give every month a distinct, recognisable weight vector
        for i, d in enumerate(idx):
            w_mkt_df.loc[d] = np.eye(N)[i % N] * 0.5 + 0.05  # distinct per row

        target_date = pd.Timestamp("2019-01-31")
        w = model._market_weights(target_date, w_mkt_df)
        expected_row = w_mkt_df.loc[w_mkt_df.index < target_date].iloc[-1][TICKERS].values
        assert np.allclose(w, expected_row / expected_row.sum())
        # And explicitly NOT the row dated exactly on target_date, if present
        if target_date in w_mkt_df.index:
            same_month_row = w_mkt_df.loc[target_date][TICKERS].values
            assert not np.allclose(w, same_month_row / same_month_row.sum())


# --------------------------------------------------------------------------- #
# 7. Input validation on the public entry point
# --------------------------------------------------------------------------- #
class TestValidateInputs:

    def test_raises_on_missing_ticker_column(self, model, dummy_w_mkt_df):
        idx = pd.date_range("2016-01-31", "2019-01-31", freq="ME")
        bad_returns = pd.DataFrame(
            np.zeros((len(idx), N - 1)), index=idx, columns=TICKERS[:-1]
        )
        with pytest.raises(ValueError, match="missing column"):
            model.run(
                date=pd.Timestamp("2019-01-31"), returns_df=bad_returns,
                w_mkt_df=dummy_w_mkt_df, w_previous=np.ones(N) / N,
            )

    def test_raises_on_wrong_w_previous_shape(self, model, dummy_returns_df, dummy_w_mkt_df):
        with pytest.raises(ValueError, match="shape"):
            model.run(
                date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns_df,
                w_mkt_df=dummy_w_mkt_df, w_previous=np.ones(N - 1),
            )

    def test_raises_on_w_previous_not_summing_to_one(self, model, dummy_returns_df, dummy_w_mkt_df):
        with pytest.raises(ValueError, match="sum to 1"):
            model.run(
                date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns_df,
                w_mkt_df=dummy_w_mkt_df, w_previous=np.full(N, 0.5),
            )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
