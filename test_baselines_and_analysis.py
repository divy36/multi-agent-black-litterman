"""Tests for baselines and analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pytest

from black_litterman import BlackLittermanModel, BLInputs, LAMBDA, TAU, N, TICKERS


# ═══════════════════════════════════════════════════════════════════════
# Pure functions copied from analysis.py / baselines.py / baselines_part2.py
# ═══════════════════════════════════════════════════════════════════════

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLB"]
PERIODS_PER_YEAR = 12
RISK_FREE_ANNUAL = 0.0
N_BOOTSTRAP = 10_000
RNG_SEED = 42


from performance_metrics import bootstrap_sharpe_ci


def max_drawdown_and_duration(cum_wealth: pd.Series):
    """Identical across all three files."""
    running_max = cum_wealth.cummax()
    dd = cum_wealth / running_max - 1.0
    max_dd = dd.min()
    trough_date = dd.idxmin() if len(dd) else None
    below = dd < 0
    longest = cur = 0
    for v in below:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    return max_dd, longest, trough_date


def hhi(weights_row: pd.Series) -> float:
    """analysis.py only."""
    w = weights_row.values.astype(float)
    w = w[~np.isnan(w)]
    if w.sum() == 0:
        return float("nan")
    w = w / w.sum()
    return float(np.sum(w ** 2))


def performance_row(name: str, r: pd.Series) -> dict:
    """Identical in baselines.py and baselines_part2.py."""
    cw = (1 + r).cumprod()
    n_months = len(r)
    years = n_months / PERIODS_PER_YEAR
    total_return = cw.iloc[-1] - 1
    ann_return = cw.iloc[-1] ** (1 / years) - 1
    ann_vol = r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    sharpe, sharpe_lo, sharpe_hi = bootstrap_sharpe_ci(r.values)
    max_dd, dd_duration, trough = max_drawdown_and_duration(cw)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else float("nan")
    downside = r[r < 0]
    sortino = ((r.mean() * PERIODS_PER_YEAR) / (downside.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
               if len(downside) > 1 else float("nan"))
    return {
        "Config": name, "Total Return": total_return, "Ann. Return": ann_return,
        "Ann. Vol": ann_vol, "Sharpe": sharpe, "Sharpe CI Low": sharpe_lo,
        "Sharpe CI High": sharpe_hi, "Sortino": sortino, "Calmar": calmar,
        "Max DD": max_dd, "DD Duration (mo)": dd_duration,
    }, cw


def load_market_weights_v1(path: Path) -> Optional[pd.Series]:
    """baselines.py's copy — NO uppercase normalisation on the wide-format path."""
    df = pd.read_csv(path)
    wide_cols = [c for c in df.columns if any(s in c.upper() for s in SECTORS)]
    if wide_cols:
        row = df[wide_cols].astype(float).iloc[0]
        return row / row.sum()
    ticker_col = None
    for c in df.columns:
        vals = df[c].astype(str).str.upper()
        if vals.isin(SECTORS).sum() >= 5:
            ticker_col = c
            break
    weight_col = None
    for c in df.columns:
        lc = c.lower()
        if "weight" in lc or "cap" in lc or "pct" in lc:
            weight_col = c
            break
    if ticker_col is not None and weight_col is not None:
        s = df.set_index(df[ticker_col].astype(str).str.upper())[weight_col].astype(float)
        s = s.reindex(SECTORS).dropna()
        if len(s) >= 5:
            return s / s.sum()
    return None


def load_market_weights_v2(path: Path) -> Optional[pd.Series]:
    """baselines_part2.py's copy — DOES uppercase the wide-format row index."""
    df = pd.read_csv(path)
    wide_cols = [c for c in df.columns if any(s in c.upper() for s in SECTORS)]
    if wide_cols:
        row = df[wide_cols].astype(float).iloc[0]
        row.index = [c.upper() for c in row.index]
        return row / row.sum()
    ticker_col = None
    for c in df.columns:
        vals = df[c].astype(str).str.upper()
        if vals.isin(SECTORS).sum() >= 5:
            ticker_col = c
            break
    weight_col = None
    for c in df.columns:
        lc = c.lower()
        if "weight" in lc or "cap" in lc or "pct" in lc:
            weight_col = c
            break
    if ticker_col is not None and weight_col is not None:
        s = df.set_index(df[ticker_col].astype(str).str.upper())[weight_col].astype(float)
        s = s.reindex(SECTORS).dropna()
        if len(s) >= 5:
            return s / s.sum()
    return None


# baselines_part2.py's independent Black-Litterman reimplementation
def bl_posterior(Pi: np.ndarray, Sigma: np.ndarray, Q: np.ndarray, Omega: np.ndarray, tau: float):
    n = len(Pi)
    P = np.eye(n)
    tau_sigma_inv = np.linalg.inv(tau * Sigma)
    omega_inv = np.linalg.inv(Omega)
    M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    mu_star = M @ (tau_sigma_inv @ Pi + P.T @ omega_inv @ Q)
    Sigma_star = Sigma + M
    return mu_star, Sigma_star


def bl_weights(mu_star: np.ndarray, Sigma_star: np.ndarray, n: int = len(SECTORS)) -> np.ndarray:
    w_raw = np.linalg.solve(Sigma_star, mu_star)
    w_raw = np.maximum(w_raw, 0.0)
    raw_sum = w_raw.sum()
    if abs(raw_sum) < 1e-12:
        return np.ones(n) / n
    return w_raw / raw_sum


def apply_turnover_cap(w_target: np.ndarray, w_prev: np.ndarray, t_max: float = 0.30) -> np.ndarray:
    delta = w_target - w_prev
    l1 = np.sum(np.abs(delta))
    if l1 > t_max:
        delta = delta * (t_max / l1)
    w_final = w_prev + delta
    return w_final / np.sum(w_final)


def shrink_covariance(Sigma: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    n = Sigma.shape[0]
    avg_var = np.trace(Sigma) / n
    target = avg_var * np.eye(n)
    return (1 - shrinkage) * Sigma + shrinkage * target


def rolling_sigma(full_returns: pd.DataFrame, target_date, window: int = 36, min_periods: int = 12):
    history = full_returns.loc[full_returns.index < target_date]
    window_data = history.tail(window)
    if len(window_data) < min_periods:
        return None, len(window_data)
    Sigma = window_data.cov().values
    return shrink_covariance(Sigma), len(window_data)


def trailing_momentum(full_returns: pd.DataFrame, target_date, window: int = 12):
    history = full_returns.loc[full_returns.index < target_date]
    trailing = history.tail(window)
    if len(trailing) < window // 2:
        return None
    cum = (1 + trailing).prod() - 1
    monthly_equiv = (1 + cum) ** (1 / len(trailing)) - 1
    return monthly_equiv.values


# ═══════════════════════════════════════════════════════════════════════
# 1. bootstrap_sharpe_ci — shared by all three files
# ═══════════════════════════════════════════════════════════════════════
class TestBootstrapSharpeCI:

    def test_matches_hand_computed_annualised_sharpe(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0.01, 0.03, size=71)  # a realistic 71-month series
        point, lo, hi = bootstrap_sharpe_ci(r)
        expected = (r.mean() / r.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR)
        assert point == pytest.approx(expected, abs=1e-10)

    def test_ci_bounds_bracket_the_point_estimate_for_well_behaved_data(self):
        rng = np.random.default_rng(3)
        r = rng.normal(0.012, 0.025, size=71)
        point, lo, hi = bootstrap_sharpe_ci(r)
        assert lo <= point <= hi

    def test_fewer_than_3_valid_returns_yields_nan(self):
        point, lo, hi = bootstrap_sharpe_ci(np.array([0.01, np.nan]))
        assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)

    def test_nans_are_dropped_not_propagated(self):
        rng = np.random.default_rng(11)
        clean = rng.normal(0.01, 0.02, size=50)
        with_nans = np.concatenate([clean, [np.nan, np.nan, np.nan]])
        point_clean, _, _ = bootstrap_sharpe_ci(clean, seed=1)
        point_with_nans, _, _ = bootstrap_sharpe_ci(with_nans, seed=1)
        assert point_clean == pytest.approx(point_with_nans, abs=1e-10)

    def test_all_zero_returns_yields_nan_not_a_crash(self):
        """The `sd == 0` guard only catches the literal all-zero case exactly."""
        r = np.zeros(20)
        point, lo, hi = bootstrap_sharpe_ci(r)
        assert np.isnan(point)

    def test_near_constant_returns_currently_gives_absurd_sharpe_not_nan(self):
        r = np.full(20, 0.01)
        point, lo, hi = bootstrap_sharpe_ci(r)
        assert np.isnan(point) or abs(point) < 100, (
            f"Expected NaN (or a sane bounded value) for a near-constant return "
            f"series, got {point}"
        )

    def test_reproducible_with_fixed_seed(self):
        rng = np.random.default_rng(5)
        r = rng.normal(0.01, 0.02, size=40)
        result1 = bootstrap_sharpe_ci(r, seed=99)
        result2 = bootstrap_sharpe_ci(r, seed=99)
        assert result1 == result2


# ═══════════════════════════════════════════════════════════════════════
# 2. max_drawdown_and_duration — shared by all three files
# ═══════════════════════════════════════════════════════════════════════
class TestMaxDrawdownAndDuration:

    def test_matches_hand_computed_drawdown(self):
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        # wealth path: 1.0 -> 1.2 -> 1.0 -> 0.9 -> 0.95 -> 1.3
        cw = pd.Series([1.0, 1.2, 1.0, 0.9, 0.95, 1.3], index=idx)
        max_dd, duration, trough = max_drawdown_and_duration(cw)
        # peak before trough is 1.2; trough is 0.9 -> dd = 0.9/1.2 - 1 = -0.25
        assert max_dd == pytest.approx(-0.25, abs=1e-10)
        assert trough == idx[3]
        # below-peak run: months at index 2,3,4 (1.0, 0.9, 0.95 all < running max of 1.2) = 3 months
        assert duration == 3

    def test_monotonically_increasing_wealth_has_zero_drawdown(self):
        idx = pd.date_range("2020-01-31", periods=5, freq="ME")
        cw = pd.Series([1.0, 1.05, 1.10, 1.20, 1.35], index=idx)
        max_dd, duration, trough = max_drawdown_and_duration(cw)
        assert max_dd == pytest.approx(0.0, abs=1e-10)
        assert duration == 0

    def test_trough_date_matches_the_minimum(self):
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        cw = pd.Series([1.0, 0.8, 0.6, 0.9], index=idx)
        max_dd, duration, trough = max_drawdown_and_duration(cw)
        assert trough == idx[2]


# ═══════════════════════════════════════════════════════════════════════
# 3. hhi — analysis.py only
# ═══════════════════════════════════════════════════════════════════════
class TestHHI:

    def test_equal_weights_gives_one_over_n(self):
        w = pd.Series([0.1] * 10)
        assert hhi(w) == pytest.approx(1.0 / 10, abs=1e-10)

    def test_fully_concentrated_gives_one(self):
        w = pd.Series([1.0, 0.0, 0.0, 0.0])
        assert hhi(w) == pytest.approx(1.0, abs=1e-10)

    def test_unnormalised_weights_are_renormalised_first(self):
        w_unnorm = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
        w_norm = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2])
        assert hhi(w_unnorm) == pytest.approx(hhi(w_norm), abs=1e-10)

    def test_nans_are_dropped(self):
        w = pd.Series([0.5, 0.5, np.nan])
        assert hhi(w) == pytest.approx(0.5, abs=1e-10)

    def test_all_zero_row_returns_nan_not_a_crash(self):
        w = pd.Series([0.0, 0.0, 0.0])
        assert np.isnan(hhi(w))


# ═══════════════════════════════════════════════════════════════════════
# 4. performance_row — baselines.py and baselines_part2.py (identical)
# ═══════════════════════════════════════════════════════════════════════
class TestPerformanceRow:

    def test_total_return_and_ann_return_agree_for_exactly_one_year(self):
        """12 months = exactly 1 year, so ann_return must equal total_return exactly."""
        idx = pd.date_range("2020-01-31", periods=12, freq="ME")
        r = pd.Series([0.01] * 12, index=idx)
        row, cw = performance_row("Test", r)
        assert row["Ann. Return"] == pytest.approx(row["Total Return"], abs=1e-10)
        expected_total = 1.01 ** 12 - 1
        assert row["Total Return"] == pytest.approx(expected_total, abs=1e-10)

    def test_ann_vol_matches_hand_computed_std(self):
        rng = np.random.default_rng(2)
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        r = pd.Series(rng.normal(0.01, 0.02, size=24), index=idx)
        row, cw = performance_row("Test", r)
        expected_vol = r.std(ddof=1) * np.sqrt(12)
        assert row["Ann. Vol"] == pytest.approx(expected_vol, abs=1e-10)

    def test_sortino_uses_only_downside_returns(self):
        idx = pd.date_range("2020-01-31", periods=10, freq="ME")
        r = pd.Series([0.05, 0.05, 0.05, -0.02, -0.03, 0.05, 0.05, -0.01, 0.05, 0.05], index=idx)
        row, cw = performance_row("Test", r)
        downside = r[r < 0]
        expected_sortino = (r.mean() * 12) / (downside.std(ddof=1) * np.sqrt(12))
        assert row["Sortino"] == pytest.approx(expected_sortino, abs=1e-10)

    def test_calmar_matches_ann_return_over_abs_max_dd(self):
        rng = np.random.default_rng(9)
        idx = pd.date_range("2020-01-31", periods=36, freq="ME")
        r = pd.Series(rng.normal(0.008, 0.04, size=36), index=idx)
        row, cw = performance_row("Test", r)
        max_dd, _, _ = max_drawdown_and_duration(cw)
        if max_dd != 0:
            expected_calmar = row["Ann. Return"] / abs(max_dd)
            assert row["Calmar"] == pytest.approx(expected_calmar, abs=1e-10)

    def test_no_negative_returns_gives_nan_sortino(self):
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        r = pd.Series([0.01, 0.02, 0.01, 0.03, 0.01, 0.02], index=idx)
        row, cw = performance_row("Test", r)
        assert np.isnan(row["Sortino"])


# ═══════════════════════════════════════════════════════════════════════
# 5. load_market_weights — CONFIRMED DIVERGENCE between the two copies
# ═══════════════════════════════════════════════════════════════════════
class TestLoadMarketWeightsDivergence:
    """
    baselines.py's copy (v1) has NO uppercase normalisation on the wide
    format; baselines_part2.py's copy (v2) does. Both are tested with a
    lowercase-header CSV to demonstrate the divergence concretely rather
    than asserting it from a diff alone.
    """

    def test_v2_handles_lowercase_wide_headers_v1_does_not(self, tmp_path):
        csv_path = tmp_path / "weights_lowercase.csv"
        # lowercase sector headers -- a plausible real-world CSV export quirk
        pd.DataFrame([{"xlk": 0.30, "xlf": 0.20, "xle": 0.10, "xlv": 0.10,
                        "xli": 0.10, "xly": 0.05, "xlp": 0.05, "xlre": 0.05,
                        "xlu": 0.03, "xlb": 0.02}]).to_csv(csv_path, index=False)

        result_v1 = load_market_weights_v1(csv_path)
        result_v2 = load_market_weights_v2(csv_path)

        assert result_v2 is not None
        assert set(result_v2.index) == set(SECTORS)
        assert result_v2["XLK"] == pytest.approx(0.30, abs=1e-6)

        # v1's index still carries the ORIGINAL lowercase headers -- it
        # matched the wide_cols filter (case-insensitively) but never
        # uppercased the resulting Series index, so downstream code
        # expecting "XLK" etc. would not find it.
        assert "XLK" not in result_v1.index or set(result_v1.index) != set(result_v2.index)

    def test_both_agree_on_already_uppercase_headers(self, tmp_path):
        csv_path = tmp_path / "weights_uppercase.csv"
        pd.DataFrame([{s: 0.10 for s in SECTORS}]).to_csv(csv_path, index=False)
        result_v1 = load_market_weights_v1(csv_path)
        result_v2 = load_market_weights_v2(csv_path)
        assert result_v1 is not None and result_v2 is not None
        pd.testing.assert_series_equal(
            result_v1.sort_index(), result_v2.sort_index(), check_names=False
        )

    def test_long_format_parsing(self, tmp_path):
        csv_path = tmp_path / "weights_long.csv"
        pd.DataFrame({"ticker": SECTORS, "weight": [0.10] * 10}).to_csv(csv_path, index=False)
        result = load_market_weights_v2(csv_path)
        assert result is not None
        assert set(result.index) == set(SECTORS)
        assert result.sum() == pytest.approx(1.0, abs=1e-10)

    def test_unrecognised_format_returns_none_not_a_crash(self, tmp_path):
        csv_path = tmp_path / "weights_junk.csv"
        pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]}).to_csv(csv_path, index=False)
        assert load_market_weights_v2(csv_path) is None


# ═══════════════════════════════════════════════════════════════════════
# 6. baselines_part2.py's independent BL reimplementation, cross-checked
#    against the real black_litterman.py on identical inputs
# ═══════════════════════════════════════════════════════════════════════
class TestMechanicalBLAgreesWithRealModel:
    """
    baselines_part2.py reimplements the BL posterior from scratch rather
    than importing black_litterman.py. Since Momentum BL and Single-Agent
    BL's reported numbers depend on this second implementation, it's
    worth confirming directly that the two independent implementations
    actually agree on the same inputs, not just that each looks correct
    in isolation.
    """

    def test_bl_posterior_matches_real_model_diagonal_case(self):
        rng = np.random.default_rng(3)
        sigma2 = rng.uniform(0.0008, 0.006, size=N)
        Sigma = np.diag(sigma2)
        w_mkt = rng.dirichlet(np.ones(N))
        Q = rng.uniform(-0.03, 0.03, size=N)
        fixed_conf = 0.5

        Pi = LAMBDA * Sigma @ w_mkt
        Omega = np.diag((1 - fixed_conf) ** 2 * sigma2)  # baselines_part2's Omega (no disagreement term)

        mu_star_mechanical, Sigma_star_mechanical = bl_posterior(Pi, Sigma, Q, Omega, TAU)

        # Real model, same inputs (c_final=0.5 fixed, d_adjusted=0 to match
        # baselines_part2's Omega formula, which has no disagreement term)
        model = BlackLittermanModel()

        def fake_cov(*a, **kw):
            return Sigma

        def fake_wmkt(*a, **kw):
            return w_mkt

        model._rolling_covariance = fake_cov
        model._market_weights = fake_wmkt

        dummy_idx = pd.date_range("2016-01-31", "2019-01-31", freq="ME")
        dummy_returns = pd.DataFrame(np.zeros((len(dummy_idx), N)), index=dummy_idx, columns=TICKERS)
        dummy_wmkt_df = pd.DataFrame(np.tile(w_mkt, (len(dummy_idx), 1)), index=dummy_idx, columns=TICKERS)

        inputs = BLInputs(Q=Q, c_final=np.full(N, fixed_conf), d_adjusted=np.zeros(N))
        out = model.run(
            date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns,
            w_mkt_df=dummy_wmkt_df, w_previous=np.ones(N) / N, bl_inputs=inputs,
        )

        assert np.allclose(mu_star_mechanical, out.mu_star, atol=1e-8), (
            "baselines_part2.py's independent bl_posterior() disagrees with "
            "the real black_litterman.py model on identical inputs."
        )
        assert np.allclose(Sigma_star_mechanical, out.Sigma_star, atol=1e-8)

    def test_bl_weights_clips_negative_like_real_model(self):
        mu_star = np.array([-0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        Sigma_star = np.diag(np.full(N, 0.002))
        w = bl_weights(mu_star, Sigma_star)
        assert np.all(w >= 0)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# 7. apply_turnover_cap, shrink_covariance, rolling_sigma, trailing_momentum
# ═══════════════════════════════════════════════════════════════════════
class TestTurnoverCapMechanical:

    def test_cap_binds_and_result_is_exactly_at_the_cap(self):
        # Sums to exactly 1: 0.9 + 9*(0.1/9)
        w_target = np.array([0.9] + [0.1 / 9] * 9)
        w_prev = np.ones(N) / N
        w_final = apply_turnover_cap(w_target, w_prev, t_max=0.30)
        realised_l1 = np.sum(np.abs(w_final - w_prev))
        assert realised_l1 == pytest.approx(0.30, abs=1e-6)
        assert w_final.sum() == pytest.approx(1.0, abs=1e-9)

    def test_cap_not_binding_returns_target_unchanged(self):
        w_prev = np.ones(N) / N
        w_target = w_prev.copy()
        w_target[0] += 0.01
        w_target[1] -= 0.01
        w_final = apply_turnover_cap(w_target, w_prev, t_max=0.30)
        assert np.allclose(w_final, w_target, atol=1e-9)


class TestShrinkCovariance:

    def test_zero_shrinkage_returns_sigma_unchanged(self):
        rng = np.random.default_rng(1)
        raw = rng.standard_normal((40, 5))
        Sigma = np.cov(raw, rowvar=False)
        assert np.allclose(shrink_covariance(Sigma, shrinkage=0.0), Sigma)

    def test_full_shrinkage_gives_scaled_identity(self):
        rng = np.random.default_rng(1)
        raw = rng.standard_normal((40, 5))
        Sigma = np.cov(raw, rowvar=False)
        result = shrink_covariance(Sigma, shrinkage=1.0)
        avg_var = np.trace(Sigma) / 5
        assert np.allclose(result, avg_var * np.eye(5))

    def test_partial_shrinkage_matches_formula(self):
        rng = np.random.default_rng(1)
        raw = rng.standard_normal((40, 5))
        Sigma = np.cov(raw, rowvar=False)
        s = 0.10
        expected = (1 - s) * Sigma + s * (np.trace(Sigma) / 5) * np.eye(5)
        assert np.allclose(shrink_covariance(Sigma, shrinkage=s), expected)


class TestRollingSigmaAndMomentumLookAhead:

    def test_rolling_sigma_excludes_data_on_or_after_target_date(self):
        idx = pd.date_range("2016-01-31", "2019-06-30", freq="ME")
        rng = np.random.default_rng(4)
        returns = pd.DataFrame(rng.normal(0, 0.01, size=(len(idx), 3)), index=idx, columns=["A", "B", "C"])
        target = pd.Timestamp("2019-01-31")
        returns.loc[returns.index >= target, :] = 5.0  # absurd anomaly on/after target
        Sigma, n_hist = rolling_sigma(returns, target, window=36, min_periods=12)
        assert Sigma is not None
        assert np.all(np.diag(Sigma) < 1.0), "look-ahead bias: future anomaly leaked into Sigma"

    def test_rolling_sigma_returns_none_below_min_periods(self):
        idx = pd.date_range("2018-06-30", "2019-01-31", freq="ME")
        rng = np.random.default_rng(2)
        returns = pd.DataFrame(rng.normal(0, 0.01, size=(len(idx), 3)), index=idx, columns=["A", "B", "C"])
        Sigma, n_hist = rolling_sigma(returns, pd.Timestamp("2019-01-31"), window=36, min_periods=12)
        assert Sigma is None

    def test_trailing_momentum_exact_identity_for_constant_returns(self):
        """If every month has the same return r, the trailing monthly-equivalent
        momentum signal must equal r exactly (up to floating point)."""
        idx = pd.date_range("2018-01-31", "2019-06-30", freq="ME")
        r = 0.015
        returns = pd.DataFrame({"A": [r] * len(idx)}, index=idx)
        result = trailing_momentum(returns, pd.Timestamp("2019-06-30"), window=12)
        assert result[0] == pytest.approx(r, abs=1e-9)

    def test_trailing_momentum_excludes_data_on_or_after_target_date(self):
        idx = pd.date_range("2018-01-31", "2019-06-30", freq="ME")
        returns = pd.DataFrame({"A": [0.01] * len(idx)}, index=idx)
        target = pd.Timestamp("2019-01-31")
        returns.loc[returns.index >= target, "A"] = 5.0  # anomaly on/after target
        result = trailing_momentum(returns, target, window=12)
        assert result[0] < 1.0, "look-ahead bias: future anomaly leaked into momentum signal"

    def test_trailing_momentum_returns_none_if_insufficient_history(self):
        idx = pd.date_range("2019-01-31", "2019-03-31", freq="ME")
        returns = pd.DataFrame({"A": [0.01] * len(idx)}, index=idx)
        result = trailing_momentum(returns, pd.Timestamp("2019-03-31"), window=12)
        assert result is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
