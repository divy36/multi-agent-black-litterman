"""Tests for guardrail ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from black_litterman import BlackLittermanModel, BLInputs, LAMBDA, TAU, N, TICKERS


SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLB"]

ALPHA = 2.0
T_MAX = 0.30
T_MAX_DYNAMIC = 0.15
VIX_DYNAMIC_THRESHOLD = 25.0
CEILING_BASE_NORMAL = 0.75
CEILING_BASE_DYNAMIC = 0.50
CEILING_DECAY = 0.02
CEILING_VIX_REF = 20.0
STABILITY_DECAY = 0.3
STABILITY_REVERSAL_MULT = 0.5
C_CLIP_MAX = 0.999


# ═══════════════════════════════════════════════════════════════════════
# Pure functions copied verbatim from guardrail_ablation.py
# ═══════════════════════════════════════════════════════════════════════

def bl_posterior_diag(Pi, Sigma, Q, Omega_diag, tau):
    Omega_diag = np.maximum(Omega_diag, 1e-12)
    tau_Sigma = tau * Sigma
    inv_Omega_diag = 1.0 / Omega_diag

    PtOinvP = np.diag(inv_Omega_diag)
    PtOinvQ = inv_Omega_diag * Q

    tau_Sigma_inv = np.linalg.inv(tau_Sigma)
    tau_Sig_inv_Pi = np.linalg.solve(tau_Sigma, Pi)

    A = tau_Sigma_inv + PtOinvP
    b = tau_Sig_inv_Pi + PtOinvQ
    M = np.linalg.inv(A)
    mu_star = M @ b

    Sigma_star = Sigma + M
    return mu_star, Sigma_star


def bl_weights_long_only(mu_star, Sigma_star, n: int) -> np.ndarray:
    Sigma_star_inv = np.linalg.inv(Sigma_star)
    w_raw = Sigma_star_inv @ mu_star
    w_raw = np.maximum(w_raw, 0.0)
    raw_sum = w_raw.sum()
    if abs(raw_sum) < 1e-12:
        return np.ones(n) / n
    return w_raw / raw_sum


def apply_turnover_cap_exact(w_target, w_prev, t_max: float):
    delta_w = w_target - w_prev
    l1_change = np.abs(delta_w).sum()
    cap_applied = l1_change > t_max
    if cap_applied:
        delta_w = delta_w * (t_max / l1_change)
        w_capped = w_prev + delta_w
        capped_sum = w_capped.sum()
        w_final = w_capped / capped_sum if abs(capped_sum) > 1e-12 \
            else np.ones(len(w_prev)) / len(w_prev)
    else:
        w_final = w_target
    return w_final, cap_applied


# The six configs, copied verbatim from guardrail_ablation.py
CONFIGS = [
    {"name": "1. No Guardrails",      "ceiling": False, "stability": False, "turnover": False, "dynamic": False},
    {"name": "2. Ceiling Only",       "ceiling": True,  "stability": False, "turnover": False, "dynamic": False},
    {"name": "3. Stability Only",     "ceiling": False, "stability": True,  "turnover": False, "dynamic": False},
    {"name": "4. Turnover Only",      "ceiling": False, "stability": False, "turnover": True,  "dynamic": False},
    {"name": "5. All Combined",       "ceiling": True,  "stability": True,  "turnover": True,  "dynamic": False},
    {"name": "6. Dynamic (VIX>25)",   "ceiling": True,  "stability": True,  "turnover": True,  "dynamic": True},
]


def run_config_step(
    cfg: dict, vix_t: float, c_raw_t: np.ndarray, sigma_diag: np.ndarray,
    d_adj_t: np.ndarray, Q_final_t: np.ndarray, Q_final_prev, Sigma_t: np.ndarray,
    Pi_t: np.ndarray, w_prev: np.ndarray, is_first_month: bool,
):
    """
    Parameterised extraction of exactly one iteration of run_config()'s
    for-loop body from guardrail_ablation.py -- same formulas, same
    order of operations, just accepting explicit arguments instead of
    reading module-level globals set by the pickle-loading code.
    """
    c = c_raw_t.copy()
    fires = {"ceiling": False, "stability": False, "turnover": False}

    dynamic_active = cfg["dynamic"] and vix_t > VIX_DYNAMIC_THRESHOLD
    ceiling_base = CEILING_BASE_DYNAMIC if dynamic_active else CEILING_BASE_NORMAL
    t_max = T_MAX_DYNAMIC if dynamic_active else T_MAX

    if cfg["ceiling"]:
        c_max = ceiling_base * np.exp(-CEILING_DECAY * max(0.0, vix_t - CEILING_VIX_REF))
        capped = c > c_max
        if capped.any():
            fires["ceiling"] = True
        c = np.minimum(c, c_max)

    if cfg["stability"] and not is_first_month:
        rolling_sigma_curr = np.sqrt(np.diag(Sigma_t))
        rolling_sigma_curr = np.where(rolling_sigma_curr == 0, 1e-8, rolling_sigma_curr)
        delta = np.abs(Q_final_t - Q_final_prev) / rolling_sigma_curr
        penalty = np.exp(-STABILITY_DECAY * np.maximum(0.0, delta - 1.0))
        sign_reversal = ((Q_final_t != 0) & (Q_final_prev != 0) &
                          (np.sign(Q_final_t) != np.sign(Q_final_prev)))
        if (penalty < 0.999).any() or sign_reversal.any():
            fires["stability"] = True
        c = c * penalty
        c = np.where(sign_reversal, c * STABILITY_REVERSAL_MULT, c)

    c = np.clip(c, 0.0, C_CLIP_MAX)

    Omega_diag_t = (1 - c) ** 2 * sigma_diag * (1 + ALPHA * d_adj_t)
    mu_star, Sigma_star = bl_posterior_diag(Pi_t, Sigma_t, Q_final_t, Omega_diag_t, TAU)
    w_target = bl_weights_long_only(mu_star, Sigma_star, len(c))

    if not np.all(np.isfinite(w_target)):
        w_target = w_prev.copy()

    raw_turnover = np.sum(np.abs(w_target - w_prev))

    if cfg["turnover"]:
        w_final, cap_applied = apply_turnover_cap_exact(w_target, w_prev, t_max)
        fires["turnover"] = cap_applied
    else:
        w_final = w_target

    return w_final, c, fires, raw_turnover, dynamic_active


def check_instability_episode(drawdown_extreme: bool, raw_turnover: float, c_raw_t: np.ndarray) -> bool:
    """The formal three-condition definition (brief Section 7.4)."""
    high_implied_turnover = raw_turnover > 0.25
    high_raw_confidence = np.any(c_raw_t > 0.80)
    return bool(drawdown_extreme and high_implied_turnover and high_raw_confidence)


def compute_drawdown_extreme(drawdown_history: list):
    """
    Copied verbatim from run_config()'s per-month drawdown tracking (the
    piece NOT covered by run_config_step above). drawdown_history is the
    cumulative wealth path up to and including the current month.

    Condition 1 of the instability-episode definition (brief Section 7.4):
    current drawdown more than 2 rolling-window standard deviations below
    the rolling mean drawdown, where the window is the last 12 months
    (or fewer, early in the series) and each point in the window is
    itself a drawdown relative to the running peak up to that point --
    NOT a peak restricted to the window.
    """
    wealth = drawdown_history[-1]
    running_max = max(drawdown_history)
    current_dd = wealth / running_max - 1.0
    window_start = max(0, len(drawdown_history) - 12)
    dd_series_window = [drawdown_history[i] / max(drawdown_history[:i + 1]) - 1.0
                         for i in range(window_start, len(drawdown_history))]
    rolling_mean_dd = np.mean(dd_series_window)
    dd_std = np.std(dd_series_window) or 1e-8
    drawdown_extreme = current_dd < (rolling_mean_dd - 2 * dd_std)
    return current_dd, rolling_mean_dd, dd_std, bool(drawdown_extreme)


# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIGS structure — does the ablation matrix encode what it claims?
# ═══════════════════════════════════════════════════════════════════════
class TestConfigsStructure:

    def test_six_configs_defined(self):
        assert len(CONFIGS) == 6

    def test_config_1_has_all_guardrails_off(self):
        cfg = CONFIGS[0]
        assert cfg["ceiling"] is False and cfg["stability"] is False and cfg["turnover"] is False

    def test_config_5_has_all_three_on_but_not_dynamic(self):
        cfg = CONFIGS[4]
        assert cfg["ceiling"] and cfg["stability"] and cfg["turnover"]
        assert cfg["dynamic"] is False

    def test_config_6_has_everything_on_including_dynamic(self):
        cfg = CONFIGS[5]
        assert cfg["ceiling"] and cfg["stability"] and cfg["turnover"] and cfg["dynamic"]

    def test_configs_2_3_4_each_isolate_exactly_one_guardrail(self):
        ceiling_only, stability_only, turnover_only = CONFIGS[1], CONFIGS[2], CONFIGS[3]
        assert (ceiling_only["ceiling"], ceiling_only["stability"], ceiling_only["turnover"]) == (True, False, False)
        assert (stability_only["ceiling"], stability_only["stability"], stability_only["turnover"]) == (False, True, False)
        assert (turnover_only["ceiling"], turnover_only["stability"], turnover_only["turnover"]) == (False, False, True)


# ═══════════════════════════════════════════════════════════════════════
# 2. Each guardrail only ever fires when its own flag is True
# ═══════════════════════════════════════════════════════════════════════
class TestGuardrailIsolation:

    def _base_month(self, seed=0):
        rng = np.random.default_rng(seed)
        sigma_diag = rng.uniform(0.001, 0.006, size=N)
        Sigma = np.diag(sigma_diag)
        Pi = rng.uniform(-0.01, 0.01, size=N)
        Q = rng.uniform(-0.1, 0.1, size=N)          # deliberately large -> confident, extreme
        c_raw = np.full(N, 0.95)                     # deliberately high confidence
        d_adj = np.zeros(N)
        w_prev = np.ones(N) / N
        return sigma_diag, Sigma, Pi, Q, c_raw, d_adj, w_prev

    def test_turnover_only_config_never_fires_ceiling_even_at_high_vix(self):
        sigma_diag, Sigma, Pi, Q, c_raw, d_adj, w_prev = self._base_month()
        cfg = CONFIGS[3]  # Turnover Only
        w_final, c_out, fires, raw_turnover, dynamic_active = run_config_step(
            cfg, vix_t=80.0, c_raw_t=c_raw, sigma_diag=sigma_diag, d_adj_t=d_adj,
            Q_final_t=Q, Q_final_prev=None, Sigma_t=Sigma, Pi_t=Pi,
            w_prev=w_prev, is_first_month=True,
        )
        assert fires["ceiling"] is False
        assert np.allclose(c_out, c_raw)

    def test_ceiling_only_config_never_applies_turnover_cap(self):
        sigma_diag, Sigma, Pi, Q, c_raw, d_adj, w_prev = self._base_month()
        Q_extreme = Q * 50
        cfg = CONFIGS[1]  # Ceiling Only
        w_final, c_out, fires, raw_turnover, dynamic_active = run_config_step(
            cfg, vix_t=15.0, c_raw_t=c_raw, sigma_diag=sigma_diag, d_adj_t=d_adj,
            Q_final_t=Q_extreme, Q_final_prev=None, Sigma_t=Sigma, Pi_t=Pi,
            w_prev=w_prev, is_first_month=True,
        )
        assert fires["turnover"] is False
        assert raw_turnover > T_MAX

    def test_stability_only_config_never_applies_ceiling(self):
        sigma_diag, Sigma, Pi, Q, c_raw, d_adj, w_prev = self._base_month()
        cfg = CONFIGS[2]  # Stability Only
        w_final, c_out, fires, raw_turnover, dynamic_active = run_config_step(
            cfg, vix_t=90.0, c_raw_t=c_raw, sigma_diag=sigma_diag, d_adj_t=d_adj,
            Q_final_t=Q, Q_final_prev=Q, Sigma_t=Sigma, Pi_t=Pi,
            w_prev=w_prev, is_first_month=False,
        )
        assert fires["ceiling"] is False
        c_max_would_be = CEILING_BASE_NORMAL * np.exp(-CEILING_DECAY * max(0.0, 90.0 - CEILING_VIX_REF))
        assert np.all(c_out > c_max_would_be + 0.1)

    def test_stability_filter_skipped_on_first_month_regardless_of_config(self):
        sigma_diag, Sigma, Pi, Q, c_raw, d_adj, w_prev = self._base_month()
        cfg = CONFIGS[2]  # Stability Only
        w_final, c_out, fires, raw_turnover, dynamic_active = run_config_step(
            cfg, vix_t=15.0, c_raw_t=c_raw, sigma_diag=sigma_diag, d_adj_t=d_adj,
            Q_final_t=Q, Q_final_prev=None, Sigma_t=Sigma, Pi_t=Pi,
            w_prev=w_prev, is_first_month=True,
        )
        assert fires["stability"] is False
        assert np.allclose(c_out, c_raw)


# ═══════════════════════════════════════════════════════════════════════
# 3. Confidence ceiling formula and sign-reversal stability penalty
# ═══════════════════════════════════════════════════════════════════════
class TestCeilingAndStabilityFormulas:

    def test_ceiling_matches_closed_form_at_calm_vix(self):
        sigma_diag = np.full(N, 0.002)
        Sigma = np.diag(sigma_diag)
        Pi = np.zeros(N)
        Q = np.zeros(N)
        c_raw = np.full(N, 0.9)
        d_adj = np.zeros(N)
        w_prev = np.ones(N) / N
        cfg = CONFIGS[1]  # Ceiling Only

        _, c_out, _, _, _ = run_config_step(
            cfg, vix_t=20.0, c_raw_t=c_raw, sigma_diag=sigma_diag, d_adj_t=d_adj,
            Q_final_t=Q, Q_final_prev=None, Sigma_t=Sigma, Pi_t=Pi,
            w_prev=w_prev, is_first_month=True,
        )
        expected_c_max = 0.75
        assert np.allclose(c_out, expected_c_max, atol=1e-9)

    def test_sign_reversal_cuts_confidence_by_exactly_half(self):
        sigma_diag = np.full(N, 0.002)
        Sigma = np.diag(sigma_diag)
        Pi = np.zeros(N)
        c_raw = np.full(N, 0.8)
        d_adj = np.zeros(N)
        w_prev = np.ones(N) / N
        cfg = CONFIGS[2]  # Stability Only

        Q_prev = np.full(N, 0.02)
        Q_now = np.full(N, -0.02)

        _, c_out, fires, _, _ = run_config_step(
            cfg, vix_t=15.0, c_raw_t=c_raw, sigma_diag=sigma_diag, d_adj_t=d_adj,
            Q_final_t=Q_now, Q_final_prev=Q_prev, Sigma_t=Sigma, Pi_t=Pi,
            w_prev=w_prev, is_first_month=False,
        )
        assert fires["stability"] is True
        assert np.allclose(c_out, c_raw * 0.5, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# 4. Dynamic (Config 6) VIX>25 threshold behaviour
# ═══════════════════════════════════════════════════════════════════════
class TestDynamicConfig:

    def test_dynamic_only_activates_for_config_6_above_threshold(self):
        sigma_diag = np.full(N, 0.002)
        Sigma = np.diag(sigma_diag)
        Pi, Q = np.zeros(N), np.zeros(N)
        c_raw, d_adj = np.full(N, 0.8), np.zeros(N)
        w_prev = np.ones(N) / N

        _, _, _, _, dynamic_active_cfg6_high_vix = run_config_step(
            CONFIGS[5], vix_t=30.0, c_raw_t=c_raw, sigma_diag=sigma_diag, d_adj_t=d_adj,
            Q_final_t=Q, Q_final_prev=None, Sigma_t=Sigma, Pi_t=Pi,
            w_prev=w_prev, is_first_month=True,
        )
        _, _, _, _, dynamic_active_cfg6_low_vix = run_config_step(
            CONFIGS[5], vix_t=15.0, c_raw_t=c_raw, sigma_diag=sigma_diag, d_adj_t=d_adj,
            Q_final_t=Q, Q_final_prev=None, Sigma_t=Sigma, Pi_t=Pi,
            w_prev=w_prev, is_first_month=True,
        )
        _, _, _, _, dynamic_active_cfg5_high_vix = run_config_step(
            CONFIGS[4], vix_t=30.0, c_raw_t=c_raw, sigma_diag=sigma_diag, d_adj_t=d_adj,
            Q_final_t=Q, Q_final_prev=None, Sigma_t=Sigma, Pi_t=Pi,
            w_prev=w_prev, is_first_month=True,
        )
        assert dynamic_active_cfg6_high_vix is True
        assert dynamic_active_cfg6_low_vix is False
        assert dynamic_active_cfg5_high_vix is False

    def test_dynamic_active_uses_tightened_ceiling(self):
        sigma_diag = np.full(N, 0.002)
        Sigma = np.diag(sigma_diag)
        Pi = np.zeros(N)
        Q = np.full(N, 0.05)
        c_raw = np.full(N, 0.9)
        d_adj = np.zeros(N)
        w_prev = np.ones(N) / N

        _, c_out, _, _, dynamic_active = run_config_step(
            CONFIGS[5], vix_t=30.0, c_raw_t=c_raw, sigma_diag=sigma_diag, d_adj_t=d_adj,
            Q_final_t=Q, Q_final_prev=None, Sigma_t=Sigma, Pi_t=Pi,
            w_prev=w_prev, is_first_month=True,
        )
        assert dynamic_active is True
        expected_c_max = CEILING_BASE_DYNAMIC * np.exp(-CEILING_DECAY * (30.0 - CEILING_VIX_REF))
        assert np.allclose(c_out, expected_c_max, atol=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# 5. Instability episode — formal three-condition definition
# ═══════════════════════════════════════════════════════════════════════
class TestInstabilityEpisodeDefinition:

    def test_requires_all_three_conditions_simultaneously(self):
        c_raw_high = np.full(N, 0.85)
        c_raw_low = np.full(N, 0.5)

        assert check_instability_episode(drawdown_extreme=True, raw_turnover=0.30, c_raw_t=c_raw_high) is True
        assert check_instability_episode(drawdown_extreme=False, raw_turnover=0.30, c_raw_t=c_raw_high) is False
        assert check_instability_episode(drawdown_extreme=True, raw_turnover=0.10, c_raw_t=c_raw_high) is False
        assert check_instability_episode(drawdown_extreme=True, raw_turnover=0.30, c_raw_t=c_raw_low) is False

    def test_turnover_threshold_is_exclusive_at_0_25(self):
        c_raw_high = np.full(N, 0.85)
        assert check_instability_episode(drawdown_extreme=True, raw_turnover=0.25, c_raw_t=c_raw_high) is False
        assert check_instability_episode(drawdown_extreme=True, raw_turnover=0.2500001, c_raw_t=c_raw_high) is True

    def test_confidence_threshold_is_exclusive_at_0_80(self):
        assert check_instability_episode(
            drawdown_extreme=True, raw_turnover=0.30, c_raw_t=np.full(N, 0.80)
        ) is False
        assert check_instability_episode(
            drawdown_extreme=True, raw_turnover=0.30, c_raw_t=np.array([0.80] * 9 + [0.801])
        ) is True


# ═══════════════════════════════════════════════════════════════════════
# 5b. drawdown_extreme — the rolling-window computation that FEEDS
#     condition 1 of the instability-episode gate above. This closes the
#     one gap previously left open: TestInstabilityEpisodeDefinition only
#     tested the boolean AND-gate given drawdown_extreme as an input;
#     these tests verify the arithmetic that actually produces it.
# ═══════════════════════════════════════════════════════════════════════
class TestDrawdownExtremeComputation:

    def test_constant_wealth_is_never_extreme(self):
        current_dd, rolling_mean_dd, dd_std, extreme = compute_drawdown_extreme([1.0] * 6)
        assert current_dd == pytest.approx(0.0, abs=1e-12)
        assert extreme is False

    def test_monotonically_increasing_wealth_is_never_extreme(self):
        current_dd, rolling_mean_dd, dd_std, extreme = compute_drawdown_extreme(
            [1.0, 1.05, 1.1, 1.2, 1.3, 1.4]
        )
        assert current_dd == pytest.approx(0.0, abs=1e-12)
        assert extreme is False

    def test_zero_variance_window_floors_std_at_1e_minus_8(self):
        """
        np.std(...) or 1e-8 relies on 0.0 being falsy in Python -- confirmed
        working here, but worth pinning down explicitly since it's a subtle
        idiom (a genuine non-zero-but-tiny std would NOT trigger the floor,
        only an exact 0.0 does).
        """
        current_dd, rolling_mean_dd, dd_std, extreme = compute_drawdown_extreme([1.0] * 6)
        assert dd_std == pytest.approx(1e-8, abs=1e-15)

    def test_matches_independently_computed_values_for_a_known_path(self):
        """
        drawdown_history = [1.0, 1.1, 1.2, 0.9, 0.85, 0.95] -- verified
        independently (not by the function itself) by direct calculation:
        running-max-to-date at each point is [1.0,1.1,1.2,1.2,1.2,1.2], so
        per-month drawdowns are [0, 0, 0, -0.25, -0.291667, -0.208333],
        mean = -0.125, current_dd (last month, 0.95 vs peak 1.2) = -0.208333.
        """
        current_dd, rolling_mean_dd, dd_std, extreme = compute_drawdown_extreme(
            [1.0, 1.1, 1.2, 0.9, 0.85, 0.95]
        )
        assert current_dd == pytest.approx(0.95 / 1.2 - 1.0, abs=1e-9)
        assert rolling_mean_dd == pytest.approx(-0.125, abs=1e-6)
        assert extreme is False  # -0.2083 is not below -0.125 - 2*0.127 = -0.379

    def test_sharp_crash_after_calm_history_is_flagged_extreme(self):
        calm = [1.0] * 11
        current_dd, rolling_mean_dd, dd_std, extreme = compute_drawdown_extreme(calm + [0.5])
        assert current_dd == pytest.approx(-0.5, abs=1e-9)
        assert extreme is True

    def test_window_truncates_to_last_12_months(self):
        """
        An old, large crash more than 12 months in the past must NOT
        affect today's rolling_mean_dd/dd_std -- only the trailing 12
        entries of drawdown_history are used in the averaging window
        (though each entry's own drawdown is still measured against the
        FULL running peak up to that point, not a windowed peak).
        """
        old_crash = [1.0, 0.3, 1.0]  # a severe crash and recovery, long ago
        calm_recent = [1.0] * 12     # last 12 months: flat, no drawdown at all
        history = old_crash + calm_recent
        current_dd, rolling_mean_dd, dd_std, extreme = compute_drawdown_extreme(history)
        # If the old crash leaked into the window, rolling_mean_dd would be
        # strongly negative; since the window is only the last 12 (all
        # calm), it must be ~0.
        assert rolling_mean_dd == pytest.approx(0.0, abs=1e-9)
        assert extreme is False

    def test_first_month_is_never_extreme_by_construction(self):
        """With only one point in history, current_dd is always 0 relative
        to itself, and the window has a single point (std=0 -> floored)."""
        current_dd, rolling_mean_dd, dd_std, extreme = compute_drawdown_extreme([1.0])
        assert current_dd == pytest.approx(0.0, abs=1e-12)
        assert extreme is False


# ═══════════════════════════════════════════════════════════════════════
# 6. bl_posterior_diag agrees with the real black_litterman.py
# ═══════════════════════════════════════════════════════════════════════
class TestBLPosteriorDiagAgreesWithRealModel:

    def test_matches_real_model_diagonal_case(self):
        rng = np.random.default_rng(6)
        sigma2 = rng.uniform(0.0008, 0.006, size=N)
        Sigma = np.diag(sigma2)
        w_mkt = rng.dirichlet(np.ones(N))
        Q = rng.uniform(-0.03, 0.03, size=N)
        c_final = rng.uniform(0.2, 0.9, size=N)
        d_adjusted = rng.uniform(0.0, 0.6, size=N)

        Pi = LAMBDA * Sigma @ w_mkt
        Omega_diag = (1 - c_final) ** 2 * sigma2 * (1 + ALPHA * d_adjusted)

        mu_star_ablation, Sigma_star_ablation = bl_posterior_diag(Pi, Sigma, Q, Omega_diag, TAU)

        model = BlackLittermanModel()
        model._rolling_covariance = lambda *a, **kw: Sigma
        model._market_weights = lambda *a, **kw: w_mkt

        dummy_idx = pd.date_range("2016-01-31", "2019-01-31", freq="ME")
        dummy_returns = pd.DataFrame(np.zeros((len(dummy_idx), N)), index=dummy_idx, columns=TICKERS)
        dummy_wmkt_df = pd.DataFrame(np.tile(w_mkt, (len(dummy_idx), 1)), index=dummy_idx, columns=TICKERS)

        inputs = BLInputs(Q=Q, c_final=c_final, d_adjusted=d_adjusted)
        out = model.run(
            date=pd.Timestamp("2019-01-31"), returns_df=dummy_returns,
            w_mkt_df=dummy_wmkt_df, w_previous=np.ones(N) / N, bl_inputs=inputs,
        )

        assert np.allclose(mu_star_ablation, out.mu_star, atol=1e-8)
        assert np.allclose(Sigma_star_ablation, out.Sigma_star, atol=1e-8)


# ═══════════════════════════════════════════════════════════════════════
# 7. Turnover cap divergence between guardrail_ablation.py and
#    baselines_part2.py's "same" function
# ═══════════════════════════════════════════════════════════════════════
class TestTurnoverCapNoRenormaliseWhenNotBinding:

    def test_returns_target_completely_unchanged_when_not_binding(self):
        """
        guardrail_ablation.py's apply_turnover_cap_exact returns w_target
        AS-IS (no renormalisation) when the cap doesn't bind -- unlike
        baselines_part2.py's apply_turnover_cap, which always renormalises
        even when not binding. Confirmed divergence, tested directly.
        """
        w_prev = np.ones(N) / N
        w_target = w_prev.copy()
        w_target[0] += 0.01
        w_target[1] -= 0.01
        w_final, cap_applied = apply_turnover_cap_exact(w_target, w_prev, t_max=0.30)
        assert bool(cap_applied) is False
        assert np.array_equal(w_final, w_target)

    def test_cap_binds_and_result_sums_to_one(self):
        w_prev = np.ones(N) / N
        w_target = np.array([0.9] + [0.1 / 9] * 9)
        w_final, cap_applied = apply_turnover_cap_exact(w_target, w_prev, t_max=0.30)
        assert bool(cap_applied) is True
        assert w_final.sum() == pytest.approx(1.0, abs=1e-9)
        realised_l1 = np.sum(np.abs(w_final - w_prev))
        assert realised_l1 == pytest.approx(0.30, abs=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
