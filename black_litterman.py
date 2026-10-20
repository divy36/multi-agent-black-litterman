"""Black litterman."""

from __future__ import annotations
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants — calibrated to research project brief Section 4.1
# --------------------------------------------------------------------------- #
TICKERS: list[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLRE", "XLU", "XLB",
]
N = len(TICKERS)

LAMBDA: float = 3.0    # market risk aversion
TAU:    float = 0.05   # prior uncertainty scalar (He-Litterman 1999)
ALPHA:  float = 2.0    # disagreement scaling for Omega inflation
T_MAX:  float = 0.30   # maximum monthly L1 turnover
COV_WINDOW: int = 36   # rolling covariance window in months


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class BLInputs:
    """
    Agent-derived inputs to the Black-Litterman model.

    All arrays are length-N (one entry per sector ETF, in TICKERS order).

    Attributes
    ----------
    Q : np.ndarray, shape (N,)
        Blended expected monthly returns from Agent 3 (Synthesis).
        Units: decimal (e.g. 0.02 = +2 % expected monthly return).
    c_final : np.ndarray, shape (N,)
        Guardrail-adjusted confidence scores in [0, 1].
        Produced by Risk Guardian after applying confidence ceiling and
        view stability filter.
    d_adjusted : np.ndarray, shape (N,)
        Adjusted inter-agent disagreement scores in [0, 1].
        d_adjusted_j = min(1.0, d_j + penalty_j) from Agent 4 (Contrarian).
        Used to inflate Omega: higher disagreement → larger Omega →
        model stays closer to equilibrium.
    """
    Q:          np.ndarray = field(default_factory=lambda: np.zeros(N))
    c_final:    np.ndarray = field(default_factory=lambda: np.full(N, 0.5))
    d_adjusted: np.ndarray = field(default_factory=lambda: np.zeros(N))

    def __post_init__(self) -> None:
        for name, arr in [("Q", self.Q),
                          ("c_final", self.c_final),
                          ("d_adjusted", self.d_adjusted)]:
            if arr.shape != (N,):
                raise ValueError(f"BLInputs.{name} must have shape ({N},), "
                                 f"got {arr.shape}")
        if not (np.all(self.c_final >= 0) and np.all(self.c_final <= 1)):
            raise ValueError("c_final values must be in [0, 1]")
        if not (np.all(self.d_adjusted >= 0) and np.all(self.d_adjusted <= 1)):
            raise ValueError("d_adjusted values must be in [0, 1]")


@dataclass
class BLOutputs:
    """
    Full output bundle from one monthly BL run.

    Every intermediate matrix is preserved so the research project's analysis
    analysis can inspect any step of the pipeline.

    Attributes
    ----------
    date : pd.Timestamp
        The month-end date this rebalance applies to.
    Pi : np.ndarray (N,)
        Equilibrium prior returns.  Pi = lambda * Sigma * w_mkt.
    Sigma : np.ndarray (N, N)
        Rolling 36-month sample covariance matrix of monthly returns.
    Omega : np.ndarray (N, N)
        Diagonal uncertainty matrix built from agent confidence and
        disagreement scores.
    mu_star : np.ndarray (N,)
        Posterior expected returns from the BL Bayesian update.
    Sigma_star : np.ndarray (N, N)
        Posterior covariance matrix.  Sigma* = Sigma + M.
    w_uncapped : np.ndarray (N,)
        Portfolio weights before turnover cap (normalised).
    w_final : np.ndarray (N,)
        Final portfolio weights after turnover cap. Use these.
    turnover : float
        L1 norm of weight change: sum(|w_final - w_previous|).
    cap_applied : bool
        True if the turnover cap was binding this month.
    used_prior_only : bool
        True if no agent views were supplied (model ran on prior alone).
    """
    date:         pd.Timestamp
    Pi:           np.ndarray
    Sigma:        np.ndarray
    Omega:        np.ndarray
    mu_star:      np.ndarray
    Sigma_star:   np.ndarray
    w_uncapped:   np.ndarray
    w_final:      np.ndarray
    turnover:     float
    cap_applied:  bool
    used_prior_only: bool


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #
class BlackLittermanModel:
    """
    Black-Litterman model — from scratch in NumPy.

    Implements the full pipeline from brief Section 4.2 exactly:

        Step 1  Pi      = lambda * Sigma * w_mkt
        Step 2  P, Q, Omega  from BLInputs
        Step 3  M       = inv( inv(tau*Sigma) + P' @ inv(Omega) @ P )
                mu*     = M @ ( inv(tau*Sigma)@Pi + P'@inv(Omega)@Q )
        Step 4  Sigma*  = Sigma + M
        Step 5  w_raw   = inv(Sigma*) @ mu*
                w_target = w_raw / sum(w_raw)
                Turnover cap applied if ||w_target - w_prev||_1 > T_max

    Reference: He & Litterman (1999), equations (1)–(5).
    """

    def __init__(
        self,
        lam:        float = LAMBDA,
        tau:        float = TAU,
        alpha:      float = ALPHA,
        t_max:      float = T_MAX,
        cov_window: int   = COV_WINDOW,
    ) -> None:
        self.lam        = lam
        self.tau        = tau
        self.alpha      = alpha
        self.t_max      = t_max
        self.cov_window = cov_window

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def run(
        self,
        date:       pd.Timestamp,
        returns_df: pd.DataFrame,
        w_mkt_df:   pd.DataFrame,
        w_previous: np.ndarray,
        bl_inputs:  Optional[BLInputs] = None,
    ) -> BLOutputs:
        """
        Execute one monthly BL rebalance.

        Parameters
        ----------
        date : pd.Timestamp
            Month-end date of the rebalance (e.g. 2019-01-31).
            Sigma is estimated from the 36 months ending the month
            *before* this date, avoiding look-ahead bias.
        returns_df : pd.DataFrame
            Monthly returns, index = month-end dates, columns = TICKERS.
            Must cover at least 36 months before `date`.
            Typically loaded from data/etf_returns_monthly.csv.
        w_mkt_df : pd.DataFrame
            Market-cap weight vectors, index = month-end dates,
            columns = TICKERS.
            Typically loaded from data/market_cap_weights.csv.
        w_previous : np.ndarray, shape (N,)
            Portfolio weights at end of prior month (used for turnover cap
            and for the view stability filter in the Risk Guardian).
        bl_inputs : BLInputs, optional
            Agent-derived views, confidence, and disagreement scores.
            If None, the model runs on the equilibrium prior alone
            (no agent views — useful as a baseline).

        Returns
        -------
        BLOutputs
            Full intermediate and final results for this month.
        """
        self._validate_inputs(date, returns_df, w_mkt_df, w_previous)

        # ── Step 1: Sigma and equilibrium prior Pi ──────────────────────
        Sigma  = self._rolling_covariance(date, returns_df)
        w_mkt  = self._market_weights(date, w_mkt_df)
        Pi     = self.lam * Sigma @ w_mkt   # (N,)

        # ── Step 2: Views matrices ───────────────────────────────────────
        used_prior_only = bl_inputs is None
        if used_prior_only:
            # No agent views — posterior IS the prior. Skip the Bayesian
            # update entirely to avoid numerical overflow from Omega → ∞.
            # M = tau*Sigma is the limiting case; Sigma* = Sigma + tau*Sigma.
            mu_star    = Pi.copy()
            M          = self.tau * Sigma
            Sigma_star = Sigma + M
            Omega      = np.diag(np.diag(Sigma) * 1e6)   # stored for output only
        else:
            Q, Omega_diag, P = self._build_views(bl_inputs, Sigma)
            Omega = np.diag(Omega_diag)   # (N, N) diagonal

            # ── Step 3: Posterior returns (He-Litterman 1999, eq. 4-5) ──
            #
            # Numerically stable: since P = identity, inv(Omega) is just
            # diag(1/Omega_diag). We never explicitly invert Omega as a full
            # matrix, avoiding overflow when Omega entries are tiny (high conf).
            #
            #   A   = inv(tau*Sigma) + diag(1/Omega_diag)
            #   b   = solve(tau*Sigma, Pi) + (1/Omega_diag) * Q
            #   M   = inv(A)
            #   mu* = M @ b
            tau_Sigma      = self.tau * Sigma
            inv_Omega_diag = 1.0 / Omega_diag           # safe: floored at 1e-12

            # P = I, so P'@inv(Omega)@P = diag(inv_Omega_diag)
            #         P'@inv(Omega)@Q   = inv_Omega_diag * Q
            PtOinvP = np.diag(inv_Omega_diag)           # (N, N)
            PtOinvQ = inv_Omega_diag * Q                # (N,)

            tau_Sigma_inv    = np.linalg.inv(tau_Sigma)
            tau_Sig_inv_Pi   = np.linalg.solve(tau_Sigma, Pi)

            A       = tau_Sigma_inv + PtOinvP
            b       = tau_Sig_inv_Pi + PtOinvQ
            M       = np.linalg.inv(A)
            mu_star = M @ b

            # ── Step 4: Posterior covariance ─────────────────────────────
            Sigma_star = Sigma + M

        # ── Step 5: Portfolio weights and turnover cap ───────────────────
        w_uncapped, w_final, turnover, cap_applied = self._portfolio_weights(
            mu_star, Sigma_star, w_previous
        )

        return BLOutputs(
            date          = date,
            Pi            = Pi,
            Sigma         = Sigma,
            Omega         = Omega,
            mu_star       = mu_star,
            Sigma_star    = Sigma_star,
            w_uncapped    = w_uncapped,
            w_final       = w_final,
            turnover      = turnover,
            cap_applied   = cap_applied,
            used_prior_only = used_prior_only,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _rolling_covariance(
        self,
        date:       pd.Timestamp,
        returns_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        36-month rolling sample covariance matrix of monthly returns.

        Uses returns *strictly before* `date` to avoid look-ahead bias.
        The window is the 36 months ending the last month-end before `date`.

        Returns
        -------
        np.ndarray, shape (N, N)
        """
        pre = returns_df[returns_df.index < date].dropna()
        if len(pre) < self.cov_window:
            raise ValueError(
                f"Only {len(pre)} return observations available before "
                f"{date.date()}; need {self.cov_window}. "
                f"Extend ETF price history further back."
            )
        window = pre.iloc[-self.cov_window:]
        Sigma  = window[TICKERS].cov().values   # (N, N)
        if np.any(np.isnan(Sigma)):
            raise ValueError(f"Sigma contains NaN at {date.date()}.")
        return Sigma

    def _market_weights(
        self,
        date:     pd.Timestamp,
        w_mkt_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Market-cap weight vector w_mkt for month t.

        Uses the row dated end of month t-1 (lag-1) to avoid look-ahead
        bias — the brief's note on item 9.3.

        Returns
        -------
        np.ndarray, shape (N,)
        """
        prior_rows = w_mkt_df[w_mkt_df.index < date]
        if prior_rows.empty:
            raise ValueError(f"No w_mkt data available before {date.date()}.")
        w = prior_rows.iloc[-1][TICKERS].values.astype(float)
        if abs(w.sum() - 1.0) > 1e-6:
            w = w / w.sum()   # renormalise for safety
        return w

    def _build_views(
        self,
        inputs: BLInputs,
        Sigma:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct Q, Omega_diag, P from agent inputs.

        Omega formula (brief Section 4.2, Step 2):
            Omega_ii = (1 - c_final_i)^2 * sigma^2_i * (1 + alpha * d_adjusted_i)

        where sigma^2_i = Sigma[i, i] (rolling variance of sector i).

        P is the N×N identity matrix (absolute views — each row picks
        exactly one sector ETF).

        Returns
        -------
        Q          : np.ndarray (N,)
        Omega_diag : np.ndarray (N,)  — diagonal entries of Omega
        P          : np.ndarray (N, N)
        """
        sigma2 = np.diag(Sigma)   # (N,) variances

        Omega_diag = (
            (1 - inputs.c_final) ** 2
            * sigma2
            * (1 + self.alpha * inputs.d_adjusted)
        )

        # Guard: Omega entries must be strictly positive
        Omega_diag = np.maximum(Omega_diag, 1e-12)

        P = np.eye(N)

        return inputs.Q, Omega_diag, P

    def _portfolio_weights(
        self,
        mu_star:    np.ndarray,
        Sigma_star: np.ndarray,
        w_previous: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Convert posterior (mu*, Sigma*) into final portfolio weights.

        Mean-variance optimal weights:
            w_raw    = inv(Sigma*) @ mu*
            w_target = w_raw / sum(w_raw)   [normalised to sum to 1]

        Then apply the L1 turnover cap (brief Section 4.2, Step 5):
            delta_w = w_target - w_previous
            if sum(|delta_w|) > T_max:
                delta_w *= T_max / sum(|delta_w|)
                w_final  = normalise(w_previous + delta_w)
            else:
                w_final  = w_target

        Returns
        -------
        w_uncapped  : weights before cap
        w_final     : weights after cap
        turnover    : L1 norm of actual weight change
        cap_applied : True if the cap was binding
        """
        Sigma_star_inv = np.linalg.inv(Sigma_star)
        w_raw          = Sigma_star_inv @ mu_star   # (N,)

        # Long-only constraint: clip negatives to zero then renormalise.
        # The ETF portfolio cannot hold short positions.
        w_raw = np.maximum(w_raw, 0.0)

        # Normalise — handle edge case where sum is near zero
        raw_sum = w_raw.sum()
        if abs(raw_sum) < 1e-12:
            w_target = np.ones(N) / N   # fallback: equal weight
        else:
            w_target = w_raw / raw_sum

        w_uncapped = w_target.copy()

        # Turnover cap
        delta_w   = w_target - w_previous
        l1_change = np.abs(delta_w).sum()
        cap_applied = l1_change > self.t_max

        if cap_applied:
            delta_w = delta_w * (self.t_max / l1_change)
            w_capped = w_previous + delta_w
            # Renormalise after cap
            capped_sum = w_capped.sum()
            w_final = w_capped / capped_sum if abs(capped_sum) > 1e-12 \
                      else np.ones(N) / N
        else:
            w_final = w_target

        turnover = np.abs(w_final - w_previous).sum()
        return w_uncapped, w_final, float(turnover), cap_applied

    def _validate_inputs(
        self,
        date:       pd.Timestamp,
        returns_df: pd.DataFrame,
        w_mkt_df:   pd.DataFrame,
        w_previous: np.ndarray,
    ) -> None:
        """Fail fast with a clear message if inputs are malformed."""
        for col in TICKERS:
            if col not in returns_df.columns:
                raise ValueError(f"returns_df missing column '{col}'")
            if col not in w_mkt_df.columns:
                raise ValueError(f"w_mkt_df missing column '{col}'")
        if w_previous.shape != (N,):
            raise ValueError(f"w_previous must have shape ({N},), "
                             f"got {w_previous.shape}")
        if abs(w_previous.sum() - 1.0) > 1e-4:
            raise ValueError(f"w_previous must sum to 1, "
                             f"got {w_previous.sum():.6f}")


# --------------------------------------------------------------------------- #
# Convenience: load standard data files
# --------------------------------------------------------------------------- #
def load_data(project_root: str = str(Path(__file__).resolve().parent)) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load etf_returns_monthly.csv and market_cap_weights.csv from the
    standard project data directory.

    Returns
    -------
    returns_df : pd.DataFrame   (108 rows × 10 cols)
    w_mkt_df   : pd.DataFrame   (73 rows  × 10 cols)
    """
    from pathlib import Path
    root = Path(project_root)
    returns_df = pd.read_csv(
        root / "data" / "etf_returns_monthly.csv",
        index_col="Date", parse_dates=True,
    )
    w_mkt_df = pd.read_csv(
        root / "data" / "market_cap_weights.csv",
        index_col="Date", parse_dates=True,
    )
    return returns_df, w_mkt_df


# --------------------------------------------------------------------------- #
# Self-test: run when executed directly
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Quick end-to-end smoke test using your real data files.
    Runs three months: Jan 2019, Mar 2020 (COVID), Sep 2022 (rate shock).
    Prints key outputs for each month so you can sanity-check the numbers.
    """
    import sys

    print("=" * 60)
    print("Black-Litterman self-test")
    print("=" * 60)

    try:
        returns_df, w_mkt_df = load_data()
    except FileNotFoundError as e:
        print(f"ERROR loading data: {e}")
        print("Run build_etf_prices.py and build_market_weights.py first.")
        sys.exit(1)

    model = BlackLittermanModel()
    w_prev = np.ones(N) / N   # equal-weight starting portfolio

    test_cases = [
        # (date, label, bl_inputs or None)
        #
        # NOTE: etf_returns_monthly.csv starts Jan 2016 with a NaN first row
        # (no prior price available), so the first valid return is Feb 2016.
        # That gives 35 clean rows before Jan 2019 and 36 before Feb 2019.
        # The simulation therefore starts Feb 2019 for the first BL call;
        # the Jan 2019 rebalance uses the Dec 2018 w_mkt row (lag-1) as intended.
        (
            pd.Timestamp("2019-02-28"),
            "Feb 2019 — prior only (no agent views)",
            None,
        ),
        (
            pd.Timestamp("2019-02-28"),
            "Feb 2019 — with agent views (moderate hawk)",
            BLInputs(
                Q          = np.array([ 0.005,  0.012, -0.008,  0.003,  0.006,
                                        0.002,  0.004, -0.018, -0.014,  0.001]),
                c_final    = np.array([ 0.65,   0.55,   0.50,   0.45,   0.50,
                                        0.55,   0.40,   0.70,   0.65,   0.40]),
                d_adjusted = np.array([ 0.40,   0.10,   0.15,   0.05,   0.10,
                                        0.20,   0.05,   0.35,   0.30,   0.10]),
            ),
        ),
        (
            pd.Timestamp("2020-03-31"),
            "Mar 2020 — COVID shock (high disagreement)",
            BLInputs(
                Q          = np.array([-0.030, -0.015, -0.040, -0.005, -0.020,
                                       -0.035,  0.010, -0.050, -0.045, -0.025]),
                c_final    = np.array([ 0.55,   0.50,   0.60,   0.45,   0.50,
                                        0.55,   0.40,   0.65,   0.60,   0.45]),
                d_adjusted = np.array([ 0.70,   0.40,   0.60,   0.20,   0.50,
                                        0.65,   0.15,   0.80,   0.75,   0.45]),
            ),
        ),
        (
            pd.Timestamp("2022-09-30"),
            "Sep 2022 — aggressive rate hike signal",
            BLInputs(
                Q          = np.array([-0.020, 0.008, -0.005, -0.003, -0.008,
                                       -0.015, 0.005, -0.030, -0.025, -0.010]),
                c_final    = np.array([ 0.70,  0.60,   0.45,   0.40,   0.50,
                                        0.65,  0.45,   0.75,   0.72,   0.40]),
                d_adjusted = np.array([ 0.50,  0.20,   0.15,   0.10,   0.20,
                                        0.40,  0.10,   0.55,   0.50,   0.20]),
            ),
        ),
    ]

    for date, label, bl_inputs in test_cases:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        try:
            out = model.run(
                date       = date,
                returns_df = returns_df,
                w_mkt_df   = w_mkt_df,
                w_previous = w_prev,
                bl_inputs  = bl_inputs,
            )

            print(f"  Prior (Pi):    {['%+.4f' % x for x in out.Pi]}")
            print(f"  Posterior mu*: {['%+.4f' % x for x in out.mu_star]}")
            print(f"\n  Portfolio weights (w_final):")
            for t, w in zip(TICKERS, out.w_final):
                bar = "█" * int(w * 40)
                print(f"    {t:5s}  {w:6.1%}  {bar}")
            print(f"\n  Weights sum to 1: {abs(out.w_final.sum() - 1.0) < 1e-9}")
            print(f"  Turnover:         {out.turnover:.4f} "
                  f"({'CAP APPLIED' if out.cap_applied else 'within limit'})")
            print(f"  All weights ≥ 0:  {np.all(out.w_final >= -1e-9)}")

            # Update w_prev for next test case's turnover calculation
            w_prev = out.w_final.copy()

        except Exception as exc:
            print(f"  ERROR: {exc}")
            raise

    print(f"\n{'=' * 60}")
    print("Self-test complete. All steps ran without errors.")
    print("=" * 60)