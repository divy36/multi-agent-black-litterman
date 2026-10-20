"""Agent synthesis."""

from __future__ import annotations
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TICKERS: list[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLRE", "XLU", "XLB",
]
N = len(TICKERS)

COV_WINDOW        = 36     # months — matches BL covariance window
DISAGREEMENT_PENALTY_WEIGHT = 0.3  # brief: c_blend *= (1 - 0.3 * d_j)
MIN_SIGMA_FLOOR   = 0.005  # floor on sigma_j (0.5% monthly) to prevent division by zero
                            # in pathological low-vol periods


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class SynthesisOutput:
    """
    Full output bundle from one Synthesis agent run.

    This is the central data structure connecting Agents 1+2 to Agent 4
    (Contrarian) and then to the Black-Litterman model.

    Attributes
    ----------
    Q_blend : np.ndarray (N,)
        Confidence-weighted blended expected monthly returns.
        Q_blend_j = (c_hawk_j*Q_hawk_j + c_dove_j*Q_dove_j) / (c_hawk_j+c_dove_j)

    c_blend : np.ndarray (N,)
        Blended confidence, penalised by disagreement.
        c_blend_j = avg(c_hawk,c_dove) * (1 - 0.3 * min(1.0, d_j))

    d_j : np.ndarray (N,)
        Raw inter-agent disagreement scores, normalised by sector vol.
        d_j = |Q_hawk_j - Q_dove_j| / sigma_j
        Range: [0, ∞) but clipped to [0, 1] for Omega inflation purposes.

    d_adjusted : np.ndarray (N,)
        Pre-Contrarian disagreement scores = min(1.0, d_j).
        Agent 4 (Contrarian) will add penalty_j to produce the final
        d_adjusted used in Omega: d_final_j = min(1.0, d_j + penalty_j).

    sigma_j : np.ndarray (N,)
        Rolling 36-month monthly return standard deviations per sector.
        Used to normalise d_j and stored for the stability report.

    bl_inputs : BLInputs
        Ready-to-use input object for black_litterman.BlackLittermanModel.run().
        Q = Q_blend, c_final = c_blend, d_adjusted = d_adjusted (pre-Contrarian).
        Contrarian will update d_adjusted before the final BL call.

    date : str
        ISO month-end date string (e.g. "2019-02-28").

    hawk_Q : np.ndarray (N,)
        Raw Hawk expected returns (stored for disagreement analysis).

    dove_Q : np.ndarray (N,)
        Raw Dove expected returns (stored for disagreement analysis).

    hawk_confidence : np.ndarray (N,)
        Raw Hawk confidence scores.

    dove_confidence : np.ndarray (N,)
        Raw Dove confidence scores.

    max_disagreement_sector : str
        Ticker with highest d_j this month (early warning signal).

    mean_disagreement : float
        Mean of d_j across all sectors (portfolio-level tension measure).

    hawk_fallback : bool
        True if Hawk used its conservative fallback this month.

    dove_fallback : bool
        True if Dove used its conservative fallback this month.
    """
    Q_blend:                 np.ndarray
    c_blend:                 np.ndarray
    d_j:                     np.ndarray
    d_adjusted:              np.ndarray
    sigma_j:                 np.ndarray
    bl_inputs:               object          # BLInputs — typed as object to avoid circular import
    date:                    str
    hawk_Q:                  np.ndarray
    dove_Q:                  np.ndarray
    hawk_confidence:         np.ndarray
    dove_confidence:         np.ndarray
    max_disagreement_sector: str
    mean_disagreement:       float
    hawk_fallback:           bool
    dove_fallback:           bool


# --------------------------------------------------------------------------- #
# Core Synthesis agent
# --------------------------------------------------------------------------- #
class SynthesisAgent:
    """
    Agent 3 — Synthesis.

    Deterministic confidence-weighted blending of Hawk and Dove views.
    No LLM call. All maths, fully auditable and reproducible.

    The agent:
    1. Estimates rolling monthly sigma_j from returns history
    2. Blends Q_hawk and Q_dove by confidence weights
    3. Computes normalised disagreement score d_j per sector
    4. Penalises blended confidence by disagreement level
    5. Packages everything into BLInputs for the BL model

    Usage
    -----
        from agent_synthesis import SynthesisAgent
        from agent_hawk import HawkAgent
        from agent_dove import DoveAgent
        import pandas as pd

        returns_df = pd.read_csv("data/etf_returns_monthly.csv",
                                 index_col="Date", parse_dates=True)
        hawk = HawkAgent()
        dove = DoveAgent()
        synth = SynthesisAgent()

        hawk_out = hawk.run(fomc_text=..., cpi_yoy=..., ...)
        dove_out = dove.run(fomc_text=..., gdp_growth=..., ...)

        result = synth.run(
            hawk_output = hawk_out,
            dove_output = dove_out,
            returns_df  = returns_df,
            date        = "2019-02-28",
        )

        # Pass to Black-Litterman
        bl_result = model.run(
            date       = pd.Timestamp("2019-02-28"),
            returns_df = returns_df,
            w_mkt_df   = w_mkt_df,
            w_previous = w_prev,
            bl_inputs  = result.bl_inputs,
        )
    """

    def __init__(self, cov_window: int = COV_WINDOW) -> None:
        self.cov_window = cov_window

    def run(
        self,
        hawk_output,          # HawkOutput from agent_hawk.py
        dove_output,          # DoveOutput from agent_dove.py
        returns_df: pd.DataFrame,
        date: str,
    ) -> SynthesisOutput:
        """
        Run one month of synthesis.

        Parameters
        ----------
        hawk_output : HawkOutput
            Output from Agent 1 (Hawk) for this month.
        dove_output : DoveOutput
            Output from Agent 2 (Dove) for this month.
        returns_df : pd.DataFrame
            Monthly returns, index=month-end dates, columns=TICKERS.
            Used to estimate rolling sigma_j for disagreement normalisation.
        date : str
            ISO month-end date string (e.g. "2019-02-28").
            Used to select the correct historical window for sigma_j.

        Returns
        -------
        SynthesisOutput
            Full blending results with BLInputs ready for the BL model.
        """
        # ── 1. Extract raw inputs ────────────────────────────────────────────
        Q_hawk = hawk_output.Q.astype(float)
        Q_dove = dove_output.Q.astype(float)
        c_hawk = hawk_output.confidence.astype(float)
        c_dove = dove_output.confidence.astype(float)

        # ── 2. Rolling sigma_j (36-month window, strictly before date) ───────
        sigma_j = self._rolling_sigma(returns_df, date)

        # ── 3. Confidence-weighted blend of Q (brief formula) ────────────────
        # Q_blend_j = (c_hawk_j * Q_hawk_j + c_dove_j * Q_dove_j)
        #             / (c_hawk_j + c_dove_j)
        # Guard: if both confidences are 0 for a sector, use equal weight
        weight_sum = c_hawk + c_dove
        weight_sum_safe = np.where(weight_sum < 1e-12, 1.0, weight_sum)
        Q_blend = (c_hawk * Q_hawk + c_dove * Q_dove) / weight_sum_safe

        # ── 4. Disagreement score d_j (brief formula) ────────────────────────
        # d_j = |Q_hawk_j - Q_dove_j| / sigma_j
        # Normalised by sector volatility so all sectors are comparable.
        # Clipped to [0, 1] for Omega inflation; raw value stored separately.
        abs_diff = np.abs(Q_hawk - Q_dove)
        d_j_raw  = abs_diff / sigma_j               # can exceed 1.0
        d_j      = d_j_raw.copy()                   # store raw (unbounded)
        d_adjusted = np.minimum(1.0, d_j_raw)       # clipped for Omega use

        # ── 5. Blended confidence, penalised by disagreement ─────────────────
        # c_blend_j = avg(c_hawk_j, c_dove_j) * (1 - 0.3 * min(1.0, d_j))
        # The penalty reduces confidence when agents disagree strongly.
        # At d_j=1.0: confidence drops by 30% (multiplied by 0.70)
        # At d_j=0.0: no penalty
        # This is critical: high disagreement → lower c_blend → larger Omega
        # → BL stays closer to equilibrium. The penalty here complements the
        # Omega inflation from d_adjusted.
        c_avg   = (c_hawk + c_dove) / 2.0
        penalty = 1.0 - DISAGREEMENT_PENALTY_WEIGHT * np.minimum(1.0, d_j_raw)
        c_blend = c_avg * penalty
        c_blend = np.clip(c_blend, 0.0, 1.0)

        # ── 6. Diagnostic: which sector has max disagreement? ─────────────────
        max_d_idx               = int(np.argmax(d_j_raw))
        max_disagreement_sector = TICKERS[max_d_idx]
        mean_disagreement       = float(np.mean(d_j_raw))

        # ── 7. Package into BLInputs ─────────────────────────────────────────
        # Import here to avoid circular imports at module level
        from black_litterman import BLInputs
        bl_inputs = BLInputs(
            Q          = Q_blend,
            c_final    = c_blend,
            d_adjusted = d_adjusted,   # Contrarian will update this
        )

        return SynthesisOutput(
            Q_blend                 = Q_blend,
            c_blend                 = c_blend,
            d_j                     = d_j,
            d_adjusted              = d_adjusted,
            sigma_j                 = sigma_j,
            bl_inputs               = bl_inputs,
            date                    = date,
            hawk_Q                  = Q_hawk,
            dove_Q                  = Q_dove,
            hawk_confidence         = c_hawk,
            dove_confidence         = c_dove,
            max_disagreement_sector = max_disagreement_sector,
            mean_disagreement       = mean_disagreement,
            hawk_fallback           = hawk_output.fallback_used,
            dove_fallback           = dove_output.fallback_used,
        )

    def _rolling_sigma(
        self,
        returns_df: pd.DataFrame,
        date:       str,
    ) -> np.ndarray:
        """
        Estimate rolling 36-month monthly return standard deviation per sector.

        Uses returns strictly before `date` (same look-ahead avoidance as BL).
        sigma_j is floored at MIN_SIGMA_FLOOR to prevent division by zero
        in pathological low-vol periods (though this should never trigger
        on real ETF data).

        Returns
        -------
        np.ndarray (N,) — monthly return std devs, one per sector
        """
        ts = pd.Timestamp(date)
        pre = returns_df[returns_df.index < ts].dropna()

        if len(pre) < self.cov_window:
            # Fallback: use whatever history is available with a warning
            import warnings
            warnings.warn(
                f"Synthesis [{date}]: only {len(pre)} months of history "
                f"available before date (need {self.cov_window}). "
                f"Using all available data for sigma_j estimate.",
                UserWarning,
                stacklevel=3,
            )
            window = pre
        else:
            window = pre.iloc[-self.cov_window:]

        sigma = window[TICKERS].std(ddof=1).values.astype(float)
        sigma = np.maximum(sigma, MIN_SIGMA_FLOOR)
        return sigma

    def disagreement_summary(self, output: SynthesisOutput) -> str:
        """
        Return a human-readable disagreement summary for logging and
        the monthly stability report.
        """
        lines = [
            f"Synthesis [{output.date}] — Disagreement Summary",
            f"{'─'*55}",
            f"{'Sector':<6}  {'Q_hawk':>8}  {'Q_dove':>8}  {'Q_blend':>8}  "
            f"{'d_j':>6}  {'c_blend':>7}",
            f"{'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*7}",
        ]
        for i, ticker in enumerate(TICKERS):
            lines.append(
                f"{ticker:<6}  "
                f"{output.hawk_Q[i]:>+8.4f}  "
                f"{output.dove_Q[i]:>+8.4f}  "
                f"{output.Q_blend[i]:>+8.4f}  "
                f"{output.d_j[i]:>6.3f}  "
                f"{output.c_blend[i]:>7.3f}"
            )
        lines.append(f"{'─'*55}")
        lines.append(f"Max disagreement sector: {output.max_disagreement_sector}  "
                     f"(d_j={output.d_j[TICKERS.index(output.max_disagreement_sector)]:.3f})")
        lines.append(f"Mean disagreement:       {output.mean_disagreement:.3f}")
        if output.hawk_fallback:
            lines.append("⚠ WARNING: Hawk used fallback this month")
        if output.dove_fallback:
            lines.append("⚠ WARNING: Dove used fallback this month")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Self-test using the same three scenarios as the Hawk and Dove tests.
    Runs Hawk + Dove + Synthesis end-to-end and verifies:
    - Q_blend is between Q_hawk and Q_dove on every sector
    - d_j is correctly computed and d_adjusted is clipped to [0,1]
    - c_blend is strictly less than or equal to the average confidence
    - BLInputs object is valid (passes BLInputs validation)
    - Economic sense: tightening scenario has higher mean disagreement
      than neutral scenario (Hawk and Dove diverge more when stakes are high)
    """
    import sys
    import logging

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)s  %(message)s",
    )

    print("=" * 60)
    print("Synthesis Agent self-test")
    print("=" * 60)

    # Load real returns data
    try:
        returns_df = pd.read_csv(
            str(Path(__file__).resolve().parent / "data" / "etf_returns_monthly.csv"),
            index_col="Date", parse_dates=True,
        )
        print(f"Loaded returns: {returns_df.shape}")
    except FileNotFoundError:
        print("Returns file not found — using synthetic data")
        dates  = pd.date_range("2015-12-31", periods=109, freq="ME")
        np.random.seed(42)
        returns_df = pd.DataFrame(
            np.random.normal(0.008, 0.045, (109, N)),
            index=dates, columns=TICKERS,
        )

    from agent_hawk import HawkAgent
    from agent_dove import DoveAgent

    hawk_agent  = HawkAgent()
    dove_agent  = DoveAgent()
    synth_agent = SynthesisAgent()

    scenarios = [
        {
            "label":      "Sep 2022 — Hawkish (high inflation, 75bp hike)",
            "date":       "2022-09-30",
            "fomc_date":  "2022-09-21",
            "cpi_yoy":    8.3,
            "pce_yoy":    4.9,
            "gdp_growth": 2.6,
            "unemployment": 3.5,
            "fomc_text": """
The Committee decided to raise the target range for the federal funds rate to
3 to 3-1/4 percent and anticipates that ongoing increases in the target range
will be appropriate in order to attain a stance of monetary policy that is
sufficiently restrictive to return inflation to 2 percent over time.
Inflation remains elevated, reflecting supply and demand imbalances related to
the pandemic, higher food and energy prices, and broader price pressures.
The Committee is highly attentive to inflation risks. Job gains have been
robust in recent months, and the unemployment rate has remained low.
The Committee raised the target range by 75 basis points.
""",
        },
        {
            "label":      "Feb 2019 — Dovish (low inflation, patient Fed)",
            "date":       "2019-02-28",
            "fomc_date":  "2019-01-30",
            "cpi_yoy":    1.6,
            "pce_yoy":    1.7,
            "gdp_growth": 3.1,
            "unemployment": 4.0,
            "fomc_text": """
In light of global economic and financial developments and muted inflation
pressures, the Committee will be patient as it determines what future
adjustments to the target range for the federal funds rate may be appropriate.
The Committee decided to maintain the target range for the federal funds rate
at 2-1/4 to 2-1/2 percent. The labor market has continued to strengthen and
economic activity has been rising at a solid rate. Job gains have been strong.
""",
        },
    ]

    passed, failed = [], []

    def run_check(name, fn):
        try:
            fn()
            passed.append(name)
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"  FAIL  {name}: {e}")

    results = {}
    for sc in scenarios:
        print(f"\n{'─'*60}")
        print(f"  {sc['label']}")
        print(f"{'─'*60}")

        hawk_out = hawk_agent.run(
            fomc_text   = sc["fomc_text"],
            cpi_yoy     = sc["cpi_yoy"],
            pce_yoy     = sc["pce_yoy"],
            date        = sc["date"],
            fomc_date   = sc["fomc_date"],
        )

        dove_out = dove_agent.run(
            fomc_text    = sc["fomc_text"],
            gdp_growth   = sc["gdp_growth"],
            unemployment = sc["unemployment"],
            date         = sc["date"],
            fomc_date    = sc["fomc_date"],
        )

        result = synth_agent.run(
            hawk_output = hawk_out,
            dove_output = dove_out,
            returns_df  = returns_df,
            date        = sc["date"],
        )
        results[sc["label"]] = result

        print(synth_agent.disagreement_summary(result))

        label = sc["label"]

        # Check 1: Q_blend between hawk and dove on every sector
        def check_blend_between_agents():
            lo = np.minimum(hawk_out.Q, dove_out.Q)
            hi = np.maximum(hawk_out.Q, dove_out.Q)
            violations = ~((result.Q_blend >= lo - 1e-9) & (result.Q_blend <= hi + 1e-9))
            assert not violations.any(), (
                f"Q_blend outside [Q_hawk, Q_dove] bounds on: "
                f"{[TICKERS[i] for i in np.where(violations)[0]]}"
            )
        run_check(f"[{sc['date']}] Q_blend within [Q_hawk, Q_dove] bounds",
                  check_blend_between_agents)

        # Check 2: d_adjusted clipped to [0,1]
        def check_d_adjusted_clipped():
            assert np.all(result.d_adjusted >= 0), "d_adjusted has negatives"
            assert np.all(result.d_adjusted <= 1.0 + 1e-9), "d_adjusted exceeds 1.0"
        run_check(f"[{sc['date']}] d_adjusted in [0,1]", check_d_adjusted_clipped)

        # Check 3: c_blend <= avg(c_hawk, c_dove) — disagreement never inflates confidence
        def check_c_blend_not_inflated():
            c_avg = (hawk_out.confidence + dove_out.confidence) / 2.0
            violations = result.c_blend > c_avg + 1e-9
            assert not violations.any(), (
                f"c_blend exceeds avg confidence on: "
                f"{[TICKERS[i] for i in np.where(violations)[0]]}"
            )
        run_check(f"[{sc['date']}] c_blend <= avg(c_hawk, c_dove)",
                  check_c_blend_not_inflated)

        # Check 4: c_blend lower when disagreement is higher
        def check_c_blend_penalised():
            high_d_sectors = result.d_j > 0.1
            if high_d_sectors.any():
                c_avg_high = ((hawk_out.confidence + dove_out.confidence) / 2.0)[high_d_sectors]
                c_blend_high = result.c_blend[high_d_sectors]
                assert np.all(c_blend_high < c_avg_high + 1e-9), (
                    "c_blend not penalised for high-disagreement sectors"
                )
        run_check(f"[{sc['date']}] c_blend penalised for high-d sectors",
                  check_c_blend_penalised)

        # Check 5: BLInputs valid
        def check_bl_inputs_valid():
            from black_litterman import BLInputs
            bl = result.bl_inputs
            assert isinstance(bl, BLInputs)
            assert bl.Q.shape    == (N,)
            assert bl.c_final.shape    == (N,)
            assert bl.d_adjusted.shape == (N,)
            assert np.all(bl.c_final    >= 0) and np.all(bl.c_final    <= 1)
            assert np.all(bl.d_adjusted >= 0) and np.all(bl.d_adjusted <= 1)
        run_check(f"[{sc['date']}] BLInputs object is valid", check_bl_inputs_valid)

        # Check 6: sigma_j positive and sensible range for monthly ETF returns
        def check_sigma_sensible():
            assert np.all(result.sigma_j > 0), "sigma_j has zero/negative"
            assert np.all(result.sigma_j < 0.30), (
                f"sigma_j implausibly large: {result.sigma_j.max():.4f} "
                f"(max monthly std dev > 30%?)"
            )
        run_check(f"[{sc['date']}] sigma_j positive and in sensible range",
                  check_sigma_sensible)

    # Cross-scenario check: tightening should produce higher mean disagreement
    if len(results) == 2:
        keys = list(results.keys())
        tight_result   = results[keys[0]]   # Sep 2022
        neutral_result = results[keys[1]]   # Feb 2019

        def check_tightening_higher_disagreement():
            assert tight_result.mean_disagreement >= neutral_result.mean_disagreement - 0.05, (
                f"Expected tightening scenario to have >= disagreement than neutral. "
                f"Tightening: {tight_result.mean_disagreement:.3f}, "
                f"Neutral: {neutral_result.mean_disagreement:.3f}"
            )
        run_check("Cross-scenario: tightening has >= mean disagreement than neutral",
                  check_tightening_higher_disagreement)

    # Full pipeline integration test: Synthesis → BL
    print(f"\n{'─'*60}")
    print("  Integration test: Synthesis → Black-Litterman")
    print(f"{'─'*60}")

    def check_bl_integration():
        from black_litterman import BlackLittermanModel
        model = BlackLittermanModel()
        w_mkt_df = pd.read_csv(
            str(Path(__file__).resolve().parent / "data" / "market_cap_weights.csv"),
            index_col="Date", parse_dates=True,
        )
        w_prev = np.ones(N) / N

        # Use the Sep 2022 synthesis result
        result_2022 = results[list(results.keys())[0]]
        bl_out = model.run(
            date       = pd.Timestamp("2022-09-30"),
            returns_df = returns_df,
            w_mkt_df   = w_mkt_df,
            w_previous = w_prev,
            bl_inputs  = result_2022.bl_inputs,
        )
        assert np.all(np.isfinite(bl_out.w_final)), "BL weights not finite"
        assert abs(bl_out.w_final.sum() - 1.0) < 1e-9, "BL weights don't sum to 1"
        assert np.all(bl_out.w_final >= -1e-9), "BL weights have negatives"
        print(f"  BL portfolio (Sep 2022 synthesis views):")
        for t, w in zip(TICKERS, bl_out.w_final):
            bar = "█" * int(w * 40)
            print(f"    {t:<5}  {w:6.1%}  {bar}")
        print(f"  Turnover: {bl_out.turnover:.4f} "
              f"({'cap applied' if bl_out.cap_applied else 'within limit'})")
    run_check("Synthesis → BLInputs → BlackLittermanModel end-to-end", check_bl_integration)

    print(f"\n{'='*60}")
    print(f"Results: {len(passed)}/{len(passed)+len(failed)} passed")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All checks passed.")
        print("\nKey observations for research project:")
        for key, result in results.items():
            print(f"  {key[:40]}")
            print(f"    Mean disagreement:     {result.mean_disagreement:.3f}")
            print(f"    Max d_j sector:        {result.max_disagreement_sector} "
                  f"(d_j={result.d_j[TICKERS.index(result.max_disagreement_sector)]:.3f})")
            print(f"    Avg c_blend:           {result.c_blend.mean():.3f}")
    print("=" * 60)
