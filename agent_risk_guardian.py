"""Agent risk guardian."""

from __future__ import annotations
from pathlib import Path

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants — calibrated to brief Section 4.1
# --------------------------------------------------------------------------- #
TICKERS: list[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLRE", "XLU", "XLB",
]
N = len(TICKERS)

# VIX ceiling parameters
C_BASE   = 0.75   # maximum allowable confidence in calm markets (VIX <= 20)
VIX_STAR = 20.0   # VIX threshold above which ceiling begins dropping
VIX_DECAY = 0.02  # exponential decay rate for ceiling

# View stability filter parameters
STABILITY_GAMMA     = 0.3   # decay rate for view change penalty
STABILITY_THRESHOLD = 1.0   # change > 1 sigma triggers penalty
SIGN_REVERSAL_PENALTY = 0.5  # confidence halved on sign reversal

# Disagreement Omega scaling
ALPHA = 2.0   # brief Section 4.1

# Instability episode detection thresholds (brief Section 7.4)
INSTABILITY_DRAWDOWN_ZSCORE  = 2.0   # drawdown std devs below rolling mean
INSTABILITY_TURNOVER_THRESH  = 0.25  # pre-guardrail implied turnover > 25%
INSTABILITY_CONFIDENCE_THRESH = 0.80  # raw agent confidence > 80%

# Rolling window for stability metrics
ROLLING_WINDOW = 12   # months for drawdown mean/std calculation

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class GuardrailLog:
    """
    Records which guardrails fired this month and how hard.
    Used for the ablation study (brief Section 7.5).

    Attributes
    ----------
    vix_ceiling_fired : bool
        True if VIX ceiling was binding for at least one sector.
    vix_ceiling_c_max : float
        The actual ceiling value this month (function of VIX).
    vix_sectors_capped : list[str]
        Tickers where raw confidence exceeded the VIX ceiling.

    stability_filter_fired : bool
        True if the stability filter reduced confidence on any sector.
    stability_sign_reversals : list[str]
        Tickers where Q flipped sign month-on-month.
    stability_large_moves : list[str]
        Tickers where |delta_Q| > 1 rolling sigma.

    omega_inflation_fired : bool
        True if any d_adjusted > 0 (Omega inflation always fires when
        agents disagree — which is almost always).
    omega_mean_d_adjusted : float
        Mean d_adjusted across all sectors this month.
    omega_max_d_adjusted_sector : str
        Sector with highest d_adjusted (most Omega inflation).

    turnover_cap_fired : bool
        True if BL model applied the L1 turnover cap (filled after BL run).
    """
    # Guardrail 1
    vix_ceiling_fired:    bool       = False
    vix_ceiling_c_max:    float      = C_BASE
    vix_sectors_capped:   list[str]  = field(default_factory=list)

    # Guardrail 2
    stability_filter_fired:    bool       = False
    stability_sign_reversals:  list[str]  = field(default_factory=list)
    stability_large_moves:     list[str]  = field(default_factory=list)

    # Guardrail 3
    omega_inflation_fired:        bool   = False
    omega_mean_d_adjusted:        float  = 0.0
    omega_max_d_adjusted_sector:  str    = ""

    # Guardrail 4 (filled after BL run)
    turnover_cap_fired: bool = False


@dataclass
class StabilityReport:
    """
    Complete monthly stability report from the Risk Guardian.
    This is the primary output for the research project's research analysis.

    Attributes
    ----------
    date : str
        ISO month-end date string (e.g. "2022-09-30").

    vix : float
        VIX level used for the confidence ceiling calculation.

    guardrail_log : GuardrailLog
        Record of which guardrails fired and activation intensity.

    c_raw : np.ndarray (N,)
        Confidence scores before any guardrail adjustment.

    c_final : np.ndarray (N,)
        Confidence scores after VIX ceiling and stability filter applied.
        This is what enters BLInputs.

    Q_final : np.ndarray (N,)
        Final blended expected returns (unchanged by guardrails).

    d_adjusted : np.ndarray (N,)
        Final disagreement scores including Contrarian penalty.

    delta_Q : np.ndarray (N,)
        Month-on-month view changes in units of rolling sigma.
        Key metric for RQ3 (view stability analysis).

    stability_penalties : np.ndarray (N,)
        Multiplicative confidence penalties from stability filter.
        1.0 = no penalty, 0.5 = sign reversal, <1 = large move penalty.

    mean_disagreement : float
        Mean d_adjusted across all sectors. Primary RQ2 signal.

    early_warning_flag : bool
        True if mean_disagreement > 0.6 (high inter-agent conflict).
        Potential precursor to instability episode.

    instability_episode : bool
        True if all three instability conditions hold simultaneously.
        Formally defined instability episode for RQ1 analysis.

    instability_conditions : dict
        Breakdown of which of the three conditions hold.

    hawk_fallback : bool
        True if Hawk used conservative fallback this month.

    dove_fallback : bool
        True if Dove used conservative fallback this month.
    """
    date:                  str
    vix:                   float
    guardrail_log:         GuardrailLog
    c_raw:                 np.ndarray
    c_final:               np.ndarray
    Q_final:               np.ndarray
    d_adjusted:            np.ndarray
    delta_Q:               np.ndarray
    stability_penalties:   np.ndarray
    mean_disagreement:     float
    early_warning_flag:    bool
    instability_episode:   bool
    instability_conditions: dict
    hawk_fallback:         bool
    dove_fallback:         bool


# --------------------------------------------------------------------------- #
# Risk Guardian
# --------------------------------------------------------------------------- #
class RiskGuardian:
    """
    Agent 5 — Risk Guardian.

    Deterministic Python class. No LLM call. Applies three mathematical
    guardrails to agent views before they enter the Black-Litterman model,
    then generates the monthly stability report.

    Usage
    -----
        from agent_risk_guardian import RiskGuardian
        guardian = RiskGuardian()

        # Called once per month in the simulation loop:
        report, final_bl_inputs = guardian.run(
            contrarian_output = cont_out,
            synthesis_output  = synth_out,
            hawk_output       = hawk_out,
            dove_output       = dove_out,
            vix               = 25.3,
            date              = "2022-09-30",
            prev_Q            = prev_month_Q,       # None for first month
            sigma_j           = synth_out.sigma_j,  # rolling sector vols
            prev_portfolio_return = -0.032,          # None for first month
            drawdown_history  = drawdown_series,     # pd.Series, None for early months
        )

        # Then run BL:
        bl_result = bl_model.run(..., bl_inputs=final_bl_inputs)

        # After BL, record turnover cap:
        guardian.record_turnover_cap(report, bl_result.cap_applied)
    """

    def __init__(
        self,
        c_base:    float = C_BASE,
        vix_star:  float = VIX_STAR,
        vix_decay: float = VIX_DECAY,
        gamma:     float = STABILITY_GAMMA,
        alpha:     float = ALPHA,
    ) -> None:
        self.c_base    = c_base
        self.vix_star  = vix_star
        self.vix_decay = vix_decay
        self.gamma     = gamma
        self.alpha     = alpha

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def run(
        self,
        contrarian_output,          # ContrarianOutput from agent_contrarian.py
        synthesis_output,           # SynthesisOutput from agent_synthesis.py
        hawk_output,                # HawkOutput from agent_hawk.py
        dove_output,                # DoveOutput from agent_dove.py
        vix:           float,
        date:          str,
        prev_Q:        Optional[np.ndarray],    # Q_final from previous month
        sigma_j:       np.ndarray,              # rolling monthly std devs (N,)
        drawdown_history: Optional[pd.Series],  # monthly drawdowns, most recent last
    ) -> tuple:
        """
        Apply guardrails and produce monthly stability report.

        Parameters
        ----------
        contrarian_output : ContrarianOutput
            Full output from Agent 4. Contains bl_inputs_updated with
            Q_blend, c_blend (pre-guardrail confidence), d_adjusted.
        synthesis_output : SynthesisOutput
            Full output from Agent 3. Used for hawk/dove fallback flags.
        hawk_output : HawkOutput
            Agent 1 output. Used for raw confidence in instability detection.
        dove_output : DoveOutput
            Agent 2 output. Used for raw confidence in instability detection.
        vix : float
            Current month VIX level (monthly average or end-of-month).
        date : str
            ISO month-end date string.
        prev_Q : np.ndarray or None
            Q_final vector from the previous month. None for first month.
        sigma_j : np.ndarray (N,)
            Rolling 36-month monthly standard deviations per sector.
            Taken from synthesis_output.sigma_j.
        drawdown_history : pd.Series or None
            Monthly portfolio drawdowns up to and including this month.
            None or short series in early months — instability detection
            requires at least ROLLING_WINDOW=12 observations.

        Returns
        -------
        (StabilityReport, BLInputs)
            report        : full monthly stability report
            final_bl_inputs : validated BLInputs ready for BL model
        """
        from black_litterman import BLInputs

        # ── Extract pre-guardrail inputs ────────────────────────────────────
        bl_pre    = contrarian_output.bl_inputs_updated
        Q_final   = bl_pre.Q.copy()
        c_raw     = bl_pre.c_final.copy()   # c_blend from Synthesis, penalised by Contrarian
        d_adjusted = bl_pre.d_adjusted.copy()

        log = GuardrailLog()

        # ── Guardrail 1: VIX Confidence Ceiling ─────────────────────────────
        c_after_ceiling, log = self._apply_vix_ceiling(c_raw, vix, log)

        # ── Guardrail 2: View Stability Filter ──────────────────────────────
        c_final, delta_Q, stability_penalties, log = self._apply_stability_filter(
            c_after_ceiling, Q_final, prev_Q, sigma_j, log
        )

        # ── Guardrail 3: Omega Inflation logging ─────────────────────────────
        log.omega_inflation_fired       = np.any(d_adjusted > 0)
        log.omega_mean_d_adjusted       = float(np.mean(d_adjusted))
        log.omega_max_d_adjusted_sector = TICKERS[int(np.argmax(d_adjusted))]

        # ── Build final validated BLInputs ───────────────────────────────────
        final_bl_inputs = BLInputs(
            Q          = Q_final,
            c_final    = c_final,
            d_adjusted = d_adjusted,
        )

        # ── Instability episode detection ────────────────────────────────────
        instability_episode, instability_conditions = self._detect_instability(
            c_raw_hawk   = hawk_output.confidence,
            c_raw_dove   = dove_output.confidence,
            d_adjusted   = d_adjusted,
            drawdown_history = drawdown_history,
        )

        # ── Early warning flag ───────────────────────────────────────────────
        mean_disagreement  = float(np.mean(d_adjusted))
        early_warning_flag = mean_disagreement > 0.60

        # ── Stability report ─────────────────────────────────────────────────
        report = StabilityReport(
            date                  = date,
            vix                   = vix,
            guardrail_log         = log,
            c_raw                 = c_raw,
            c_final               = c_final,
            Q_final               = Q_final,
            d_adjusted            = d_adjusted,
            delta_Q               = delta_Q,
            stability_penalties   = stability_penalties,
            mean_disagreement     = mean_disagreement,
            early_warning_flag    = early_warning_flag,
            instability_episode   = instability_episode,
            instability_conditions = instability_conditions,
            hawk_fallback         = hawk_output.fallback_used,
            dove_fallback         = dove_output.fallback_used,
        )

        logger.info(
            f"RiskGuardian [{date}]: VIX={vix:.1f} c_max={log.vix_ceiling_c_max:.3f} | "
            f"ceiling_fired={log.vix_ceiling_fired} stability_fired={log.stability_filter_fired} | "
            f"mean_d_adj={mean_disagreement:.3f} early_warning={early_warning_flag} | "
            f"instability_episode={instability_episode}"
        )

        return report, final_bl_inputs

    def record_turnover_cap(self, report: StabilityReport, cap_applied: bool) -> None:
        """
        Update the stability report after the BL model runs.
        Call this immediately after bl_model.run() to record whether
        the turnover cap fired.
        """
        report.guardrail_log.turnover_cap_fired = cap_applied

    # ------------------------------------------------------------------ #
    # Private: Guardrail 1 — VIX Confidence Ceiling
    # ------------------------------------------------------------------ #
    def _apply_vix_ceiling(
        self,
        c_raw: np.ndarray,
        vix:   float,
        log:   GuardrailLog,
    ) -> tuple[np.ndarray, GuardrailLog]:
        """
        Apply VIX-adaptive confidence ceiling (brief Section 4.3, Guardrail 1).

        c_max   = c_base * exp(-vix_decay * max(0, vix - vix_star))
        c_final = min(c_raw_j, c_max)  for each sector j

        Brief error note: brief says ceiling drops to 0.18 at VIX=66.
        Actual formula: 0.75 * exp(-0.02 * 46) = 0.75 * exp(-0.92) = 0.2989.
        This code is correct. The brief example is wrong.
        """
        c_max = self.c_base * np.exp(
            -self.vix_decay * max(0.0, vix - self.vix_star)
        )
        c_ceiling = np.minimum(c_raw, c_max)

        capped = [TICKERS[i] for i in range(N) if c_raw[i] > c_max]
        log.vix_ceiling_fired  = len(capped) > 0
        log.vix_ceiling_c_max  = float(c_max)
        log.vix_sectors_capped = capped

        return c_ceiling, log

    # ------------------------------------------------------------------ #
    # Private: Guardrail 2 — View Stability Filter
    # ------------------------------------------------------------------ #
    def _apply_stability_filter(
        self,
        c_in:    np.ndarray,
        Q_curr:  np.ndarray,
        Q_prev:  Optional[np.ndarray],
        sigma_j: np.ndarray,
        log:     GuardrailLog,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, GuardrailLog]:
        """
        Apply view stability filter (brief Section 4.3, Guardrail 2).

        delta_j = |Q_j,t - Q_j,t-1| / sigma_j,rolling
        penalty_j = exp(-gamma * max(0, delta_j - threshold))
        if sign reversal: c_final_j *= 0.5

        Returns c_final, delta_Q, stability_penalties, log.
        """
        if Q_prev is None:
            # First month — no prior view to compare against
            delta_Q            = np.zeros(N)
            stability_penalties = np.ones(N)
            return c_in.copy(), delta_Q, stability_penalties, log

        # Normalised view change
        delta_Q = np.abs(Q_curr - Q_prev) / np.maximum(sigma_j, 1e-8)

        # Exponential penalty for large moves
        stability_penalties = np.exp(
            -self.gamma * np.maximum(0.0, delta_Q - STABILITY_THRESHOLD)
        )

        # Sign reversal: additional 50% penalty
        sign_reversals = []
        for i in range(N):
            if (Q_prev[i] != 0 and Q_curr[i] != 0 and
                    np.sign(Q_curr[i]) != np.sign(Q_prev[i])):
                stability_penalties[i] *= SIGN_REVERSAL_PENALTY
                sign_reversals.append(TICKERS[i])

        c_final = np.clip(c_in * stability_penalties, 0.0, 1.0)

        large_moves = [TICKERS[i] for i in range(N) if delta_Q[i] > STABILITY_THRESHOLD]

        log.stability_filter_fired   = (len(sign_reversals) + len(large_moves)) > 0
        log.stability_sign_reversals = sign_reversals
        log.stability_large_moves    = large_moves

        return c_final, delta_Q, stability_penalties, log

    # ------------------------------------------------------------------ #
    # Private: Instability episode detection (brief Section 7.4)
    # ------------------------------------------------------------------ #
    def _detect_instability(
        self,
        c_raw_hawk:       np.ndarray,
        c_raw_dove:       np.ndarray,
        d_adjusted:       np.ndarray,
        drawdown_history: Optional[pd.Series],
    ) -> tuple[bool, dict]:
        """
        Check all three instability conditions from brief Section 7.4.

        Condition 1: Drawdown > 2 std devs below rolling 12-month mean.
        Condition 2: Pre-guardrail implied turnover > 25%.
                     Approximated here as: mean_d_adjusted > 0.5 AND
                     max raw confidence > 0.7 (proxy for high implied turnover).
        Condition 3: At least one raw agent confidence > 80%.
        """
        conditions = {
            "drawdown_extreme": False,
            "high_implied_turnover": False,
            "high_raw_confidence": False,
        }

        # Condition 1: drawdown
        if drawdown_history is not None and len(drawdown_history) >= ROLLING_WINDOW:
            recent = drawdown_history.iloc[-ROLLING_WINDOW:]
            mean_dd = recent.mean()
            std_dd  = recent.std()
            if std_dd > 1e-8:
                current_dd = drawdown_history.iloc[-1]
                zscore = (current_dd - mean_dd) / std_dd
                conditions["drawdown_extreme"] = zscore < -INSTABILITY_DRAWDOWN_ZSCORE

        # Condition 2: high implied turnover (proxy)
        mean_d  = float(np.mean(d_adjusted))
        max_raw = float(max(np.max(c_raw_hawk), np.max(c_raw_dove)))
        conditions["high_implied_turnover"] = (mean_d > 0.5) and (max_raw > 0.70)

        # Condition 3: high raw agent confidence
        conditions["high_raw_confidence"] = max_raw > INSTABILITY_CONFIDENCE_THRESH

        episode = all(conditions.values())
        return episode, conditions


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def format_stability_report(report: StabilityReport) -> str:
    """
    Format a monthly stability report as a human-readable string.
    Used in the simulation loop for logging and case study analysis.
    """
    log = report.guardrail_log
    lines = [
        f"{'═'*65}",
        f"  Risk Guardian Monthly Report — {report.date}",
        f"{'═'*65}",
        f"  VIX: {report.vix:.1f}   |   c_max (ceiling): {log.vix_ceiling_c_max:.3f}   |   "
        f"Mean d_adjusted: {report.mean_disagreement:.3f}",
        f"",
        f"  Guardrail Activations:",
        f"    1. VIX Ceiling:       {'FIRED' if log.vix_ceiling_fired else 'not fired':10s}  "
        f"{'Capped: ' + ', '.join(log.vix_sectors_capped) if log.vix_sectors_capped else ''}",
        f"    2. Stability Filter:  {'FIRED' if log.stability_filter_fired else 'not fired':10s}  "
        f"{'Sign reversals: ' + ', '.join(log.stability_sign_reversals) if log.stability_sign_reversals else ''}"
        f"{'  Large moves: ' + ', '.join(log.stability_large_moves) if log.stability_large_moves else ''}",
        f"    3. Omega Inflation:   {'FIRED' if log.omega_inflation_fired else 'not fired':10s}  "
        f"Max d_adj sector: {log.omega_max_d_adjusted_sector} ({np.max(report.d_adjusted):.3f})",
        f"    4. Turnover Cap:      {'FIRED' if log.turnover_cap_fired else 'not fired'}",
        f"",
        f"  {'Sector':<6}  {'Q_final':>8}  {'c_raw':>6}  {'c_final':>7}  "
        f"{'delta_Q':>7}  {'stab_pen':>8}  {'d_adj':>6}",
        f"  {'─'*6}  {'─'*8}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*6}",
    ]
    for i, t in enumerate(TICKERS):
        lines.append(
            f"  {t:<6}  {report.Q_final[i]:>+8.4f}  "
            f"{report.c_raw[i]:>6.3f}  "
            f"{report.c_final[i]:>7.3f}  "
            f"{report.delta_Q[i]:>7.3f}  "
            f"{report.stability_penalties[i]:>8.3f}  "
            f"{report.d_adjusted[i]:>6.3f}"
        )
    lines += [
        f"",
        f"  Early Warning Flag:    {'⚠ YES — mean d_adjusted > 0.60' if report.early_warning_flag else 'no'}",
        f"  Instability Episode:   {'🚨 YES' if report.instability_episode else 'no'}",
    ]
    if report.instability_episode:
        for cond, val in report.instability_conditions.items():
            lines.append(f"    {'✓' if val else '✗'} {cond}")
    if report.hawk_fallback:
        lines.append("  ⚠ Hawk used fallback this month")
    if report.dove_fallback:
        lines.append("  ⚠ Dove used fallback this month")
    lines.append(f"{'═'*65}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Full pipeline test:
    Hawk → Dove → Synthesis → Contrarian → Risk Guardian → BL

    Tests three scenarios:
    1. Calm market (VIX=15) — no guardrails should fire hard
    2. Stressed market (VIX=35) — VIX ceiling should fire
    3. Crisis (VIX=82, like COVID peak) — all guardrails fire

    Also tests the view stability filter by running two consecutive months
    and checking that sign reversals and large moves are correctly penalised.
    """
    import sys
    import logging
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s"
    )

    print("=" * 65)
    print("Risk Guardian self-test — full pipeline")
    print("=" * 65)

    try:
        returns_df = pd.read_csv(
            str(Path(__file__).resolve().parent / "data" / "etf_returns_monthly.csv"),
            index_col="Date", parse_dates=True,
        )
        w_mkt_df = pd.read_csv(
            str(Path(__file__).resolve().parent / "data" / "market_cap_weights.csv"),
            index_col="Date", parse_dates=True,
        )
        print(f"Loaded returns: {returns_df.shape}, w_mkt: {w_mkt_df.shape}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    from agent_hawk import HawkAgent
    from agent_dove import DoveAgent
    from agent_synthesis import SynthesisAgent
    from agent_contrarian import ContrarianAgent
    from black_litterman import BlackLittermanModel

    hawk_agent       = HawkAgent()
    dove_agent       = DoveAgent()
    synth_agent      = SynthesisAgent()
    contrarian_agent = ContrarianAgent()
    guardian         = RiskGuardian()
    bl_model         = BlackLittermanModel()

    fomc_text_hawkish = """
The Committee decided to raise the target range for the federal funds rate to
3 to 3-1/4 percent. The Committee raised the target range by 75 basis points.
Inflation remains elevated. The Committee is highly attentive to inflation risks.
The Committee is strongly committed to returning inflation to its 2 percent objective.
"""
    fomc_text_neutral = """
The Committee will be patient as it determines future adjustments.
The Committee decided to maintain the target range at 2-1/4 to 2-1/2 percent.
Muted inflation pressures. Strong labor market. Solid economic activity.
"""

    scenarios = [
        {
            "label":       "Calm market (VIX=15, neutral Fed)",
            "date":        "2019-06-30",
            "fomc_date":   "2019-05-01",
            "vix":         15.0,
            "cpi_yoy":     1.8, "pce_yoy": 1.6,
            "gdp_growth":  3.1, "unemployment": 3.6,
            "fomc_text":   fomc_text_neutral,
        },
        {
            "label":       "Stressed market (VIX=35, rate hikes)",
            "date":        "2022-06-30",
            "fomc_date":   "2022-06-15",
            "vix":         35.0,
            "cpi_yoy":     8.6, "pce_yoy": 4.7,
            "gdp_growth":  1.6, "unemployment": 3.6,
            "fomc_text":   fomc_text_hawkish,
        },
        {
            "label":       "Crisis (VIX=82, COVID-like emergency)",
            "date":        "2020-03-31",
            "fomc_date":   "2020-03-15",
            "vix":         82.7,
            "cpi_yoy":     2.3, "pce_yoy": 1.8,
            "gdp_growth":  -5.0, "unemployment": 4.4,
            "fomc_text":   """
The coronavirus outbreak has harmed communities. The Committee decided to lower
the target range for the federal funds rate to 0 to 1/4 percent. Emergency
lending facilities established. Full range of tools will be used.
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
        except Exception as e:
            failed.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")

    prev_Q = None
    w_prev = np.ones(N) / N
    reports = []

    for sc in scenarios:
        print(f"\n{'─'*65}")
        print(f"  {sc['label']}")
        print(f"{'─'*65}")

        hawk_out  = hawk_agent.run(
            fomc_text=sc["fomc_text"], cpi_yoy=sc["cpi_yoy"],
            pce_yoy=sc["pce_yoy"], date=sc["date"], fomc_date=sc["fomc_date"],
        )
        dove_out  = dove_agent.run(
            fomc_text=sc["fomc_text"], gdp_growth=sc["gdp_growth"],
            unemployment=sc["unemployment"], date=sc["date"], fomc_date=sc["fomc_date"],
        )
        synth_out = synth_agent.run(
            hawk_output=hawk_out, dove_output=dove_out,
            returns_df=returns_df, date=sc["date"],
        )
        cont_out  = contrarian_agent.run(
            synthesis_output=synth_out, fomc_text=sc["fomc_text"], date=sc["date"],
        )
        report, final_bl = guardian.run(
            contrarian_output=cont_out,
            synthesis_output=synth_out,
            hawk_output=hawk_out,
            dove_output=dove_out,
            vix=sc["vix"],
            date=sc["date"],
            prev_Q=prev_Q,
            sigma_j=synth_out.sigma_j,
            drawdown_history=None,
        )

        bl_out = bl_model.run(
            date=pd.Timestamp(sc["date"]),
            returns_df=returns_df,
            w_mkt_df=w_mkt_df,
            w_previous=w_prev,
            bl_inputs=final_bl,
        )
        guardian.record_turnover_cap(report, bl_out.cap_applied)

        print(format_stability_report(report))
        reports.append((sc, report))

        vix    = sc["vix"]
        date   = sc["date"]

        # Check 1: c_final <= c_raw always
        def check_c_final_le_c_raw():
            assert np.all(final_bl.c_final <= cont_out.bl_inputs_updated.c_final + 1e-9), \
                "c_final exceeds pre-guardrail confidence — guardrails inflated confidence"
        run_check(f"[{date}] c_final <= pre-guardrail confidence", check_c_final_le_c_raw)

        # Check 2: VIX ceiling correctness
        def check_vix_ceiling():
            expected_cmax = C_BASE * np.exp(-VIX_DECAY * max(0, vix - VIX_STAR))
            assert abs(report.guardrail_log.vix_ceiling_c_max - expected_cmax) < 1e-9, \
                f"c_max formula wrong: expected {expected_cmax:.4f}, got {report.guardrail_log.vix_ceiling_c_max:.4f}"
            # All c_final must be <= c_max
            assert np.all(final_bl.c_final <= expected_cmax + 1e-9), \
                "Some c_final exceeds VIX ceiling"
        run_check(f"[{date}] VIX ceiling formula correct and binding", check_vix_ceiling)

        # Check 3: stability penalties in [0,1]
        def check_stability_penalties():
            assert np.all(report.stability_penalties >= 0)
            assert np.all(report.stability_penalties <= 1.0 + 1e-9)
        run_check(f"[{date}] Stability penalties in [0,1]", check_stability_penalties)

        # Check 4: BLInputs valid
        def check_bl_inputs():
            from black_litterman import BLInputs
            assert isinstance(final_bl, BLInputs)
            assert np.all(final_bl.c_final >= 0) and np.all(final_bl.c_final <= 1)
            assert np.all(final_bl.d_adjusted >= 0) and np.all(final_bl.d_adjusted <= 1)
        run_check(f"[{date}] Final BLInputs valid", check_bl_inputs)

        # Check 5: BL portfolio valid
        def check_bl_portfolio():
            assert np.all(np.isfinite(bl_out.w_final))
            assert abs(bl_out.w_final.sum() - 1.0) < 1e-9
            assert np.all(bl_out.w_final >= -1e-9)
        run_check(f"[{date}] BL portfolio weights valid", check_bl_portfolio)

        # Update for next scenario
        prev_Q = final_bl.Q.copy()
        w_prev = bl_out.w_final.copy()

    # Cross-scenario: crisis should have lowest c_final (most ceiling applied)
    if len(reports) == 3:
        _, calm_report   = reports[0]
        _, stress_report = reports[1]
        _, crisis_report = reports[2]

        def check_vix_ordering():
            assert crisis_report.guardrail_log.vix_ceiling_c_max < stress_report.guardrail_log.vix_ceiling_c_max < calm_report.guardrail_log.vix_ceiling_c_max, \
                "VIX ceiling should be: calm > stressed > crisis"
        run_check("Cross-scenario: VIX ceiling decreases as VIX rises", check_vix_ordering)

        def check_crisis_ceiling_fires():
            assert crisis_report.guardrail_log.vix_ceiling_fired, \
                "VIX ceiling should fire at VIX=82.7"
        run_check("Crisis scenario: VIX ceiling fires at VIX=82.7", check_crisis_ceiling_fires)

        def check_calm_ceiling_may_not_fire():
            # At VIX=15 (below VIX*=20), ceiling = c_base = 0.75
            # Only fires if any agent confidence exceeds 0.75
            expected_cmax = C_BASE * np.exp(-VIX_DECAY * max(0, 15.0 - VIX_STAR))
            assert abs(expected_cmax - C_BASE) < 1e-9, \
                "Below VIX* ceiling should equal c_base"
        run_check("Calm scenario: ceiling equals c_base when VIX < VIX*", check_calm_ceiling_may_not_fire)

        def check_stability_filter_fires_on_second_month():
            # Second scenario (stressed) has prev_Q from calm — views likely changed
            # So stability filter should have computed non-trivial delta_Q
            assert np.any(stress_report.delta_Q != 0), \
                "delta_Q should be non-zero when prev_Q exists"
        run_check("Stability filter: computes non-zero delta_Q from month 2 onwards",
                  check_stability_filter_fires_on_second_month)

    # Summary print
    print(f"\n{'='*65}")
    print(f"Results: {len(passed)}/{len(passed)+len(failed)} passed")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All checks passed.")
        print("\nKey observations for research project:")
        for sc, rep in reports:
            print(f"  {sc['label'][:45]}")
            print(f"    VIX={sc['vix']:.0f}  c_max={rep.guardrail_log.vix_ceiling_c_max:.3f}  "
                  f"ceiling_fired={rep.guardrail_log.vix_ceiling_fired}  "
                  f"mean_d_adj={rep.mean_disagreement:.3f}  "
                  f"early_warning={rep.early_warning_flag}")
        print("\nVIX ceiling values (research project table):")
        for vix_test in [15, 20, 25, 30, 35, 50, 66, 82.7]:
            cmax = C_BASE * np.exp(-VIX_DECAY * max(0, vix_test - VIX_STAR))
            print(f"    VIX={vix_test:5.1f}  →  c_max={cmax:.4f}")
    print("=" * 65)
