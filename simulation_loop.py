"""Simulation loop."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Module-level logger
# --------------------------------------------------------------------------- #
# BUG FIX: `logger` was previously only ever assigned as a LOCAL variable
# inside setup_logging(), run_simulation(), and the __main__ block -- never
# at module level. load_fomc_text() calls bare logger.warning()/logger.info()
# in five places with no local `logger` of its own, so Python resolved the
# name against the module globals, found nothing, and raised
# NameError: name 'logger' is not defined -- every time load_fomc_text() hit
# one of its intended graceful-fallback warning paths (missing FOMC folder,
# no matching file found, pdfplumber extraction failure). This crash only
# fires on the failure path, so it never triggered during the real 71-month
# runs (every month's FOMC text was always found via a success path), but
# it's a landmine for any future partial rerun or reproduction. The local
# `logger` variables inside setup_logging()/run_simulation()/__main__ still
# shadow this one within their own scope, unchanged -- this only adds the
# missing global that load_fomc_text() (and anything else outside those
# three functions) actually needs.
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

# FOMC text files — one per meeting, stored in year subfolders
FOMC_ROOTS = {
    year: PROJECT_ROOT / f"{year} FO"
    for year in range(2019, 2025)
}

# Simulation window
SIM_START = pd.Timestamp("2019-02-28")
SIM_END   = pd.Timestamp("2024-12-31")

TICKERS: list[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLRE", "XLU", "XLB",
]
N = len(TICKERS)

# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
def setup_logging(results_dir: Path, dry_run: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "simulation_log.txt"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a"),
    ]
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        handlers = handlers,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"{'='*65}")
    logger.info(f"Simulation {'DRY RUN' if dry_run else 'START'} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Window: {SIM_START.date()} → {SIM_END.date()}")
    logger.info(f"{'='*65}")

# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
def load_alignment_table() -> pd.DataFrame:
    """
    Load the master monthly alignment table produced by data_pipeline.py.

    Columns include: month_end, fomc_date, fomc_file, cpi_yoy, pce_yoy,
    gdp_growth, unemployment, vix_close, vix_max, and more.
    """
    path = DATA_DIR / "monthly_alignment.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Monthly alignment table not found at {path}. "
            "Run data_pipeline.py first."
        )
    df = pd.read_csv(path, parse_dates=["month_end"])
    df = df.set_index("month_end").sort_index()
    return df


def load_fomc_text(fomc_file: str, fomc_date: str) -> str:
    """
    Load FOMC minutes text from the correct year subfolder.

    Tries multiple filename patterns to handle both naming conventions:
    - fomcminutes{YYYYMMDD}.txt  (2021-2024)
    - fomcminutes{YYYYMMDD}meeting.txt  (2019-2020)
    - monetary{YYYYMMDD}*.txt  (emergency statements)
    """
    year = int(str(fomc_date)[:4])
    folder = PROJECT_ROOT / f"{year} FO"

    if not folder.exists():
        logger.warning(f"FOMC folder not found: {folder}. Using empty string.")
        return ""

    # Try the exact filename from alignment table first
    if fomc_file:
        # Convert PDF filename to text filename
        txt_name = Path(fomc_file).stem + ".txt"
        txt_path = folder / txt_name
        if txt_path.exists():
            return txt_path.read_text(encoding="utf-8", errors="replace")

    # Fallback: search folder for any file matching the date
    date_str = str(fomc_date).replace("-", "")
    for f in sorted(folder.iterdir()):
        if f.suffix == ".txt" and date_str in f.name:
            return f.read_text(encoding="utf-8", errors="replace")

    # Extract directly from PDF using pdfplumber
    def extract_pdf(pdf_path):
        try:
            import pdfplumber
            parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return "\n".join(parts)
        except Exception as e:
            logger.warning(f"pdfplumber failed on {pdf_path}: {e}")
            return ""

    if fomc_file:
        pdf_path = folder / fomc_file
        if pdf_path.exists():
            logger.info(f"Extracting text from PDF: {fomc_file}")
            text = extract_pdf(pdf_path)
            if text:
                return text

    for f in sorted(folder.iterdir()):
        if f.suffix == ".pdf" and date_str in f.name:
            logger.info(f"Extracting text from PDF: {f.name}")
            text = extract_pdf(f)
            if text:
                return text

    logger.warning(f"No FOMC text found for {fomc_date} in {folder}")
    return ""


def load_returns_and_weights() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ETF monthly returns and market cap weights."""
    returns_df = pd.read_csv(
        DATA_DIR / "etf_returns_monthly.csv",
        index_col="Date", parse_dates=True,
    )
    w_mkt_df = pd.read_csv(
        DATA_DIR / "market_cap_weights.csv",
        index_col="Date", parse_dates=True,
    )
    # Finding B guard (2026-07-22): the BL prior Pi = lambda*Sigma*w_mkt must
    # be anchored by the REAL time-varying index-weight series
    # (build_market_weights_v2.py). A constant-row file means the old static
    # ~2026 snapshot has been regenerated by mistake — refuse to run on it.
    if len(w_mkt_df.drop_duplicates()) <= 1:
        raise RuntimeError(
            "data/market_cap_weights.csv contains a single repeated weight row "
            "(the tainted static snapshot). Rebuild it with "
            "build_market_weights_v2.py before running the simulation."
        )
    return returns_df, w_mkt_df


# --------------------------------------------------------------------------- #
# PnL context builder
# --------------------------------------------------------------------------- #
def build_pnl_context(
    month_idx:       int,
    monthly_returns: list[float],
    monthly_weights: list[np.ndarray],
    dates:           list[str],
    returns_df:      pd.DataFrame,
) -> str:
    """
    Build a structured PnL feedback string for the LLM agents.

    Injects previous month's portfolio return, the best and worst
    performing sectors, and a simple directional assessment.
    Sent to Hawk and Dove prompts for RQ3 (feedback recalibration).
    """
    if month_idx == 0 or len(monthly_returns) == 0:
        return ""   # no feedback on first month

    prev_return = monthly_returns[-1]
    prev_date   = dates[-1]
    prev_w      = monthly_weights[-1]

    # Compute sector returns for previous month
    prev_ts = pd.Timestamp(prev_date)
    sector_rets_row = returns_df[returns_df.index == prev_ts]

    if sector_rets_row.empty:
        direction = "positive" if prev_return > 0 else "negative"
        return (
            f"Last month ({prev_date}): portfolio returned {prev_return*100:+.2f}%. "
            f"Direction was {direction}."
        )

    sector_rets = sector_rets_row.iloc[0][TICKERS].values.astype(float)
    contrib     = prev_w * sector_rets   # weight × return = contribution
    best_idx    = int(np.nanargmax(contrib))
    worst_idx   = int(np.nanargmin(contrib))

    pnl_str = (
        f"Last month ({prev_date}): portfolio returned {prev_return*100:+.2f}%. "
        f"Best contributor: {TICKERS[best_idx]} ({contrib[best_idx]*100:+.2f}%). "
        f"Worst contributor: {TICKERS[worst_idx]} ({contrib[worst_idx]*100:+.2f}%). "
    )
    if prev_return > 0.02:
        pnl_str += "Strong positive month — consider whether views were too conservative."
    elif prev_return < -0.02:
        pnl_str += "Significant drawdown — consider whether views were too aggressive or missed key risks."
    else:
        pnl_str += "Returns were moderate."

    return pnl_str


# --------------------------------------------------------------------------- #
# Drawdown tracker
# --------------------------------------------------------------------------- #
def compute_drawdown(cumulative_returns: list[float]) -> float:
    """
    Compute current portfolio drawdown from peak.
    Returns negative value (e.g. -0.12 for 12% drawdown from peak).
    """
    if len(cumulative_returns) < 2:
        return 0.0
    cum = np.array(cumulative_returns)
    peak = np.maximum.accumulate(cum)
    drawdown_series = (cum - peak) / np.maximum(peak, 1e-8)
    return float(drawdown_series[-1])


# --------------------------------------------------------------------------- #
# Results storage
# --------------------------------------------------------------------------- #
class SimulationResults:
    """Accumulates monthly outputs across the full simulation."""

    def __init__(self) -> None:
        self.dates:            list[str]         = []
        self.weights:          list[np.ndarray]  = []
        # Realized-return series. return_dates holds the CALENDAR MONTH each
        # return was earned in (= decision month + 1); the final decision
        # month has no realized return inside the window, so these four lists
        # can be one entry shorter than self.dates.
        self.return_dates:     list[str]         = []
        self.portfolio_returns: list[float]       = []
        self.cumulative_returns: list[float]      = []
        self.drawdowns:        list[float]        = []

        # Agent outputs (per month)
        self.Q_final:          list[np.ndarray]  = []
        self.c_final:          list[np.ndarray]  = []
        self.d_adjusted:       list[np.ndarray]  = []
        self.d_j:              list[np.ndarray]  = []
        self.hawk_Q:           list[np.ndarray]  = []
        self.dove_Q:           list[np.ndarray]  = []

        # Guardrail logs
        self.vix_levels:            list[float]  = []
        self.ceiling_fired:         list[bool]   = []
        self.stability_fired:       list[bool]   = []
        self.turnover_cap_fired:    list[bool]   = []
        self.mean_disagreement:     list[float]  = []
        self.early_warning_flags:   list[bool]   = []
        self.instability_episodes:  list[bool]   = []

        # Fallback tracking
        self.hawk_fallbacks:   list[bool]        = []
        self.dove_fallbacks:   list[bool]        = []

        # Full objects (for deep analysis)
        self.stability_reports: list             = []
        self.bl_outputs:        list             = []

    def record(
        self,
        date:            str,
        w_final:         np.ndarray,
        port_return:     Optional[float],
        cum_return:      float,
        drawdown:        float,
        synth_out,
        cont_out,
        report,
        bl_out,
        return_date:     Optional[str] = None,
    ) -> None:
        self.dates.append(date)
        self.weights.append(w_final.copy())
        # Return decided at `date` is realized in month `return_date`
        # (= date + 1 month); the final decision month has none.
        if return_date is not None and port_return is not None:
            self.return_dates.append(return_date)
            self.portfolio_returns.append(port_return)
            self.cumulative_returns.append(cum_return)
            self.drawdowns.append(drawdown)

        self.Q_final.append(report.Q_final.copy())
        self.c_final.append(report.c_final.copy())
        self.d_adjusted.append(report.d_adjusted.copy())
        self.d_j.append(synth_out.d_j.copy())
        self.hawk_Q.append(synth_out.hawk_Q.copy())
        self.dove_Q.append(synth_out.dove_Q.copy())

        log = report.guardrail_log
        self.vix_levels.append(report.vix)
        self.ceiling_fired.append(log.vix_ceiling_fired)
        self.stability_fired.append(log.stability_filter_fired)
        self.turnover_cap_fired.append(log.turnover_cap_fired)
        self.mean_disagreement.append(report.mean_disagreement)
        self.early_warning_flags.append(report.early_warning_flag)
        self.instability_episodes.append(report.instability_episode)

        self.hawk_fallbacks.append(report.hawk_fallback)
        self.dove_fallbacks.append(report.dove_fallback)

        self.stability_reports.append(report)
        self.bl_outputs.append(bl_out)

    def save(self, results_dir: Path) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        idx = pd.DatetimeIndex(self.dates, name="Date")

        # Portfolio weights
        pd.DataFrame(
            np.vstack(self.weights), index=idx, columns=TICKERS
        ).round(6).to_csv(results_dir / "monthly_weights.csv")

        # Portfolio returns — indexed by the calendar month each return was
        # actually earned in (decision month + 1), NOT the decision month.
        ridx = pd.DatetimeIndex(self.return_dates, name="Date")
        pd.DataFrame({
            "portfolio_return":    self.portfolio_returns,
            "cumulative_return":   self.cumulative_returns,
            "drawdown":            self.drawdowns,
        }, index=ridx).round(6).to_csv(results_dir / "monthly_returns.csv")

        # Agent views
        pd.DataFrame(
            np.vstack(self.Q_final), index=idx, columns=TICKERS
        ).round(6).to_csv(results_dir / "monthly_views.csv")

        # Confidence
        pd.DataFrame(
            np.vstack(self.c_final), index=idx, columns=TICKERS
        ).round(6).to_csv(results_dir / "monthly_confidence.csv")

        # Disagreement
        pd.DataFrame(
            np.vstack(self.d_adjusted), index=idx, columns=TICKERS
        ).round(6).to_csv(results_dir / "monthly_disagreement.csv")

        # Hawk/Dove raw views
        pd.DataFrame(
            np.vstack(self.hawk_Q), index=idx, columns=[f"hawk_{t}" for t in TICKERS]
        ).round(6).to_csv(results_dir / "monthly_hawk_views.csv")

        pd.DataFrame(
            np.vstack(self.dove_Q), index=idx, columns=[f"dove_{t}" for t in TICKERS]
        ).round(6).to_csv(results_dir / "monthly_dove_views.csv")

        # Raw disagreement (pre-Contrarian)
        pd.DataFrame(
            np.vstack(self.d_j), index=idx, columns=[f"dj_{t}" for t in TICKERS]
        ).round(6).to_csv(results_dir / "monthly_raw_disagreement.csv")

        # Stability summary
        pd.DataFrame({
            "vix":                  self.vix_levels,
            "ceiling_fired":        self.ceiling_fired,
            "stability_fired":      self.stability_fired,
            "turnover_cap_fired":   self.turnover_cap_fired,
            "mean_disagreement":    self.mean_disagreement,
            "early_warning_flag":   self.early_warning_flags,
            "instability_episode":  self.instability_episodes,
            "hawk_fallback":        self.hawk_fallbacks,
            "dove_fallback":        self.dove_fallbacks,
        }, index=idx).to_csv(results_dir / "monthly_stability.csv")

        # Full objects
        with open(results_dir / "agent_outputs.pkl", "wb") as f:
            pickle.dump({
                "stability_reports": self.stability_reports,
                "bl_outputs":        self.bl_outputs,
                "dates":             self.dates,
            }, f)

        logging.getLogger(__name__).info(f"Results saved to {results_dir}/")


# --------------------------------------------------------------------------- #
# Mock agents for dry run
# --------------------------------------------------------------------------- #
class MockHawkOutput:
    Q = np.array([-0.01, 0.005, -0.005, -0.003, -0.008,
                  -0.012, 0.003, -0.018, -0.015, -0.006])
    confidence = np.full(10, 0.55)
    fallback_used = False

class MockDoveOutput:
    Q = np.array([0.015, 0.005, 0.008, 0.003, 0.012,
                   0.018, 0.002, 0.005, 0.003, 0.008])
    confidence = np.full(10, 0.60)
    fallback_used = False


# --------------------------------------------------------------------------- #
# Main simulation loop
# --------------------------------------------------------------------------- #
def run_simulation(
    backend:     str = "qwen2",
    dry_run:     bool = False,
    test_month:  Optional[str] = None,
    resume_from: Optional[str] = None,
) -> SimulationResults:

    logger = logging.getLogger(__name__)

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading data files...")
    alignment   = load_alignment_table()
    returns_df, w_mkt_df = load_returns_and_weights()

    # ── Build simulation month index ─────────────────────────────────────────
    try:
        sim_months = pd.date_range(SIM_START, SIM_END, freq="ME")
    except Exception:
        sim_months = pd.date_range(SIM_START, SIM_END, freq="M")

    if test_month:
        sim_months = pd.DatetimeIndex([pd.Timestamp(test_month)])
        logger.info(f"TEST MODE: single month {test_month}")

    if resume_from:
        resume_ts  = pd.Timestamp(resume_from)
        sim_months = sim_months[sim_months >= resume_ts]
        logger.info(f"RESUME MODE: starting from {resume_from}")

    logger.info(f"Simulation months: {len(sim_months)} ({sim_months[0].date()} → {sim_months[-1].date()})")

    # ── Initialise agents ────────────────────────────────────────────────────
    # BUG FIX (2026-07-22, mirrors simulation_loop_postfix.py): the model
    # string MUST be passed explicitly to Hawk, Dove AND Contrarian — their
    # constructors default to the module-level OLLAMA_MODEL ("qwen2:7b"), so
    # an unparameterized run silently executed Qwen2 regardless of intent.
    ollama_model = "llama3:latest" if backend == "llama3" else "qwen2:7b"
    logger.info(f"Backend: {backend} (model={ollama_model})")

    if not dry_run:
        from agent_hawk_v2         import HawkAgent
        from agent_dove_v2          import DoveAgent
        from agent_synthesis     import SynthesisAgent
        from agent_contrarian    import ContrarianAgent
        from agent_risk_guardian import RiskGuardian
        from black_litterman     import BlackLittermanModel

        hawk_agent       = HawkAgent(model=ollama_model)
        dove_agent       = DoveAgent(model=ollama_model)
        synth_agent      = SynthesisAgent()
        contrarian_agent = ContrarianAgent(model=ollama_model)
        guardian         = RiskGuardian()
        bl_model         = BlackLittermanModel()
    else:
        from agent_synthesis     import SynthesisAgent
        from agent_contrarian    import ContrarianAgent
        from agent_risk_guardian import RiskGuardian
        from black_litterman     import BlackLittermanModel

        synth_agent      = SynthesisAgent()
        contrarian_agent = ContrarianAgent(model=ollama_model)
        guardian         = RiskGuardian()
        bl_model         = BlackLittermanModel()
        logger.info("DRY RUN: LLM agents replaced with mock outputs")

    # ── State variables ──────────────────────────────────────────────────────
    results  = SimulationResults()
    w_prev   = np.ones(N) / N          # start equal-weight
    prev_Q   = None                     # no prior views on month 1
    cum_return = 1.0                    # £1 initial value

    # ── Main loop ────────────────────────────────────────────────────────────
    total   = len(sim_months)
    t_start = time.time()

    for month_idx, month_ts in enumerate(sim_months):
        date_str = month_ts.strftime("%Y-%m-%d")
        logger.info(f"\n{'─'*65}")
        logger.info(f"Month {month_idx+1}/{total}: {date_str}")

        # ── Look up alignment row ─────────────────────────────────────────
        if month_ts not in alignment.index:
            logger.warning(f"No alignment row for {date_str} — skipping")
            continue

        row = alignment.loc[month_ts]

        # Extract FRED macro data
        # Handles both possible column name variants from data_pipeline
        def get_col(row, *names):
            for name in names:
                if name in row.index and pd.notna(row[name]):
                    return float(row[name])
            return 0.0

        cpi_yoy      = get_col(row, "cpi_yoy", "CPI_YOY", "cpi")
        pce_yoy      = get_col(row, "pce_yoy", "PCE_YOY", "pce")
        gdp_growth   = get_col(row, "gdp_growth", "GDP_GROWTH", "gdp")
        unemployment = get_col(row, "unemployment", "UNEMPLOYMENT", "unrate")
        vix          = get_col(row, "vix_close", "VIX_CLOSE", "vix", "VIX")

        fomc_date    = str(row.get("fomc_date", row.get("FOMC_DATE", "")))
        fomc_file    = str(row.get("fomc_file", row.get("FOMC_FILE", "")))

        if vix <= 0:
            vix = 20.0   # fallback VIX
            logger.warning(f"VIX not found for {date_str}, using default 20.0")

        # ── Load FOMC text ────────────────────────────────────────────────
        fomc_text = load_fomc_text(fomc_file, fomc_date)
        if not fomc_text:
            logger.warning(f"Empty FOMC text for {date_str} (fomc_date={fomc_date})")

        # ── PnL context ───────────────────────────────────────────────────
        pnl_context = build_pnl_context(
            month_idx       = month_idx,
            monthly_returns = results.portfolio_returns,
            monthly_weights = results.weights,
            dates           = results.dates,
            returns_df      = returns_df,
        )

        # ── Agents 1 & 2 ─────────────────────────────────────────────────
        if dry_run:
            hawk_out = MockHawkOutput()
            dove_out = MockDoveOutput()
        else:
            hawk_out = hawk_agent.run(
                fomc_text   = fomc_text,
                cpi_yoy     = cpi_yoy,
                pce_yoy     = pce_yoy,
                date        = date_str,
                fomc_date   = fomc_date,
                pnl_context = pnl_context,
            )
            dove_out = dove_agent.run(
                fomc_text    = fomc_text,
                gdp_growth   = gdp_growth,
                unemployment = unemployment,
                date         = date_str,
                fomc_date    = fomc_date,
                pnl_context  = pnl_context,
            )

        # ── Agent 3: Synthesis ────────────────────────────────────────────
        synth_out = synth_agent.run(
            hawk_output = hawk_out,
            dove_output = dove_out,
            returns_df  = returns_df,
            date        = date_str,
        )

        # ── Agent 4: Contrarian ───────────────────────────────────────────
        cont_out = contrarian_agent.run(
            synthesis_output = synth_out,
            fomc_text        = fomc_text,
            date             = date_str,
        )

        # ── Agent 5: Risk Guardian ────────────────────────────────────────
        drawdown_series = (
            pd.Series(results.drawdowns, index=pd.DatetimeIndex(results.return_dates))
            if len(results.drawdowns) >= 2 else None
        )

        report, final_bl_inputs = guardian.run(
            contrarian_output = cont_out,
            synthesis_output  = synth_out,
            hawk_output       = hawk_out,
            dove_output       = dove_out,
            vix               = vix,
            date              = date_str,
            prev_Q            = prev_Q,
            sigma_j           = synth_out.sigma_j,
            drawdown_history  = drawdown_series,
        )

        # ── Black-Litterman model ─────────────────────────────────────────
        try:
            bl_out = bl_model.run(
                date       = month_ts,
                returns_df = returns_df,
                w_mkt_df   = w_mkt_df,
                w_previous = w_prev,
                bl_inputs  = final_bl_inputs,
            )
            guardian.record_turnover_cap(report, bl_out.cap_applied)
            w_new = bl_out.w_final

        except Exception as e:
            logger.error(f"BL model failed for {date_str}: {e}. Using previous weights.")
            w_new = w_prev.copy()
            bl_out = None

        # ── Monthly portfolio return ──────────────────────────────────────
        next_idx  = month_idx + 1
        if next_idx < len(sim_months):
            next_ts = sim_months[next_idx]
        else:
            next_ts = None

        # BUG FIX (return labeling): w_new is decided at month-end t and held
        # through month t+1, so its realized return belongs to calendar month
        # t+1 and is recorded under THAT label. The final decision month has
        # no t+1 inside the window and gets NO return row — the old fallback
        # re-used month t's own return there, double-counting Dec 2024.
        port_return = None
        return_date = None
        if next_ts is not None and next_ts in returns_df.index:
            sector_rets = returns_df.loc[next_ts, TICKERS].values.astype(float)
            valid       = ~np.isnan(sector_rets)
            if valid.any():
                port_return = float(np.dot(w_new[valid], sector_rets[valid]))
                return_date = next_ts.strftime("%Y-%m-%d")

        if port_return is not None:
            cum_return *= (1 + port_return)
        drawdown = compute_drawdown(results.cumulative_returns + [cum_return])

        # ── Record ────────────────────────────────────────────────────────
        results.record(
            date        = date_str,
            w_final     = w_new,
            port_return = port_return,
            cum_return  = cum_return,
            drawdown    = drawdown,
            synth_out   = synth_out,
            cont_out    = cont_out,
            report      = report,
            bl_out      = bl_out,
            return_date = return_date,
        )

        # ── Update state ──────────────────────────────────────────────────
        w_prev = w_new.copy()
        prev_Q = final_bl_inputs.Q.copy()

        # ── Progress log ──────────────────────────────────────────────────
        elapsed = time.time() - t_start
        pct     = (month_idx + 1) / total * 100
        eta_s   = elapsed / (month_idx + 1) * (total - month_idx - 1)
        eta_min = eta_s / 60

        logger.info(
            f"  [{month_idx+1}/{total}  {pct:.0f}%]  "
            f"return={(port_return * 100 if port_return is not None else float('nan')):+.2f}%  "
            f"cum={cum_return:.4f}  "
            f"VIX={vix:.1f}  "
            f"mean_d={report.mean_disagreement:.3f}  "
            f"early_warn={report.early_warning_flag}  "
            f"ETA={eta_min:.1f}min"
        )

        # Save checkpoint every 12 months
        if (month_idx + 1) % 12 == 0:
            results.save(RESULTS_DIR)
            logger.info(f"  Checkpoint saved at month {month_idx+1}")

    # ── Final save ────────────────────────────────────────────────────────────
    results.save(RESULTS_DIR)

    elapsed_total = (time.time() - t_start) / 60
    logger.info(f"\n{'='*65}")
    logger.info(f"Simulation complete in {elapsed_total:.1f} minutes")
    logger.info(f"Months completed: {len(results.dates)}/{total}")
    logger.info(f"Fallbacks — Hawk: {sum(results.hawk_fallbacks)}  Dove: {sum(results.dove_fallbacks)}")
    logger.info(f"Early warning flags: {sum(results.early_warning_flags)}")
    logger.info(f"Instability episodes: {sum(results.instability_episodes)}")
    logger.info(f"Guardrail activations — Ceiling: {sum(results.ceiling_fired)}  "
                f"Stability: {sum(results.stability_fired)}  "
                f"Turnover: {sum(results.turnover_cap_fired)}")
    logger.info(f"Final cumulative return: {(results.cumulative_returns[-1]-1)*100:+.2f}%")
    logger.info(f"Results saved to: {RESULTS_DIR}/")
    logger.info(f"{'='*65}")

    return results


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the 71-month LLM-BL simulation (ORIGINAL meeting-date "
                    "alignment; parameterized 2026-07-22)")
    parser.add_argument("--backend",      type=str, required=True,
                        choices=["llama3", "qwen2"],
                        help="Which LLM backend to run: llama3 or qwen2")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Run without LLM calls using mock agent outputs")
    parser.add_argument("--test-month",   type=str, default=None,
                        help="Run a single month only (e.g. 2022-09-30)")
    parser.add_argument("--resume-from",  type=str, default=None,
                        help="Resume simulation from this month (e.g. 2021-06-30)")
    args = parser.parse_args()

    # SAFETY (2026-07-22): output now goes to a per-backend directory.
    # The old hardcoded RESULTS_DIR = PROJECT_ROOT/"results" would have
    # clobbered the preserved original-run data in results/. RESULTS_DIR must
    # be rebound before setup_logging()/run_simulation(), which read this
    # module global at call time.
    RESULTS_DIR = PROJECT_ROOT / f"results_{args.backend}_original"

    setup_logging(RESULTS_DIR, args.dry_run)
    logger = logging.getLogger(__name__)

    try:
        results = run_simulation(
            backend     = args.backend,
            dry_run     = args.dry_run,
            test_month  = args.test_month,
            resume_from = args.resume_from,
        )

        # Quick summary print
        print(f"\n{'='*65}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*65}")
        print(f"Months:              {len(results.dates)}")
        if results.cumulative_returns:
            total_ret = (results.cumulative_returns[-1] - 1) * 100
            ann_ret   = ((results.cumulative_returns[-1]) ** (12/len(results.portfolio_returns)) - 1) * 100
            print(f"Total return:        {total_ret:+.2f}%")
            print(f"Annualised return:   {ann_ret:+.2f}%")
        print(f"Early warnings:      {sum(results.early_warning_flags)}")
        print(f"Instability episodes:{sum(results.instability_episodes)}")
        print(f"Hawk fallbacks:      {sum(results.hawk_fallbacks)}")
        print(f"Dove fallbacks:      {sum(results.dove_fallbacks)}")
        print(f"Results in:          {RESULTS_DIR}/")
        print(f"{'='*65}")

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user. Saving partial results...")
        if 'results' in locals():
            results.save(RESULTS_DIR)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        sys.exit(1)
