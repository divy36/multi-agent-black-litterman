"""Build market weights."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent  # adjust if needed
OUT_DIR = PROJECT_ROOT / "data"

# 10-sector universe (fixed). XLC / Communication Services is intentionally
# excluded (item 9.5) -- it launched in 2018 and lacks the long history the
# 36-month covariance seed needs.
TICKERS: List[str] = ["XLK", "XLF", "XLE", "XLV", "XLI",
                      "XLY", "XLP", "XLRE", "XLU", "XLB"]

SIM_START = "2019-01-31"   # first simulation month-end
SIM_END = "2024-12-31"     # last simulation month-end
LAG_START = "2018-12-31"   # one prior month, so a t-1 lag is always available

# Static fallback: approximate S&P 500 GICS sector weights, ex-Communication
# Services, as a percentage. Values are renormalised to sum to 1 in code.
# Source: S&P 500 sector breakdown, ~late March 2026 (S&P Global / Slickcharts);
# rough figures -- VERIFY/UPDATE from SSGA fund fact sheets (ssga.com) or
# S&P DJI before relying on the static path. Primary path is dynamic, below.
STATIC_SNAPSHOT: Dict[str, float] = {
    "XLK": 32.0,   # Information Technology
    "XLF": 13.0,   # Financials
    "XLV": 12.0,   # Health Care
    "XLY": 10.0,   # Consumer Discretionary
    "XLI": 8.0,    # Industrials
    "XLP": 6.0,    # Consumer Staples
    "XLE": 3.0,    # Energy
    "XLU": 2.5,    # Utilities
    "XLRE": 2.0,   # Real Estate
    "XLB": 2.0,    # Materials
}


# --------------------------------------------------------------------------- #
# Pure helpers (no network -- unit-tested)
# --------------------------------------------------------------------------- #
def _month_end_index(start: str, end: str) -> pd.DatetimeIndex:
    """Month-end DatetimeIndex, version-agnostic ('ME' on pandas>=2.2, else 'M')."""
    try:
        return pd.date_range(start, end, freq="ME")
    except ValueError:
        return pd.date_range(start, end, freq="M")


def fund_values_to_weights(fund_values: pd.DataFrame) -> pd.DataFrame:
    """Row-normalise positive fund values into weights that sum to 1 per row."""
    fv = fund_values.astype(float).copy()
    if (fv < 0).any().any():
        raise ValueError("Negative fund values encountered.")
    row_sums = fv.sum(axis=1)
    if (row_sums <= 0).any():
        bad = row_sums[row_sums <= 0].index.tolist()
        raise ValueError(f"Non-positive total fund value on rows: {bad}")
    return fv.div(row_sums, axis=0)


def backfill_shares(shares_me: pd.Series, full_index: pd.DatetimeIndex
                    ) -> Tuple[pd.Series, int]:
    """
    Align a month-end shares-outstanding series to `full_index`:
    forward-fill within coverage, then back-fill the earliest known value to
    cover months before yfinance coverage begins (constant-shares assumption).

    Returns (aligned_series, n_backfilled_months).
    """
    s = shares_me.reindex(full_index).ffill()
    n_backfilled = int(s.isna().sum())          # leading gap before first datum
    s = s.bfill()
    return s, n_backfilled


def static_snapshot_weights(snapshot: Dict[str, float],
                            tickers: List[str]) -> pd.Series:
    """Renormalise the editable static snapshot to a weight vector summing to 1."""
    s = pd.Series({t: float(snapshot[t]) for t in tickers})
    if s.sum() <= 0:
        raise ValueError("Static snapshot sums to <= 0.")
    return s / s.sum()


def validate_weights(weights: pd.DataFrame, tol: float = 1e-9) -> None:
    """Assert every row sums to 1 and all weights are in [0, 1]."""
    sums = weights.sum(axis=1)
    off = sums[(sums - 1.0).abs() > tol]
    if len(off):
        raise AssertionError(f"{len(off)} rows do not sum to 1 (e.g. {off.head().to_dict()})")
    if (weights < -tol).any().any() or (weights > 1 + tol).any().any():
        raise AssertionError("Weights outside [0, 1] found.")


# --------------------------------------------------------------------------- #
# Data fetch (yfinance -- requires network; run locally)
# --------------------------------------------------------------------------- #
def get_monthly_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Month-end close prices (one column per ticker)."""
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=pd.Timestamp(end) + pd.Timedelta(days=5),
                      interval="1mo", auto_adjust=False, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    # snap to canonical month-end labels ('ME' on pandas >= 2.2, 'M' before)
    try:
        close = close.resample("ME").last()
    except ValueError:
        close = close.resample("M").last()
    return close.reindex(columns=tickers)


def get_shares_month_end(ticker: str, start: str) -> pd.Series:
    """Month-end shares outstanding from yfinance; empty Series if unavailable."""
    import yfinance as yf
    try:
        s = yf.Ticker(ticker).get_shares_full(start=start)
    except Exception:
        return pd.Series(dtype=float)
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    try:
        return s.resample("ME").last()
    except ValueError:      # pandas < 2.2
        return s.resample("M").last()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_dynamic_weights() -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Primary path: weights from shares_outstanding x price."""
    full_index = _month_end_index(LAG_START, SIM_END)
    prices = get_monthly_prices(TICKERS, LAG_START, SIM_END).reindex(full_index).ffill()

    fund_values = pd.DataFrame(index=full_index, columns=TICKERS, dtype=float)
    backfill_report: Dict[str, int] = {}
    for t in TICKERS:
        shares_me = get_shares_month_end(t, LAG_START)
        if shares_me.empty:
            raise RuntimeError(f"No shares-outstanding data for {t}; use static fallback.")
        shares_aligned, n_bf = backfill_shares(shares_me, full_index)
        backfill_report[t] = n_bf
        fund_values[t] = shares_aligned * prices[t]

    fund_values = fund_values.ffill().bfill()
    weights = fund_values_to_weights(fund_values)
    validate_weights(weights)
    return weights, backfill_report


def build_static_weights() -> pd.DataFrame:
    """Fallback path: constant snapshot across the window."""
    full_index = _month_end_index(LAG_START, SIM_END)
    w = static_snapshot_weights(STATIC_SNAPSHOT, TICKERS)
    weights = pd.DataFrame([w.values] * len(full_index), index=full_index, columns=TICKERS)
    validate_weights(weights)
    return weights


def main() -> None:
    # ------------------------------------------------------------------ #
    # DEPRECATED (2026-07-22, Finding B fix). Running this script would
    # overwrite data/market_cap_weights.csv — which now holds the REAL
    # time-varying S&P 500 index-weight series built by
    # build_market_weights_v2.py from dated, sourced anchors — with either
    # (a) the fund-AUM proxy (a different, rejected methodology) or
    # (b) on any fetch failure, the old static ~2026 snapshot whose silent
    # fallback caused Finding B in the first place. It now fails loudly
    # instead. The pure helper functions above remain importable for tests.
    # ------------------------------------------------------------------ #
    raise SystemExit(
        "build_market_weights.py is DEPRECATED and disabled: it would clobber "
        "the corrected data/market_cap_weights.csv (real index weights). "
        "Use build_market_weights_v2.py instead. If you truly intend to "
        "rebuild the old AUM-proxy file, edit this guard out consciously."
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    method = "dynamic (shares x price)"
    try:
        weights, backfill_report = build_dynamic_weights()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] dynamic path failed ({exc}). Falling back to static snapshot.")
        weights = build_static_weights()
        backfill_report = {}
        method = "static snapshot (constant)"

    out_dyn = OUT_DIR / "market_cap_weights.csv"
    weights.round(6).to_csv(out_dyn, index_label="Date")

    static_vec = static_snapshot_weights(STATIC_SNAPSHOT, TICKERS)
    static_vec.round(6).to_csv(OUT_DIR / "market_cap_weights_static.csv",
                               header=["weight"], index_label="ticker")

    # ----------------------------- diagnostics ----------------------------- #
    print(f"Method used:        {method}")
    print(f"Saved:              {out_dyn}  ({weights.shape[0]} months x {weights.shape[1]} sectors)")
    print(f"Window:             {weights.index.min().date()} -> {weights.index.max().date()}")
    print(f"Row sums all == 1:  {bool(np.allclose(weights.sum(axis=1), 1.0))}")
    if backfill_report:
        bf = {k: v for k, v in backfill_report.items() if v}
        print(f"Constant-shares back-filled months (pre-coverage): {bf if bf else 'none'}")
    largest = weights.idxmax(axis=1)
    print(f"Largest sector by month (should be mostly XLK): "
          f"{largest.value_counts().to_dict()}")
    print("\nFirst row (use end of 2018-12 for the Jan-2019 rebalance):")
    print(weights.iloc[0].round(4).to_string())
    print("\nLatest row vs. static snapshot:")
    cmp = pd.DataFrame({"dynamic_latest": weights.iloc[-1].round(4),
                        "static_snapshot": static_vec.round(4)})
    print(cmp.to_string())


if __name__ == "__main__":
    main()