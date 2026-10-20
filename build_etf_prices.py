"""Build etf prices."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration — edit here if paths change
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR      = PROJECT_ROOT / "data"

TICKERS: List[str] = ["XLK", "XLF", "XLE", "XLV", "XLI",
                       "XLY", "XLP", "XLRE", "XLU", "XLB"]

# Download window: 3 years before sim start → end of sim
DOWNLOAD_START = "2015-12-01"   # yfinance end-exclusive, so Dec gives Jan 2016 bar
DOWNLOAD_END   = "2025-01-01"   # captures Dec 2024 month-end bar

# Canonical output window (month-end labels)
SERIES_START = "2016-01-31"
SERIES_END   = "2024-12-31"

# Simulation start — first month that must have a full 36M covariance window
SIM_START = "2019-01-31"

# Rolling covariance window length (months)
COV_WINDOW = 36


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _month_end_index(start: str, end: str) -> pd.DatetimeIndex:
    """Version-safe month-end DatetimeIndex ('ME' on pandas ≥ 2.2, else 'M')."""
    try:
        return pd.date_range(start, end, freq="ME")
    except ValueError:
        return pd.date_range(start, end, freq="M")


def download_prices(tickers: List[str],
                    start: str,
                    end: str) -> pd.DataFrame:
    """
    Download monthly adjusted-close prices via yfinance.

    Returns a DataFrame indexed by month-end dates, one column per ticker.
    Raises RuntimeError if any ticker has more than 2 consecutive NaNs
    after forward-fill (signals a genuine data gap, not just a missing bar).
    """
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,       # already-adjusted close; splits/divs handled
        progress=False,
    )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # Single-ticker download: exactly one Close column to rename.
        prices = raw[["Close"]].copy()
        prices.columns = tickers[: prices.shape[1]]

    # Strip timezone, snap to month-end labels ('ME' on pandas >= 2.2, 'M' before)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    try:
        prices = prices.resample("ME").last()
    except ValueError:
        prices = prices.resample("M").last()

    # Clip to canonical window
    idx = _month_end_index(SERIES_START, SERIES_END)
    prices = prices.reindex(idx)

    # Forward-fill up to 1 month (handles occasional missing bars)
    prices = prices.ffill(limit=1)

    # Validate: no ticker should still have >2 consecutive NaNs
    for col in prices.columns:
        consec = (prices[col].isna().astype(int)
                             .groupby((prices[col].notna()).cumsum())
                             .cumsum()
                             .max())
        if consec > 2:
            raise RuntimeError(
                f"{col} has a run of {consec} consecutive NaNs — "
                "check ETF launch date or yfinance data quality."
            )

    return prices[tickers]   # enforce column order


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns: r_t = P_t / P_{t-1} - 1. First row is NaN."""
    return prices.pct_change()


def check_covariance_seed(returns: pd.DataFrame,
                           sim_start: str,
                           window: int) -> None:
    """
    Verify the first simulation month has a full rolling covariance window.

    Raises AssertionError if any element of Sigma is NaN at sim_start.
    """
    sim_ts = pd.Timestamp(sim_start)

    # Returns available up to and including the month before sim_start
    pre_sim = returns[returns.index < sim_ts]

    if len(pre_sim) < window:
        raise AssertionError(
            f"Only {len(pre_sim)} months of pre-simulation returns available; "
            f"need {window}. Extend DOWNLOAD_START further back."
        )

    seed_window = pre_sim.iloc[-window:]   # last 36 months before sim
    Sigma = seed_window.cov().values

    if np.any(np.isnan(Sigma)):
        bad = [TICKERS[i] for i in range(len(TICKERS)) if np.isnan(np.diag(Sigma)[i])]
        raise AssertionError(
            f"Sigma has NaN entries at simulation start for: {bad}. "
            "Check price history for these tickers."
        )

    print(f"\nCovariance seed check PASSED")
    print(f"  Window used:   {seed_window.index.min().date()} → "
          f"{seed_window.index.max().date()} ({len(seed_window)} months)")
    print(f"  Sigma shape:   {Sigma.shape}")
    print(f"  Sigma finite:  {np.all(np.isfinite(Sigma))}")
    print(f"  Annualised vol (diagonal sqrt * sqrt(12)):")
    vols = pd.Series(np.sqrt(np.diag(Sigma) * 12), index=TICKERS)
    print(vols.round(4).to_string(header=False))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Download prices ──────────────────────────────────────────────────
    print("Downloading monthly adjusted-close prices …")
    print(f"  Tickers:  {TICKERS}")
    print(f"  Window:   {SERIES_START} → {SERIES_END}  ({(pd.Timestamp(SERIES_END).year - pd.Timestamp(SERIES_START).year) * 12 + pd.Timestamp(SERIES_END).month - pd.Timestamp(SERIES_START).month + 1} months)")

    prices = download_prices(TICKERS, DOWNLOAD_START, DOWNLOAD_END)

    # ── 2. Compute returns ──────────────────────────────────────────────────
    returns = prices_to_returns(prices)

    # ── 3. Save ─────────────────────────────────────────────────────────────
    prices_path  = OUT_DIR / "etf_prices_monthly.csv"
    returns_path = OUT_DIR / "etf_returns_monthly.csv"

    prices.round(4).to_csv(prices_path,  index_label="Date")
    returns.round(6).to_csv(returns_path, index_label="Date")

    # ── 4. Diagnostics ──────────────────────────────────────────────────────
    print(f"\nFiles saved:")
    print(f"  {prices_path}")
    print(f"  {returns_path}")

    print(f"\nPrice data summary:")
    print(f"  Shape:         {prices.shape}  (months × tickers)")
    print(f"  NaNs total:    {prices.isna().sum().sum()}")
    print(f"  Date range:    {prices.index.min().date()} → {prices.index.max().date()}")

    print(f"\nFirst prices (Jan 2016 — seed start):")
    print(prices.iloc[0].round(2).to_string())

    print(f"\nFirst sim-window prices (Jan 2019):")
    sim_start_ts = pd.Timestamp(SIM_START)
    print(prices.loc[prices.index == sim_start_ts].iloc[0].round(2).to_string())

    print(f"\nLast prices (Dec 2024 — sim end):")
    print(prices.iloc[-1].round(2).to_string())

    # ── 5. Covariance seed check ────────────────────────────────────────────
    check_covariance_seed(returns, SIM_START, COV_WINDOW)

    # ── 6. Return data quality per ticker ───────────────────────────────────
    sim_returns = returns[returns.index >= sim_start_ts]
    print(f"\nSimulation-window returns ({SIM_START} → {SERIES_END}):")
    print(f"  Months:        {len(sim_returns)}")
    print(f"  NaNs:          {sim_returns.isna().sum().sum()}  (must be 0)")
    print(f"\n  Annualised return (geometric mean):")
    ann_ret = ((1 + sim_returns).prod() ** (12 / len(sim_returns)) - 1)
    print(ann_ret.round(4).to_string())

    print(f"\nDone. build_etf_prices.py completed successfully.")


if __name__ == "__main__":
    main()
