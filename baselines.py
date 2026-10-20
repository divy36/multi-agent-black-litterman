"""Baselines."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

# The results directory MUST be passed explicitly — the historical default
# of results/ is frozen generation-1 legacy data.
# NOTE (return-labeling fix, 2026-07-21): this script assumes monthly_returns.csv
# uses the CORRECTED convention where row t holds the return realized in
# calendar month t — raw ETF returns are aligned on the same labels below.
if len(sys.argv) < 2:
    raise SystemExit("Usage: python3 baselines.py <results_dir>   "
                     "e.g. python3 baselines.py results_llama3_postfix")
RESULTS_DIR = Path(sys.argv[1])
OUT_DIR = RESULTS_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLB"]
PERIODS_PER_YEAR = 12
RISK_FREE_ANNUAL = 0.0
N_BOOTSTRAP = 10_000
RNG_SEED = 42

# Candidate locations for raw ETF price/return data, tried in order.
# Confirmed real files first, old guesses kept as fallback.
ETF_PRICE_CANDIDATES = [
    "data/etf_prices_monthly.csv", "processed/etf_prices.csv",
    "data/etf_prices.csv", "data/sector_etf_prices.csv",
    "etf_prices.csv", "data/etf_price_history.csv",
]
ETF_RETURN_CANDIDATES = [
    "data/etf_returns_monthly.csv", "processed/etf_returns.csv",
    "data/monthly_alignment.csv",
    "data/etf_returns.csv", "data/sector_etf_returns.csv",
    "etf_returns.csv", "data/master_alignment_table.csv",
    "master_alignment_table.csv", "data/alignment_table.csv",
]
# Finding B fix (2026-07-21): the Market-Cap baseline uses the TIME-VARYING
# real index-weight series (see build_market_weights_v2.py), applied with a
# one-month lag (row t-1 held during month t). The static-snapshot file is
# deliberately no longer a candidate — it was a ~2026 snapshot backdated to
# 2019 (hindsight bias).
MARKET_WEIGHT_CANDIDATES = [
    "data/market_cap_weights.csv",
]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "figure.dpi": 140,
    "savefig.dpi": 300,   # print-quality output for analysis figures
})
NAVY, RED, GREEN, GOLD, GREY = "#1a2b4a", "#b5433a", "#3a7d5c", "#c9962f", "#7a7a7a"


# ──────────────────────────────────────────────────────────────────────
# HELPERS (shared logic with analysis.py, kept standalone here)
# ──────────────────────────────────────────────────────────────────────

def find_col(df: pd.DataFrame, *keywords: str, exclude: Optional[list] = None) -> Optional[str]:
    """First column whose name contains ALL given keywords (case-insensitive)."""
    exclude = exclude or []
    for col in df.columns:
        lc = col.lower()
        if all(k.lower() in lc for k in keywords) and not any(e.lower() in lc for e in exclude):
            return col
    return None


def load_market_weights(path: Path) -> Optional[pd.Series]:
    """Handles both wide format (one column per sector) and long format
    (a ticker column + a weight column, one row per sector)."""
    df = pd.read_csv(path)

    wide_cols = sector_numeric_cols(df)
    if wide_cols:
        row = df[wide_cols].astype(float).iloc[0]
        return row / row.sum()

    ticker_col = None
    for c in df.columns:
        vals = df[c].astype(str).str.upper()
        if vals.isin(SECTORS).sum() >= 5:  # most rows look like sector tickers
            ticker_col = c
            break
    weight_col = find_col(df, "weight") or find_col(df, "cap") or find_col(df, "pct")

    if ticker_col is not None and weight_col is not None:
        s = df.set_index(df[ticker_col].astype(str).str.upper())[weight_col].astype(float)
        s = s.reindex(SECTORS).dropna()
        if len(s) >= 5:
            return s / s.sum()

    print(f"  [WARN] {path} was found but its columns didn't match a recognised "
          f"wide or long weight format. Columns: {list(df.columns)}")
    return None


def find_first_existing(candidates: list) -> Optional[Path]:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def sector_numeric_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if any(s in c.upper() for s in SECTORS)]


def find_date_col(df: pd.DataFrame) -> Optional[str]:
    for cand in ["date", "month", "period", "Date", "Month"]:
        if cand in df.columns:
            return cand
    for col in df.columns:
        if "date" in col.lower() or "month" in col.lower():
            return col
    return None


def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    date_col = find_date_col(df)
    if date_col is None:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).set_index(date_col)


from performance_metrics import bootstrap_sharpe_ci as _bootstrap_sharpe_ci


def bootstrap_sharpe_ci(returns, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    return _bootstrap_sharpe_ci(returns, n_boot, seed, RISK_FREE_ANNUAL, PERIODS_PER_YEAR)


def max_drawdown_and_duration(cum_wealth: pd.Series):
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


def performance_row(name: str, r: pd.Series) -> dict:
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


def df_to_markdown(df: pd.DataFrame) -> str:
    """Manual markdown table writer — avoids the optional 'tabulate' dependency."""
    header = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, row in df.iterrows():
        vals = [str(idx)]
        for v in row:
            vals.append(f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# 1. LOAD THE LLM-BL SYSTEM'S OWN RETURN SERIES (defines the date window)
# ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("LOADING FIVE-AGENT LLM-BL PORTFOLIO RETURNS (defines date window)")
print("=" * 70)

returns_path = RESULTS_DIR / "monthly_returns.csv"
if not returns_path.exists():
    raise SystemExit(f"[FATAL] {returns_path} not found — run analysis.py's data first.")

llm_bl_df = to_datetime_index(pd.read_csv(returns_path))
ret_col = "portfolio_return" if "portfolio_return" in llm_bl_df.columns else \
    next((c for c in llm_bl_df.columns if "return" in c.lower() and "cum" not in c.lower()), None)
if ret_col is None:
    raise SystemExit(f"[FATAL] Could not find a return column in {returns_path}. "
                      f"Columns: {list(llm_bl_df.columns)}")

llm_bl_returns = llm_bl_df[ret_col].astype(float)
date_index = llm_bl_returns.index
print(f"  [OK] {len(llm_bl_returns)} months, {date_index.min().date()} -> {date_index.max().date()}")


# ──────────────────────────────────────────────────────────────────────
# 2. LOCATE RAW ETF RETURN DATA
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("LOCATING RAW ETF RETURN / PRICE DATA")
print("=" * 70)

etf_returns = None

return_path = find_first_existing(ETF_RETURN_CANDIDATES)
if return_path is not None:
    df = to_datetime_index(pd.read_csv(return_path))
    cols = sector_numeric_cols(df)
    if cols:
        etf_returns = df[cols].astype(float)
        print(f"  [OK] Found returns directly: {return_path}  ->  columns: {cols}")

if etf_returns is None:
    price_path = find_first_existing(ETF_PRICE_CANDIDATES)
    if price_path is not None:
        df = to_datetime_index(pd.read_csv(price_path))
        cols = sector_numeric_cols(df)
        if cols:
            etf_returns = df[cols].astype(float).pct_change().dropna(how="all")
            print(f"  [OK] Found prices, computed returns via pct_change(): {price_path}  ->  columns: {cols}")

if etf_returns is None:
    print("  [MISSING] No ETF price/return file found at any of:")
    for c in ETF_RETURN_CANDIDATES + ETF_PRICE_CANDIDATES:
        print(f"             {c}")
    print("  Point me at the real path/filename (e.g. wherever build_etf_prices.py")
    print("  wrote its output) and I'll wire it in — one-line fix once I know the path.")
else:
    etf_returns = etf_returns.reindex(date_index).dropna(how="all")
    print(f"  [OK] Aligned to {len(etf_returns)} months matching the LLM-BL window")


# ──────────────────────────────────────────────────────────────────────
# 3. LOCATE STATIC MARKET-CAP WEIGHTS
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("LOCATING MARKET-CAP WEIGHTS")
print("=" * 70)

market_weights_df = None
weight_path = find_first_existing(MARKET_WEIGHT_CANDIDATES)
if weight_path is not None:
    print(f"  [file found] {weight_path}")
    _w = to_datetime_index(pd.read_csv(weight_path))
    _wcols = sector_numeric_cols(_w)
    if _wcols and len(_w) > 1:
        market_weights_df = _w[_wcols].astype(float)
        market_weights_df.columns = [c.upper() for c in market_weights_df.columns]
        print(f"  [OK] time-varying weights: {len(market_weights_df)} rows, "
              f"{market_weights_df.index.min().date()} -> {market_weights_df.index.max().date()}")
    else:
        print(f"  [WARN] {weight_path} is not a usable time-varying wide file. "
              f"Columns: {list(_w.columns)}")
else:
    print("  [MISSING] No file present at any of:")
    for c in MARKET_WEIGHT_CANDIDATES:
        print(f"             {c}")


# ──────────────────────────────────────────────────────────────────────
# 4. COMPUTE BASELINES + COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────

if etf_returns is None:
    print("\n[STOPPED] Cannot compute baselines without raw ETF return data. "
          "Send me the correct file path and I'll finish this in one edit.")
else:
    print("\n" + "=" * 70)
    print("COMPUTING BASELINES")
    print("=" * 70)

    rows = []
    curves = {}

    row, cw = performance_row("Five-Agent LLM-BL", llm_bl_returns)
    rows.append(row)
    curves["Five-Agent LLM-BL"] = cw

    equal_weight_returns = etf_returns.mean(axis=1)
    row, cw = performance_row("Equal-Weight (1/N)", equal_weight_returns)
    rows.append(row)
    curves["Equal-Weight (1/N)"] = cw

    if market_weights_df is not None:
        # Row t-1 (last weight row strictly before month t) held during
        # month t — no look-ahead.
        wm = market_weights_df.reindex(columns=[c.upper() for c in etf_returns.columns])
        mc = {}
        for t in etf_returns.index:
            prior = wm[wm.index < t]
            if prior.empty:
                continue
            w_t = prior.iloc[-1].values.astype(float)
            w_t = w_t / w_t.sum()
            mc[t] = float(np.nansum(w_t * etf_returns.loc[t].values))
        market_cap_returns = pd.Series(mc)
        row, cw = performance_row("Market-Cap Weight", market_cap_returns)
        rows.append(row)
        curves["Market-Cap Weight"] = cw

    table = pd.DataFrame(rows).set_index("Config")
    pct_cols = ["Total Return", "Ann. Return", "Ann. Vol", "Sharpe CI Low", "Sharpe CI High", "Max DD"]
    print("\n" + table.to_string(float_format=lambda x: f"{x:.4f}"))

    table.to_csv(OUT_DIR / "baseline_comparison_table.csv")
    print(f"\n  [saved] {OUT_DIR / 'baseline_comparison_table.csv'}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"Five-Agent LLM-BL": NAVY, "Equal-Weight (1/N)": GOLD, "Market-Cap Weight": GREEN}
    for name, cw in curves.items():
        ax.plot(cw.index, cw.values, label=name, linewidth=1.6,
                color=colors.get(name, GREY))
    ax.axhline(1.0, color=GREY, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Cumulative wealth (x)")
    ax.set_title("Five-Agent LLM-BL vs. Baselines — Cumulative Performance")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_baseline_comparison.png")
    plt.close(fig)
    print(f"  [chart] saved 07_baseline_comparison.png")

    report_path = OUT_DIR / "baselines_report.md"
    with open(report_path, "w") as f:
        f.write("# Baseline Comparison\n\n")
        f.write(df_to_markdown(table))
        f.write("\n")
    print(f"  [saved] {report_path}")

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)
