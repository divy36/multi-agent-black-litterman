"""Baselines part2."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ──────────────────────────────────────────────────────────────────────
# CONFIG — matches the calibrated parameters in the research project brief
# ──────────────────────────────────────────────────────────────────────

# The results directory MUST be passed explicitly — the historical default
# of results/ is frozen generation-1 legacy data.
# NOTE (return-labeling fix, 2026-07-21): this script assumes monthly_returns.csv
# uses the CORRECTED convention where row t holds the return realized in
# calendar month t. Under that convention the mechanical baselines here
# (signal from data strictly before t, applied to month t's return) share the
# five-agent system's timing exactly.
if len(sys.argv) < 2:
    raise SystemExit("Usage: python3 baselines_part2.py <results_dir>   "
                     "e.g. python3 baselines_part2.py results_llama3_postfix")
RESULTS_DIR = Path(sys.argv[1])
OUT_DIR = RESULTS_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLB"]
PERIODS_PER_YEAR = 12
RISK_FREE_ANNUAL = 0.0
N_BOOTSTRAP = 10_000
RNG_SEED = 42

LAMBDA = 3.0            # market risk aversion
TAU = 0.05               # prior uncertainty scalar
T_MAX = 0.30              # max monthly L1 turnover
FIXED_CONFIDENCE = 0.5    # stated simplification — see module docstring
SIGMA_WINDOW = 36         # rolling covariance window, months
SIGMA_MIN_PERIODS = 12    # fall back gracefully if full history unavailable
SIGMA_SHRINKAGE = 0.10    # Ledoit & Wolf (2003) style shrinkage toward scaled identity —
                          # guards against near-singular covariance from 10 correlated
                          # sectors estimated over only 36 monthly observations
MOMENTUM_WINDOW = 12      # trailing months for momentum signal

ETF_RETURN_CANDIDATES = [
    "data/etf_returns_monthly.csv", "processed/etf_returns.csv",
]
# Finding B fix (2026-07-21): use the TIME-VARYING real index-weight series
# (build_market_weights_v2.py); the static-snapshot file is no longer a
# candidate (it was a ~2026 snapshot backdated to 2019).
MARKET_WEIGHT_CANDIDATES = [
    "data/market_cap_weights.csv",
]
# Hawk views must come from the SAME run being analysed (they are backend-
# specific LLM output) — resolved against RESULTS_DIR, not hardcoded to
# "results/", which silently pointed every run at whichever backend's views
# happened to live there.
HAWK_VIEWS_CANDIDATES = [
    str(RESULTS_DIR / "monthly_hawk_views.csv"),
]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "figure.dpi": 140,
    "savefig.dpi": 300,   # print-quality output for analysis figures
})
NAVY, RED, GREEN, GOLD, GREY, PURPLE = "#1a2b4a", "#b5433a", "#3a7d5c", "#c9962f", "#7a7a7a", "#6b4c8a"


# ──────────────────────────────────────────────────────────────────────
# HELPERS (self-contained — no dependency on other project files)
# ──────────────────────────────────────────────────────────────────────

def find_first_existing(candidates: list) -> Optional[Path]:
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def sector_numeric_cols(df: pd.DataFrame, prefix: str = "") -> list:
    return [c for c in df.columns if any((prefix + s) in c.upper() for s in SECTORS)]


def find_col(df: pd.DataFrame, *keywords: str, exclude: Optional[list] = None) -> Optional[str]:
    exclude = exclude or []
    for col in df.columns:
        lc = col.lower()
        if all(k.lower() in lc for k in keywords) and not any(e.lower() in lc for e in exclude):
            return col
    return None


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


def load_market_weights(path: Path) -> Optional[pd.Series]:
    df = pd.read_csv(path)
    wide_cols = sector_numeric_cols(df)
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
    weight_col = find_col(df, "weight") or find_col(df, "cap") or find_col(df, "pct")
    if ticker_col is not None and weight_col is not None:
        s = df.set_index(df[ticker_col].astype(str).str.upper())[weight_col].astype(float)
        s = s.reindex(SECTORS).dropna()
        if len(s) >= 5:
            return s / s.sum()
    print(f"  [WARN] {path} found but columns unrecognised. Columns: {list(df.columns)}")
    return None


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
# BLACK-LITTERMAN CORE (from scratch, matches brief Section 4.2 exactly)
# ──────────────────────────────────────────────────────────────────────

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
    """Mirrors black_litterman.py's real _portfolio_weights exactly: clips
    negative weights to zero (long-only constraint — this ETF portfolio
    cannot short) before normalising. Missing this previously allowed
    unbounded short leverage when Omega got small, which is what actually
    drove the earlier Hawk-only collapse, not view quality alone."""
    w_raw = np.linalg.solve(Sigma_star, mu_star)
    w_raw = np.maximum(w_raw, 0.0)
    raw_sum = w_raw.sum()
    if abs(raw_sum) < 1e-12:
        return np.ones(n) / n
    return w_raw / raw_sum


def apply_turnover_cap(w_target: np.ndarray, w_prev: np.ndarray, t_max: float = T_MAX) -> np.ndarray:
    delta = w_target - w_prev
    l1 = np.sum(np.abs(delta))
    if l1 > t_max:
        delta = delta * (t_max / l1)
    w_final = w_prev + delta
    return w_final / np.sum(w_final)


def shrink_covariance(Sigma: np.ndarray, shrinkage: float = SIGMA_SHRINKAGE) -> np.ndarray:
    """Ledoit & Wolf (2003) style shrinkage toward a scaled identity matrix.
    Guards against the near-singular sample covariance that arises from
    estimating a 10x10 matrix over highly correlated US sector ETFs using
    only 36 monthly observations — a textbook small-sample conditioning
    problem, not a bug in the BL equations themselves."""
    n = Sigma.shape[0]
    avg_var = np.trace(Sigma) / n
    target = avg_var * np.eye(n)
    return (1 - shrinkage) * Sigma + shrinkage * target


def rolling_sigma(full_returns: pd.DataFrame, target_date, window: int = SIGMA_WINDOW,
                   min_periods: int = SIGMA_MIN_PERIODS):
    """Covariance using only data strictly before target_date — no look-ahead.
    Applies shrinkage to guard against near-singular matrices (see
    shrink_covariance)."""
    history = full_returns.loc[full_returns.index < target_date]
    window_data = history.tail(window)
    if len(window_data) < min_periods:
        return None, len(window_data)
    Sigma = window_data.cov().values
    return shrink_covariance(Sigma), len(window_data)


def trailing_momentum(full_returns: pd.DataFrame, target_date, window: int = MOMENTUM_WINDOW):
    """Monthly-equivalent trailing return using only data strictly before target_date."""
    history = full_returns.loc[full_returns.index < target_date]
    trailing = history.tail(window)
    if len(trailing) < window // 2:
        return None
    cum = (1 + trailing).prod() - 1
    monthly_equiv = (1 + cum) ** (1 / len(trailing)) - 1
    return monthly_equiv.values


def _wmkt_at(w_mkt_df: pd.DataFrame, date) -> np.ndarray:
    """w_mkt row for month `date`: last weight row strictly BEFORE date
    (lag-1, mirrors black_litterman._market_weights — no look-ahead)."""
    prior = w_mkt_df[w_mkt_df.index < date]
    w = prior.iloc[-1].values.astype(float)
    return w / w.sum()


def run_mechanical_bl(dates, get_Q_fn, full_returns: pd.DataFrame, w_mkt_df: pd.DataFrame,
                       fixed_conf: float = FIXED_CONFIDENCE, label: str = ""):
    """Runs the BL pipeline sequentially month by month. Returns a Series of
    realised monthly portfolio returns indexed by date, and a diagnostics
    dict describing the worst-conditioned month encountered — used to
    pinpoint numerical instability precisely instead of guessing."""
    w_prev = np.full(len(SECTORS), 1.0 / len(SECTORS))
    realised_returns = {}
    skipped = 0

    diag = {
        "worst_cond": -1.0, "worst_cond_date": None,
        "min_omega_diag": np.inf, "min_omega_diag_date": None,
        "n_nonfinite_Q": 0, "n_nonfinite_weights": 0,
        "n_months": 0,
    }

    for date in dates:
        Sigma_t, n_hist = rolling_sigma(full_returns, date)
        Q_t = get_Q_fn(date)

        if Sigma_t is None or Q_t is None:
            skipped += 1
            realised_returns[date] = float(full_returns.loc[date, SECTORS].values @ w_prev)
            continue

        if not np.all(np.isfinite(Q_t)):
            diag["n_nonfinite_Q"] += 1
            realised_returns[date] = float(full_returns.loc[date, SECTORS].values @ w_prev)
            continue

        diag["n_months"] += 1

        Pi_t = LAMBDA * Sigma_t @ _wmkt_at(w_mkt_df, date)
        sigma_diag = np.diag(Sigma_t)
        Omega_t = np.diag((1 - fixed_conf) ** 2 * sigma_diag)

        cond = np.linalg.cond(TAU * Sigma_t)
        if cond > diag["worst_cond"]:
            diag["worst_cond"] = cond
            diag["worst_cond_date"] = date

        min_om = np.min(np.diag(Omega_t))
        if min_om < diag["min_omega_diag"]:
            diag["min_omega_diag"] = min_om
            diag["min_omega_diag_date"] = date

        mu_star, Sigma_star = bl_posterior(Pi_t, Sigma_t, Q_t, Omega_t, TAU)
        w_target = bl_weights(mu_star, Sigma_star)

        if not np.all(np.isfinite(w_target)):
            diag["n_nonfinite_weights"] += 1
            realised_returns[date] = float(full_returns.loc[date, SECTORS].values @ w_prev)
            continue

        w_final = apply_turnover_cap(w_target, w_prev, T_MAX)

        realised_returns[date] = float(full_returns.loc[date, SECTORS].values @ w_final)
        w_prev = w_final

    return pd.Series(realised_returns).sort_index(), skipped, diag


# ──────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

llm_bl_df = to_datetime_index(pd.read_csv(RESULTS_DIR / "monthly_returns.csv"))
ret_col = "portfolio_return" if "portfolio_return" in llm_bl_df.columns else \
    next((c for c in llm_bl_df.columns if "return" in c.lower() and "cum" not in c.lower()), None)
llm_bl_returns = llm_bl_df[ret_col].astype(float)
date_index = llm_bl_returns.index
print(f"  [OK] Target window: {len(date_index)} months, {date_index.min().date()} -> {date_index.max().date()}")

return_path = find_first_existing(ETF_RETURN_CANDIDATES)
if return_path is None:
    raise SystemExit("[FATAL] No ETF returns file found. Checked: " + str(ETF_RETURN_CANDIDATES))
full_returns_raw = to_datetime_index(pd.read_csv(return_path))
ret_cols = sector_numeric_cols(full_returns_raw)
full_returns = full_returns_raw[ret_cols].astype(float)
full_returns.columns = [c.upper() for c in full_returns.columns]
full_returns = full_returns[SECTORS]
print(f"  [OK] {return_path}  ->  full history {full_returns.index.min().date()} -> {full_returns.index.max().date()} "
      f"({len(full_returns)} months)")

months_before_window = (full_returns.index < date_index.min()).sum()
print(f"  [check] {months_before_window} months of history available before the target window "
      f"(need {SIGMA_WINDOW} for a full covariance window, {MOMENTUM_WINDOW} for momentum)")
if months_before_window < SIGMA_WINDOW:
    print(f"  [WARN] Fewer than {SIGMA_WINDOW} months of pre-window history — early months in the "
          f"target window will use a shorter (expanding) covariance window. Document this in Methodology.")

weight_path = find_first_existing(MARKET_WEIGHT_CANDIDATES)
if weight_path is None:
    raise SystemExit("[FATAL] No market-cap weight file found. Checked: " + str(MARKET_WEIGHT_CANDIDATES))
w_mkt_df = to_datetime_index(pd.read_csv(weight_path))
w_mkt_df = w_mkt_df[[c for c in w_mkt_df.columns if c.upper() in SECTORS]].astype(float)
w_mkt_df.columns = [c.upper() for c in w_mkt_df.columns]
w_mkt_df = w_mkt_df[SECTORS]
print(f"  [OK] Time-varying market weights loaded from {weight_path}: "
      f"{len(w_mkt_df)} rows, {w_mkt_df.index.min().date()} -> {w_mkt_df.index.max().date()}")

hawk_path = find_first_existing(HAWK_VIEWS_CANDIDATES)
hawk_views = None
if hawk_path is not None:
    hawk_df = to_datetime_index(pd.read_csv(hawk_path))
    hawk_cols = [c for c in hawk_df.columns if any(s in c.upper() for s in SECTORS)]
    if hawk_cols:
        hawk_views = hawk_df[hawk_cols].astype(float)
        hawk_views.columns = [next(s for s in SECTORS if s in c.upper()) for c in hawk_cols]
        hawk_views = hawk_views[SECTORS]
        # BUG FIX (look-ahead): monthly_hawk_views.csv is labeled by DECISION
        # month — views formed at month-end t. Applying them to month t's own
        # return would score a view against a month that was already over
        # when the view was formed. Shift labels one month forward so the
        # view formed at end of t is applied to month t+1's return, matching
        # the five-agent system's timing.
        hawk_views.index = hawk_views.index + pd.offsets.MonthEnd(1)
        print(f"  [OK] {hawk_path}  ->  {len(hawk_views)} months, columns mapped to {SECTORS} "
              f"(views shifted +1 month: decision-month label -> realization month)")
    else:
        print(f"  [WARN] {hawk_path} found but no sector columns recognised. Columns: {list(hawk_df.columns)}")
else:
    print(f"  [MISSING] No Hawk views file found. Checked: {HAWK_VIEWS_CANDIDATES}")

print(f"\n  [assumption] Fixed confidence = {FIXED_CONFIDENCE} used for both Hawk-only and "
      f"Momentum Omega calibration (see module docstring)")
print(f"  [assumption] Initial portfolio (month 1) = equal weight, {1/len(SECTORS):.4f} per sector")


# ──────────────────────────────────────────────────────────────────────
# 2. RUN MOMENTUM BL
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RUNNING MOMENTUM BL")
print("=" * 70)

def print_diag(diag: dict):
    print(f"  [diag] Months with valid BL solve: {diag['n_months']}")
    print(f"  [diag] Non-finite Q_t (skipped, held previous weights): {diag['n_nonfinite_Q']}")
    print(f"  [diag] Non-finite weights after BL solve (skipped): {diag['n_nonfinite_weights']}")
    print(f"  [diag] Worst condition number of (tau*Sigma): {diag['worst_cond']:.3e} "
          f"at {diag['worst_cond_date'].date() if diag['worst_cond_date'] is not None else 'n/a'}")
    print(f"  [diag] Smallest Omega diagonal entry seen: {diag['min_omega_diag']:.3e} "
          f"at {diag['min_omega_diag_date'].date() if diag['min_omega_diag_date'] is not None else 'n/a'}")


def momentum_Q(date):
    return trailing_momentum(full_returns, date)

momentum_returns, momentum_skipped, momentum_diag = run_mechanical_bl(date_index, momentum_Q, full_returns, w_mkt_df)
print(f"  [OK] {len(momentum_returns)} months computed, {momentum_skipped} used insufficient-history fallback "
      f"(previous month's weights held)")
print_diag(momentum_diag)


# ──────────────────────────────────────────────────────────────────────
# 3. RUN SINGLE-AGENT (HAWK-ONLY) BL
# ──────────────────────────────────────────────────────────────────────

hawk_returns = None
if hawk_views is not None:
    print("\n" + "=" * 70)
    print("RUNNING SINGLE-AGENT (HAWK-ONLY) BL")
    print("=" * 70)

    def hawk_Q(date):
        if date not in hawk_views.index:
            return None
        return hawk_views.loc[date].values

    hawk_returns, hawk_skipped, hawk_diag = run_mechanical_bl(date_index, hawk_Q, full_returns, w_mkt_df)
    print(f"  [OK] {len(hawk_returns)} months computed, {hawk_skipped} used insufficient-history fallback")
    print_diag(hawk_diag)
else:
    print("\n[SKIPPED] Single-Agent Hawk-only BL — no Hawk views file found. "
          "Point me at the real path and I'll finish this in one edit.")


# ──────────────────────────────────────────────────────────────────────
# 4. COMBINED COMPARISON TABLE + CHART
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("COMBINED BASELINE COMPARISON")
print("=" * 70)

rows = []
curves = {}

row, cw = performance_row("Five-Agent LLM-BL", llm_bl_returns)
rows.append(row); curves["Five-Agent LLM-BL"] = cw

row, cw = performance_row("Momentum BL", momentum_returns)
rows.append(row); curves["Momentum BL"] = cw

if hawk_returns is not None:
    row, cw = performance_row("Single-Agent (Hawk-only) BL", hawk_returns)
    rows.append(row); curves["Single-Agent (Hawk-only) BL"] = cw

table = pd.DataFrame(rows).set_index("Config")
print("\n" + table.to_string(float_format=lambda x: f"{x:.4f}"))

table.to_csv(OUT_DIR / "baseline_part2_table.csv")
print(f"\n  [saved] {OUT_DIR / 'baseline_part2_table.csv'}")

fig, ax = plt.subplots(figsize=(10, 5.5))
colors = {"Five-Agent LLM-BL": NAVY, "Momentum BL": RED, "Single-Agent (Hawk-only) BL": PURPLE}
for name, cw in curves.items():
    ax.plot(cw.index, cw.values, label=name, linewidth=1.6, color=colors.get(name, GREY))
ax.axhline(1.0, color=GREY, linewidth=0.8, linestyle="--")
ax.set_ylabel("Cumulative wealth (x)")
ax.set_title("Five-Agent LLM-BL vs. Mechanical Baselines")
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig(OUT_DIR / "08_mechanical_baseline_comparison.png")
plt.close(fig)
print("  [chart] saved 08_mechanical_baseline_comparison.png")

with open(OUT_DIR / "baseline_part2_report.md", "w") as f:
    f.write("# Momentum BL and Single-Agent (Hawk-only) BL Comparison\n\n")
    f.write(f"Fixed confidence = {FIXED_CONFIDENCE} used for Omega calibration in both mechanical baselines.\n\n")
    f.write(df_to_markdown(table))
    f.write("\n")
print(f"  [saved] {OUT_DIR / 'baseline_part2_report.md'}")

print("\n" + "=" * 70)
print("DONE.")
print("=" * 70)
