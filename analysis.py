"""Analysis."""

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
# of results/ is frozen generation-1 legacy data and silently analysing it
# was a live trap.
if len(sys.argv) < 2:
    raise SystemExit("Usage: python3 analysis.py <results_dir>   "
                     "e.g. python3 analysis.py results_llama3_postfix")
RESULTS_DIR = Path(sys.argv[1])
OUT_DIR = RESULTS_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLB"]

RISK_FREE_ANNUAL = 0.0  # set e.g. 0.02 if you want to net out a T-bill yield
PERIODS_PER_YEAR = 12
N_BOOTSTRAP = 10_000
RNG_SEED = 42

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "figure.dpi": 140,
    "savefig.dpi": 300,   # print-quality output for analysis figures
})

NAVY, RED, GREEN, GOLD, GREY = "#1a2b4a", "#b5433a", "#3a7d5c", "#c9962f", "#7a7a7a"


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

def load_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"  [MISSING] {path} not found — skipping anything that depends on it.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  [OK] {name}  ->  {df.shape[0]} rows, columns: {list(df.columns)}")
    return df


def find_col(df: pd.DataFrame, *keywords: str, exclude: Optional[list] = None) -> Optional[str]:
    """First column whose name contains ALL given keywords (case-insensitive)."""
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
        print("  [WARN] No date-like column found — using default RangeIndex.")
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).set_index(date_col)


def sector_numeric_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if any(s in c.upper() for s in SECTORS)]


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


def hhi(weights_row: pd.Series) -> float:
    w = weights_row.values.astype(float)
    w = w[~np.isnan(w)]
    if w.sum() == 0:
        return float("nan")
    w = w / w.sum()
    return float(np.sum(w ** 2))


# ──────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("LOADING SIMULATION OUTPUT")
print("=" * 70)

returns_df = to_datetime_index(load_csv("monthly_returns.csv"))
weights_df = to_datetime_index(load_csv("monthly_weights.csv"))
stability_df = to_datetime_index(load_csv("monthly_stability.csv"))
disagreement_df = to_datetime_index(load_csv("monthly_disagreement.csv"))
raw_disagreement_df = to_datetime_index(load_csv("monthly_raw_disagreement.csv"))
views_df = to_datetime_index(load_csv("monthly_views.csv"))
hawk_df = to_datetime_index(load_csv("monthly_hawk_views.csv"))
dove_df = to_datetime_index(load_csv("monthly_dove_views.csv"))

report_lines: list = []


def log(msg: str = ""):
    print(msg)
    report_lines.append(msg)


# ──────────────────────────────────────────────────────────────────────
# 1. PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────────────────

log("\n" + "=" * 70)
log("1. PERFORMANCE METRICS")
log("=" * 70)

if not returns_df.empty:
    ret_col = find_col(returns_df, "return", exclude=["cumulative", "cum"])
    cum_col = find_col(returns_df, "cumulative") or find_col(returns_df, "cum")

    if ret_col is None:
        log("  [ERROR] Could not identify the monthly return column in monthly_returns.csv.")
        log(f"          Available columns: {list(returns_df.columns)}")
    else:
        r = returns_df[ret_col].astype(float)

        # Always derive cumulative wealth directly from monthly returns —
        # unambiguous regardless of how the CSV's own cumulative_return
        # column happens to be scaled. cw.iloc[-1] is the final wealth
        # multiple assuming a starting wealth of 1.0 before the first
        # month in the sample (NOT cw.iloc[0], which is already 1+r_0).
        cw = (1 + r).cumprod()

        if cum_col is not None:
            csv_series = returns_df[cum_col].astype(float)
            csv_final = csv_series.iloc[-1]
            csv_first = csv_series.iloc[0]
            # Detect whether the CSV column is a wealth index (starts ~1.0,
            # e.g. 2.1944 = grew to 2.19x) or an already-computed return
            # fraction (starts ~0, e.g. 1.1944 = +119.44%). Comparing the
            # wrong pair here produces a false "disagreement" that's off
            # by exactly 1.0.
            if csv_final > 1.5 or abs(csv_first - 1) < 0.3:
                csv_total_return = csv_final - 1  # wealth-index convention
            else:
                csv_total_return = csv_final       # already a return fraction
            recomputed_total_return = cw.iloc[-1] - 1
            log(f"  [cross-check] CSV '{cum_col}' implies total return: {csv_total_return:+.4f}   "
                f"vs recomputed from '{ret_col}' compounding: {recomputed_total_return:+.4f}")
            if abs(csv_total_return - recomputed_total_return) > 0.02:
                log(f"  [WARN] These disagree by more than 2 percentage points — "
                    f"check how simulation_loop.py computes '{cum_col}'.")
            else:
                log("  [OK] Cross-check passed — CSV cumulative_return agrees with "
                    "compounding portfolio_return directly.")

        n_months = len(r)
        total_return = cw.iloc[-1] - 1
        years = n_months / PERIODS_PER_YEAR
        ann_return = cw.iloc[-1] ** (1 / years) - 1
        ann_vol = r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)

        sharpe, sharpe_lo, sharpe_hi = bootstrap_sharpe_ci(r.values)
        max_dd, dd_duration, trough = max_drawdown_and_duration(cw)
        calmar = ann_return / abs(max_dd) if max_dd != 0 else float("nan")

        downside = r[r < 0]
        sortino = ((r.mean() * PERIODS_PER_YEAR) / (downside.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
                   if len(downside) > 1 else float("nan"))

        log(f"  Months in sample:            {n_months}")
        log(f"  Total cumulative return:     {total_return:+.2%}")
        log(f"  Annualised return:           {ann_return:+.2%}")
        log(f"  Annualised volatility:       {ann_vol:.2%}")
        log(f"  Sharpe ratio (annualised):   {sharpe:.3f}   [95% bootstrap CI: {sharpe_lo:.3f}, {sharpe_hi:.3f}]")
        log(f"  Sortino ratio (annualised):  {sortino:.3f}")
        log(f"  Calmar ratio:                {calmar:.3f}")
        log(f"  Maximum drawdown:            {max_dd:.2%}  (trough: {trough.date() if trough is not None else 'n/a'})")
        log(f"  Longest drawdown duration:   {dd_duration} months")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]})
        ax1.plot(cw.index, cw.values, color=NAVY, linewidth=1.6)
        ax1.set_ylabel("Cumulative wealth (x)")
        ax1.set_title("Five-Agent LLM-BL Portfolio — Cumulative Performance")
        ax1.axhline(1.0, color=GREY, linewidth=0.8, linestyle="--")

        running_max = cw.cummax()
        dd_series = cw / running_max - 1.0
        ax2.fill_between(dd_series.index, dd_series.values, 0, color=RED, alpha=0.35)
        ax2.plot(dd_series.index, dd_series.values, color=RED, linewidth=1.0)
        ax2.set_ylabel("Drawdown")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

        for ax in (ax1, ax2):
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        fig.tight_layout()
        fig.savefig(OUT_DIR / "01_cumulative_return_and_drawdown.png")
        plt.close(fig)
        log("  [chart] saved 01_cumulative_return_and_drawdown.png")
else:
    log("  [SKIPPED] monthly_returns.csv not found.")


# ──────────────────────────────────────────────────────────────────────
# 2. GUARDRAIL ACTIVATION ANALYSIS
# ──────────────────────────────────────────────────────────────────────

log("\n" + "=" * 70)
log("2. GUARDRAIL ACTIVATION ANALYSIS")
log("=" * 70)

if not stability_df.empty:
    ceiling_col = find_col(stability_df, "ceiling")
    stability_col = find_col(stability_df, "stability")
    turnover_col = find_col(stability_df, "turnover")
    ew_col = find_col(stability_df, "early") or find_col(stability_df, "warning")
    instab_col = find_col(stability_df, "instability")

    def bool_count(col):
        if col is None:
            return None
        s = stability_df[col]
        return int(s.sum()) if s.dtype == bool else int((s.astype(float) != 0).sum())

    n = len(stability_df)
    for label, col in [("Confidence ceiling", ceiling_col), ("Stability filter", stability_col),
                        ("Turnover cap", turnover_col), ("Early warning flags", ew_col),
                        ("Instability episodes", instab_col)]:
        cnt = bool_count(col)
        if cnt is not None:
            log(f"  {label:26s}: {cnt:3d} / {n} months  ({cnt/n:.1%})")
        else:
            log(f"  {label:26s}: column not found (searched, no match)")

    fig, ax = plt.subplots(figsize=(10, 3.2))
    y_labels, y_pos = [], []
    colors = [NAVY, GOLD, RED]
    for i, (label, col) in enumerate([("Ceiling", ceiling_col), ("Stability", stability_col), ("Turnover", turnover_col)]):
        if col is None:
            continue
        active = stability_df[stability_df[col].astype(float) != 0].index
        ax.scatter(active, [i] * len(active), color=colors[i % 3], s=22, marker="s", label=label)
        y_labels.append(label)
        y_pos.append(i)
    if y_pos:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        ax.set_title("Guardrail Activation Timeline")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(OUT_DIR / "02_guardrail_activation_timeline.png")
        log("  [chart] saved 02_guardrail_activation_timeline.png")
    plt.close(fig)
else:
    log("  [SKIPPED] monthly_stability.csv not found.")


# ──────────────────────────────────────────────────────────────────────
# 3. DISAGREEMENT SCORE ANALYSIS
# ──────────────────────────────────────────────────────────────────────

log("\n" + "=" * 70)
log("3. DISAGREEMENT SCORE ANALYSIS")
log("=" * 70)

if not disagreement_df.empty:
    d_cols = sector_numeric_cols(disagreement_df)
    if d_cols:
        d_vals = disagreement_df[d_cols].astype(float)
        mean_d_by_month = d_vals.mean(axis=1)
        log(f"  Sector disagreement columns found: {d_cols}")
        log(f"  Mean d_adjusted (overall):    {d_vals.values.mean():.3f}")
        log(f"  Mean d_adjusted (max month):  {mean_d_by_month.max():.3f}  ({mean_d_by_month.idxmax().date()})")
        log(f"  Mean d_adjusted (min month):  {mean_d_by_month.min():.3f}  ({mean_d_by_month.idxmin().date()})")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            frac = (mean_d_by_month > thresh).mean()
            log(f"  Months with mean d_adjusted > {thresh:.1f}: {frac:.1%}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(mean_d_by_month.index, mean_d_by_month.values, color=NAVY, linewidth=1.3,
                label="Mean d_adjusted (all sectors)")
        ax.axhline(0.60, color=RED, linestyle="--", linewidth=1.0, label="Early-warning threshold (0.60)")
        ax.fill_between(mean_d_by_month.index, mean_d_by_month.values, 0, color=NAVY, alpha=0.12)
        ax.set_title("Inter-Agent Disagreement Score Over Time")
        ax.set_ylabel("Mean d_adjusted")
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(OUT_DIR / "03_disagreement_over_time.png")
        plt.close(fig)
        log("  [chart] saved 03_disagreement_over_time.png")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        im = ax.imshow(d_vals.T.values, aspect="auto", cmap="Reds",
                        extent=[0, len(d_vals), 0, len(d_cols)])
        ax.set_yticks(np.arange(len(d_cols)) + 0.5)
        ax.set_yticklabels(list(reversed(d_cols)))
        ax.set_title("Per-Sector Disagreement Heatmap")
        ax.set_xlabel("Month index (Feb 2019 -> Dec 2024)")
        fig.colorbar(im, ax=ax, label="d_adjusted")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "04_disagreement_heatmap.png")
        plt.close(fig)
        log("  [chart] saved 04_disagreement_heatmap.png")
    else:
        log("  [WARN] No sector-named columns found in monthly_disagreement.csv.")
        log(f"          Available columns: {list(disagreement_df.columns)}")
else:
    log("  [SKIPPED] monthly_disagreement.csv not found.")


# ──────────────────────────────────────────────────────────────────────
# 4. PORTFOLIO CONCENTRATION (HHI) AND SECTOR ALLOCATION
# ──────────────────────────────────────────────────────────────────────

log("\n" + "=" * 70)
log("4. PORTFOLIO CONCENTRATION AND ALLOCATION")
log("=" * 70)

if not weights_df.empty:
    w_cols = sector_numeric_cols(weights_df)
    if w_cols:
        w_vals = weights_df[w_cols].astype(float)
        hhi_series = w_vals.apply(hhi, axis=1)
        log(f"  Sector weight columns found: {w_cols}")
        log(f"  Mean HHI:      {hhi_series.mean():.4f}  (1/N equal-weight HHI = {1/len(w_cols):.4f})")
        log(f"  Max HHI:       {hhi_series.max():.4f}  ({hhi_series.idxmax().date()})")
        log(f"  Min HHI:       {hhi_series.min():.4f}  ({hhi_series.idxmin().date()})")

        turnover_series = w_vals.diff().abs().sum(axis=1)
        log(f"  Mean monthly L1 turnover (from weights): {turnover_series.mean():.2%}")
        log(f"  Max monthly L1 turnover (from weights):  {turnover_series.max():.2%}  ({turnover_series.idxmax().date()})")

        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(hhi_series.index, hhi_series.values, color=GOLD, linewidth=1.3)
        ax.axhline(1 / len(w_cols), color=GREY, linestyle="--", linewidth=1.0, label="Equal-weight (1/N) HHI")
        ax.set_title("Portfolio Concentration (Herfindahl-Hirschman Index)")
        ax.set_ylabel("HHI")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(OUT_DIR / "05_hhi_concentration.png")
        plt.close(fig)
        log("  [chart] saved 05_hhi_concentration.png")

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.stackplot(w_vals.index, [w_vals[c].values for c in w_cols], labels=w_cols,
                     colors=plt.cm.tab10.colors[:len(w_cols)])
        ax.set_title("Sector Weight Allocation Over Time")
        ax.set_ylabel("Weight")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        fig.tight_layout()
        fig.savefig(OUT_DIR / "06_sector_allocation_stacked.png")
        plt.close(fig)
        log("  [chart] saved 06_sector_allocation_stacked.png")
    else:
        log("  [WARN] No sector-named columns found in monthly_weights.csv.")
        log(f"          Available columns: {list(weights_df.columns)}")
else:
    log("  [SKIPPED] monthly_weights.csv not found.")


# ──────────────────────────────────────────────────────────────────────
# 5. SAVE MARKDOWN SUMMARY REPORT
# ──────────────────────────────────────────────────────────────────────

report_path = OUT_DIR / "summary_report.md"
with open(report_path, "w") as f:
    f.write("# Simulation Analysis Summary\n\nGenerated by `analysis.py`.\n\n```\n")
    f.write("\n".join(report_lines))
    f.write("\n```\n")

print("\n" + "=" * 70)
print(f"DONE. Charts and summary saved to: {OUT_DIR.resolve()}")
print("=" * 70)
