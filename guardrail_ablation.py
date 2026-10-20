"""Guardrail ablation."""

from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=RuntimeWarning)

# agent_outputs.pkl was pickled using the real StabilityReport/BLOutputs
# classes from these modules — pickle needs them importable to reconstruct
# the objects, even though this script never calls their methods directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import agent_risk_guardian  # noqa: F401
    import black_litterman      # noqa: F401
except ImportError as e:
    print(f"[FATAL] Could not import agent_risk_guardian / black_litterman: {e}")
    print("        Make sure this script is run from the FinalProject root directory.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

# The results directory MUST be passed explicitly — the historical default
# of results/ is frozen generation-1 legacy data.
# NOTE (return-labeling fix, 2026-07-21): weights decided at month t are now
# realized on month t+1's returns and recorded under label t+1, matching the
# corrected monthly_returns.csv convention. The final decision month (Dec
# 2024) has no in-window t+1 and produces no return row.
if len(sys.argv) < 2:
    raise SystemExit("Usage: python3 guardrail_ablation.py <results_dir>   "
                     "e.g. python3 guardrail_ablation.py results_llama3_postfix")
RESULTS_DIR = Path(sys.argv[1])
OUT_DIR = RESULTS_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLB"]
PERIODS_PER_YEAR = 12
RISK_FREE_ANNUAL = 0.0
N_BOOTSTRAP = 10_000
RNG_SEED = 42

LAMBDA = 3.0              # market risk aversion (matches black_litterman.py)
TAU = 0.05
ALPHA = 2.0               # disagreement Omega inflation — always on (see docstring)
T_MAX = 0.30              # base turnover cap
T_MAX_DYNAMIC = 0.15      # Config 6: tightened cap when VIX > 25
VIX_DYNAMIC_THRESHOLD = 25.0
CEILING_BASE_NORMAL = 0.75
CEILING_BASE_DYNAMIC = 0.50   # Config 6: tightened ceiling coefficient when VIX > 25
CEILING_DECAY = 0.02
CEILING_VIX_REF = 20.0
STABILITY_DECAY = 0.3
STABILITY_REVERSAL_MULT = 0.5
C_CLIP_MAX = 0.999        # numerical safety — avoid Omega diagonal hitting exactly 0

ETF_RETURN_CANDIDATES = ["data/etf_returns_monthly.csv", "processed/etf_returns.csv"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "figure.dpi": 140,
    "savefig.dpi": 300,   # print-quality output for analysis figures
})
COLORS = ["#7a7a7a", "#c9962f", "#3a7d5c", "#6b4c8a", "#1a2b4a", "#b5433a"]


# ──────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────

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


def bootstrap_sharpe_ci(returns: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    r = returns[~np.isnan(returns)]
    if len(r) < 3:
        return float("nan"), float("nan"), float("nan")
    rf_m = (1 + RISK_FREE_ANNUAL) ** (1 / PERIODS_PER_YEAR) - 1
    excess = r - rf_m

    def ann_sharpe(x):
        mu, sd = x.mean(), x.std(ddof=1)
        return float("nan") if sd == 0 else (mu / sd) * np.sqrt(PERIODS_PER_YEAR)

    point = ann_sharpe(excess)
    n = len(excess)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = ann_sharpe(excess[rng.integers(0, n, n)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return point, lo, hi


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


def bl_posterior_diag(Pi, Sigma, Q, Omega_diag, tau):
    """Mirrors black_litterman.py's BlackLittermanModel.run() Step 3 exactly —
    numerically stable diagonal-Omega approach (never inverts Omega as a
    full matrix), with the same 1e-12 floor on Omega_diag."""
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
    """Mirrors black_litterman.py's _portfolio_weights exactly: THE
    LONG-ONLY CONSTRAINT (clip negative weights to zero) that was missing
    before, then normalise, with the same near-zero-sum fallback to
    equal weight."""
    Sigma_star_inv = np.linalg.inv(Sigma_star)
    w_raw = Sigma_star_inv @ mu_star
    w_raw = np.maximum(w_raw, 0.0)
    raw_sum = w_raw.sum()
    if abs(raw_sum) < 1e-12:
        return np.ones(n) / n
    return w_raw / raw_sum


def apply_turnover_cap_exact(w_target, w_prev, t_max: float):
    """Mirrors black_litterman.py's turnover cap exactly, including that
    it only renormalises when the cap actually binds."""
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


# ──────────────────────────────────────────────────────────────────────
# 1. LOAD PICKLE + RAW RETURNS
# ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("LOADING AGENT OUTPUTS")
print("=" * 70)

with open(RESULTS_DIR / "agent_outputs.pkl", "rb") as f:
    data = pickle.load(f)

dates = pd.to_datetime(data["dates"])
stability_reports = data["stability_reports"]
bl_outputs = data["bl_outputs"]
n_months = len(dates)
print(f"  [OK] {n_months} months loaded, {dates.min().date()} -> {dates.max().date()}")

vix = np.array([sr.vix for sr in stability_reports])
c_raw = np.array([sr.c_raw for sr in stability_reports])         # (n_months, 10)
d_adjusted = np.array([sr.d_adjusted for sr in stability_reports])
Q_final = np.array([sr.Q_final for sr in stability_reports])
Sigma_all = np.array([bo.Sigma for bo in bl_outputs])            # (n_months, 10, 10)

# Finding B fix (2026-07-21): the pickled bo.Pi was built from the tainted
# static ~2026 w_mkt snapshot. Pi is instead recomputed here each month as
# LAMBDA * Sigma @ w_mkt(t) from the corrected time-varying weight file
# (build_market_weights_v2.py), using the last weight row strictly before
# each decision date (lag-1, mirrors black_litterman._market_weights).
_w_mkt_df = pd.read_csv("data/market_cap_weights.csv",
                        index_col="Date", parse_dates=True)[SECTORS]
assert len(_w_mkt_df.drop_duplicates()) > 1, \
    "market_cap_weights.csv looks constant — rebuild with build_market_weights_v2.py"

def _wmkt_at(date) -> np.ndarray:
    w = _w_mkt_df[_w_mkt_df.index < date].iloc[-1].values.astype(float)
    return w / w.sum()

Pi_all = np.array([LAMBDA * Sigma_all[t] @ _wmkt_at(dates[t])
                   for t in range(len(dates))])

# Sanity check: does Config 5 (all guardrails) reproduce c_final in the pickle?
c_final_stored = np.array([sr.c_final for sr in stability_reports])
print(f"  [check] c_raw == c_final for month 1 (expected, VIX low, no prior month): "
      f"{np.allclose(c_raw[0], c_final_stored[0])}")

return_path = find_first_existing(ETF_RETURN_CANDIDATES)
if return_path is None:
    raise SystemExit("[FATAL] No ETF returns file found. Checked: " + str(ETF_RETURN_CANDIDATES))
full_returns_raw = to_datetime_index(pd.read_csv(return_path))
ret_cols = sector_numeric_cols(full_returns_raw)
full_returns = full_returns_raw[ret_cols].astype(float)
full_returns.columns = [c.upper() for c in full_returns.columns]
full_returns = full_returns[SECTORS]
etf_returns = full_returns.reindex(dates).values  # (n_months, 10), aligned to sim dates
print(f"  [OK] {return_path} aligned to {n_months} months")


# ──────────────────────────────────────────────────────────────────────
# 2. CONFIG DEFINITIONS
# ──────────────────────────────────────────────────────────────────────

CONFIGS = [
    {"name": "1. No Guardrails",      "ceiling": False, "stability": False, "turnover": False, "dynamic": False},
    {"name": "2. Ceiling Only",       "ceiling": True,  "stability": False, "turnover": False, "dynamic": False},
    {"name": "3. Stability Only",     "ceiling": False, "stability": True,  "turnover": False, "dynamic": False},
    {"name": "4. Turnover Only",      "ceiling": False, "stability": False, "turnover": True,  "dynamic": False},
    {"name": "5. All Combined",       "ceiling": True,  "stability": True,  "turnover": True,  "dynamic": False},
    {"name": "6. Dynamic (VIX>25)",   "ceiling": True,  "stability": True,  "turnover": True,  "dynamic": True},
]


def run_config(cfg: dict):
    """Replays all n_months using stored Pi/Sigma/Q_final/d_adjusted, with
    Omega and turnover recomputed according to this config's guardrail
    toggles. Returns (realised_returns Series, instability_episode_count,
    guardrail_fire_counts dict)."""
    w_prev = np.full(len(SECTORS), 1.0 / len(SECTORS))
    realised = {}
    fires = {"ceiling": 0, "stability": 0, "turnover": 0}
    instability_episodes = 0
    drawdown_history = []
    wealth = 1.0

    for t in range(n_months):
        date = dates[t]
        vix_t = vix[t]
        Pi_t, Sigma_t, Q_t, d_adj_t = Pi_all[t], Sigma_all[t], Q_final[t], d_adjusted[t]
        sigma_diag = np.diag(Sigma_t)
        c = c_raw[t].copy()

        dynamic_active = cfg["dynamic"] and vix_t > VIX_DYNAMIC_THRESHOLD
        ceiling_base = CEILING_BASE_DYNAMIC if dynamic_active else CEILING_BASE_NORMAL
        t_max = T_MAX_DYNAMIC if dynamic_active else T_MAX

        if cfg["ceiling"]:
            c_max = ceiling_base * np.exp(-CEILING_DECAY * max(0.0, vix_t - CEILING_VIX_REF))
            capped = c > c_max
            if capped.any():
                fires["ceiling"] += 1
            c = np.minimum(c, c_max)

        if cfg["stability"] and t > 0:
            rolling_sigma_curr = np.sqrt(np.diag(Sigma_all[t]))
            rolling_sigma_curr = np.where(rolling_sigma_curr == 0, 1e-8, rolling_sigma_curr)
            delta = np.abs(Q_final[t] - Q_final[t - 1]) / rolling_sigma_curr
            penalty = np.exp(-STABILITY_DECAY * np.maximum(0.0, delta - 1.0))
            sign_reversal = ((Q_final[t] != 0) & (Q_final[t - 1] != 0) &
                              (np.sign(Q_final[t]) != np.sign(Q_final[t - 1])))
            if (penalty < 0.999).any() or sign_reversal.any():
                fires["stability"] += 1
            c = c * penalty
            c = np.where(sign_reversal, c * STABILITY_REVERSAL_MULT, c)

        c = np.clip(c, 0.0, C_CLIP_MAX)

        Omega_diag_t = (1 - c) ** 2 * sigma_diag * (1 + ALPHA * d_adj_t)
        mu_star, Sigma_star = bl_posterior_diag(Pi_t, Sigma_t, Q_t, Omega_diag_t, TAU)
        w_target = bl_weights_long_only(mu_star, Sigma_star, len(SECTORS))

        if not np.all(np.isfinite(w_target)):
            w_target = w_prev.copy()

        raw_turnover = np.sum(np.abs(w_target - w_prev))

        if cfg["turnover"]:
            w_final, cap_applied = apply_turnover_cap_exact(w_target, w_prev, t_max)
            if cap_applied:
                fires["turnover"] += 1
        else:
            w_final = w_target

        # BUG FIX (return labeling): weights decided at month t are held
        # through month t+1 — realize them on month t+1's returns and record
        # under that label, exactly as the (corrected) real simulation does.
        # The old code applied w(t) to month t's OWN return, a one-month
        # look-ahead that alone accounted for the ~20pp gap between the
        # reconstructed 'All Combined' config and the real run.
        if t + 1 < n_months:
            month_return = float(etf_returns[t + 1] @ w_final)
            realised[dates[t + 1]] = month_return
            wealth *= (1 + month_return)

            drawdown_history.append(wealth)
            running_max = max(drawdown_history)
            current_dd = wealth / running_max - 1.0
            window_start = max(0, len(drawdown_history) - 12)
            dd_series_window = [drawdown_history[i] / max(drawdown_history[:i + 1]) - 1.0
                                 for i in range(window_start, len(drawdown_history))]
            rolling_mean_dd = np.mean(dd_series_window)
            dd_std = np.std(dd_series_window) or 1e-8
            drawdown_extreme = current_dd < (rolling_mean_dd - 2 * dd_std)
            high_raw_confidence = np.any(c_raw[t] > 0.80)
            high_implied_turnover = raw_turnover > 0.25

            if drawdown_extreme and high_implied_turnover and high_raw_confidence:
                instability_episodes += 1

        w_prev = w_final

    return pd.Series(realised).sort_index(), instability_episodes, fires


# ──────────────────────────────────────────────────────────────────────
# 3. RUN ALL 6 CONFIGS
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RUNNING 6-CONFIGURATION GUARDRAIL ABLATION")
print("=" * 70)

rows = []
curves = {}
instability_summary = []

for cfg in CONFIGS:
    r, n_instab, fires = run_config(cfg)
    row, cw = performance_row(cfg["name"], r)
    rows.append(row)
    curves[cfg["name"]] = cw
    instability_summary.append({
        "Config": cfg["name"],
        "Instability Episodes": n_instab,
        "Ceiling Fires": fires["ceiling"],
        "Stability Fires": fires["stability"],
        "Turnover Fires": fires["turnover"],
    })
    print(f"  [OK] {cfg['name']:24s} done — {n_instab} instability episodes, "
          f"fires C:{fires['ceiling']} S:{fires['stability']} T:{fires['turnover']}")

table = pd.DataFrame(rows).set_index("Config")
instab_table = pd.DataFrame(instability_summary).set_index("Config")

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print("\n" + table.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 70)
print("INSTABILITY AND GUARDRAIL ACTIVATION")
print("=" * 70)
print("\n" + instab_table.to_string())

# Validation check against the real 71-month simulation result
real_returns_path = RESULTS_DIR / "monthly_returns.csv"
if real_returns_path.exists():
    real_df = to_datetime_index(pd.read_csv(real_returns_path))
    real_ret_col = "portfolio_return" if "portfolio_return" in real_df.columns else None
    if real_ret_col:
        real_total = (1 + real_df[real_ret_col]).prod() - 1
        config5_total = table.loc["5. All Combined", "Total Return"]
        print("\n" + "=" * 70)
        print("VALIDATION — Config 5 (All Combined) vs. actual simulation result")
        print("=" * 70)
        print(f"  Actual simulation total return:  {real_total:+.4f}")
        print(f"  Config 5 (reconstructed) return: {config5_total:+.4f}")
        diff = abs(real_total - config5_total)
        if diff < 0.02:
            print(f"  [OK] Match within {diff:.4f} — reconstruction is consistent with the real run.")
        else:
            print(f"  [WARN] Differ by {diff:.4f} — worth checking the c_final composition "
                  f"assumption (multiplicative ceiling x stability x reversal) against "
                  f"how agent_risk_guardian.py actually combines them.")


# ──────────────────────────────────────────────────────────────────────
# 4. SAVE OUTPUTS
# ──────────────────────────────────────────────────────────────────────

table.to_csv(OUT_DIR / "guardrail_ablation_table.csv")
instab_table.to_csv(OUT_DIR / "guardrail_ablation_instability.csv")
print(f"\n  [saved] {OUT_DIR / 'guardrail_ablation_table.csv'}")
print(f"  [saved] {OUT_DIR / 'guardrail_ablation_instability.csv'}")

fig, ax = plt.subplots(figsize=(11, 6))
for i, (name, cw) in enumerate(curves.items()):
    ax.plot(cw.index, cw.values, label=name, linewidth=1.6, color=COLORS[i % len(COLORS)])
ax.axhline(1.0, color="#7a7a7a", linewidth=0.8, linestyle="--")
ax.set_ylabel("Cumulative wealth (x)")
ax.set_title("Guardrail Ablation — 6 Configurations")
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
fig.savefig(OUT_DIR / "09_guardrail_ablation_comparison.png")
plt.close(fig)
print("  [chart] saved 09_guardrail_ablation_comparison.png")

with open(OUT_DIR / "guardrail_ablation_report.md", "w") as f:
    f.write("# Guardrail Ablation Study\n\n")
    f.write("## Performance by Configuration\n\n")
    f.write(df_to_markdown(table))
    f.write("\n\n## Instability Episodes and Guardrail Activation\n\n")
    f.write(df_to_markdown(instab_table))
    f.write("\n")
print(f"  [saved] {OUT_DIR / 'guardrail_ablation_report.md'}")

print("\n" + "=" * 70)
print("DONE.")
print("=" * 70)
