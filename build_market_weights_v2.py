"""Build market weights v2."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data"

# ETF order used throughout the project
SEC = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLB"]

# 11-sector anchor weights (%), keyed by the month-end row they anchor.
# Sector order here: Tech, Fin, Energy, HealthCare, Industrials, ConsDisc,
# Staples, RealEstate, Utilities, Materials, CommSvcs (dropped below).
ANCHORS_RAW = {
    # source A: Bespoke p.4 year-end table (exact published values)
    "2018-12-31": [20.12, 13.31, 5.32, 15.54, 9.20,  9.94, 7.41, 2.96, 3.34, 2.73, 10.12],
    "2019-12-31": [23.20, 12.95, 4.35, 14.20, 9.05,  9.75, 7.20, 2.93, 3.32, 2.65, 10.39],
    "2020-12-31": [27.61, 10.44, 2.28, 13.46, 8.40, 12.72, 6.51, 2.42, 2.76, 2.63, 10.77],
    "2021-12-31": [29.17, 10.69, 2.67, 13.29, 7.78, 12.54, 5.88, 2.77, 2.50, 2.56, 10.16],
    "2022-12-31": [25.74, 11.66, 5.23, 15.82, 8.65,  9.80, 7.20, 2.71, 3.18, 2.73,  7.28],
    # source A: Bespoke p.1, "current" as of 2023-06-30
    "2023-06-30": [28.10, 12.45, 4.13, 13.30, 8.48, 10.79, 6.69, 2.52, 2.59, 2.51,  8.36],
    # source B: iShares IVV exposure as of 2023-12-19 (cash line excluded)
    "2023-12-31": [28.85, 12.94, 3.95, 12.45, 8.74, 10.98, 6.13, 2.50, 2.34, 2.40,  8.49],
    # source B: iShares IVV exposure as of 2024-12-27 (cash line excluded)
    "2024-12-31": [32.63, 13.50, 3.07, 10.02, 8.11, 11.36, 5.50, 2.06, 2.30, 1.88,  9.37],
}
ANCHOR_SOURCE = {
    "2018-12-31": "Bespoke 2023-07-03 report, p.4 year-end table",
    "2019-12-31": "Bespoke 2023-07-03 report, p.4 year-end table",
    "2020-12-31": "Bespoke 2023-07-03 report, p.4 year-end table",
    "2021-12-31": "Bespoke 2023-07-03 report, p.4 year-end table",
    "2022-12-31": "Bespoke 2023-07-03 report, p.4 year-end table",
    "2023-06-30": "Bespoke 2023-07-03 report, p.1 current weights (as of 2023-06-30)",
    "2023-12-31": "iShares IVV sector exposure as of 2023-12-19 (Wayback 20231220225606)",
    "2024-12-31": "iShares IVV sector exposure as of 2024-12-27 (Wayback 20241231072103)",
}


def anchors_ex_comm() -> dict[pd.Timestamp, np.ndarray]:
    """Drop Communication Services, renormalise the 10 remaining sectors."""
    out = {}
    for date, vals in ANCHORS_RAW.items():
        ten = np.array(vals[:10], dtype=float)   # last entry is CommSvcs
        out[pd.Timestamp(date)] = ten / ten.sum()
    return out


def main() -> None:
    anchors = anchors_ex_comm()
    prices = pd.read_csv(DATA / "etf_prices_monthly.csv",
                         index_col="Date", parse_dates=True)[SEC]

    months = pd.date_range("2018-12-31", "2024-12-31", freq="ME")
    a_dates = sorted(anchors)

    rows, prov = {}, {}
    print("Drift-only error at each anchor (method-accuracy evidence, "
          "max abs pp across 10 sectors BEFORE reconciliation):")
    for k in range(len(a_dates) - 1):
        A, B = a_dates[k], a_dates[k + 1]
        wA, wB = anchors[A], anchors[B]
        drift_B = wA * (prices.loc[B].values / prices.loc[A].values)
        drift_B = drift_B / drift_B.sum()
        K = wB / drift_B
        print(f"  {A.date()} -> {B.date()}: {np.abs(drift_B - wB).max()*100:.2f} pp")
        span = [m for m in months if A < m < B]
        n_span = (B.to_period('M') - A.to_period('M')).n
        for m in span:
            f = (m.to_period('M') - A.to_period('M')).n / n_span
            w = wA * (prices.loc[m].values / prices.loc[A].values)
            w = w / w.sum()
            w = w * (K ** f)
            w = w / w.sum()
            rows[m] = w
            prov[m] = (f"DRIFT-INTERPOLATED between anchors {A.date()} and "
                       f"{B.date()} (ETF-price drift, geometric reconciliation)")
    for d in a_dates:
        rows[d] = anchors[d]
        prov[d] = f"ANCHOR — {ANCHOR_SOURCE[d.strftime('%Y-%m-%d')]}"

    w_df = pd.DataFrame({m: rows[m] for m in months}, index=SEC).T
    w_df.index.name = "Date"

    # sanity
    assert len(w_df) == 73, len(w_df)
    assert np.allclose(w_df.sum(axis=1), 1.0, atol=1e-9)
    assert not w_df.isna().any().any()

    # back up the old tainted file, then write
    old = DATA / "market_cap_weights.csv"
    backup = DATA / "market_cap_weights_STATIC_TAINTED_BACKUP.csv"
    if old.exists() and not backup.exists():
        old_df = pd.read_csv(old)
        if len(old_df.drop_duplicates(subset=[c for c in old_df.columns if c != "Date"])) == 1:
            shutil.copy2(old, backup)
            print(f"\n[backup] old constant-row file -> {backup.name}")
    w_df.round(6).to_csv(old)

    prov_df = pd.DataFrame({"Date": [m.date() for m in months],
                            "provenance": [prov[m] for m in months]})
    prov_df.to_csv(DATA / "market_cap_weights_provenance.csv", index=False)

    n_anchor = sum(1 for m in months if m in anchors)
    print(f"\nSaved {old}  ({len(w_df)} rows, {w_df.index.min().date()} -> "
          f"{w_df.index.max().date()})")
    print(f"  ANCHOR rows: {n_anchor}   DRIFT-INTERPOLATED rows: {len(w_df)-n_anchor}")
    print(f"  Provenance:  {DATA/'market_cap_weights_provenance.csv'}")
    print("\nXLK path (was constant 0.3536 in the tainted file):")
    show = ["2018-12-31", "2019-06-30", "2019-12-31", "2020-03-31", "2021-12-31",
            "2022-12-31", "2023-12-31", "2024-12-31"]
    for s in show:
        m = pd.Timestamp(s)
        tag = "ANCHOR" if m in anchors else "interp"
        print(f"  {s}  XLK {w_df.loc[m,'XLK']:.4f}  XLE {w_df.loc[m,'XLE']:.4f}  "
              f"XLV {w_df.loc[m,'XLV']:.4f}   [{tag}]")


if __name__ == "__main__":
    main()
