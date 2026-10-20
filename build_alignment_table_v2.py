"""Build alignment table v2."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"

# FOMC PDF folders per year
FOMC_FOLDERS = {yr: PROJECT_ROOT / f"{yr} FO" for yr in range(2019, 2025)}

# Simulation window (Feb 2019 = first month with full 36-month covariance seed)
SIM_START = "2019-02-28"
SIM_END   = "2024-12-31"

# FRED series IDs
FRED_SERIES = {
    "cpi_yoy":      "CPIAUCSL",        # CPI all urban consumers
    "pce_yoy":      "PCEPILFE",        # Core PCE (ex food & energy)
    "gdp_growth":   "A191RL1Q225SBEA", # Real GDP growth rate annualised
    "unemployment": "UNRATE",           # Unemployment rate
}

# --------------------------------------------------------------------------- #
# REAL FOMC minutes release dates, verified directly against the Fed's own
# published calendars (fetched 2026-07-21):
#   2019-2020: https://www.federalreserve.gov/monetarypolicy/fomchistorical2019.htm
#              https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm
#   2021-2024: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
#
# Keyed on the MEETING date parsed from the filename (meeting's final day),
# mapped to the REAL date the minutes were actually released to the public.
#
# Two special cases, resolved by checking the Fed's own notes on each page:
#   - The 2019-10-04 and 2020-03-02 unscheduled conference calls were never
#     released as standalone minutes documents -- the Fed's calendar notes
#     say their content was folded into the minutes of the next regularly
#     scheduled meeting (Oct 29-30 2019 and Mar 15 2020 respectively). Their
#     real public-availability date is therefore that later meeting's real
#     release date, not their own meeting date + 21 days.
#   - monetary20200323a1.pdf is an emergency FOMC *statement*, not minutes.
#     Statements are released the same day as the decision, so its real
#     pub_date equals its own filename date -- it is deliberately excluded
#     from this lookup and handled separately below.
# --------------------------------------------------------------------------- #
REAL_MINUTES_RELEASE_DATES: dict[str, str] = {
    # 2019
    "2019-01-30": "2019-02-20",
    "2019-03-20": "2019-04-10",
    "2019-05-01": "2019-05-22",
    "2019-06-19": "2019-07-10",
    "2019-07-31": "2019-08-21",
    "2019-09-18": "2019-10-09",
    "2019-10-04": "2019-11-20",  # unscheduled call, folded into Oct 29-30 minutes
    "2019-10-30": "2019-11-20",
    "2019-12-11": "2020-01-03",
    # 2020
    "2020-01-29": "2020-02-19",
    "2020-03-02": "2020-04-08",  # unscheduled call, folded into Mar 15 minutes
    "2020-03-15": "2020-04-08",
    "2020-04-29": "2020-05-20",
    "2020-06-10": "2020-07-01",
    "2020-07-29": "2020-08-19",
    "2020-09-16": "2020-10-07",
    "2020-11-05": "2020-11-25",
    "2020-12-16": "2021-01-06",
    # 2021
    "2021-01-27": "2021-02-17",
    "2021-03-17": "2021-04-07",
    "2021-04-28": "2021-05-19",
    "2021-06-16": "2021-07-07",
    "2021-07-28": "2021-08-18",
    "2021-09-22": "2021-10-13",
    "2021-11-03": "2021-11-24",
    "2021-12-15": "2022-01-05",
    # 2022
    "2022-01-26": "2022-02-16",
    "2022-03-16": "2022-04-06",
    "2022-05-04": "2022-05-25",
    "2022-06-15": "2022-07-06",
    "2022-07-27": "2022-08-17",
    "2022-09-21": "2022-10-12",
    "2022-11-02": "2022-11-23",
    "2022-12-14": "2023-01-04",
    # 2023
    "2023-02-01": "2023-02-22",
    "2023-03-22": "2023-04-12",
    "2023-05-03": "2023-05-24",
    "2023-06-14": "2023-07-05",
    "2023-07-26": "2023-08-16",
    "2023-09-20": "2023-10-11",
    "2023-11-01": "2023-11-21",
    "2023-12-13": "2024-01-03",
    # 2024
    "2024-01-31": "2024-02-21",
    "2024-03-20": "2024-04-10",
    "2024-05-01": "2024-05-22",
    "2024-06-12": "2024-07-03",
    "2024-07-31": "2024-08-21",
    "2024-09-18": "2024-10-09",
    "2024-11-07": "2024-11-26",
    "2024-12-18": "2025-01-08",
}


def _approximate_release_date(meeting_date: pd.Timestamp) -> pd.Timestamp:
    """
    FALLBACK ONLY -- not exercised by the current dataset (every file present
    is covered by REAL_MINUTES_RELEASE_DATES above). Documented approximation
    for any future meeting date not in the real-dates table: FOMC minutes are
    released on the Wednesday three weeks (21 days) after the meeting's final
    day. Compute meeting_date + 21 days, then roll forward to the next
    Wednesday if that lands elsewhere (in practice the Fed's actual release
    dates are already Wednesdays close to but not always exactly 21 days out,
    so this is a reasoned approximation, not a confirmed real date).
    """
    candidate = meeting_date + pd.Timedelta(days=21)
    days_to_wednesday = (2 - candidate.weekday()) % 7  # Monday=0 ... Wednesday=2
    return candidate + pd.Timedelta(days=days_to_wednesday)


# --------------------------------------------------------------------------- #
# Step 1 — Scan all FOMC PDF files and extract their dates
# --------------------------------------------------------------------------- #
def scan_fomc_files() -> pd.DataFrame:
    """
    Scan all year folders for FOMC PDF files.

    FIX: returns TWO date columns now, not one ambiguous "pub_date":
      - meeting_date : the date parsed from the filename (the meeting's own
                        final day -- this is what the OLD code incorrectly
                        called "pub_date").
      - pub_date     : the REAL publication date, looked up from
                        REAL_MINUTES_RELEASE_DATES (falling back to
                        _approximate_release_date() if a meeting date isn't
                        in the table -- not triggered for the current files).
    Statement-type files (monetary*.pdf) are same-day releases, so their
    pub_date equals their own meeting_date directly.

    Returns DataFrame with columns: meeting_date, pub_date (Timestamp),
    filename, year. Sorted by pub_date ascending (this is now the correct
    sort key for the look-ahead-safe join in map_months_to_fomc()).
    """
    records = []

    # Regex patterns for different filename conventions
    patterns = [
        re.compile(r"fomcminutes(\d{8})\.pdf"),        # 2021-2024: fomcminutes20220126.pdf
        re.compile(r"FOMC(\d{8})meeting\.pdf"),         # 2019-2020: FOMC20190130meeting.pdf
        re.compile(r"fomcminutes(\d{8})meeting\.pdf"),  # variant
        re.compile(r"monetary(\d{8})\w*\.pdf"),         # emergency statements
    ]
    statement_pattern = re.compile(r"monetary(\d{8})\w*\.pdf")

    for year, folder in FOMC_FOLDERS.items():
        if not folder.exists():
            print(f"  WARNING: folder not found: {folder}")
            continue

        for pdf_file in sorted(folder.glob("*.pdf")):
            date_str = None
            for pat in patterns:
                m = pat.match(pdf_file.name)
                if m:
                    date_str = m.group(1)
                    break

            if date_str is None:
                # Try any 8-digit sequence in the filename
                m = re.search(r"(\d{8})", pdf_file.name)
                if m:
                    date_str = m.group(1)

            if date_str:
                try:
                    meeting_date = pd.Timestamp(
                        f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    )
                    if statement_pattern.match(pdf_file.name):
                        # Same-day-release statement, not lagged minutes.
                        pub_date = meeting_date
                    else:
                        key = meeting_date.strftime("%Y-%m-%d")
                        if key in REAL_MINUTES_RELEASE_DATES:
                            pub_date = pd.Timestamp(REAL_MINUTES_RELEASE_DATES[key])
                        else:
                            pub_date = _approximate_release_date(meeting_date)
                            print(f"  WARNING: no real release date on file for "
                                  f"{pdf_file.name} (meeting {key}) -- using "
                                  f"21-day+Wednesday approximation: {pub_date.date()}")
                    records.append({
                        "meeting_date": meeting_date,
                        "pub_date": pub_date,
                        "filename": pdf_file.name,
                        "year":     year,
                    })
                except Exception:
                    print(f"  WARNING: could not parse date from {pdf_file.name}")

    if not records:
        print(f"  WARNING: no FOMC files matched any pattern in any folder.")
        return pd.DataFrame(columns=["meeting_date", "pub_date", "filename", "year"])

    df = pd.DataFrame(records).sort_values("pub_date").reset_index(drop=True)
    print(f"Found {len(df)} FOMC files across {len(FOMC_FOLDERS)} folders")
    return df


# --------------------------------------------------------------------------- #
# Step 2 — Map each simulation month to the most recent prior FOMC publication
# --------------------------------------------------------------------------- #
def map_months_to_fomc(fomc_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each simulation month-end, find the most recent FOMC minutes
    published BEFORE or ON that month-end date.

    This function's join logic is UNCHANGED from the original -- it always
    joined on whatever column is named "pub_date". The look-ahead-bias fix
    lives entirely in scan_fomc_files(), which now populates "pub_date" with
    the REAL release date instead of the meeting date. The practical effect:
    any month whose old (meeting-date-based) fomc_file assignment relied on a
    document released only 1-3 weeks before that month-end, but which crosses
    into the FOLLOWING month once the real ~3-week release lag is accounted
    for, will now be assigned the PRIOR available document instead -- see
    Phase 2/3 validation output for the concrete months this changes.

    Returns DataFrame indexed by month_end with fomc_date and fomc_file columns.
    """
    try:
        months = pd.date_range(SIM_START, SIM_END, freq="ME")
    except Exception:
        months = pd.date_range(SIM_START, SIM_END, freq="M")

    records = []
    for month_end in months:
        # All FOMC docs published on or before this month-end
        prior = fomc_df[fomc_df["pub_date"] <= month_end]
        if prior.empty:
            print(f"  WARNING: no FOMC doc found before {month_end.date()}")
            records.append({
                "month_end": month_end,
                "fomc_date": None,
                "fomc_file": None,
            })
        else:
            latest = prior.iloc[-1]
            records.append({
                "month_end": month_end,
                "fomc_date": latest["pub_date"].strftime("%Y-%m-%d"),
                "fomc_file": latest["filename"],
            })

    return pd.DataFrame(records).set_index("month_end")


# --------------------------------------------------------------------------- #
# Step 3 — Download FRED macro data
# --------------------------------------------------------------------------- #
def fetch_fred_data() -> pd.DataFrame:
    """
    Download CPI, Core PCE, GDP growth, and unemployment from FRED.
    Returns monthly DataFrame covering the full simulation window.
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("ERROR: fredapi not installed. Run: pip install fredapi")
        sys.exit(1)

    # Try to find API key
    api_key = None

    # Check environment variable
    import os
    api_key = os.environ.get("FRED_API_KEY")

    # Check common config locations
    if not api_key:
        for path in [
            PROJECT_ROOT / "fred_api_key.txt",
            Path.home() / ".fred_api_key",
            PROJECT_ROOT / ".env",
        ]:
            if path.exists():
                text = path.read_text().strip()
                # Handle .env format: FRED_API_KEY=abc123
                if "=" in text:
                    for line in text.splitlines():
                        if "FRED_API_KEY" in line:
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                else:
                    api_key = text
                if api_key:
                    print(f"  Found FRED API key in {path}")
                    break

    if not api_key:
        print("\nFRED API key not found.")
        print("Options:")
        print("  1. Set environment variable: export FRED_API_KEY=your_key_here")
        print("  2. Save key to: <project root>/fred_api_key.txt")
        print("  3. Enter it now:")
        api_key = input("  FRED API key: ").strip()
        if not api_key:
            print("No key provided — using synthetic macro data as fallback.")
            return _synthetic_macro_data()

    fred = Fred(api_key=api_key)

    # Download window — slightly wider to allow YoY computation
    fetch_start = "2017-01-01"
    fetch_end   = "2025-01-31"

    macro_frames = {}
    for col_name, series_id in FRED_SERIES.items():
        try:
            print(f"  Downloading {series_id} ({col_name})...")
            s = fred.get_series(series_id, observation_start=fetch_start,
                                observation_end=fetch_end)
            s.index = pd.to_datetime(s.index)
            macro_frames[col_name] = s
            print(f"    Got {len(s)} observations")
        except Exception as e:
            print(f"  WARNING: failed to download {series_id}: {e}")
            macro_frames[col_name] = pd.Series(dtype=float)

    # ── Process CPI → YoY % ─────────────────────────────────────────────────
    try:
        cpi_monthly = pd.date_range(SIM_START, SIM_END, freq="ME")
        cpi = macro_frames["cpi_yoy"].resample("ME").last().ffill()
        cpi_yoy = cpi.pct_change(12) * 100   # 12-month % change
        macro_frames["cpi_yoy"] = cpi_yoy
    except Exception as e:
        print(f"  WARNING: CPI YoY computation failed: {e}")

    # ── Process Core PCE → YoY % ────────────────────────────────────────────
    try:
        pce = macro_frames["pce_yoy"].resample("ME").last().ffill()
        pce_yoy = pce.pct_change(12) * 100
        macro_frames["pce_yoy"] = pce_yoy
    except Exception as e:
        print(f"  WARNING: PCE YoY computation failed: {e}")

    # ── GDP: already annualised quarterly growth rate ────────────────────────
    # Forward-fill quarterly to monthly
    try:
        gdp = macro_frames["gdp_growth"]
        gdp.index = pd.to_datetime(gdp.index)
        gdp_monthly = gdp.resample("ME").last().ffill()
        macro_frames["gdp_growth"] = gdp_monthly
    except Exception as e:
        print(f"  WARNING: GDP processing failed: {e}")

    # ── Unemployment: already monthly ───────────────────────────────────────
    try:
        unemp = macro_frames["unemployment"]
        unemp.index = pd.to_datetime(unemp.index)
        unemp_monthly = unemp.resample("ME").last().ffill()
        macro_frames["unemployment"] = unemp_monthly
    except Exception as e:
        print(f"  WARNING: Unemployment processing failed: {e}")

    # ── Combine into monthly DataFrame ──────────────────────────────────────
    try:
        months = pd.date_range(SIM_START, SIM_END, freq="ME")
    except Exception:
        months = pd.date_range(SIM_START, SIM_END, freq="M")

    result = pd.DataFrame(index=months)
    for col, series in macro_frames.items():
        if isinstance(series, pd.Series) and len(series) > 0:
            result[col] = series.reindex(months, method="ffill")
        else:
            result[col] = np.nan

    print(f"  Macro data: {result.shape[0]} months, "
          f"{result.isna().sum().sum()} NaN values")
    return result


def _synthetic_macro_data() -> pd.DataFrame:
    """
    Synthetic macro data fallback when FRED is unavailable.
    Uses approximate real values for 2019-2024.
    Only used if FRED download fails entirely.
    """
    print("  Using synthetic macro data (FRED unavailable)")
    try:
        months = pd.date_range(SIM_START, SIM_END, freq="ME")
    except Exception:
        months = pd.date_range(SIM_START, SIM_END, freq="M")

    n = len(months)
    np.random.seed(42)

    # Approximate real macro trajectory 2019-2024
    cpi_path  = np.concatenate([
        np.linspace(1.8, 1.2, 12),   # 2019: low stable
        np.linspace(1.5, 1.2, 11),   # 2020: COVID
        np.linspace(2.0, 7.0, 12),   # 2021: transitory inflation
        np.linspace(7.5, 4.0, 12),   # 2022: peak and decline
        np.linspace(3.5, 2.5, 12),   # 2023: continuing decline
        np.linspace(2.5, 2.4, 12),   # 2024: near target
    ])[:n]

    return pd.DataFrame({
        "cpi_yoy":      cpi_path,
        "pce_yoy":      cpi_path * 0.85,  # Core PCE typically below CPI
        "gdp_growth":   np.where(months.year == 2020,
                                 np.random.uniform(-8, 2, n),
                                 np.random.uniform(1.5, 3.5, n)),
        "unemployment": _unemployment_path(n),
    }, index=months)


def _unemployment_path(n: int) -> np.ndarray:
    """
    Approximate 2019 -> COVID spike -> recovery unemployment path.

    Guards the split point at min(15, n) so this never crashes for a
    simulation window shorter than 15 months (e.g. a short smoke-test
    window) — the original hardcoded split of 15 assumed n >= 15, which
    np.linspace(14.7, 3.5, n - 15) violates for any shorter n.
    """
    split = min(15, n)
    return np.concatenate([
        np.linspace(3.8, 14.7, split),        # pre-spike -> COVID spike
        np.linspace(14.7, 3.5, max(n - split, 0)),  # spike -> recovery
    ])[:n]


# --------------------------------------------------------------------------- #
# Step 4 — Add VIX from yfinance
# --------------------------------------------------------------------------- #
def fetch_vix() -> pd.Series:
    """Download monthly VIX close from yfinance."""
    try:
        import yfinance as yf
        print("  Downloading VIX from yfinance...")
        vix = yf.download("^VIX", start="2019-01-01", end="2025-01-31",
                          interval="1mo", progress=False, auto_adjust=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix["Close"]["^VIX"]
        else:
            vix = vix["Close"]
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        try:
            vix = vix.resample("ME").last()
        except Exception:
            vix = vix.resample("M").last()
        print(f"  VIX: {len(vix)} months")
        return vix
    except Exception as e:
        print(f"  WARNING: VIX download failed ({e}), using default 20.0")
        try:
            months = pd.date_range(SIM_START, SIM_END, freq="ME")
        except Exception:
            months = pd.date_range(SIM_START, SIM_END, freq="M")
        return pd.Series(20.0, index=months)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Building monthly alignment table...")
    print(f"Window: {SIM_START} → {SIM_END}")
    print()

    # Step 1: Scan FOMC files
    print("Step 1: Scanning FOMC PDF files...")
    fomc_df = scan_fomc_files()
    print(f"  Date range: {fomc_df['pub_date'].min().date()} → "
          f"{fomc_df['pub_date'].max().date()}")
    print()

    # Step 2: Map months to FOMC docs
    print("Step 2: Mapping simulation months to FOMC documents...")
    mapping = map_months_to_fomc(fomc_df)
    mapped = mapping["fomc_date"].notna().sum()
    print(f"  {mapped}/{len(mapping)} months mapped to an FOMC document")
    print()

    # Step 3: FRED macro data
    print("Step 3: Fetching FRED macro data...")
    macro = fetch_fred_data()
    print()

    # Step 4: VIX
    print("Step 4: Fetching VIX...")
    vix = fetch_vix()
    print()

    # Step 5: Combine
    print("Step 5: Building alignment table...")
    alignment = mapping.copy()
    for col in macro.columns:
        alignment[col] = macro[col].reindex(alignment.index, method="ffill")
    alignment["vix_close"] = vix.reindex(alignment.index, method="ffill")

    # Add text file column (same name, .txt extension)
    alignment["fomc_text_file"] = alignment["fomc_file"].apply(
        lambda f: Path(f).stem + ".txt" if pd.notna(f) and f else None
    )

    # Step 6: Save
    # NOTE: written to a new "_postfix" filename, NOT the original
    # monthly_alignment.csv, so the input that produced the already-validated
    # results_llama3_backup/ and results/ output remains untouched on disk.
    out_path = DATA_DIR / "monthly_alignment_postfix.csv"
    alignment.to_csv(out_path, index_label="month_end")
    print(f"Saved: {out_path}")
    print(f"Shape: {alignment.shape}")
    print(f"NaN counts:\n{alignment.isna().sum().to_string()}")
    print()

    # Step 7: Validation
    print("Validation:")
    print(f"  Months in window:      {len(alignment)}")
    print(f"  FOMC docs mapped:      {alignment['fomc_date'].notna().sum()}")
    print(f"  CPI data coverage:     {alignment['cpi_yoy'].notna().sum()}")
    print(f"  VIX data coverage:     {alignment['vix_close'].notna().sum()}")

    print()
    print("Sample rows:")
    sample_dates = ["2019-02-28", "2020-03-31", "2022-09-30", "2024-12-31"]
    for d in sample_dates:
        ts = pd.Timestamp(d)
        if ts in alignment.index:
            row = alignment.loc[ts]
            print(f"  {d}: fomc={row['fomc_date']}  "
                  f"cpi={row['cpi_yoy']:.2f}%  "
                  f"unemp={row['unemployment']:.1f}%  "
                  f"vix={row['vix_close']:.1f}")

    print()
    print("Done. Run simulation_loop.py --dry-run to verify.")


if __name__ == "__main__":
    main()
