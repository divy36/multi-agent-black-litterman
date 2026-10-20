"""Tests for market weights."""

import numpy as np
import pandas as pd
import pytest
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Pure functions copied from build_market_weights.py
# If you change the logic there, update these copies too.
# ---------------------------------------------------------------------------

def _month_end_index(start: str, end: str) -> pd.DatetimeIndex:
    try:
        return pd.date_range(start, end, freq="ME")
    except ValueError:
        return pd.date_range(start, end, freq="M")


def fund_values_to_weights(fund_values: pd.DataFrame) -> pd.DataFrame:
    fv = fund_values.astype(float).copy()
    if (fv < 0).any().any():
        raise ValueError("Negative fund values encountered.")
    row_sums = fv.sum(axis=1)
    if (row_sums <= 0).any():
        bad = row_sums[row_sums <= 0].index.tolist()
        raise ValueError(f"Non-positive total fund value on rows: {bad}")
    return fv.div(row_sums, axis=0)


def backfill_shares(shares_me: pd.Series, full_index: pd.DatetimeIndex) -> Tuple[pd.Series, int]:
    s = shares_me.reindex(full_index).ffill()
    n_backfilled = int(s.isna().sum())
    s = s.bfill()
    return s, n_backfilled


def static_snapshot_weights(snapshot: Dict[str, float], tickers: List[str]) -> pd.Series:
    s = pd.Series({t: float(snapshot[t]) for t in tickers})
    if s.sum() <= 0:
        raise ValueError("Static snapshot sums to <= 0.")
    return s / s.sum()


def validate_weights(weights: pd.DataFrame, tol: float = 1e-9) -> None:
    sums = weights.sum(axis=1)
    off = sums[(sums - 1.0).abs() > tol]
    if len(off):
        raise AssertionError(f"{len(off)} rows do not sum to 1")
    if (weights < -tol).any().any() or (weights > 1 + tol).any().any():
        raise AssertionError("Weights outside [0, 1] found.")


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLB"]

STATIC_SNAPSHOT = {
    "XLK": 32.0, "XLF": 13.0, "XLV": 12.0, "XLY": 10.0,
    "XLI":  8.0, "XLP":  6.0, "XLE":  3.0, "XLU":  2.5,
    "XLRE": 2.0, "XLB":  2.0,
}


# ---------------------------------------------------------------------------
# 1. _month_end_index
# ---------------------------------------------------------------------------

def test_month_end_index_correct_length_and_endpoints():
    idx = _month_end_index("2018-12-31", "2024-12-31")
    # Dec 2018 through Dec 2024 = 73 month-ends
    assert len(idx) == 73, f"expected 73, got {len(idx)}"
    assert idx[0] == pd.Timestamp("2018-12-31"), f"first={idx[0]}"
    assert idx[-1] == pd.Timestamp("2024-12-31"), f"last={idx[-1]}"


def test_month_end_index_72_months_for_2019_2024_sim_window():
    idx = _month_end_index("2019-01-31", "2024-12-31")
    assert len(idx) == 72, f"expected 72, got {len(idx)}"


# ---------------------------------------------------------------------------
# 2. fund_values_to_weights
# ---------------------------------------------------------------------------

def test_fund_values_to_weights_rows_sum_to_one():
    idx = _month_end_index("2019-01-31", "2019-03-31")
    data = {"A": [100.0, 200.0, 150.0],
            "B": [300.0, 100.0, 250.0],
            "C": [50.0, 50.0, 50.0]}
    fv = pd.DataFrame(data, index=idx)
    w = fund_values_to_weights(fv)
    assert np.allclose(w.sum(axis=1), 1.0), "rows don't sum to 1"


def test_fund_values_to_weights_proportional():
    idx = _month_end_index("2019-01-31", "2019-01-31")
    fv = pd.DataFrame({"A": [300.0], "B": [100.0], "C": [100.0], "D": [500.0]}, index=idx)
    w = fund_values_to_weights(fv)
    assert abs(w.iloc[0]["A"] - 0.30) < 1e-10, f"A weight wrong: {w.iloc[0]['A']}"


def test_fund_values_to_weights_rejects_negative():
    idx = _month_end_index("2019-01-31", "2019-01-31")
    fv = pd.DataFrame({"A": [-10.0], "B": [100.0]}, index=idx)
    with pytest.raises(ValueError):
        fund_values_to_weights(fv)


def test_fund_values_to_weights_rejects_zero_row():
    idx = _month_end_index("2019-01-31", "2019-01-31")
    fv = pd.DataFrame({"A": [0.0], "B": [0.0]}, index=idx)
    with pytest.raises(ValueError):
        fund_values_to_weights(fv)


# ---------------------------------------------------------------------------
# 3. backfill_shares
# ---------------------------------------------------------------------------

def test_backfill_shares_no_gap():
    full_idx = _month_end_index("2019-01-31", "2019-06-30")
    shares = pd.Series([1e9] * 6, index=full_idx)
    out, n = backfill_shares(shares, full_idx)
    assert n == 0, f"expected 0 backfilled, got {n}"
    assert len(out) == 6
    assert not out.isna().any()


def test_backfill_shares_2_month_leading_gap():
    full_idx = _month_end_index("2019-01-31", "2019-06-30")
    partial_idx = _month_end_index("2019-03-31", "2019-06-30")
    shares = pd.Series([5e8] * 4, index=partial_idx)
    out, n = backfill_shares(shares, full_idx)
    assert n == 2, f"expected 2 backfilled (Jan+Feb), got {n}"
    assert not out.isna().any(), "NaNs remain after backfill"
    assert np.allclose(out.values, 5e8), f"unexpected values: {out.values}"


def test_backfill_shares_quarterly_forward_fills():
    """Quarterly share counts should forward-fill across intermediate months."""
    full_idx = _month_end_index("2019-01-31", "2019-06-30")
    quarterly_idx = pd.DatetimeIndex([pd.Timestamp("2019-01-31"), pd.Timestamp("2019-04-30")])
    shares = pd.Series([1e9, 1.2e9], index=quarterly_idx)
    out, n = backfill_shares(shares, full_idx)
    assert n == 0
    assert np.isclose(out.iloc[2], 1e9), f"March should be 1e9, got {out.iloc[2]}"
    assert np.isclose(out.iloc[4], 1.2e9), f"May should be 1.2e9, got {out.iloc[4]}"


# ---------------------------------------------------------------------------
# 4. static_snapshot_weights
# ---------------------------------------------------------------------------

def test_static_snapshot_weights_sums_to_one():
    w = static_snapshot_weights(STATIC_SNAPSHOT, TICKERS)
    assert abs(w.sum() - 1.0) < 1e-10, f"sums to {w.sum()}"


def test_static_snapshot_weights_all_positive():
    w = static_snapshot_weights(STATIC_SNAPSHOT, TICKERS)
    assert (w > 0).all(), "not all positive"


def test_static_snapshot_weights_index_matches_tickers_order():
    """Output index must match TICKERS list order, not snapshot dict order."""
    w = static_snapshot_weights(STATIC_SNAPSHOT, TICKERS)
    assert list(w.index) == TICKERS, f"index mismatch: {list(w.index)}"


def test_static_snapshot_weights_xlk_is_largest():
    w = static_snapshot_weights(STATIC_SNAPSHOT, TICKERS)
    assert w.idxmax() == "XLK", f"largest is {w.idxmax()}, expected XLK"


# ---------------------------------------------------------------------------
# 5. validate_weights
# ---------------------------------------------------------------------------

def test_validate_weights_accepts_valid_silently():
    idx = _month_end_index("2019-01-31", "2019-03-31")
    w = pd.DataFrame({"A": [0.3, 0.4, 0.5], "B": [0.7, 0.6, 0.5]}, index=idx)
    validate_weights(w)  # must not raise


def test_validate_weights_catches_bad_sum():
    idx = _month_end_index("2019-01-31", "2019-01-31")
    w = pd.DataFrame({"A": [0.4], "B": [0.4]}, index=idx)  # sums to 0.8
    with pytest.raises(AssertionError):
        validate_weights(w)


def test_validate_weights_catches_negative():
    idx = _month_end_index("2019-01-31", "2019-01-31")
    w = pd.DataFrame({"A": [-0.1], "B": [1.1]}, index=idx)
    with pytest.raises(AssertionError):
        validate_weights(w)


# ---------------------------------------------------------------------------
# 6. BL integration smoke test
# ---------------------------------------------------------------------------

def test_bl_prior_shape():
    """Pi = lambda * Sigma * w_mkt — verify shapes and finiteness."""
    lam = 3.0
    n = len(TICKERS)
    w = static_snapshot_weights(STATIC_SNAPSHOT, TICKERS).values  # (10,)
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((36, n))
    Sigma = np.cov(raw, rowvar=False)  # (10,10)
    Pi = lam * Sigma @ w  # (10,)
    assert Pi.shape == (n,), f"Pi shape wrong: {Pi.shape}"
    assert np.all(np.isfinite(Pi)), "Pi contains inf/nan"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
