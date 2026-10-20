"""Tests for build etf prices."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import build_etf_prices as bep


TICKERS = bep.TICKERS


# ═══════════════════════════════════════════════════════════════════════
# 1. _month_end_index
# ═══════════════════════════════════════════════════════════════════════
class TestMonthEndIndex:

    def test_correct_length_and_endpoints(self):
        idx = bep._month_end_index("2016-01-31", "2024-12-31")
        assert idx[0] == pd.Timestamp("2016-01-31")
        assert idx[-1] == pd.Timestamp("2024-12-31")
        assert len(idx) == 108  # 9 years * 12 months


# ═══════════════════════════════════════════════════════════════════════
# 2. prices_to_returns
# ═══════════════════════════════════════════════════════════════════════
class TestPricesToReturns:

    def test_first_row_is_nan(self):
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        prices = pd.DataFrame({"XLK": [100.0, 110.0, 121.0, 108.9]}, index=idx)
        returns = bep.prices_to_returns(prices)
        assert pd.isna(returns.iloc[0]["XLK"])

    def test_matches_hand_computed_pct_change(self):
        idx = pd.date_range("2020-01-31", periods=3, freq="ME")
        prices = pd.DataFrame({"XLK": [100.0, 110.0, 99.0]}, index=idx)
        returns = bep.prices_to_returns(prices)
        assert returns.iloc[1]["XLK"] == pytest.approx(0.10, abs=1e-9)
        assert returns.iloc[2]["XLK"] == pytest.approx(-0.10, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# 3. check_covariance_seed
# ═══════════════════════════════════════════════════════════════════════
class TestCheckCovarianceSeed:

    def _make_returns(self, n_months, seed=0):
        idx = pd.date_range("2016-01-31", periods=n_months, freq="ME")
        rng = np.random.default_rng(seed)
        data = rng.normal(0, 0.02, size=(n_months, len(TICKERS)))
        return pd.DataFrame(data, index=idx, columns=TICKERS)

    def test_passes_silently_with_full_window(self, capsys):
        # 36 months before "2019-01-31" -> needs Jan 2016 through Dec 2018 (36 months)
        returns = self._make_returns(36)
        sim_start = (returns.index[-1] + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        bep.check_covariance_seed(returns, sim_start, window=36)  # must not raise
        captured = capsys.readouterr()
        assert "PASSED" in captured.out

    def test_raises_when_insufficient_pre_simulation_history(self):
        returns = self._make_returns(20)  # fewer than the required window
        sim_start = (returns.index[-1] + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        with pytest.raises(AssertionError, match="Only 20 months"):
            bep.check_covariance_seed(returns, sim_start, window=36)

    def test_raises_and_names_bad_tickers_when_sigma_has_nan(self):
        returns = self._make_returns(36)
        returns["XLRE"] = np.nan  # one ticker entirely NaN -> NaN covariance
        sim_start = (returns.index[-1] + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        with pytest.raises(AssertionError, match="XLRE"):
            bep.check_covariance_seed(returns, sim_start, window=36)

    def test_uses_only_the_last_window_months_not_all_history(self, capsys):
        """A much longer history than the window should still only report
        the trailing `window` months as the seed window used."""
        returns = self._make_returns(60)  # far more than the 36-month window
        sim_start = (returns.index[-1] + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        bep.check_covariance_seed(returns, sim_start, window=36)
        captured = capsys.readouterr()
        assert "36 months" in captured.out


# ═══════════════════════════════════════════════════════════════════════
# 4. download_prices — yfinance mocked, not hit over the network
# ═══════════════════════════════════════════════════════════════════════
class TestDownloadPricesWithMockedYfinance:

    def _multi_ticker_raw(self, tickers, n_months=110, seed=0):
        """Builds a fake yf.download-style MultiIndex-column DataFrame."""
        idx = pd.date_range(bep.SERIES_START, periods=n_months, freq="ME")
        rng = np.random.default_rng(seed)
        cols = pd.MultiIndex.from_product([["Close", "Open"], tickers])
        data = rng.uniform(50, 200, size=(n_months, len(cols)))
        return pd.DataFrame(data, index=idx, columns=cols)

    def test_multi_ticker_extracts_close_and_enforces_column_order(self, monkeypatch):
        tickers = TICKERS[:3]
        raw = self._multi_ticker_raw(tickers)

        def fake_download(*args, **kwargs):
            return raw

        import yfinance
        monkeypatch.setattr(yfinance, "download", fake_download)

        prices = bep.download_prices(tickers, bep.DOWNLOAD_START, bep.DOWNLOAD_END)
        assert list(prices.columns) == tickers
        assert prices.index.min() >= pd.Timestamp(bep.SERIES_START)
        assert prices.index.max() <= pd.Timestamp(bep.SERIES_END)

    def test_single_ticker_non_multiindex_path(self, monkeypatch):
        ticker = [TICKERS[0]]
        idx = pd.date_range(bep.SERIES_START, periods=110, freq="ME")
        rng = np.random.default_rng(1)
        raw = pd.DataFrame({"Close": rng.uniform(50, 200, size=110),
                             "Open": rng.uniform(50, 200, size=110)}, index=idx)

        def fake_download(*args, **kwargs):
            return raw

        import yfinance
        monkeypatch.setattr(yfinance, "download", fake_download)

        prices = bep.download_prices(ticker, bep.DOWNLOAD_START, bep.DOWNLOAD_END)
        assert list(prices.columns) == ticker

    def test_raises_runtime_error_on_4_plus_consecutive_nans(self, monkeypatch):
        """
        ffill(limit=1) fills only the FIRST NaN of a consecutive run, so a
        raw run of 3 leaves 2 remaining (not > 2, does not raise) -- a raw
        run of 4 leaves 3 remaining, which does trigger the RuntimeError.
        """
        tickers = TICKERS[:2]
        idx = pd.date_range(bep.SERIES_START, periods=110, freq="ME")
        rng = np.random.default_rng(2)
        cols = pd.MultiIndex.from_product([["Close"], tickers])
        data = rng.uniform(50, 200, size=(110, len(tickers)))
        raw = pd.DataFrame(data, index=idx, columns=cols)
        raw.loc[raw.index[10:14], ("Close", tickers[0])] = np.nan  # 4 in a row

        def fake_download(*args, **kwargs):
            return raw

        import yfinance
        monkeypatch.setattr(yfinance, "download", fake_download)

        with pytest.raises(RuntimeError, match=tickers[0]):
            bep.download_prices(tickers, bep.DOWNLOAD_START, bep.DOWNLOAD_END)

    def test_does_not_raise_on_exactly_2_consecutive_nans(self, monkeypatch):
        """The check is `consec > 2`, so exactly 2 consecutive NaNs (which
        ffill(limit=1) reduces to 1 remaining NaN) must NOT raise."""
        tickers = TICKERS[:2]
        idx = pd.date_range(bep.SERIES_START, periods=110, freq="ME")
        rng = np.random.default_rng(3)
        cols = pd.MultiIndex.from_product([["Close"], tickers])
        data = rng.uniform(50, 200, size=(110, len(tickers)))
        raw = pd.DataFrame(data, index=idx, columns=cols)
        raw.loc[raw.index[10:12], ("Close", tickers[0])] = np.nan  # exactly 2 in a row

        def fake_download(*args, **kwargs):
            return raw

        import yfinance
        monkeypatch.setattr(yfinance, "download", fake_download)

        prices = bep.download_prices(tickers, bep.DOWNLOAD_START, bep.DOWNLOAD_END)  # must not raise
        assert prices is not None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
