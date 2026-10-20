"""Tests for build alignment table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import build_alignment_table as bat


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _touch(path: Path) -> None:
    """Create an empty file — scan_fomc_files only reads the filename, not
    the file content, so an empty placeholder is sufficient."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def make_fomc_df(pub_dates: list[str], filenames: list[str] | None = None) -> pd.DataFrame:
    """Build a synthetic fomc_df in the exact shape scan_fomc_files() returns,
    bypassing the filesystem entirely — used for testing map_months_to_fomc
    in isolation from scan_fomc_files."""
    if filenames is None:
        filenames = [f"fomcminutes{d.replace('-', '')}.pdf" for d in pub_dates]
    df = pd.DataFrame({
        "pub_date": [pd.Timestamp(d) for d in pub_dates],
        "filename": filenames,
        "year": [pd.Timestamp(d).year for d in pub_dates],
    })
    return df.sort_values("pub_date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 1. scan_fomc_files — filename parsing across all four regex conventions
# --------------------------------------------------------------------------- #
class TestScanFomcFiles:

    @pytest.mark.parametrize("filename,expected_date", [
        ("fomcminutes20220126.pdf", "2022-01-26"),      # 2021-2024 convention
        ("FOMC20190130meeting.pdf", "2019-01-30"),       # 2019-2020 convention
        ("fomcminutes20200315meeting.pdf", "2020-03-15"),  # variant
        ("monetary20200315statement.pdf", "2020-03-15"),   # emergency statement
    ])
    def test_recognised_filename_patterns_parse_correctly(
        self, tmp_path, monkeypatch, filename, expected_date
    ):
        folder = tmp_path / "2020 FO"
        _touch(folder / filename)
        monkeypatch.setattr(bat, "FOMC_FOLDERS", {2020: folder})

        df = bat.scan_fomc_files()

        assert len(df) == 1
        assert df.iloc[0]["pub_date"] == pd.Timestamp(expected_date)
        assert df.iloc[0]["filename"] == filename

    def test_unparseable_filename_is_skipped_not_crashed(self, tmp_path, monkeypatch, capsys):
        """
        FIXED (was a discovered bug): scan_fomc_files() used to raise
        KeyError: 'pub_date' when zero files matched, because
        pd.DataFrame([]) has no columns to sort on. Now guarded with an
        explicit empty-result return before .sort_values() is called.
        """
        folder = tmp_path / "2020 FO"
        _touch(folder / "no_date_here.pdf")
        monkeypatch.setattr(bat, "FOMC_FOLDERS", {2020: folder})

        df = bat.scan_fomc_files()  # must not raise
        assert len(df) == 0
        assert list(df.columns) == ["pub_date", "filename", "year"]

    def test_missing_folder_is_skipped_not_crashed(self, tmp_path, monkeypatch):
        """FIXED (was a discovered bug) — same root cause as above."""
        monkeypatch.setattr(bat, "FOMC_FOLDERS", {2020: tmp_path / "does_not_exist"})
        df = bat.scan_fomc_files()  # must not raise
        assert len(df) == 0

    def test_results_sorted_ascending_by_pub_date(self, tmp_path, monkeypatch):
        folder = tmp_path / "2019 FO"
        _touch(folder / "FOMC20190918meeting.pdf")
        _touch(folder / "FOMC20190130meeting.pdf")
        _touch(folder / "FOMC20190501meeting.pdf")
        monkeypatch.setattr(bat, "FOMC_FOLDERS", {2019: folder})

        df = bat.scan_fomc_files()

        assert list(df["pub_date"]) == sorted(df["pub_date"])
        assert df.iloc[0]["pub_date"] == pd.Timestamp("2019-01-30")
        assert df.iloc[-1]["pub_date"] == pd.Timestamp("2019-09-18")

    def test_multiple_years_combined_and_sorted(self, tmp_path, monkeypatch):
        folder_2019 = tmp_path / "2019 FO"
        folder_2020 = tmp_path / "2020 FO"
        _touch(folder_2019 / "FOMC20190918meeting.pdf")
        _touch(folder_2020 / "fomcminutes20200129.pdf")
        monkeypatch.setattr(bat, "FOMC_FOLDERS", {2019: folder_2019, 2020: folder_2020})

        df = bat.scan_fomc_files()

        assert len(df) == 2
        assert list(df["pub_date"]) == sorted(df["pub_date"])


# --------------------------------------------------------------------------- #
# 2. map_months_to_fomc — the look-ahead-bias-critical logic
# --------------------------------------------------------------------------- #
class TestMapMonthsToFomcLookAheadBias:
    """
    These are the highest-value tests in this file: they directly defend
    the "no look-ahead bias" claim made in the Methodology analysis.
    """

    def test_month_end_maps_to_most_recent_prior_publication(self, monkeypatch):
        monkeypatch.setattr(bat, "SIM_START", "2019-01-31")
        monkeypatch.setattr(bat, "SIM_END", "2019-07-31")

        fomc_df = make_fomc_df(["2019-01-30", "2019-03-20", "2019-06-19"])
        mapping = bat.map_months_to_fomc(fomc_df)

        assert mapping.loc[pd.Timestamp("2019-01-31"), "fomc_date"] == "2019-01-30"
        # Feb: no new doc since Jan -> still the Jan doc, NOT skipped/None
        assert mapping.loc[pd.Timestamp("2019-02-28"), "fomc_date"] == "2019-01-30"
        # Mar: the Mar 20 doc is now available (published before Mar 31 month-end)
        assert mapping.loc[pd.Timestamp("2019-03-31"), "fomc_date"] == "2019-03-20"
        # Apr, May: still the Mar doc, not the June one
        assert mapping.loc[pd.Timestamp("2019-04-30"), "fomc_date"] == "2019-03-20"
        assert mapping.loc[pd.Timestamp("2019-05-31"), "fomc_date"] == "2019-03-20"
        # Jun: the Jun 19 doc is now available
        assert mapping.loc[pd.Timestamp("2019-06-30"), "fomc_date"] == "2019-06-19"

    def test_doc_published_exactly_on_month_end_is_included(self, monkeypatch):
        """The mapping uses <=, so a doc published exactly on the month-end
        date must be available for that same month-end's rebalance."""
        monkeypatch.setattr(bat, "SIM_START", "2019-01-31")
        monkeypatch.setattr(bat, "SIM_END", "2019-01-31")

        fomc_df = make_fomc_df(["2019-01-31"])
        mapping = bat.map_months_to_fomc(fomc_df)

        assert mapping.loc[pd.Timestamp("2019-01-31"), "fomc_date"] == "2019-01-31"

    def test_doc_published_one_day_after_month_end_is_excluded(self, monkeypatch):
        """
        THE critical regression guard: a document published even one day
        after a month-end must never be used for that month-end's
        rebalance, no matter how close. This is precisely the bug class
        that would silently reintroduce look-ahead bias.
        """
        monkeypatch.setattr(bat, "SIM_START", "2019-01-31")
        monkeypatch.setattr(bat, "SIM_END", "2019-01-31")

        # Only a "future" doc exists relative to the sole simulation month
        fomc_df = make_fomc_df(["2019-02-01"])
        mapping = bat.map_months_to_fomc(fomc_df)

        row = mapping.loc[pd.Timestamp("2019-01-31")]
        assert pd.isna(row["fomc_date"]) or row["fomc_date"] is None, (
            "A document published AFTER the month-end was used for that "
            "month-end's rebalance — this is look-ahead bias."
        )

    def test_no_prior_document_yields_none_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(bat, "SIM_START", "2019-01-31")
        monkeypatch.setattr(bat, "SIM_END", "2019-01-31")

        fomc_df = make_fomc_df(["2020-01-30"])  # far in the future
        mapping = bat.map_months_to_fomc(fomc_df)  # must not raise

        row = mapping.loc[pd.Timestamp("2019-01-31")]
        assert pd.isna(row["fomc_date"]) or row["fomc_date"] is None
        assert pd.isna(row["fomc_file"]) or row["fomc_file"] is None

    def test_multiple_docs_in_same_window_picks_the_latest_not_the_first(self, monkeypatch):
        """
        If two documents both qualify (both published on or before the
        month-end), the mapping must pick the most recent one, not
        arbitrarily the first in the table.
        """
        monkeypatch.setattr(bat, "SIM_START", "2019-01-31")
        monkeypatch.setattr(bat, "SIM_END", "2019-01-31")

        fomc_df = make_fomc_df(["2018-11-01", "2018-12-15", "2019-01-10"])
        mapping = bat.map_months_to_fomc(fomc_df)

        assert mapping.loc[pd.Timestamp("2019-01-31"), "fomc_date"] == "2019-01-10"

    def test_every_simulation_month_present_in_output(self, monkeypatch):
        monkeypatch.setattr(bat, "SIM_START", "2019-01-31")
        monkeypatch.setattr(bat, "SIM_END", "2019-12-31")

        fomc_df = make_fomc_df(["2019-01-30"])
        mapping = bat.map_months_to_fomc(fomc_df)

        expected_months = pd.date_range("2019-01-31", "2019-12-31", freq="ME")
        assert len(mapping) == len(expected_months)
        assert set(mapping.index) == set(expected_months)


# --------------------------------------------------------------------------- #
# 3. Synthetic macro-data fallback (deterministic, network-free path)
# --------------------------------------------------------------------------- #
class TestSyntheticMacroDataFallback:
    """
    _synthetic_macro_data() is only ever called with the real project
    SIM_START/SIM_END (Feb 2019 - Dec 2024, 71 months), so the "realistic
    window" tests below use a window of that length rather than the
    project's actual dates, to keep the test self-contained.
    """

    def test_shape_and_index_match_simulation_window(self, monkeypatch):
        monkeypatch.setattr(bat, "SIM_START", "2019-02-28")
        monkeypatch.setattr(bat, "SIM_END", "2024-12-31")

        df = bat._synthetic_macro_data()

        expected_months = pd.date_range("2019-02-28", "2024-12-31", freq="ME")
        assert len(df) == len(expected_months)
        assert set(df.index) == set(expected_months)

    def test_no_nans_and_expected_columns_present(self, monkeypatch):
        monkeypatch.setattr(bat, "SIM_START", "2019-02-28")
        monkeypatch.setattr(bat, "SIM_END", "2024-12-31")

        df = bat._synthetic_macro_data()

        for col in ["cpi_yoy", "pce_yoy", "gdp_growth", "unemployment"]:
            assert col in df.columns
        assert df.isna().sum().sum() == 0

    def test_is_deterministic_across_calls(self, monkeypatch):
        """Uses np.random.seed(42) internally — confirm this actually makes
        it reproducible, since a future refactor to a Generator instance
        could silently break that guarantee."""
        monkeypatch.setattr(bat, "SIM_START", "2019-02-28")
        monkeypatch.setattr(bat, "SIM_END", "2024-12-31")

        df1 = bat._synthetic_macro_data()
        df2 = bat._synthetic_macro_data()

        pd.testing.assert_frame_equal(df1, df2)

    @pytest.mark.parametrize("start,end", [
        ("2019-01-31", "2019-06-30"),  # 6 months -- shorter than the old hardcoded split of 15
        ("2019-01-31", "2019-01-31"),  # 1 month -- extreme edge case
    ])
    def test_short_window_no_longer_crashes(self, monkeypatch, start, end):
        """
        FIXED (was a discovered bug): _synthetic_macro_data()'s unemployment
        path used to hardcode a 15-month split
        (np.linspace(3.8, 14.7, 15) then np.linspace(14.7, 3.5, n-15)),
        which raised ValueError for any window shorter than 15 months.
        Now guarded via _unemployment_path()'s min(15, n) split.
        """
        monkeypatch.setattr(bat, "SIM_START", start)
        monkeypatch.setattr(bat, "SIM_END", end)
        df = bat._synthetic_macro_data()  # must not raise
        assert df["unemployment"].notna().all()
        assert len(df) >= 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
