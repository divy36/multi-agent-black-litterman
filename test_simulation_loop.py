"""Tests for simulation loop."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import simulation_loop as sl


TICKERS = sl.TICKERS


# ═══════════════════════════════════════════════════════════════════════
# 1. load_fomc_text — including regression tests for the fixed logger bug
# ═══════════════════════════════════════════════════════════════════════
class TestLoadFomcTextSuccessPaths:

    def test_exact_txt_filename_from_alignment_table_is_used(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sl, "PROJECT_ROOT", tmp_path)
        folder = tmp_path / "2022 FO"
        folder.mkdir()
        (folder / "fomcminutes20220921.txt").write_text("Real FOMC content here.", encoding="utf-8")

        text = sl.load_fomc_text("fomcminutes20220921.pdf", "2022-09-21")
        assert text == "Real FOMC content here."

    def test_fallback_date_based_search_when_exact_name_not_in_alignment(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sl, "PROJECT_ROOT", tmp_path)
        folder = tmp_path / "2019 FO"
        folder.mkdir()
        (folder / "some_other_name_20190130.txt").write_text("Found via date fallback.", encoding="utf-8")

        text = sl.load_fomc_text("", "2019-01-30")
        assert text == "Found via date fallback."


class TestLoadFomcTextFailurePaths:
    """
    THE regression tests for the logger NameError fix. Before the fix,
    every one of these would have raised NameError instead of returning
    "" with a logged warning.
    """

    def test_missing_folder_returns_empty_string_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sl, "PROJECT_ROOT", tmp_path)
        # No "2099 FO" folder created at all
        text = sl.load_fomc_text("", "2099-01-31")  # must not raise NameError
        assert text == ""

    def test_folder_exists_but_no_matching_file_returns_empty_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sl, "PROJECT_ROOT", tmp_path)
        folder = tmp_path / "2020 FO"
        folder.mkdir()
        (folder / "unrelated_file.txt").write_text("nothing relevant", encoding="utf-8")

        text = sl.load_fomc_text("nonexistent.pdf", "2020-06-15")  # must not raise NameError
        assert text == ""

    def test_pdf_extraction_failure_falls_through_gracefully(self, tmp_path, monkeypatch):
        """A .pdf file matching the date exists but isn't a real PDF (so
        pdfplumber will fail internally) -- must still return "" via the
        warning path, not crash."""
        monkeypatch.setattr(sl, "PROJECT_ROOT", tmp_path)
        folder = tmp_path / "2021 FO"
        folder.mkdir()
        (folder / "fomcminutes20210127.pdf").write_bytes(b"not a real pdf")

        text = sl.load_fomc_text("fomcminutes20210127.pdf", "2021-01-27")  # must not raise
        assert text == ""


# ═══════════════════════════════════════════════════════════════════════
# 2. build_pnl_context
# ═══════════════════════════════════════════════════════════════════════
class TestBuildPnlContext:

    def test_first_month_returns_empty_string(self):
        result = sl.build_pnl_context(0, [], [], [], pd.DataFrame())
        assert result == ""

    def test_empty_monthly_returns_gives_empty_string(self):
        result = sl.build_pnl_context(3, [], [np.ones(10) / 10], ["2019-05-31"], pd.DataFrame())
        assert result == ""

    def test_normal_case_identifies_best_and_worst_contributor(self):
        prev_date = "2022-09-30"
        returns_df = pd.DataFrame(
            {t: [0.0] for t in TICKERS}, index=[pd.Timestamp(prev_date)]
        )
        returns_df.loc[pd.Timestamp(prev_date), "XLK"] = 0.10   # best
        returns_df.loc[pd.Timestamp(prev_date), "XLE"] = -0.08  # worst
        w = np.full(10, 0.1)

        result = sl.build_pnl_context(
            1, [0.015], [w], [prev_date], returns_df,
        )
        assert "XLK" in result  # best contributor
        assert "XLE" in result  # worst contributor
        assert "+1.50%" in result

    def test_strong_positive_month_message_branch(self):
        prev_date = "2022-09-30"
        returns_df = pd.DataFrame({t: [0.01] for t in TICKERS}, index=[pd.Timestamp(prev_date)])
        w = np.full(10, 0.1)
        result = sl.build_pnl_context(1, [0.03], [w], [prev_date], returns_df)
        assert "too conservative" in result

    def test_significant_drawdown_message_branch(self):
        prev_date = "2022-09-30"
        returns_df = pd.DataFrame({t: [-0.01] for t in TICKERS}, index=[pd.Timestamp(prev_date)])
        w = np.full(10, 0.1)
        result = sl.build_pnl_context(1, [-0.03], [w], [prev_date], returns_df)
        assert "too aggressive" in result or "missed key risks" in result

    def test_moderate_return_message_branch(self):
        prev_date = "2022-09-30"
        returns_df = pd.DataFrame({t: [0.001] for t in TICKERS}, index=[pd.Timestamp(prev_date)])
        w = np.full(10, 0.1)
        result = sl.build_pnl_context(1, [0.005], [w], [prev_date], returns_df)
        assert "moderate" in result.lower()

    def test_falls_back_to_simple_message_when_date_not_in_returns_df(self):
        result = sl.build_pnl_context(
            1, [0.02], [np.ones(10) / 10], ["2022-09-30"], pd.DataFrame({t: [] for t in TICKERS})
        )
        assert "positive" in result
        assert "+2.00%" in result


# ═══════════════════════════════════════════════════════════════════════
# 3. compute_drawdown
# ═══════════════════════════════════════════════════════════════════════
class TestComputeDrawdown:

    def test_fewer_than_2_points_returns_zero(self):
        assert sl.compute_drawdown([]) == 0.0
        assert sl.compute_drawdown([1.05]) == 0.0

    def test_monotonic_increase_has_zero_drawdown(self):
        result = sl.compute_drawdown([1.0, 1.05, 1.1, 1.2])
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_matches_hand_computed_drawdown(self):
        # peak so far at each point: [1.0, 1.2, 1.2, 1.2]; last point 0.9
        # drawdown = (0.9 - 1.2) / 1.2 = -0.25
        result = sl.compute_drawdown([1.0, 1.2, 1.1, 0.9])
        assert result == pytest.approx(-0.25, abs=1e-9)

    def test_returns_negative_value_for_a_real_drawdown(self):
        result = sl.compute_drawdown([1.0, 1.1, 0.8])
        assert result < 0


# ═══════════════════════════════════════════════════════════════════════
# 4. load_alignment_table
# ═══════════════════════════════════════════════════════════════════════
class TestLoadAlignmentTable:

    def test_raises_informative_error_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sl, "DATA_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="monthly_alignment.csv"):
            sl.load_alignment_table()

    def test_loads_and_indexes_by_month_end_when_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sl, "DATA_DIR", tmp_path)
        df = pd.DataFrame({
            "month_end": ["2019-02-28", "2019-01-31"],
            "fomc_date": ["2019-01-30", "2019-01-30"],
        })
        df.to_csv(tmp_path / "monthly_alignment.csv", index=False)

        result = sl.load_alignment_table()
        assert result.index.name == "month_end"
        # sorted ascending despite input being out of order
        assert list(result.index) == sorted(result.index)


# ═══════════════════════════════════════════════════════════════════════
# 5. SimulationResults — light record()/save() round-trip
# ═══════════════════════════════════════════════════════════════════════
# Module-level (not local to a test) so pickle can actually serialise them --
# pickle requires classes to be importable by their qualified module path,
# which a class defined inside a test function is not.
class _FakeGuardrailLog:
    vix_ceiling_fired = False
    stability_filter_fired = False
    turnover_cap_fired = False


class _FakeReport:
    Q_final = np.ones(10) * 0.01
    c_final = np.full(10, 0.5)
    d_adjusted = np.zeros(10)
    vix = 15.0
    guardrail_log = _FakeGuardrailLog()
    mean_disagreement = 0.1
    early_warning_flag = False
    instability_episode = False
    hawk_fallback = False
    dove_fallback = False


class _FakeSynthOut:
    d_j = np.zeros(10)
    hawk_Q = np.ones(10) * 0.01
    dove_Q = np.ones(10) * 0.02


class _FakeBLOut:
    pass


class TestSimulationResultsRoundTrip:

    def test_record_and_save_produces_expected_files_and_shapes(self, tmp_path):
        # Return-labeling fix: record() now takes return_date — the calendar
        # month the return was earned in (decision month + 1). Rows recorded
        # without a return_date (the final decision month) produce NO return
        # row, and monthly_returns.csv is indexed by realization months.
        results = sl.SimulationResults()
        decision_to_realization = {
            "2019-02-28": "2019-03-31",
            "2019-03-31": "2019-04-30",
        }
        for i, (date, ret_date) in enumerate(decision_to_realization.items()):
            results.record(
                date=date, w_final=np.full(10, 0.1), port_return=0.01 * (i + 1),
                cum_return=0.01 * (i + 1), drawdown=0.0,
                synth_out=_FakeSynthOut(), cont_out=None,
                report=_FakeReport(), bl_out=_FakeBLOut(),
                return_date=ret_date,
            )
        # Final decision month of a window: no realized return inside it.
        results.record(
            date="2019-04-30", w_final=np.full(10, 0.1), port_return=None,
            cum_return=0.02, drawdown=0.0,
            synth_out=_FakeSynthOut(), cont_out=None,
            report=_FakeReport(), bl_out=_FakeBLOut(),
            return_date=None,
        )

        results.save(tmp_path)

        weights = pd.read_csv(tmp_path / "monthly_weights.csv", index_col="Date")
        assert weights.shape == (3, 10)   # one row per DECISION month

        returns = pd.read_csv(tmp_path / "monthly_returns.csv", index_col="Date")
        assert list(returns["portfolio_return"]) == pytest.approx([0.01, 0.02], abs=1e-9)
        # Indexed by REALIZATION month, and no row for the final decision month
        assert list(returns.index) == ["2019-03-31", "2019-04-30"]

        with open(tmp_path / "agent_outputs.pkl", "rb") as f:
            pkl = pickle.load(f)
        assert len(pkl["dates"]) == 3


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
