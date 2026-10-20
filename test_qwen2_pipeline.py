"""Tests for qwen2 pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import ollama_is_reachable

from agent_hawk_v2 import HawkAgent
from agent_dove_v2 import DoveAgent
from agent_synthesis import SynthesisAgent
from agent_contrarian import ContrarianAgent
from agent_risk_guardian import RiskGuardian

TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLU", "XLB"]
N = len(TICKERS)

_OLLAMA_UP = ollama_is_reachable()
_SKIP_REASON = (
    None if _OLLAMA_UP else
    "Ollama is not reachable at http://localhost:11434 -- start it with "
    "'ollama serve' before running the full Qwen2 pipeline integration test."
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Backend consistency — no Ollama needed, just checks configuration
# ═══════════════════════════════════════════════════════════════════════
class TestQwen2BackendConsistency:
    """
    Confirms the pipeline this file wires together is genuinely
    single-backend, before ever making a real call. Cheap, fast, and
    catches the exact mismatch found in the existing self-tests: if
    anyone later swaps one agent's OLLAMA_MODEL without updating the
    others, this fails immediately rather than silently mixing backends.
    """

    def test_hawk_v2_is_qwen2(self):
        assert HawkAgent().model == "qwen2:7b"

    def test_dove_v2_is_qwen2(self):
        assert DoveAgent().model == "qwen2:7b"

    def test_contrarian_is_qwen2(self):
        import agent_contrarian
        assert agent_contrarian.OLLAMA_MODEL == "qwen2:7b"

    def test_all_three_llm_calling_agents_agree_on_backend(self):
        import agent_contrarian
        models = {HawkAgent().model, DoveAgent().model, agent_contrarian.OLLAMA_MODEL}
        assert len(models) == 1, (
            f"Backend mismatch across the pipeline: {models}. Hawk, Dove, and "
            f"Contrarian must all use the same model for this to be a valid "
            f"single-backend test of either the Llama3 or Qwen2 configuration."
        )


# ═══════════════════════════════════════════════════════════════════════
# 2. Full pipeline — real Qwen2 calls throughout, gated on Ollama
# ═══════════════════════════════════════════════════════════════════════
def _make_synthetic_returns_df(start="2016-01-31", n_months=60, seed=0):
    """
    Starts in 2016 (matching how the real pipeline is always seeded --
    see build_etf_prices.py's 3-year lookback) so that any realistic
    2019-era test date has genuine history behind it for sigma_j
    estimation. Starting this too close to the test dates was the actual
    cause of a NaN-propagation failure caught during verification --
    Synthesis's sigma_j estimate goes to NaN with zero months of prior
    history, which BLInputs then correctly rejects rather than silently
    continuing -- fixed here by seeding realistically, not by loosening
    any validation.
    """
    idx = pd.date_range(start, periods=n_months, freq="ME")
    rng = np.random.default_rng(seed)
    data = rng.normal(0.005, 0.04, size=(n_months, N))
    return pd.DataFrame(data, index=idx, columns=TICKERS)


@pytest.mark.integration
@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
class TestQwen2FullPipelineIntegration:

    def test_single_month_pipeline_produces_valid_bl_inputs(self):
        fomc_text = (
            "The Committee decided to raise the target range for the federal "
            "funds rate by 75 basis points to 3 to 3-1/4 percent. Inflation "
            "remains elevated. The Committee is strongly committed to "
            "returning inflation to its 2 percent objective."
        )
        date, fomc_date = "2022-09-30", "2022-09-21"
        returns_df = _make_synthetic_returns_df()

        hawk = HawkAgent()
        dove = DoveAgent()
        synthesis = SynthesisAgent()
        contrarian = ContrarianAgent()
        risk_guardian = RiskGuardian()

        hawk_out = hawk.run(fomc_text, cpi_yoy=8.3, pce_yoy=4.9, date=date, fomc_date=fomc_date)
        dove_out = dove.run(fomc_text, gdp_growth=2.6, unemployment=3.5, date=date, fomc_date=fomc_date)

        assert hawk_out.fallback_used is False, "Hawk (qwen2) fell back -- check Ollama/model availability"
        assert dove_out.fallback_used is False, "Dove (qwen2) fell back -- check Ollama/model availability"

        synth_out = synthesis.run(hawk_out, dove_out, returns_df, date)
        cont_out = contrarian.run(synth_out, fomc_text, date)

        assert cont_out.fallback_used is False, "Contrarian (qwen2) fell back -- check Ollama/model availability"

        report, final_bl_inputs = risk_guardian.run(
            contrarian_output=cont_out, synthesis_output=synth_out,
            hawk_output=hawk_out, dove_output=dove_out,
            vix=28.5, date=date, prev_Q=None,
            sigma_j=synth_out.sigma_j, drawdown_history=None,
        )

        assert final_bl_inputs.Q.shape == (N,)
        assert np.all(np.isfinite(final_bl_inputs.Q))
        assert np.all((0.0 <= final_bl_inputs.c_final) & (final_bl_inputs.c_final <= 1.0))
        assert np.all((0.0 <= final_bl_inputs.d_adjusted) & (final_bl_inputs.d_adjusted <= 1.0))

    def test_two_sequential_months_exercises_stability_filter_with_real_prev_q(self):
        """
        Runs two consecutive months for real, feeding month 1's Q_final as
        month 2's prev_Q -- exercising the stability filter's delta/sign-
        reversal logic against genuine Qwen2 output, not synthetic Q values.
        """
        returns_df = _make_synthetic_returns_df()
        hawk, dove = HawkAgent(), DoveAgent()
        synthesis, contrarian, risk_guardian = SynthesisAgent(), ContrarianAgent(), RiskGuardian()

        scenarios = [
            {"date": "2019-01-31", "fomc_date": "2019-01-30", "cpi_yoy": 1.6, "pce_yoy": 1.7,
             "gdp_growth": 3.1, "unemployment": 4.0,
             "fomc_text": "The Committee will be patient. Maintained target range at "
                          "2-1/4 to 2-1/2 percent. Strong labor market."},
            {"date": "2019-02-28", "fomc_date": "2019-01-30", "cpi_yoy": 1.5, "pce_yoy": 1.6,
             "gdp_growth": 3.0, "unemployment": 3.9,
             "fomc_text": "The Committee continues to be patient. Economic activity "
                          "rising at a solid rate. Muted inflation pressures."},
        ]

        prev_Q = None
        for sc in scenarios:
            hawk_out = hawk.run(sc["fomc_text"], sc["cpi_yoy"], sc["pce_yoy"], sc["date"], sc["fomc_date"])
            dove_out = dove.run(sc["fomc_text"], sc["gdp_growth"], sc["unemployment"], sc["date"], sc["fomc_date"])
            synth_out = synthesis.run(hawk_out, dove_out, returns_df, sc["date"])
            cont_out = contrarian.run(synth_out, sc["fomc_text"], sc["date"])
            report, final_bl_inputs = risk_guardian.run(
                contrarian_output=cont_out, synthesis_output=synth_out,
                hawk_output=hawk_out, dove_output=dove_out,
                vix=16.0, date=sc["date"], prev_Q=prev_Q,
                sigma_j=synth_out.sigma_j, drawdown_history=None,
            )
            assert np.all(np.isfinite(final_bl_inputs.Q))
            prev_Q = final_bl_inputs.Q.copy()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
