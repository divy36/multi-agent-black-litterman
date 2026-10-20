"""Tests for agent pipelines."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from conftest import ollama_is_reachable

PROJECT_ROOT = Path(__file__).resolve().parent
RETURNS_FILE = PROJECT_ROOT / "data" / "etf_returns_monthly.csv"
W_MKT_FILE = PROJECT_ROOT / "data" / "market_cap_weights.csv"

_OLLAMA_UP = ollama_is_reachable()
_DATA_PRESENT = RETURNS_FILE.exists() and W_MKT_FILE.exists()

_SKIP_REASON = None
if not _OLLAMA_UP:
    _SKIP_REASON = (
        "Ollama is not reachable at http://localhost:11434 — start it with "
        "'ollama serve' (or open the Ollama app) before running integration tests."
    )
elif not _DATA_PRESENT:
    _SKIP_REASON = (
        f"Required data files not found "
        f"({RETURNS_FILE.name}, {W_MKT_FILE.name} under {PROJECT_ROOT / 'data'}). "
        f"Run the data pipeline scripts first."
    )


def _run_self_test(script_name: str) -> None:
    """
    Execute a script's __main__ block exactly as `python3 script_name.py`
    would, and fail the pytest test if it exits non-zero (each of the
    three scripts calls sys.exit(1) if any internal check failed).
    """
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        pytest.skip(f"{script_name} not found at {script_path}")

    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            pytest.fail(
                f"{script_name}'s self-test reported failures "
                f"(exited with code {code}) — see captured output above for "
                f"which specific check(s) failed."
            )
    # SystemExit not raised at all == script's own success path (no sys.exit call)


@pytest.mark.integration
@pytest.mark.skipif(_SKIP_REASON is not None, reason=str(_SKIP_REASON))
class TestAgentPipelineSelfTests:
    """
    Each test below wraps one of the three existing agent self-tests.
    Run from the FinalProject root directory so the scripts' internal
    `from agent_hawk import HawkAgent`-style imports resolve correctly.
    """

    def test_risk_guardian_full_pipeline_self_test(self):
        """
        Wraps agent_risk_guardian.py's __main__: Hawk -> Dove -> Synthesis
        -> Contrarian -> Risk Guardian -> BL across calm/stressed/crisis
        VIX scenarios. Checks (inside the wrapped script) include: VIX
        ceiling formula exactness, c_final never exceeding pre-guardrail
        confidence, stability penalty bounds, valid BLInputs, valid BL
        portfolio weights, and cross-scenario VIX ordering.
        """
        _run_self_test("agent_risk_guardian.py")

    def test_synthesis_full_pipeline_self_test(self):
        """
        Wraps agent_synthesis.py's __main__: Hawk -> Dove -> Synthesis
        across a tightening and a neutral scenario. Checks include:
        Q_blend bounded by [Q_hawk, Q_dove], d_adjusted in [0,1], c_blend
        never inflated above the average confidence, valid BLInputs,
        sensible sigma_j range, and a full Synthesis -> BL integration
        producing a valid portfolio.
        """
        _run_self_test("agent_synthesis.py")

    def test_contrarian_full_pipeline_self_test(self):
        """
        Wraps agent_contrarian.py's __main__: Hawk -> Dove -> Synthesis
        -> Contrarian across a tightening and a neutral scenario. Checks
        include: d_adjusted = min(1.0, d_j + penalty_j) as an exact
        formula match, penalty_j bounds, Contrarian only ever inflating
        disagreement (never deflating it), and a full pipeline producing
        a valid BL portfolio.
        """
        _run_self_test("agent_contrarian.py")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-m", "integration"]))
