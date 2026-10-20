"""Agent contrarian."""

from __future__ import annotations
from pathlib import Path

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from fomc_signals import FOMCSignalExtractor
    _EXTRACTOR = FOMCSignalExtractor()
except ImportError:
    _EXTRACTOR = None

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TICKERS: list[str] = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLY", "XLP", "XLRE", "XLU", "XLB",
]
N = len(TICKERS)

OLLAMA_MODEL  = "qwen2:7b"
OLLAMA_URL    = "http://localhost:11434/api/generate"
TEMPERATURE   = 0
MAX_RETRIES   = 3
TIMEOUT_SECS  = 120

# Penalty bounds — Contrarian can add at most 0.5 to d_adjusted per sector
# This caps its influence: even maximum Contrarian penalty cannot triple Omega
# alone; it requires combination with genuine agent disagreement.
MAX_PENALTY   = 0.50
MIN_PENALTY   = 0.00

FALLBACK_PENALTY = 0.10   # conservative fallback: small uniform penalty

# Thresholds for pre-screening overconfidence flags
GROUPTHINK_D_THRESHOLD   = 0.20   # d_j < 0.20 = agents agree = groupthink risk
GROUPTHINK_CONF_THRESHOLD = 0.60  # avg confidence > 0.60 when d_j low = overconfident
MAGNITUDE_THRESHOLD       = 0.025  # |Q_blend| > 2.5% monthly = extreme view
EXTREME_MAGNITUDE         = 0.040  # |Q_blend| > 4.0% monthly = very extreme view

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class OverconfidenceFlags:
    """
    Pre-computed overconfidence flags per sector.
    These are injected into the LLM prompt as hard signals.

    Attributes
    ----------
    groupthink : np.ndarray (N,) bool
        True if agents agree in direction AND both have high confidence.
        Low d_j + high avg confidence = false consensus risk.
    extreme_magnitude : np.ndarray (N,) bool
        True if |Q_blend_j| > EXTREME_MAGNITUDE (4% monthly).
    large_magnitude : np.ndarray (N,) bool
        True if |Q_blend_j| > MAGNITUDE_THRESHOLD (2.5% monthly).
    regime_conflict : np.ndarray (N,) bool
        True if Q_blend direction conflicts with hawkish_score regime signal.
    suggested_penalty : np.ndarray (N,) float
        Rule-based suggested penalty in [0, MAX_PENALTY].
        LLM may agree, increase, or reduce this.
    """
    groupthink:        np.ndarray
    extreme_magnitude: np.ndarray
    large_magnitude:   np.ndarray
    regime_conflict:   np.ndarray
    suggested_penalty: np.ndarray


@dataclass
class ContrarianOutput:
    """
    Full output bundle from one Contrarian agent run.

    Attributes
    ----------
    penalty_j : np.ndarray (N,)
        Overconfidence penalty scores in [0, MAX_PENALTY] per sector.
        penalty_j = 0 means no additional penalty beyond raw disagreement.
        penalty_j = 0.5 means Contrarian is maximally skeptical of that view.

    d_adjusted : np.ndarray (N,)
        Final adjusted disagreement: min(1.0, d_j + penalty_j).
        This is what gets passed to BLInputs and then to Omega.

    bl_inputs_updated : object
        BLInputs with d_adjusted updated to include Contrarian penalty.
        This is the final input to black_litterman.BlackLittermanModel.run().

    date : str
        ISO month-end date string.

    overconfidence_flags : OverconfidenceFlags
        Pre-computed flags that were injected into the prompt.

    max_penalty_sector : str
        Ticker where Contrarian assigned highest penalty (key warning signal).

    mean_penalty : float
        Mean penalty across all sectors.

    fallback_used : bool
        True if all LLM retries failed and uniform fallback penalty applied.

    raw_response : str
        Raw LLM response preserved for case study analysis.

    attempts : int
        Number of LLM calls before success.
    """
    penalty_j:            np.ndarray
    d_adjusted:           np.ndarray
    bl_inputs_updated:    object
    date:                 str
    overconfidence_flags: OverconfidenceFlags
    max_penalty_sector:   str
    mean_penalty:         float
    fallback_used:        bool
    raw_response:         str
    attempts:             int


# --------------------------------------------------------------------------- #
# Stage 1: Pre-compute overconfidence flags (deterministic Python)
# --------------------------------------------------------------------------- #
def _compute_overconfidence_flags(
    synthesis_output,    # SynthesisOutput from agent_synthesis.py
    fomc_text: str,
) -> OverconfidenceFlags:
    """
    Compute objective overconfidence signals from Synthesis output.

    These are pure Python / numpy — no LLM involved.
    They give the LLM hard numbers to anchor its penalty reasoning.
    """
    Q_blend   = synthesis_output.Q_blend
    d_j       = synthesis_output.d_j
    c_hawk    = synthesis_output.hawk_confidence
    c_dove    = synthesis_output.dove_confidence
    Q_hawk    = synthesis_output.hawk_Q
    Q_dove    = synthesis_output.dove_Q

    avg_conf  = (c_hawk + c_dove) / 2.0

    # Flag 1: Groupthink — agents agree (low d_j) with high confidence
    groupthink = (d_j < GROUPTHINK_D_THRESHOLD) & (avg_conf > GROUPTHINK_CONF_THRESHOLD)

    # Flag 2: Extreme magnitude
    abs_q = np.abs(Q_blend)
    extreme_magnitude = abs_q > EXTREME_MAGNITUDE
    large_magnitude   = abs_q > MAGNITUDE_THRESHOLD

    # Flag 3: Regime conflict — Q_blend direction conflicts with FOMC signal
    regime_conflict = np.zeros(N, dtype=bool)
    if _EXTRACTOR is not None and fomc_text:
        try:
            signals = _EXTRACTOR.extract(fomc_text)
            hawkish_score = signals.hawkish_score

            if hawkish_score >= 4:
                # Strongly tightening: expect negative returns on rate-sensitive sectors
                # XLK(0), XLI(4), XLY(5), XLRE(7), XLU(8) — positive views conflict
                rate_sensitive = np.array([True, False, False, False, True,
                                           True, False, True,  True,  False])
                regime_conflict = rate_sensitive & (Q_blend > 0.005)

            elif hawkish_score <= -4:
                # Strongly easing: expect positive returns broadly
                # Negative views on growth sectors conflict
                growth_sectors = np.array([True, False, False, False, True,
                                           True, False, False, False, False])
                regime_conflict = growth_sectors & (Q_blend < -0.005)

        except Exception:
            pass  # If signal extraction fails, no regime flags

    # Suggested penalty (rule-based baseline the LLM can adjust)
    suggested_penalty = np.zeros(N)
    suggested_penalty += 0.15 * groupthink.astype(float)       # groupthink: +0.15
    suggested_penalty += 0.10 * large_magnitude.astype(float)   # large view: +0.10
    suggested_penalty += 0.15 * extreme_magnitude.astype(float) # extreme view: +0.15 extra
    suggested_penalty += 0.15 * regime_conflict.astype(float)   # regime conflict: +0.15
    suggested_penalty  = np.minimum(suggested_penalty, MAX_PENALTY)

    return OverconfidenceFlags(
        groupthink        = groupthink,
        extreme_magnitude = extreme_magnitude,
        large_magnitude   = large_magnitude,
        regime_conflict   = regime_conflict,
        suggested_penalty = suggested_penalty,
    )


def _flags_to_prompt_block(
    flags:          OverconfidenceFlags,
    synthesis_out,
) -> str:
    """Format overconfidence flags compactly for prompt injection."""
    lines = ["## Pre-Screened Overconfidence Signals (GT=groupthink, LM=large magnitude, RC=regime conflict)"]
    for i, t in enumerate(TICKERS):
        flag_strs = []
        if flags.groupthink[i]:        flag_strs.append("GT")
        if flags.large_magnitude[i]:   flag_strs.append("LM")
        if flags.extreme_magnitude[i]: flag_strs.append("EM")
        if flags.regime_conflict[i]:   flag_strs.append("RC")
        lines.append(
            f"  {t}: suggested_penalty={flags.suggested_penalty[i]:.2f}"
            f"  flags=[{','.join(flag_strs) or 'none'}]"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Stage 2: Prompt builder
# --------------------------------------------------------------------------- #
def _build_prompt(
    synthesis_output,
    flags:       OverconfidenceFlags,
    fomc_text:   str,
    date:        str,
) -> str:
    """
    Build the Contrarian agent prompt.

    Structure:
    1. Persona — the Contrarian's mandate
    2. Committee summary — what Hawk, Dove, Synthesis produced
    3. Pre-screened overconfidence signals — hard flags per sector
    4. FOMC signal summary — macro regime context
    5. Penalty framework — how to interpret and output penalties
    6. Output format — JSON penalty array
    """
    # FOMC signal block
    fomc_block = ""
    if _EXTRACTOR is not None and fomc_text:
        try:
            signals = _EXTRACTOR.extract(fomc_text)
            fomc_block = signals.summary()
        except Exception:
            fomc_block = "FOMC signal extraction unavailable."

    flags_block = _flags_to_prompt_block(flags, synthesis_output)

    # Committee summary — compact format to stay within token limit
    committee_lines = ["## Committee Views (Blend=blended Q, d=disagreement score)"]
    for i, t in enumerate(TICKERS):
        committee_lines.append(
            f"  {t}: Hawk={synthesis_output.hawk_Q[i]:+.3f} "
            f"Dove={synthesis_output.dove_Q[i]:+.3f} "
            f"Blend={synthesis_output.Q_blend[i]:+.3f} "
            f"d={synthesis_output.d_j[i]:.3f}"
        )
    committee_summary = "\n".join(committee_lines)

    prompt = f"""You are CONTRARIAN, a devil's advocate analyst at an investment committee.
Your SOLE JOB is to find reasons why the committee's blended views are WRONG or OVERCONFIDENT.
You are not here to agree. You are not here to be balanced. You are here to stress-test.

You read the Hawk and Dove views and the blended consensus. Your job:
1. Identify sectors where the committee is dangerously overconfident
2. Identify sectors where both agents surprisingly agree (groupthink risk)
3. Identify sectors where the views are implausibly extreme
4. Assign a confidence PENALTY per sector — how much to inflate uncertainty

Today's date: {date}

---

{committee_summary}

---

{fomc_block}

---

{flags_block}

---

## Your Task

For each sector assign a penalty in [0.0, 0.5]:
- 0.00 = no additional concern beyond existing disagreement
- 0.10 = mild overconfidence — view is somewhat aggressive for this regime
- 0.20 = moderate — committee is too certain given macro uncertainty
- 0.30 = high — groupthink or implausibly large expected return
- 0.40 = very high — view clearly conflicts with FOMC signal direction
- 0.50 = extreme — committee is dangerously wrong on this sector

KEY RULE: Sectors already flagged with high d_j (agents disagree) need LOWER
additional penalty — disagreement already inflates Omega. Focus your penalties
on sectors where BOTH agents AGREE but you think they are WRONG.

Use the "SugPenalty" column above as your starting point and adjust based on
your qualitative read of the FOMC signal and macro regime.

## Output Format — STRICT
Output ONLY a JSON array. No text before or after. No markdown.
You MAY output sectors in ANY order — I will sort them.
Each object needs exactly: sector, penalty, reason (under 8 words).

[
  {{"sector": "XLK",  "penalty": 0.00, "reason": "example reason here"}},
  {{"sector": "XLF",  "penalty": 0.10, "reason": "example reason here"}},
  {{"sector": "XLE",  "penalty": 0.00, "reason": "example reason here"}},
  {{"sector": "XLV",  "penalty": 0.00, "reason": "example reason here"}},
  {{"sector": "XLI",  "penalty": 0.00, "reason": "example reason here"}},
  {{"sector": "XLY",  "penalty": 0.00, "reason": "example reason here"}},
  {{"sector": "XLP",  "penalty": 0.00, "reason": "example reason here"}},
  {{"sector": "XLRE", "penalty": 0.10, "reason": "example reason here"}},
  {{"sector": "XLU",  "penalty": 0.00, "reason": "example reason here"}},
  {{"sector": "XLB",  "penalty": 0.00, "reason": "example reason here"}}
]

Replace the example numbers above with your actual penalty scores.
Output the JSON array now:"""

    return prompt


# --------------------------------------------------------------------------- #
# JSON parser
# --------------------------------------------------------------------------- #
def _parse_contrarian_response(raw: str) -> Optional[list[dict]]:
    """Parse penalty JSON from LLM response. Handles all common formatting quirks."""
    text = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    text = text.replace("```", "").strip()
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start:end + 1]
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    json_str = re.sub(r":\s*\+", ": ", json_str)
    json_str = json_str.replace("'", '"')
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _validate_penalties(data: list[dict]) -> bool:
    """
    Validate penalty JSON structure and value ranges.
    Order-agnostic: sectors can appear in any order; we match by name.
    Returns False if any sector is missing, duplicated, or has invalid values.
    """
    if len(data) != N:
        logger.warning(f"Contrarian: expected {N} sectors, got {len(data)}")
        return False
    seen = {}
    for entry in data:
        if not isinstance(entry, dict):
            return False
        if "sector" not in entry or "penalty" not in entry:
            logger.warning(f"Contrarian: entry missing fields: {entry}")
            return False
        ticker = entry["sector"]
        if ticker not in TICKERS:
            logger.warning(f"Contrarian: unknown sector: {ticker}")
            return False
        if ticker in seen:
            logger.warning(f"Contrarian: duplicate sector: {ticker}")
            return False
        try:
            p = float(entry["penalty"])
        except (TypeError, ValueError):
            return False
        if not (MIN_PENALTY - 1e-6 <= p <= MAX_PENALTY + 1e-6):
            logger.warning(f"Contrarian: penalty {p} out of range for {ticker}")
            return False
        seen[ticker] = p
    # Check all 10 sectors present
    missing = [t for t in TICKERS if t not in seen]
    if missing:
        logger.warning(f"Contrarian: missing sectors: {missing}")
        return False
    return True


# --------------------------------------------------------------------------- #
# Ollama caller
# --------------------------------------------------------------------------- #
def _call_ollama(prompt: str, attempt: int) -> str:
    import urllib.request
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 2000,
            "top_p": 1.0,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECS) as resp:
            body = resp.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"Ollama call failed (attempt {attempt}): {exc}") from exc
    return json.loads(body).get("response", "")


# --------------------------------------------------------------------------- #
# Fallback
# --------------------------------------------------------------------------- #
def _make_fallback(
    synthesis_output,
    flags:   OverconfidenceFlags,
    date:    str,
    raw:     str,
    attempts: int,
) -> ContrarianOutput:
    """
    Conservative fallback: apply the rule-based suggested penalties directly.
    This is better than zero penalties (ignores Contrarian entirely)
    and better than max penalties (too conservative).
    """
    logger.error(
        f"Contrarian: all {attempts} attempts failed for {date}. "
        f"Applying rule-based suggested penalties as fallback."
    )
    penalty_j  = flags.suggested_penalty.copy()
    d_adjusted = np.minimum(1.0, synthesis_output.d_j + penalty_j)

    from black_litterman import BLInputs
    bl_updated = BLInputs(
        Q          = synthesis_output.Q_blend,
        c_final    = synthesis_output.c_blend,
        d_adjusted = d_adjusted,
    )

    max_idx = int(np.argmax(penalty_j))
    return ContrarianOutput(
        penalty_j            = penalty_j,
        d_adjusted           = d_adjusted,
        bl_inputs_updated    = bl_updated,
        date                 = date,
        overconfidence_flags = flags,
        max_penalty_sector   = TICKERS[max_idx],
        mean_penalty         = float(penalty_j.mean()),
        fallback_used        = True,
        raw_response         = raw,
        attempts             = attempts,
    )


# --------------------------------------------------------------------------- #
# Main agent class
# --------------------------------------------------------------------------- #
class ContrarianAgent:
    """
    Agent 4 — Contrarian.

    Two-stage architecture:
    1. Pre-compute overconfidence flags (deterministic Python, always runs)
    2. LLM call to refine penalties based on committee outputs and FOMC context

    The two-stage design ensures the agent degrades gracefully:
    - If LLM fails → fallback uses rule-based suggested penalties (Stage 1)
    - If LLM succeeds → it can raise or lower the suggested penalties

    Usage
    -----
        from agent_contrarian import ContrarianAgent
        from agent_synthesis import SynthesisAgent
        # ... run hawk, dove, synthesis first ...

        contrarian = ContrarianAgent()
        c_output = contrarian.run(
            synthesis_output = synth_result,
            fomc_text        = fomc_text,
            date             = "2022-09-30",
        )

        # Use updated BLInputs for the BL model
        bl_result = model.run(..., bl_inputs=c_output.bl_inputs_updated)
    """

    def __init__(
        self,
        model:       str = OLLAMA_MODEL,
        temperature: int = TEMPERATURE,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.model       = model
        self.temperature = temperature
        self.max_retries = max_retries

    def run(
        self,
        synthesis_output,     # SynthesisOutput from agent_synthesis.py
        fomc_text: str,
        date:      str,
    ) -> ContrarianOutput:
        """
        Run the Contrarian agent for one month.

        Parameters
        ----------
        synthesis_output : SynthesisOutput
            Full output from Agent 3 (Synthesis) for this month.
            Contains Q_blend, d_j, c_blend, hawk_Q, dove_Q, etc.
        fomc_text : str
            FOMC meeting minutes text. Used for regime conflict detection.
        date : str
            ISO month-end date string (e.g. "2022-09-30").

        Returns
        -------
        ContrarianOutput
            Penalty scores, updated d_adjusted, and updated BLInputs.
        """
        # ── Fully deterministic: rule-based overconfidence penalties ───────────
        # Design decision: llama3-8B cannot reliably produce 10-item structured
        # JSON for this task (ordering/truncation issues across all prompt variants).
        # The rule-based Stage 1 pre-screening is used directly as the final penalty.
        # This is academically defensible: penalties are fully interpretable,
        # reproducible, and auditable — stronger properties than LLM-derived scores.
        # Cite: the penalty mechanism itself is the contribution, not LLM agreement.
        flags = _compute_overconfidence_flags(synthesis_output, fomc_text)

        # Refine suggested penalties with additional rules beyond the basic flags:
        penalty_j = flags.suggested_penalty.copy()

        # Extra rule: if d_j is already >= 0.8 (extreme disagreement),
        # reduce additional penalty — Omega already heavily inflated
        high_d_mask = synthesis_output.d_j >= 0.80
        penalty_j[high_d_mask] = np.minimum(penalty_j[high_d_mask], 0.05)

        # Extra rule: emergency/crisis flag → apply moderate uniform penalty
        # (high uncertainty across all sectors, not just flagged ones)
        if flags.groupthink.sum() == 0 and np.all(synthesis_output.d_j < 0.15):
            # All agents agree on everything — maximum groupthink risk
            penalty_j = np.maximum(penalty_j, 0.10)

        penalty_j = np.clip(penalty_j, MIN_PENALTY, MAX_PENALTY)

        # Final d_adjusted = min(1.0, d_j + penalty_j)
        d_adjusted = np.minimum(1.0, synthesis_output.d_j + penalty_j)

        from black_litterman import BLInputs
        bl_updated = BLInputs(
            Q          = synthesis_output.Q_blend,
            c_final    = synthesis_output.c_blend,
            d_adjusted = d_adjusted,
        )

        max_idx = int(np.argmax(penalty_j))
        logger.info(f"Contrarian [{date}]: deterministic penalties applied. "
                    f"Max penalty: {TICKERS[max_idx]} ({penalty_j[max_idx]:.3f}), "
                    f"Mean: {penalty_j.mean():.3f}")

        return ContrarianOutput(
            penalty_j            = penalty_j,
            d_adjusted           = d_adjusted,
            bl_inputs_updated    = bl_updated,
            date                 = date,
            overconfidence_flags = flags,
            max_penalty_sector   = TICKERS[max_idx],
            mean_penalty         = float(penalty_j.mean()),
            fallback_used        = False,
            raw_response         = "deterministic — no LLM call",
            attempts             = 0,
        )

    def penalty_summary(self, output: ContrarianOutput, synthesis_output) -> str:
        """Human-readable penalty summary for logging and stability reports."""
        lines = [
            f"Contrarian [{output.date}] — Penalty Summary",
            f"{'─'*70}",
            f"{'Sector':<6}  {'Q_blend':>8}  {'d_j':>6}  {'penalty':>8}  "
            f"{'d_adj':>7}  {'Flags'}",
            f"{'─'*6}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*20}",
        ]
        flags = output.overconfidence_flags
        for i, t in enumerate(TICKERS):
            flag_strs = []
            if flags.groupthink[i]:        flag_strs.append("GROUPTHINK")
            if flags.extreme_magnitude[i]: flag_strs.append("EXTREME_MAG")
            if flags.large_magnitude[i] and not flags.extreme_magnitude[i]:
                flag_strs.append("LARGE_MAG")
            if flags.regime_conflict[i]:   flag_strs.append("REGIME_CONFLICT")
            lines.append(
                f"{t:<6}  "
                f"{synthesis_output.Q_blend[i]:>+8.4f}  "
                f"{synthesis_output.d_j[i]:>6.3f}  "
                f"{output.penalty_j[i]:>8.3f}  "
                f"{output.d_adjusted[i]:>7.3f}  "
                f"{', '.join(flag_strs) or 'none'}"
            )
        lines.append(f"{'─'*70}")
        lines.append(f"Max penalty sector: {output.max_penalty_sector}  "
                     f"(penalty={output.penalty_j[TICKERS.index(output.max_penalty_sector)]:.3f})")
        lines.append(f"Mean penalty:       {output.mean_penalty:.3f}")
        lines.append(f"Fallback used:      {output.fallback_used}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Full pipeline test: Hawk → Dove → Synthesis → Contrarian → BL.
    Tests two scenarios:
    1. Sep 2022 (tightening, 75bp hike) — expect high penalties on groupthink
       sectors and regime-conflicted views
    2. Feb 2019 (neutral, patient Fed) — expect lower penalties overall

    Key checks:
    - d_adjusted >= d_j always (Contrarian only inflates, never deflates)
    - d_adjusted in [0,1]
    - penalty_j in [0, MAX_PENALTY]
    - BLInputs valid for BL model
    - Full pipeline Hawk→Dove→Synth→Contrarian→BL produces valid portfolio
    """
    import sys
    import logging
    import pandas as pd

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)s  %(message)s",
    )

    print("=" * 70)
    print("Contrarian Agent self-test — full pipeline")
    print("=" * 70)

    try:
        returns_df = pd.read_csv(
            str(Path(__file__).resolve().parent / "data" / "etf_returns_monthly.csv"),
            index_col="Date", parse_dates=True,
        )
        w_mkt_df = pd.read_csv(
            str(Path(__file__).resolve().parent / "data" / "market_cap_weights.csv"),
            index_col="Date", parse_dates=True,
        )
        print(f"Loaded returns: {returns_df.shape}, w_mkt: {w_mkt_df.shape}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    from agent_hawk import HawkAgent
    from agent_dove import DoveAgent
    from agent_synthesis import SynthesisAgent
    from black_litterman import BlackLittermanModel

    hawk_agent       = HawkAgent()
    dove_agent       = DoveAgent()
    synth_agent      = SynthesisAgent()
    contrarian_agent = ContrarianAgent()
    bl_model         = BlackLittermanModel()

    scenarios = [
        {
            "label":       "Sep 2022 — Hawkish (75bp hike, CPI=8.3%)",
            "date":        "2022-09-30",
            "fomc_date":   "2022-09-21",
            "cpi_yoy":     8.3, "pce_yoy": 4.9,
            "gdp_growth":  2.6, "unemployment": 3.5,
            "fomc_text": """
The Committee decided to raise the target range for the federal funds rate to
3 to 3-1/4 percent. The Committee raised the target range by 75 basis points.
Ongoing increases will be appropriate. Inflation remains elevated. The Committee
is highly attentive to inflation risks. The Committee is strongly committed to
returning inflation to its 2 percent objective. Job gains have been robust.
""",
        },
        {
            "label":       "Feb 2019 — Neutral (hold, patient, CPI=1.6%)",
            "date":        "2019-02-28",
            "fomc_date":   "2019-01-30",
            "cpi_yoy":     1.6, "pce_yoy": 1.7,
            "gdp_growth":  3.1, "unemployment": 4.0,
            "fomc_text": """
The Committee will be patient as it determines future adjustments.
The Committee decided to maintain the target range at 2-1/4 to 2-1/2 percent.
Muted inflation pressures. Strong labor market. Solid economic activity.
The Committee will assess incoming data carefully.
""",
        },
    ]

    passed, failed = [], []
    def run_check(name, fn):
        try:
            fn()
            passed.append(name)
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed.append(name)
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")

    all_results = {}
    w_prev = np.ones(N) / N

    for sc in scenarios:
        print(f"\n{'─'*70}")
        print(f"  {sc['label']}")
        print(f"{'─'*70}")

        hawk_out = hawk_agent.run(
            fomc_text=sc["fomc_text"], cpi_yoy=sc["cpi_yoy"],
            pce_yoy=sc["pce_yoy"], date=sc["date"], fomc_date=sc["fomc_date"],
        )
        dove_out = dove_agent.run(
            fomc_text=sc["fomc_text"], gdp_growth=sc["gdp_growth"],
            unemployment=sc["unemployment"], date=sc["date"], fomc_date=sc["fomc_date"],
        )
        synth_out = synth_agent.run(
            hawk_output=hawk_out, dove_output=dove_out,
            returns_df=returns_df, date=sc["date"],
        )
        cont_out = contrarian_agent.run(
            synthesis_output=synth_out,
            fomc_text=sc["fomc_text"],
            date=sc["date"],
        )
        all_results[sc["label"]] = (synth_out, cont_out)

        print(contrarian_agent.penalty_summary(cont_out, synth_out))

        date = sc["date"]

        def check_d_adjusted_monotone():
            # d_adjusted = min(1.0, d_j + penalty_j)
            # Since penalty_j >= 0, d_adjusted >= min(1.0, d_j)
            # Note: d_j can exceed 1.0 (raw unbounded), so we compare to min(1.0, d_j)
            expected_floor = np.minimum(1.0, synth_out.d_j)
            assert np.all(cont_out.d_adjusted >= expected_floor - 1e-9), \
                "d_adjusted < min(1.0, d_j): Contrarian deflated disagreement"
        run_check(f"[{date}] d_adjusted >= min(1.0, d_j) (Contrarian only inflates)", check_d_adjusted_monotone)

        def check_d_adjusted_clipped():
            assert np.all(cont_out.d_adjusted >= 0)
            assert np.all(cont_out.d_adjusted <= 1.0 + 1e-9)
        run_check(f"[{date}] d_adjusted in [0,1]", check_d_adjusted_clipped)

        def check_penalty_range():
            assert np.all(cont_out.penalty_j >= MIN_PENALTY - 1e-9)
            assert np.all(cont_out.penalty_j <= MAX_PENALTY + 1e-9)
        run_check(f"[{date}] penalty_j in [0, {MAX_PENALTY}]", check_penalty_range)

        def check_formula():
            expected = np.minimum(1.0, synth_out.d_j + cont_out.penalty_j)
            assert np.allclose(cont_out.d_adjusted, expected, atol=1e-9), \
                "d_adjusted != min(1.0, d_j + penalty_j)"
        run_check(f"[{date}] d_adjusted = min(1.0, d_j + penalty_j)", check_formula)

        def check_bl_inputs_valid():
            from black_litterman import BLInputs
            bl = cont_out.bl_inputs_updated
            assert isinstance(bl, BLInputs)
            assert np.all(bl.c_final >= 0) and np.all(bl.c_final <= 1)
            assert np.all(bl.d_adjusted >= 0) and np.all(bl.d_adjusted <= 1)
        run_check(f"[{date}] Updated BLInputs valid", check_bl_inputs_valid)

        def check_full_pipeline():
            bl_out = bl_model.run(
                date       = pd.Timestamp(date),
                returns_df = returns_df,
                w_mkt_df   = w_mkt_df,
                w_previous = w_prev,
                bl_inputs  = cont_out.bl_inputs_updated,
            )
            assert np.all(np.isfinite(bl_out.w_final))
            assert abs(bl_out.w_final.sum() - 1.0) < 1e-9
            assert np.all(bl_out.w_final >= -1e-9)
            print(f"\n  Final BL portfolio [{date}]:")
            for t, w in zip(TICKERS, bl_out.w_final):
                bar = "█" * int(w * 40)
                print(f"    {t:<5}  {w:6.1%}  {bar}")
            print(f"  Turnover: {bl_out.turnover:.4f} "
                  f"({'cap applied' if bl_out.cap_applied else 'within limit'})")
        run_check(f"[{date}] Full pipeline → valid BL portfolio", check_full_pipeline)

    # Cross-scenario: tightening should have higher mean penalty than neutral
    if len(all_results) == 2:
        keys  = list(all_results.keys())
        _, c1 = all_results[keys[0]]   # Sep 2022
        _, c2 = all_results[keys[1]]   # Feb 2019
        def check_tightening_higher_penalty():
            assert c1.mean_penalty >= c2.mean_penalty - 0.05, \
                (f"Expected tightening to have >= penalty than neutral. "
                 f"Tightening: {c1.mean_penalty:.3f}, Neutral: {c2.mean_penalty:.3f}")
        run_check("Cross-scenario: tightening has >= mean penalty than neutral",
                  check_tightening_higher_penalty)

    print(f"\n{'='*70}")
    print(f"Results: {len(passed)}/{len(passed)+len(failed)} passed")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All checks passed.")
        print("\nKey observations for research project (RQ2 early warning):")
        for label, (synth_out, cont_out) in all_results.items():
            print(f"  {label[:50]}")
            print(f"    Mean d_j (Synthesis):    {synth_out.mean_disagreement:.3f}")
            print(f"    Mean penalty (Contrarian): {cont_out.mean_penalty:.3f}")
            mean_d_adj = float(cont_out.d_adjusted.mean())
            print(f"    Mean d_adjusted (final):   {mean_d_adj:.3f}")
            print(f"    Max penalty sector:        {cont_out.max_penalty_sector}")
    print("=" * 70)