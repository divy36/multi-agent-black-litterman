"""Agent hawk."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Signal extractor — must be in same directory
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

OLLAMA_MODEL   = "llama3:latest"
OLLAMA_URL     = "http://localhost:11434/api/generate"
TEMPERATURE    = 0          # pinned to 0 for reproducibility (brief requirement)
MAX_RETRIES    = 3          # attempts before falling back to conservative defaults
TIMEOUT_SECS   = 120        # per Ollama request

# Conservative fallback returned when all retries fail
FALLBACK_RETURN     = 0.0
FALLBACK_CONFIDENCE = 0.3

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class HawkOutput:
    """
    Full output bundle from one Hawk agent call.

    Attributes
    ----------
    Q : np.ndarray (N,)
        Expected monthly returns per sector, in TICKERS order.
        Units: decimal (e.g. -0.012 = -1.2% expected monthly return).
    confidence : np.ndarray (N,)
        Raw confidence scores in [0, 1] before any guardrail adjustment.
    date : str
        ISO month string this output corresponds to (e.g. "2019-02-28").
    fomc_date : str
        Publication date of the FOMC minutes used as input.
    cpi_yoy : float
        Year-on-year CPI inflation rate provided as macro context.
    pce_yoy : float
        Year-on-year Core PCE inflation rate provided as macro context.
    pnl_context : str
        PnL feedback string injected into the prompt (empty string if
        this is the first month of the simulation).
    fallback_used : bool
        True if all LLM retries failed and conservative defaults were used.
        Flagged as a hallucination/failure event in the stability report.
    raw_response : str
        The raw LLM response string (preserved for case study analysis).
    attempts : int
        Number of LLM calls made before a valid response was obtained.
    """
    Q:            np.ndarray
    confidence:   np.ndarray
    date:         str
    fomc_date:    str
    cpi_yoy:      float
    pce_yoy:      float
    pnl_context:  str
    fallback_used: bool
    raw_response: str
    attempts:     int


# --------------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------------- #
def _build_prompt(
    fomc_text:   str,
    cpi_yoy:     float,
    pce_yoy:     float,
    date:        str,
    pnl_context: str,
) -> str:
    """
    Build the full Hawk agent prompt.

    Structure
    ---------
    1. Persona — who the Hawk is and what it fears
    2. Macro context — FOMC text excerpt + CPI/PCE figures
    3. PnL feedback — previous month performance (if available)
    4. Sector guide — brief rate sensitivity reminder per ETF
    5. Output format — exact JSON schema required

    The FOMC text is truncated to 6000 characters to stay within the
    llama3:latest context window while preserving the most policy-relevant
    content (the minutes tend to front-load the key decisions).
    """
    fomc_excerpt = fomc_text[:3000].strip()

    # Pre-extract structured signals — forces regime differentiation
    signals_block = ""
    if _EXTRACTOR is not None:
        try:
            signals = _EXTRACTOR.extract(fomc_text)
            signals_block = signals.summary()
        except Exception:
            signals_block = "## FOMC Signal Extraction: unavailable"
    else:
        signals_block = "## FOMC Signal Extraction: fomc_signals.py not found"

    pnl_section = ""
    if pnl_context:
        pnl_section = f"""
## Last Month's Performance Feedback
{pnl_context}

Reflect on whether your previous views were directionally correct.
If you were wrong, consider whether you were too aggressive or too conservative.
"""

    prompt = f"""You are HAWK, a risk-averse macro analyst at an investment committee.
You fear inflation above all else. When inflation is rising or the Federal Reserve
is tightening monetary policy, you expect equities to fall — especially
interest-rate-sensitive sectors. You are sceptical of growth stories when
rates are rising and always ask: "what does this do to discount rates?"

Today's date: {date}
CPI inflation (year-on-year): {cpi_yoy:.2f}%
Core PCE inflation (year-on-year): {pce_yoy:.2f}%

---

{signals_block}

---

## Step 1 — Read and quote the minutes
Read the Federal Reserve meeting minutes below carefully.
Then identify and write down (in your own words, 2-3 sentences):
a) What did the Fed say about the direction of interest rates?
b) What did the Fed say about inflation?
c) What did the Fed say about economic growth and employment?

## Federal Reserve Meeting Minutes (excerpt)
{fomc_excerpt}

---
{pnl_section}
---

## Step 2 — Assess the macro regime
Based ONLY on what you just read in these specific minutes and the inflation
figures above, classify the current regime as one of:
- TIGHTENING (rates rising, inflation above target, hawkish tone)
- NEUTRAL (rates on hold, inflation near target, balanced tone)  
- EASING (rates falling or cut, growth concerns, dovish tone)
- EMERGENCY (unprecedented action, crisis language, extreme uncertainty)

Write your regime classification and a one-sentence justification.

---

## Step 3 — Output your sector views
Now output your views. Your expected_return values MUST reflect the specific
regime you identified above. Do not use the same numbers across different
regimes — a TIGHTENING regime must produce more negative returns on
rate-sensitive sectors than a NEUTRAL or EASING regime.

Regime-return guidance (adjust magnitude based on how extreme the language is):
- TIGHTENING:  XLK [-0.03 to -0.01], XLF [+0.01 to +0.02], XLRE [-0.04 to -0.02], XLU [-0.03 to -0.02]
- NEUTRAL:     XLK [-0.01 to +0.01], XLF [0.00 to +0.01],  XLRE [-0.01 to +0.01], XLU [-0.01 to 0.00]
- EASING:      XLK [+0.01 to +0.03], XLF [-0.01 to +0.01], XLRE [+0.01 to +0.03], XLU [+0.01 to +0.02]
- EMERGENCY:   all sectors negative, confidence 0.30-0.50, high uncertainty

Confidence MUST reflect certainty of the Fed signal:
- Clear, forceful language (e.g. "strongly committed", specific rate path) → confidence 0.65-0.85
- Balanced or uncertain language (e.g. "patient", "data dependent") → confidence 0.40-0.60
- Crisis or unprecedented language → confidence 0.25-0.45

## Sector Rate Sensitivity
- XLK (Technology): Very high NEGATIVE rate sensitivity
- XLF (Financials): POSITIVE rate sensitivity — benefits from rate rises
- XLE (Energy): Low rate sensitivity — commodity driven, moderate views
- XLV (Healthcare): Low rate sensitivity — defensive, moderate views
- XLI (Industrials): Medium rate sensitivity — cautious in tightening
- XLY (Consumer Discretionary): Medium NEGATIVE rate sensitivity
- XLP (Consumer Staples): Defensive safe haven — rotate in during stress
- XLRE (Real Estate): Very high NEGATIVE rate sensitivity — most bearish
- XLU (Utilities): Very high NEGATIVE rate sensitivity
- XLB (Materials): Low rate sensitivity — commodity driven, moderate views

---

## Output Format
After your Step 1 and Step 2 analysis, output the JSON array.
The JSON must be the LAST thing you write.
Output exactly 10 objects in this order: XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLRE, XLU, XLB

[
  {{"sector": "XLK",  "expected_return": <float>, "confidence": <float>}},
  {{"sector": "XLF",  "expected_return": <float>, "confidence": <float>}},
  {{"sector": "XLE",  "expected_return": <float>, "confidence": <float>}},
  {{"sector": "XLV",  "expected_return": <float>, "confidence": <float>}},
  {{"sector": "XLI",  "expected_return": <float>, "confidence": <float>}},
  {{"sector": "XLY",  "expected_return": <float>, "confidence": <float>}},
  {{"sector": "XLP",  "expected_return": <float>, "confidence": <float>}},
  {{"sector": "XLRE", "expected_return": <float>, "confidence": <float>}},
  {{"sector": "XLU",  "expected_return": <float>, "confidence": <float>}},
  {{"sector": "XLB",  "expected_return": <float>, "confidence": <float>}}
]

CRITICAL: Always end your response with the complete JSON array above with real numbers filled in.
Do NOT stop after Step 1 or Step 2 — the JSON is mandatory."""

    return prompt


# --------------------------------------------------------------------------- #
# JSON parser — robust to common LLM formatting quirks
# --------------------------------------------------------------------------- #
def _parse_hawk_response(raw: str) -> Optional[list[dict]]:
    """
    Extract and validate the JSON array from an LLM response.

    Handles:
    - Markdown code fences (```json ... ```)
    - Extra text before or after the JSON array
    - Trailing commas before closing brackets
    - Single quotes instead of double quotes

    Returns None if no valid parseable JSON array is found.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    text = text.replace("```", "").strip()

    # Find the outermost JSON array
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = text[start:end + 1]

    # Fix trailing commas (e.g. [{"a":1},] → [{"a":1}])
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    # Fix single quotes
    json_str = json_str.replace("'", '"')

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    return data


def _validate_views(data: list[dict]) -> bool:
    """
    Validate that the parsed JSON has the expected structure.

    Checks:
    - Exactly 10 entries
    - Each entry has sector, expected_return, confidence
    - Sectors match TICKERS in order
    - expected_return in [-0.20, +0.20] (hard clip — LLM sometimes hallucinates huge numbers)
    - confidence in [0.0, 1.0]
    """
    if len(data) != N:
        logger.warning(f"Hawk: expected {N} sectors, got {len(data)}")
        return False

    for i, (entry, ticker) in enumerate(zip(data, TICKERS)):
        if not isinstance(entry, dict):
            return False
        if "sector" not in entry or "expected_return" not in entry or "confidence" not in entry:
            logger.warning(f"Hawk: entry {i} missing required fields: {entry}")
            return False
        if entry["sector"] != ticker:
            logger.warning(f"Hawk: sector mismatch at position {i}: "
                           f"expected {ticker}, got {entry['sector']}")
            return False
        try:
            r = float(entry["expected_return"])
            c = float(entry["confidence"])
        except (TypeError, ValueError):
            return False
        if not (-0.20 <= r <= 0.20):
            logger.warning(f"Hawk: expected_return {r} out of range for {ticker}")
            return False
        if not (0.0 <= c <= 1.0):
            logger.warning(f"Hawk: confidence {c} out of range for {ticker}")
            return False

    return True


# --------------------------------------------------------------------------- #
# Ollama caller
# --------------------------------------------------------------------------- #
def _call_ollama(prompt: str, attempt: int) -> str:
    """
    Call the local Ollama API and return the raw response string.

    Uses streaming=False for simplicity. Temperature is pinned to 0.
    Raises RuntimeError on connection failure or timeout.
    """
    import urllib.request

    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 2500,    # enough for 10 JSON entries
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

    response_data = json.loads(body)
    return response_data.get("response", "")


# --------------------------------------------------------------------------- #
# Fallback constructor
# --------------------------------------------------------------------------- #
def _make_fallback(
    date:        str,
    fomc_date:   str,
    cpi_yoy:     float,
    pce_yoy:     float,
    pnl_context: str,
    raw:         str,
    attempts:    int,
) -> HawkOutput:
    """Conservative fallback when all LLM retries are exhausted."""
    logger.error(
        f"Hawk: all {attempts} attempts failed for {date}. "
        f"Using conservative fallback (return=0, confidence={FALLBACK_CONFIDENCE})."
    )
    return HawkOutput(
        Q             = np.zeros(N),
        confidence    = np.full(N, FALLBACK_CONFIDENCE),
        date          = date,
        fomc_date     = fomc_date,
        cpi_yoy       = cpi_yoy,
        pce_yoy       = pce_yoy,
        pnl_context   = pnl_context,
        fallback_used = True,
        raw_response  = raw,
        attempts      = attempts,
    )


# --------------------------------------------------------------------------- #
# Main agent class
# --------------------------------------------------------------------------- #
class HawkAgent:
    """
    Agent 1 — Hawk.

    Reads FOMC minutes and CPI/PCE data, produces hawkish sector return views.

    Usage
    -----
        agent = HawkAgent()
        output = agent.run(
            fomc_text   = open("2019 FO/fomcminutes20190130.txt").read(),
            cpi_yoy     = 1.9,
            pce_yoy     = 1.7,
            date        = "2019-02-28",
            fomc_date   = "2019-01-30",
            pnl_context = "",   # empty string for first month
        )
        print(output.Q)           # expected returns array
        print(output.confidence)  # confidence array
        print(output.fallback_used)
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
        fomc_text:   str,
        cpi_yoy:     float,
        pce_yoy:     float,
        date:        str,
        fomc_date:   str,
        pnl_context: str = "",
    ) -> HawkOutput:
        """
        Run the Hawk agent for one month.

        Parameters
        ----------
        fomc_text : str
            Full text of the FOMC meeting minutes, as extracted by
            data_pipeline.py. Will be truncated to 6000 chars in the prompt.
        cpi_yoy : float
            Year-on-year CPI inflation rate for this month (e.g. 1.9 for 1.9%).
        pce_yoy : float
            Year-on-year Core PCE inflation rate for this month.
        date : str
            ISO month-end date string (e.g. "2019-02-28").
        fomc_date : str
            Publication date of the FOMC minutes being used (e.g. "2019-01-30").
            Used for look-ahead bias auditing.
        pnl_context : str
            Structured string describing previous month's portfolio performance.
            Injected into the prompt as feedback. Empty string for month 1.

        Returns
        -------
        HawkOutput
            Full output bundle including Q vector, confidence scores,
            metadata, and fallback flag.
        """
        prompt = _build_prompt(fomc_text, cpi_yoy, pce_yoy, date, pnl_context)
        raw    = ""

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Hawk [{date}]: attempt {attempt}/{self.max_retries}")
            try:
                raw = _call_ollama(prompt, attempt)
            except RuntimeError as exc:
                logger.warning(str(exc))
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)   # exponential backoff
                continue

            parsed = _parse_hawk_response(raw)
            if parsed is None:
                logger.warning(f"Hawk [{date}]: JSON parse failed on attempt {attempt}. "
                               f"Raw: {raw[:200]}")
                continue

            if not _validate_views(parsed):
                logger.warning(f"Hawk [{date}]: validation failed on attempt {attempt}.")
                continue

            # Success
            Q          = np.array([float(e["expected_return"]) for e in parsed])
            confidence = np.array([float(e["confidence"])       for e in parsed])

            logger.info(f"Hawk [{date}]: success on attempt {attempt}.")
            return HawkOutput(
                Q             = Q,
                confidence    = confidence,
                date          = date,
                fomc_date     = fomc_date,
                cpi_yoy       = cpi_yoy,
                pce_yoy       = pce_yoy,
                pnl_context   = pnl_context,
                fallback_used = False,
                raw_response  = raw,
                attempts      = attempt,
            )

        # All retries exhausted
        return _make_fallback(date, fomc_date, cpi_yoy, pce_yoy,
                              pnl_context, raw, self.max_retries)

    def describe(self) -> dict:
        """Return agent metadata for logging and stability reports."""
        return {
            "agent":       "HAWK",
            "model":       self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "persona":     "Inflation-sensitive, rate-rise pessimist",
        }


# --------------------------------------------------------------------------- #
# Self-test — run directly against Ollama
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Live test against your local Ollama instance.

    Uses a short synthetic FOMC excerpt so the test runs in ~30 seconds.
    Tests three scenarios:
      1. Hawkish environment (high inflation, rate hikes signalled)
      2. Dovish environment (low inflation, rate cuts signalled)
      3. COVID emergency (ambiguous, high uncertainty)

    Expected behaviour:
      - Scenario 1: negative returns on XLK, XLRE, XLU; positive on XLF
      - Scenario 2: positive returns across most sectors
      - Scenario 3: mixed, low confidence, possibly high XLRE/XLU bearishness
    """
    import sys

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)s  %(message)s",
    )

    agent = HawkAgent()
    print("=" * 60)
    print("Hawk Agent self-test")
    print(f"Model: {agent.model}  |  Temperature: {agent.temperature}")
    print("=" * 60)

    scenarios = [
        {
            "label": "Scenario 1 — Hawkish (high inflation, rate hikes)",
            "date":       "2022-09-30",
            "fomc_date":  "2022-09-21",
            "cpi_yoy":    8.3,
            "pce_yoy":    4.9,
            "pnl_context": "Last month: portfolio returned -3.2%. "
                           "Your bearish XLRE view was correct (-8.1%). "
                           "Your XLF bullish view was partially correct (+1.2%).",
            "fomc_text": """
The Committee decided to raise the target range for the federal funds rate to
3 to 3-1/4 percent and anticipates that ongoing increases in the target range
will be appropriate in order to attain a stance of monetary policy that is
sufficiently restrictive to return inflation to 2 percent over time.
In determining the pace of future increases in the target range, the Committee
will take into account the cumulative tightening of monetary policy, the lags
with which monetary policy affects economic activity and inflation, and economic
and financial developments. Inflation remains elevated, reflecting supply and
demand imbalances related to the pandemic, higher food and energy prices, and
broader price pressures. The Committee is highly attentive to inflation risks.
Recent indicators point to modest growth in spending and production. Job gains
have been robust in recent months, and the unemployment rate has remained low.
The Committee seeks to achieve maximum employment and inflation at the rate of
2 percent over the longer run. In support of these goals, the Committee decided
to raise the target range for the federal funds rate. The Committee is strongly
committed to returning inflation to its 2 percent objective.
""",
        },
        {
            "label": "Scenario 2 — Dovish (low inflation, accommodative)",
            "date":       "2019-02-28",
            "fomc_date":  "2019-01-30",
            "cpi_yoy":    1.6,
            "pce_yoy":    1.7,
            "pnl_context": "",
            "fomc_text": """
In light of global economic and financial developments and muted inflation
pressures, the Committee will be patient as it determines what future
adjustments to the target range for the federal funds rate may be appropriate
to support these outcomes. The Committee continues to view sustained expansion
of economic activity, strong labor market conditions, and inflation near the
Committee's symmetric 2 percent objective as the most likely outcomes.
The labor market has continued to strengthen and that economic activity has
been rising at a solid rate. Job gains have been strong, on average, in recent
months, and the unemployment rate has remained low. Household spending has
continued to grow strongly, while growth of business fixed investment has
moderated from its rapid pace earlier in the year.
""",
        },
        {
            "label": "Scenario 3 — COVID emergency (March 2020)",
            "date":       "2020-03-31",
            "fomc_date":  "2020-03-15",
            "cpi_yoy":    2.3,
            "pce_yoy":    1.8,
            "pnl_context": "Last month: portfolio returned -8.4%. "
                           "Market conditions unprecedented. All views were wrong.",
            "fomc_text": """
The coronavirus outbreak has harmed communities and disrupted economic activity
in many countries, including the United States. Global financial conditions have
also been significantly affected. Available economic data show that the U.S.
economy came into this challenging period on a strong footing. The Committee
expects that the effects of the coronavirus will weigh on economic activity in
the near term and poses considerable risks to the economic outlook over the
medium term. In light of these developments, the Committee decided to lower the
target range for the federal funds rate to 0 to 1/4 percent. The Committee
expects to maintain this target range until it is confident that the economy has
weathered recent events and is on track to achieve its maximum employment and
price stability goals. The Federal Reserve is prepared to use its full range of
tools to support the flow of credit to households and businesses and thereby
promote its maximum employment and price stability goals.
""",
        },
    ]

    all_passed = True
    for sc in scenarios:
        print(f"\n{'─' * 60}")
        print(f"  {sc['label']}")
        print(f"{'─' * 60}")

        output = agent.run(
            fomc_text   = sc["fomc_text"],
            cpi_yoy     = sc["cpi_yoy"],
            pce_yoy     = sc["pce_yoy"],
            date        = sc["date"],
            fomc_date   = sc["fomc_date"],
            pnl_context = sc["pnl_context"],
        )

        print(f"  Fallback used: {output.fallback_used}")
        print(f"  Attempts:      {output.attempts}")
        print(f"\n  {'Sector':<6}  {'Exp Return':>10}  {'Confidence':>10}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}")
        for ticker, q, c in zip(TICKERS, output.Q, output.confidence):
            direction = "▲" if q > 0 else "▼"
            print(f"  {ticker:<6}  {q:>+10.4f}  {c:>10.2f}  {direction}")

        print(f"\n  Return range:  [{output.Q.min():+.4f}, {output.Q.max():+.4f}]")
        print(f"  Conf range:    [{output.confidence.min():.2f}, {output.confidence.max():.2f}]")
        print(f"  All returns finite: {np.all(np.isfinite(output.Q))}")
        print(f"  All conf in [0,1]:  {np.all((output.confidence >= 0) & (output.confidence <= 1))}")

        if output.fallback_used:
            print("  WARNING: fallback used — check Ollama is running")
            all_passed = False

    print(f"\n{'=' * 60}")
    if all_passed:
        print("All scenarios completed. Check outputs above for economic sense:")
        print("  Scenario 1: XLK/XLRE/XLU should be negative, XLF positive")
        print("  Scenario 2: Most sectors should be positive or neutral")
        print("  Scenario 3: Mixed views, generally negative, lower confidence")
    else:
        print("WARNING: one or more scenarios used the fallback.")
        print("Check that Ollama is running: ollama serve")
    print("=" * 60)