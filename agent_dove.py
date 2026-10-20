"""Agent dove."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Signal extractor — must be in same directory as agent_hawk.py
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
TEMPERATURE    = 0
MAX_RETRIES    = 3
TIMEOUT_SECS   = 120

FALLBACK_RETURN     = 0.0
FALLBACK_CONFIDENCE = 0.3

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data class
# --------------------------------------------------------------------------- #
@dataclass
class DoveOutput:
    """
    Full output bundle from one Dove agent call.

    Attributes mirror HawkOutput exactly so Synthesis can process both
    with the same interface.
    """
    Q:            np.ndarray   # expected monthly returns (N,)
    confidence:   np.ndarray   # raw confidence scores (N,)
    date:         str
    fomc_date:    str
    gdp_growth:   float        # annualised GDP growth rate (%)
    unemployment: float        # unemployment rate (%)
    pnl_context:  str
    fallback_used: bool
    raw_response: str
    attempts:     int


# --------------------------------------------------------------------------- #
# Prompt builder
# --------------------------------------------------------------------------- #
def _build_prompt(
    fomc_text:    str,
    gdp_growth:   float,
    unemployment: float,
    date:         str,
    pnl_context:  str,
) -> str:
    """
    Build the full Dove agent prompt with pre-extracted FOMC signals.

    The Dove interprets the same FOMC signals as the Hawk but through
    an optimistic lens:
    - Low hawkish score (dovish Fed) → bullish across most sectors
    - High hawkish score (tightening) → cautious but still more optimistic
      than the Hawk, especially on growth sectors
    - Emergency → very uncertain, but sees stimulus as equity-positive
    """
    fomc_excerpt = fomc_text[:2000].strip()

    # Pre-extract signals
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
If growth came in stronger than expected, increase your conviction.
If growth disappointed, moderate your optimism accordingly.
"""

    prompt = f"""You are DOVE, a growth-focused macro analyst at an investment committee.
You prioritise employment, economic expansion, and consumer strength above all else.
When the Federal Reserve is accommodative or growth signals are strong, you expect
equities to rise — especially cyclical and growth sectors. You always ask:
"what does this mean for corporate earnings and consumer spending?"
You are the optimistic counterpart to a hawkish colleague and deliberately
look for the bullish case even in uncertain environments.

Today's date: {date}
GDP growth (annualised): {gdp_growth:.2f}%
Unemployment rate: {unemployment:.2f}%

---

{signals_block}

---

## Step 1 — Read and interpret the minutes through a growth lens
Read the Federal Reserve meeting minutes below carefully.
Then identify (2-3 sentences):
a) What signals does the Fed give about the strength of the economy and jobs?
b) Is the Fed's tone supportive of continued economic expansion?
c) Are there any risks to growth that concern you as a growth-focused analyst?

## Federal Reserve Meeting Minutes (excerpt)
{fomc_excerpt}

---
{pnl_section}
---

## Step 2 — Assess the growth regime
Based on what you read AND the macro data above, classify as one of:
- EXPANSION (strong growth, low unemployment, Fed supportive or neutral)
- SLOWDOWN (growth weakening, unemployment rising, Fed tightening into weakness)
- RECOVERY (growth rebounding after weakness, Fed accommodative)
- CRISIS (severe disruption, emergency Fed action, extreme uncertainty)

Write your regime classification and a one-sentence justification.

---

## Step 3 — Output your sector views
Your expected_return values MUST reflect the specific regime above.
As a Dove you are systematically MORE OPTIMISTIC than your hawkish colleague.
Your views should differ meaningfully from a hawkish analyst — especially on
growth-sensitive sectors like XLK, XLI, XLY where you should be more bullish.

Regime-return guidance for DOVE (adjust magnitude based on signal strength):
- EXPANSION:  XLK [+0.01 to +0.03], XLI [+0.01 to +0.02], XLY [+0.01 to +0.02], XLF [0.00 to +0.01]
- SLOWDOWN:   XLK [-0.01 to +0.01], XLI [-0.01 to 0.00],  XLY [-0.01 to 0.00],  XLP [+0.01 to +0.02]
- RECOVERY:   XLK [+0.02 to +0.04], XLI [+0.01 to +0.03], XLY [+0.02 to +0.03], XLRE [+0.01 to +0.02]
- CRISIS:     all sectors cautious, confidence 0.25-0.45, emphasise XLP/XLV defensives

Confidence MUST reflect certainty of the growth signal:
- Strong, clear growth data (GDP >2.5%, unemployment <4%) → confidence 0.60-0.80
- Mixed signals or moderate growth → confidence 0.40-0.60
- Deteriorating or crisis conditions → confidence 0.20-0.45

## Sector Growth Sensitivity (Dove perspective)
- XLK (Technology): HIGH positive growth sensitivity — bull market leader in expansion
- XLF (Financials): Moderate — benefits from growth but hurt by rapid rate rises
- XLE (Energy): Moderate — benefits from strong demand, global growth driven
- XLV (Healthcare): Defensive — stable in all regimes, moderate bullish always
- XLI (Industrials): HIGH positive growth sensitivity — classic cycle play
- XLY (Consumer Discretionary): HIGH positive — strongest in consumer expansion
- XLP (Consumer Staples): Defensive safe haven — rotate in during stress only
- XLRE (Real Estate): Hurt by rate rises but benefits from growth and low rates
- XLU (Utilities): Defensive — low growth sensitivity, hold in uncertainty
- XLB (Materials): Moderate — benefits from global growth and infrastructure spend

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
Do NOT stop after Step 1 or Step 2 — the JSON is mandatory.
Begin your JSON output now with the opening bracket ["""

    return prompt


# --------------------------------------------------------------------------- #
# JSON parser — identical to Hawk
# --------------------------------------------------------------------------- #
def _parse_dove_response(raw: str) -> Optional[list[dict]]:
    text = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    text = text.replace("```", "").strip()
    start = text.find("[")
    end   = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start:end + 1]
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    json_str = json_str.replace("'", '"')
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return data


def _validate_views(data: list[dict]) -> bool:
    if len(data) != N:
        logger.warning(f"Dove: expected {N} sectors, got {len(data)}")
        return False
    for i, (entry, ticker) in enumerate(zip(data, TICKERS)):
        if not isinstance(entry, dict):
            return False
        if "sector" not in entry or "expected_return" not in entry or "confidence" not in entry:
            logger.warning(f"Dove: entry {i} missing fields: {entry}")
            return False
        if entry["sector"] != ticker:
            logger.warning(f"Dove: sector mismatch at {i}: expected {ticker}, got {entry['sector']}")
            return False
        try:
            r = float(entry["expected_return"])
            c = float(entry["confidence"])
        except (TypeError, ValueError):
            return False
        if not (-0.20 <= r <= 0.20):
            logger.warning(f"Dove: expected_return {r} out of range for {ticker}")
            return False
        if not (0.0 <= c <= 1.0):
            logger.warning(f"Dove: confidence {c} out of range for {ticker}")
            return False
    return True


# --------------------------------------------------------------------------- #
# Ollama caller — identical to Hawk
# --------------------------------------------------------------------------- #
def _call_ollama(prompt: str, attempt: int) -> str:
    import urllib.request
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 3000,
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
    date: str, fomc_date: str, gdp_growth: float,
    unemployment: float, pnl_context: str, raw: str, attempts: int,
) -> DoveOutput:
    logger.error(
        f"Dove: all {attempts} attempts failed for {date}. "
        f"Using conservative fallback (return=0, confidence={FALLBACK_CONFIDENCE})."
    )
    return DoveOutput(
        Q             = np.zeros(N),
        confidence    = np.full(N, FALLBACK_CONFIDENCE),
        date          = date,
        fomc_date     = fomc_date,
        gdp_growth    = gdp_growth,
        unemployment  = unemployment,
        pnl_context   = pnl_context,
        fallback_used = True,
        raw_response  = raw,
        attempts      = attempts,
    )


# --------------------------------------------------------------------------- #
# Main agent class
# --------------------------------------------------------------------------- #
class DoveAgent:
    """
    Agent 2 — Dove.

    Reads FOMC minutes and GDP/unemployment data, produces growth-optimistic
    sector return views that are systematically more bullish than the Hawk's
    on cyclical and growth sectors.

    Usage
    -----
        agent = DoveAgent()
        output = agent.run(
            fomc_text    = open("2019 FO/fomcminutes20190130.txt").read(),
            gdp_growth   = 2.9,
            unemployment = 4.0,
            date         = "2019-02-28",
            fomc_date    = "2019-01-30",
            pnl_context  = "",
        )
        print(output.Q)
        print(output.confidence)
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
        fomc_text:    str,
        gdp_growth:   float,
        unemployment: float,
        date:         str,
        fomc_date:    str,
        pnl_context:  str = "",
    ) -> DoveOutput:
        """
        Run the Dove agent for one month.

        Parameters
        ----------
        fomc_text : str
            Full text of the FOMC meeting minutes.
        gdp_growth : float
            Annualised GDP growth rate for this month (e.g. 2.9 for 2.9%).
        unemployment : float
            Unemployment rate for this month (e.g. 4.0 for 4.0%).
        date : str
            ISO month-end date string (e.g. "2019-02-28").
        fomc_date : str
            Publication date of the FOMC minutes (for look-ahead audit).
        pnl_context : str
            Previous month portfolio performance feedback. Empty for month 1.

        Returns
        -------
        DoveOutput
        """
        prompt = _build_prompt(fomc_text, gdp_growth, unemployment,
                               date, pnl_context)
        raw = ""

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Dove [{date}]: attempt {attempt}/{self.max_retries}")
            try:
                raw = _call_ollama(prompt, attempt)
            except RuntimeError as exc:
                logger.warning(str(exc))
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                continue

            parsed = _parse_dove_response(raw)
            if parsed is None:
                logger.warning(f"Dove [{date}]: JSON parse failed on attempt {attempt}. "
                               f"Raw: {raw[:200]}")
                continue

            if not _validate_views(parsed):
                logger.warning(f"Dove [{date}]: validation failed on attempt {attempt}.")
                continue

            Q          = np.array([float(e["expected_return"]) for e in parsed])
            confidence = np.array([float(e["confidence"])       for e in parsed])

            logger.info(f"Dove [{date}]: success on attempt {attempt}.")
            return DoveOutput(
                Q             = Q,
                confidence    = confidence,
                date          = date,
                fomc_date     = fomc_date,
                gdp_growth    = gdp_growth,
                unemployment  = unemployment,
                pnl_context   = pnl_context,
                fallback_used = False,
                raw_response  = raw,
                attempts      = attempt,
            )

        return _make_fallback(date, fomc_date, gdp_growth, unemployment,
                              pnl_context, raw, self.max_retries)

    def describe(self) -> dict:
        return {
            "agent":       "DOVE",
            "model":       self.model,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "persona":     "Growth-focused, employment optimist",
        }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    Live test against Ollama. Same three scenarios as the Hawk self-test
    so you can directly compare outputs and verify the Dove is more bullish
    on growth sectors in expansionary regimes.

    Expected behaviour vs Hawk:
    - Scenario 1 (tightening): Dove less negative than Hawk on XLK/XLI/XLY
    - Scenario 2 (dovish):     Dove positive on XLK/XLI/XLY where Hawk is flat/negative
    - Scenario 3 (COVID):      Both cautious but Dove sees stimulus as growth-positive
    """
    import sys

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s  %(levelname)s  %(message)s",
    )

    agent = DoveAgent()
    print("=" * 60)
    print("Dove Agent self-test")
    print(f"Model: {agent.model}  |  Temperature: {agent.temperature}")
    print("=" * 60)

    scenarios = [
        {
            "label":       "Scenario 1 — Hawkish (high inflation, rate hikes)",
            "date":        "2022-09-30",
            "fomc_date":   "2022-09-21",
            "gdp_growth":  2.6,
            "unemployment": 3.5,
            "pnl_context": "Last month: portfolio returned -3.2%. "
                           "Growth sectors underperformed. Reassess conviction.",
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
2 percent over the longer run. The Committee is strongly committed to returning
inflation to its 2 percent objective.
""",
        },
        {
            "label":       "Scenario 2 — Dovish (low inflation, accommodative)",
            "date":        "2019-02-28",
            "fomc_date":   "2019-01-30",
            "gdp_growth":  3.1,
            "unemployment": 4.0,
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
            "label":       "Scenario 3 — COVID emergency (March 2020)",
            "date":        "2020-03-31",
            "fomc_date":   "2020-03-15",
            "gdp_growth":  -5.0,
            "unemployment": 4.4,
            "pnl_context": "Last month: portfolio returned -8.4%. "
                           "All growth views were wrong. Unprecedented conditions.",
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

    # Also import Hawk for direct comparison
    hawk_outputs = {}
    try:
        from agent_hawk import HawkAgent
        hawk = HawkAgent()
        hawk_available = True
    except ImportError:
        hawk_available = False

    all_passed = True
    for sc in scenarios:
        print(f"\n{'─' * 60}")
        print(f"  {sc['label']}")
        print(f"{'─' * 60}")

        dove_out = agent.run(
            fomc_text    = sc["fomc_text"],
            gdp_growth   = sc["gdp_growth"],
            unemployment = sc["unemployment"],
            date         = sc["date"],
            fomc_date    = sc["fomc_date"],
            pnl_context  = sc["pnl_context"],
        )

        # Run Hawk on same scenario for comparison
        hawk_Q = None
        if hawk_available:
            hawk_out = hawk.run(
                fomc_text   = sc["fomc_text"],
                cpi_yoy     = 8.3 if "Hawkish" in sc["label"] else
                              1.6 if "Dovish"  in sc["label"] else 2.3,
                pce_yoy     = 4.9 if "Hawkish" in sc["label"] else
                              1.7 if "Dovish"  in sc["label"] else 1.8,
                date        = sc["date"],
                fomc_date   = sc["fomc_date"],
                pnl_context = sc["pnl_context"],
            )
            hawk_Q = hawk_out.Q

        print(f"  Fallback used: {dove_out.fallback_used}")
        print(f"  Attempts:      {dove_out.attempts}")

        if hawk_Q is not None:
            print(f"\n  {'Sector':<6}  {'Dove':>8}  {'Hawk':>8}  {'Diff':>8}  {'Direction'}")
            print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")
            for ticker, dq, hq, dc in zip(TICKERS, dove_out.Q, hawk_Q, dove_out.confidence):
                diff = dq - hq
                direction = "Dove↑" if diff > 0.001 else "Dove↓" if diff < -0.001 else "Same "
                print(f"  {ticker:<6}  {dq:>+8.4f}  {hq:>+8.4f}  {diff:>+8.4f}  {direction}  conf={dc:.2f}")
        else:
            print(f"\n  {'Sector':<6}  {'Exp Return':>10}  {'Confidence':>10}")
            print(f"  {'─'*6}  {'─'*10}  {'─'*10}")
            for ticker, q, c in zip(TICKERS, dove_out.Q, dove_out.confidence):
                direction = "▲" if q > 0 else "▼"
                print(f"  {ticker:<6}  {q:>+10.4f}  {c:>10.2f}  {direction}")

        print(f"\n  Return range:  [{dove_out.Q.min():+.4f}, {dove_out.Q.max():+.4f}]")
        print(f"  Conf range:    [{dove_out.confidence.min():.2f}, {dove_out.confidence.max():.2f}]")
        print(f"  All finite:    {np.all(np.isfinite(dove_out.Q))}")

        if dove_out.fallback_used:
            all_passed = False

    print(f"\n{'=' * 60}")
    if all_passed:
        print("All scenarios completed.")
        print("Key checks:")
        print("  Scenario 1: Dove should be LESS negative than Hawk on XLK/XLI/XLY")
        print("  Scenario 2: Dove should be POSITIVE on XLK/XLI/XLY")
        print("  Scenario 3: Both cautious but Dove sees stimulus as growth-positive")
    else:
        print("WARNING: fallback used. Check Ollama is running: ollama serve")
    print("=" * 60)