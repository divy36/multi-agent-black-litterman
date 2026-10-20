"""Agent dove v2."""

from __future__ import annotations

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

OLLAMA_MODEL   = "qwen2:7b"
OLLAMA_URL     = "http://localhost:11434/api/generate"
TEMPERATURE    = 0
MAX_RETRIES    = 3
TIMEOUT_SECS   = 120

FALLBACK_RETURN     = 0.0
FALLBACK_CONFIDENCE = 0.3

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data class — identical to v1
# --------------------------------------------------------------------------- #
@dataclass
class DoveOutput:
    Q:            np.ndarray
    confidence:   np.ndarray
    date:         str
    fomc_date:    str
    gdp_growth:   float
    unemployment: float
    pnl_context:  str
    fallback_used: bool
    raw_response: str
    attempts:     int


# --------------------------------------------------------------------------- #
# Prompt builder — STRIPPED, direct JSON only
# --------------------------------------------------------------------------- #
def _build_prompt(
    fomc_text:    str,
    gdp_growth:   float,
    unemployment: float,
    date:         str,
    pnl_context:  str,
) -> str:
    """
    Stripped Dove prompt — no chain-of-thought, straight to JSON.

    Structure:
    1. One-line persona
    2. Macro data (GDP, unemployment)
    3. FOMC signal block
    4. Regime-return lookup table (Dove version — bullish on growth)
    5. JSON output block
    """
    signals_block = ""
    if _EXTRACTOR is not None and fomc_text:
        try:
            signals = _EXTRACTOR.extract(fomc_text)
            s = signals
            signals_block = (
                f"Rate change: {s.rate_change_bps:+d}bp | "
                f"Hawkish score: {s.hawkish_score:+d} | "
                f"Guidance: {s.forward_guidance.upper()} | "
                f"Emergency: {s.emergency_flag}"
            )
        except Exception:
            signals_block = "FOMC signals: unavailable"
    else:
        signals_block = "FOMC signals: unavailable"

    pnl_line = f"\nLast month context: {pnl_context}" if pnl_context else ""

    prompt = f"""You are DOVE — a growth-focused macro analyst who prioritises employment and expansion.
Date: {date} | GDP growth: {gdp_growth:.1f}% | Unemployment: {unemployment:.1f}%{pnl_line}

{signals_block}

DOVE RETURN RULES (use hawkish_score above to pick the regime row):
Score +5 to +10 (TIGHTENING):  XLK=-0.005, XLF=+0.010, XLE=+0.005, XLV=+0.003, XLI=-0.003, XLY=+0.002, XLP=+0.005, XLRE=-0.008, XLU=-0.005, XLB=+0.005  conf=0.52
Score +2 to +4  (MILD HAWK):   XLK=+0.008, XLF=+0.006, XLE=+0.006, XLV=+0.004, XLI=+0.008, XLY=+0.010, XLP=+0.002, XLRE=+0.002, XLU=0.000, XLB=+0.006   conf=0.57
Score -1 to +1  (NEUTRAL):     XLK=+0.012, XLF=+0.005, XLE=+0.006, XLV=+0.004, XLI=+0.010, XLY=+0.014, XLP=+0.002, XLRE=+0.008, XLU=+0.005, XLB=+0.007  conf=0.62
Score -2 to -4  (MILD DOVE):   XLK=+0.018, XLF=+0.004, XLE=+0.008, XLV=+0.005, XLI=+0.015, XLY=+0.020, XLP=0.000, XLRE=+0.012, XLU=+0.008, XLB=+0.010   conf=0.67
Score -5 to -10 (EASING):      XLK=+0.025, XLF=+0.003, XLE=+0.010, XLV=+0.006, XLI=+0.020, XLY=+0.028, XLP=-0.002, XLRE=+0.018, XLU=+0.012, XLB=+0.012  conf=0.72
EMERGENCY flag=True:            XLK=+0.010, XLF=+0.005, XLV=+0.008, others mixed, conf=0.35

Adjust individual sector returns within ±0.010 of the table values based on your dovish judgment.
Higher GDP growth → more positive values. Higher unemployment → more cautious values.
DOVE is ALWAYS more optimistic than Hawk — your returns should be higher on growth sectors.

Output ONLY this JSON array. No text before or after. No markdown. No explanation:
[
  {{"sector": "XLK",  "expected_return": 0.000, "confidence": 0.00}},
  {{"sector": "XLF",  "expected_return": 0.000, "confidence": 0.00}},
  {{"sector": "XLE",  "expected_return": 0.000, "confidence": 0.00}},
  {{"sector": "XLV",  "expected_return": 0.000, "confidence": 0.00}},
  {{"sector": "XLI",  "expected_return": 0.000, "confidence": 0.00}},
  {{"sector": "XLY",  "expected_return": 0.000, "confidence": 0.00}},
  {{"sector": "XLP",  "expected_return": 0.000, "confidence": 0.00}},
  {{"sector": "XLRE", "expected_return": 0.000, "confidence": 0.00}},
  {{"sector": "XLU",  "expected_return": 0.000, "confidence": 0.00}},
  {{"sector": "XLB",  "expected_return": 0.000, "confidence": 0.00}}
]
Replace the 0.000 values with your actual numbers based on the regime table above.
Output the JSON now:"""

    return prompt


# --------------------------------------------------------------------------- #
# JSON parser — identical to v1
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
    json_str = re.sub(r":\s*\+", ": ", json_str)
    json_str = json_str.replace("'", '"')
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


def _validate_views(data: list[dict]) -> bool:
    if len(data) != N:
        logger.warning(f"Dove v2: expected {N} sectors, got {len(data)}")
        return False
    for i, (entry, ticker) in enumerate(zip(data, TICKERS)):
        if not isinstance(entry, dict):
            return False
        if "sector" not in entry or "expected_return" not in entry or "confidence" not in entry:
            logger.warning(f"Dove v2: entry {i} missing fields: {entry}")
            return False
        if entry["sector"] != ticker:
            logger.warning(f"Dove v2: sector mismatch at {i}: expected {ticker}, got {entry['sector']}")
            return False
        try:
            r = float(entry["expected_return"])
            c = float(entry["confidence"])
        except (TypeError, ValueError):
            return False
        if not (-0.20 <= r <= 0.20):
            logger.warning(f"Dove v2: return {r} out of range for {ticker}")
            return False
        if not (0.0 <= c <= 1.0):
            logger.warning(f"Dove v2: confidence {c} out of range for {ticker}")
            return False
    return True


# --------------------------------------------------------------------------- #
# Ollama caller
# --------------------------------------------------------------------------- #
def _call_ollama(prompt: str, attempt: int, model: str = OLLAMA_MODEL) -> str:
    # BUG FIX: see agent_hawk_v2.py's identical fix -- model was hardcoded to
    # OLLAMA_MODEL here, ignoring DoveAgent's self.model entirely. Now
    # required from the caller and logged explicitly for independent
    # verification.
    logger.info(f"Dove v2 [_call_ollama attempt {attempt}]: model={model}")
    import urllib.request
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 1000,
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
def _make_fallback(date, fomc_date, gdp_growth, unemployment, pnl_context, raw, attempts):
    logger.error(f"Dove v2: all {attempts} attempts failed for {date}. Using fallback.")
    return DoveOutput(
        Q=np.zeros(N), confidence=np.full(N, FALLBACK_CONFIDENCE),
        date=date, fomc_date=fomc_date, gdp_growth=gdp_growth,
        unemployment=unemployment, pnl_context=pnl_context,
        fallback_used=True, raw_response=raw, attempts=attempts,
    )


# --------------------------------------------------------------------------- #
# Main agent class
# --------------------------------------------------------------------------- #
class DoveAgent:
    """
    Agent 2 — Dove v2.
    Stripped prompt: signal block → regime table → JSON only.
    No chain-of-thought. Fallback rate target: < 10%.
    """

    def __init__(self, model=OLLAMA_MODEL, temperature=TEMPERATURE, max_retries=MAX_RETRIES):
        self.model       = model
        self.temperature = temperature
        self.max_retries = max_retries

    def run(self, fomc_text, gdp_growth, unemployment, date, fomc_date, pnl_context=""):
        prompt = _build_prompt(fomc_text, gdp_growth, unemployment, date, pnl_context)
        raw = ""

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Dove v2 [{date}]: attempt {attempt}/{self.max_retries}  model={self.model}")
            try:
                raw = _call_ollama(prompt, attempt, self.model)
            except RuntimeError as exc:
                logger.warning(str(exc))
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                continue

            parsed = _parse_dove_response(raw)
            if parsed is None:
                logger.warning(f"Dove v2 [{date}]: JSON parse failed. Raw: {raw[:150]}")
                continue
            if not _validate_views(parsed):
                logger.warning(f"Dove v2 [{date}]: validation failed.")
                continue

            Q          = np.array([float(e["expected_return"]) for e in parsed])
            confidence = np.array([float(e["confidence"])       for e in parsed])
            logger.info(f"Dove v2 [{date}]: success on attempt {attempt}.")
            return DoveOutput(
                Q=Q, confidence=confidence, date=date, fomc_date=fomc_date,
                gdp_growth=gdp_growth, unemployment=unemployment,
                pnl_context=pnl_context, fallback_used=False,
                raw_response=raw, attempts=attempt,
            )

        return _make_fallback(date, fomc_date, gdp_growth, unemployment,
                              pnl_context, raw, self.max_retries)

    def describe(self):
        return {"agent": "DOVE_V2", "model": self.model,
                "temperature": self.temperature, "prompt": "stripped_no_cot"}


# --------------------------------------------------------------------------- #
# Quick test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    agent = DoveAgent()
    print("=" * 55)
    print("Dove v2 — quick test (3 scenarios)")
    print("=" * 55)

    scenarios = [
        {"label": "Sep 2022 — 75bp hike, GDP=2.6%, U=3.5%",
         "date": "2022-09-30", "fomc_date": "2022-09-21",
         "gdp_growth": 2.6, "unemployment": 3.5,
         "fomc_text": "The Committee raised the target range by 75 basis points to 3 to 3-1/4 percent. Inflation remains elevated. Strongly committed to returning inflation to 2 percent."},
        {"label": "Feb 2019 — hold, GDP=3.1%, U=4.0%",
         "date": "2019-02-28", "fomc_date": "2019-01-30",
         "gdp_growth": 3.1, "unemployment": 4.0,
         "fomc_text": "The Committee will be patient. Maintained target range at 2-1/4 to 2-1/2 percent. Strong labor market. Economic activity rising at solid rate."},
        {"label": "Mar 2020 — emergency cut, GDP=-5.0%, U=4.4%",
         "date": "2020-03-31", "fomc_date": "2020-03-15",
         "gdp_growth": -5.0, "unemployment": 4.4,
         "fomc_text": "The Committee decided to lower the target range to 0 to 1/4 percent. Emergency lending facilities. Coronavirus unprecedented disruption to economy."},
    ]

    all_passed = True
    for sc in scenarios:
        print(f"\n  {sc['label']}")
        out = agent.run(sc["fomc_text"], sc["gdp_growth"], sc["unemployment"],
                        sc["date"], sc["fomc_date"])
        status = "FALLBACK" if out.fallback_used else f"OK (attempt {out.attempts})"
        print(f"  Status: {status}")
        if not out.fallback_used:
            print(f"  XLK={out.Q[0]:+.4f} XLI={out.Q[4]:+.4f} XLY={out.Q[5]:+.4f} XLRE={out.Q[7]:+.4f}")
            print(f"  Conf range: [{out.confidence.min():.2f}, {out.confidence.max():.2f}]")
        else:
            all_passed = False

    print(f"\n{'='*55}")
    print("All scenarios completed." if all_passed else
          "WARNING: some fallbacks — check Ollama is running.")
