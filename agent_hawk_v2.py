"""Agent hawk v2."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
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
class HawkOutput:
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
# Prompt builder — STRIPPED, direct JSON only
# --------------------------------------------------------------------------- #
def _build_prompt(
    fomc_text:   str,
    cpi_yoy:     float,
    pce_yoy:     float,
    date:        str,
    pnl_context: str,
) -> str:
    """
    Stripped Hawk prompt — no chain-of-thought, straight to JSON.

    Structure:
    1. One-line persona
    2. Macro data (CPI, PCE)
    3. FOMC signal block (pre-computed by FOMCSignalExtractor)
    4. Regime-return lookup table
    5. JSON output block

    The model only needs to:
    a) Read the hawkish_score and regime signals
    b) Map them to returns using the lookup table
    c) Output the JSON array

    No reasoning steps, no analysis paragraphs — just numbers in, JSON out.
    """
    # Pre-extract signals
    signals_block = ""
    if _EXTRACTOR is not None and fomc_text:
        try:
            signals = _EXTRACTOR.extract(fomc_text)
            signals_block = signals.summary()
        except Exception:
            signals_block = "FOMC signals: unavailable"
    else:
        signals_block = "FOMC signals: unavailable"

    pnl_line = f"\nLast month context: {pnl_context}" if pnl_context else ""

    prompt = f"""You are HAWK — a risk-averse macro analyst who fears inflation above all else.
Date: {date} | CPI YoY: {cpi_yoy:.1f}% | Core PCE YoY: {pce_yoy:.1f}%{pnl_line}

{signals_block}

HAWK RETURN RULES (use hawkish_score above to pick the regime row):
Score +5 to +10 (TIGHTENING):  XLK=-0.025, XLF=+0.012, XLE=-0.008, XLV=-0.005, XLI=-0.015, XLY=-0.022, XLP=+0.005, XLRE=-0.035, XLU=-0.028, XLB=-0.010  conf=0.70-0.85
Score +2 to +4  (MILD HAWK):   XLK=-0.012, XLF=+0.006, XLE=-0.004, XLV=-0.002, XLI=-0.008, XLY=-0.010, XLP=+0.003, XLRE=-0.018, XLU=-0.014, XLB=-0.005  conf=0.55-0.70
Score -1 to +1  (NEUTRAL):     XLK=-0.005, XLF=+0.002, XLE=0.000, XLV=0.000, XLI=-0.003, XLY=-0.004, XLP=+0.002, XLRE=-0.008, XLU=-0.006, XLB=-0.002   conf=0.40-0.55
Score -2 to -4  (MILD DOVE):   XLK=+0.005, XLF=0.000, XLE=+0.003, XLV=+0.002, XLI=+0.005, XLY=+0.006, XLP=0.000, XLRE=+0.005, XLU=+0.003, XLB=+0.004   conf=0.40-0.55
Score -5 to -10 (EASING):      XLK=+0.018, XLF=0.000, XLE=+0.008, XLV=+0.005, XLI=+0.012, XLY=+0.015, XLP=-0.002, XLRE=+0.015, XLU=+0.012, XLB=+0.008  conf=0.55-0.70
EMERGENCY flag=True:            all negative, magnitude -0.01 to -0.04, conf=0.25-0.45

Adjust individual sector returns within ±0.010 of the table values based on your hawkish judgment.
Higher CPI/PCE → more negative values. Lower CPI/PCE → less negative values.

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
def _parse_hawk_response(raw: str) -> Optional[list[dict]]:
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
        logger.warning(f"Hawk v2: expected {N} sectors, got {len(data)}")
        return False
    for i, (entry, ticker) in enumerate(zip(data, TICKERS)):
        if not isinstance(entry, dict):
            return False
        if "sector" not in entry or "expected_return" not in entry or "confidence" not in entry:
            logger.warning(f"Hawk v2: entry {i} missing fields: {entry}")
            return False
        if entry["sector"] != ticker:
            logger.warning(f"Hawk v2: sector mismatch at {i}: expected {ticker}, got {entry['sector']}")
            return False
        try:
            r = float(entry["expected_return"])
            c = float(entry["confidence"])
        except (TypeError, ValueError):
            return False
        if not (-0.20 <= r <= 0.20):
            logger.warning(f"Hawk v2: return {r} out of range for {ticker}")
            return False
        if not (0.0 <= c <= 1.0):
            logger.warning(f"Hawk v2: confidence {c} out of range for {ticker}")
            return False
    return True


# --------------------------------------------------------------------------- #
# Ollama caller
# --------------------------------------------------------------------------- #
def _call_ollama(prompt: str, attempt: int, model: str = OLLAMA_MODEL) -> str:
    # BUG FIX: model used to be hardcoded to the module-level OLLAMA_MODEL
    # constant here, ignoring any model= passed to HawkAgent's constructor --
    # self.model was stored but never actually reached this function. Now
    # the caller's model is required and logged explicitly (not just the
    # "Hawk v2" label, which turned out to say nothing about which model
    # actually ran) so this is independently verifiable from the log.
    logger.info(f"Hawk v2 [_call_ollama attempt {attempt}]: model={model}")
    import urllib.request
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": 600,   # much smaller — no reasoning needed
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
def _make_fallback(date, fomc_date, cpi_yoy, pce_yoy, pnl_context, raw, attempts):
    logger.error(f"Hawk v2: all {attempts} attempts failed for {date}. Using fallback.")
    return HawkOutput(
        Q=np.zeros(N), confidence=np.full(N, FALLBACK_CONFIDENCE),
        date=date, fomc_date=fomc_date, cpi_yoy=cpi_yoy, pce_yoy=pce_yoy,
        pnl_context=pnl_context, fallback_used=True, raw_response=raw, attempts=attempts,
    )


# --------------------------------------------------------------------------- #
# Main agent class
# --------------------------------------------------------------------------- #
class HawkAgent:
    """
    Agent 1 — Hawk v2.
    Stripped prompt: signal block → regime table → JSON only.
    No chain-of-thought. Fallback rate target: < 10%.
    """

    def __init__(self, model=OLLAMA_MODEL, temperature=TEMPERATURE, max_retries=MAX_RETRIES):
        self.model       = model
        self.temperature = temperature
        self.max_retries = max_retries

    def run(self, fomc_text, cpi_yoy, pce_yoy, date, fomc_date, pnl_context=""):
        prompt = _build_prompt(fomc_text, cpi_yoy, pce_yoy, date, pnl_context)
        raw = ""

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Hawk v2 [{date}]: attempt {attempt}/{self.max_retries}  model={self.model}")
            try:
                raw = _call_ollama(prompt, attempt, self.model)
            except RuntimeError as exc:
                logger.warning(str(exc))
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                continue

            parsed = _parse_hawk_response(raw)
            if parsed is None:
                logger.warning(f"Hawk v2 [{date}]: JSON parse failed. Raw: {raw[:150]}")
                continue
            if not _validate_views(parsed):
                logger.warning(f"Hawk v2 [{date}]: validation failed.")
                continue

            Q          = np.array([float(e["expected_return"]) for e in parsed])
            confidence = np.array([float(e["confidence"])       for e in parsed])
            logger.info(f"Hawk v2 [{date}]: success on attempt {attempt}.")
            return HawkOutput(
                Q=Q, confidence=confidence, date=date, fomc_date=fomc_date,
                cpi_yoy=cpi_yoy, pce_yoy=pce_yoy, pnl_context=pnl_context,
                fallback_used=False, raw_response=raw, attempts=attempt,
            )

        return _make_fallback(date, fomc_date, cpi_yoy, pce_yoy, pnl_context, raw, self.max_retries)

    def describe(self):
        return {"agent": "HAWK_V2", "model": self.model,
                "temperature": self.temperature, "prompt": "stripped_no_cot"}


# --------------------------------------------------------------------------- #
# Quick test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

    agent = HawkAgent()
    print("=" * 55)
    print("Hawk v2 — quick test (3 scenarios)")
    print("=" * 55)

    scenarios = [
        {"label": "Sep 2022 — 75bp hike, CPI=8.3%",
         "date": "2022-09-30", "fomc_date": "2022-09-21",
         "cpi_yoy": 8.3, "pce_yoy": 4.9,
         "fomc_text": "The Committee raised the target range by 75 basis points to 3 to 3-1/4 percent. Inflation remains elevated. The Committee is strongly committed to returning inflation to 2 percent."},
        {"label": "Feb 2019 — hold, patient, CPI=1.6%",
         "date": "2019-02-28", "fomc_date": "2019-01-30",
         "cpi_yoy": 1.6, "pce_yoy": 1.7,
         "fomc_text": "The Committee will be patient. The Committee decided to maintain the target range at 2-1/4 to 2-1/2 percent. Muted inflation pressures. Strong labor market."},
        {"label": "Mar 2020 — emergency cut, COVID",
         "date": "2020-03-31", "fomc_date": "2020-03-15",
         "cpi_yoy": 2.3, "pce_yoy": 1.8,
         "fomc_text": "The Committee decided to lower the target range to 0 to 1/4 percent. Emergency lending facilities. Full range of tools. Coronavirus unprecedented disruption."},
    ]

    all_passed = True
    for sc in scenarios:
        print(f"\n  {sc['label']}")
        out = agent.run(sc["fomc_text"], sc["cpi_yoy"], sc["pce_yoy"],
                        sc["date"], sc["fomc_date"])
        status = "FALLBACK" if out.fallback_used else f"OK (attempt {out.attempts})"
        print(f"  Status: {status}")
        if not out.fallback_used:
            print(f"  XLK={out.Q[0]:+.4f} XLF={out.Q[1]:+.4f} XLRE={out.Q[7]:+.4f} XLU={out.Q[8]:+.4f}")
            print(f"  Conf range: [{out.confidence.min():.2f}, {out.confidence.max():.2f}]")
        else:
            all_passed = False

    print(f"\n{'='*55}")
    print("All scenarios completed." if all_passed else
          "WARNING: some fallbacks — check Ollama is running.")
