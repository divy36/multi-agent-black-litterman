"""Fomc signals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Signal data class
# --------------------------------------------------------------------------- #
@dataclass
class FOMCSignals:
    """
    Structured signals extracted from one FOMC minutes document.

    All fields have sensible defaults so the object is always usable
    even if extraction partially fails.
    """
    # Rate decision
    rate_change_bps:  int   = 0          # e.g. +75 for 75bp hike, -25 for cut
    rate_level_low:   float = 0.0        # lower bound of target range (e.g. 3.00)
    rate_level_high:  float = 0.0        # upper bound of target range (e.g. 3.25)
    rate_found:       bool  = False      # True if a rate level was found in text

    # Directional signals
    inflation_signal:    str = "neutral"   # "hawkish", "dovish", "neutral"
    growth_signal:       str = "neutral"   # "hawkish", "dovish", "neutral"
    forward_guidance:    str = "hold"      # "tightening", "easing", "hold", "uncertain"

    # Uncertainty and emergency
    uncertainty_level:   str  = "medium"  # "high", "medium", "low"
    emergency_flag:      bool = False      # True if crisis language detected

    # Composite score: positive = hawkish, negative = dovish, range [-10, +10]
    hawkish_score:       int  = 0

    # Key sentences for prompt context (up to 3)
    key_phrases:         list[str] = field(default_factory=list)

    def summary(self) -> str:
        """
        Return a structured text block for injection into the LLM prompt.
        Designed to be compact, numerical, and unambiguous.
        """
        rate_str = (
            f"{self.rate_level_low:.2f}%-{self.rate_level_high:.2f}%"
            if self.rate_found
            else "not found in text"
        )

        change_str = (
            f"+{self.rate_change_bps}bp hike"  if self.rate_change_bps > 0 else
            f"{self.rate_change_bps}bp cut"    if self.rate_change_bps < 0 else
            "no change (hold)"
        )

        emergency_str = "YES — EMERGENCY/CRISIS LANGUAGE DETECTED" if self.emergency_flag else "No"

        phrases_str = ""
        if self.key_phrases:
            phrases_str = "\nKey policy sentences from the minutes:\n"
            for i, p in enumerate(self.key_phrases, 1):
                phrases_str += f'  {i}. "{p}"\n'

        return f"""## FOMC Signal Extraction (computed from minutes text)
Rate decision this meeting : {change_str}
Current fed funds rate     : {rate_str}
Inflation signal           : {self.inflation_signal.upper()}
Growth/employment signal   : {self.growth_signal.upper()}
Forward guidance           : {self.forward_guidance.upper()}
Uncertainty level          : {self.uncertainty_level.upper()}
Emergency flag             : {emergency_str}
Hawkish composite score    : {self.hawkish_score:+d} (range -10 to +10, positive=hawkish)
{phrases_str}
IMPORTANT: Your expected_return values MUST be consistent with the signals above.
A hawkish score of {self.hawkish_score:+d} means {"significant" if abs(self.hawkish_score) >= 5 else "moderate"} {"tightening" if self.hawkish_score > 0 else "easing" if self.hawkish_score < 0 else "neutral"} pressure."""


# --------------------------------------------------------------------------- #
# Keyword lists
# --------------------------------------------------------------------------- #
_HAWKISH_INFLATION = [
    "inflation elevated", "inflation remains elevated", "inflation well above",
    "inflation risk", "price pressures", "persistent inflation",
    "inflation expectations", "above target", "above 2 percent",
    "highly attentive to inflation", "strongly committed to returning inflation",
    "supply and demand imbalances", "broader price pressures",
    "inflation has not yet come down", "uncomfortably high",
]

_DOVISH_INFLATION = [
    "inflation near", "inflation at 2", "muted inflation", "inflation has declined",
    "inflation is declining", "inflation moving toward", "inflation expectations anchored",
    "below target", "disinflation", "deflationary", "price stability achieved",
    "inflation has eased", "inflation returning to",
]

_HAWKISH_GROWTH = [
    "labor market tight", "job gains robust", "strong labor market",
    "unemployment low", "labor market remains strong", "wage growth elevated",
    "demand strong", "economy resilient", "output above potential",
]

_DOVISH_GROWTH = [
    "economic slowdown", "growth slowing", "recession risk", "output gap",
    "labor market softening", "unemployment rising", "layoffs", "job losses",
    "weak demand", "contraction", "below potential", "economic weakness",
    "financial conditions tightened significantly",
]

_TIGHTENING_GUIDANCE = [
    "ongoing increases", "further increases", "additional increases",
    "continue to raise", "anticipates raising", "more restrictive",
    "sufficiently restrictive", "will be appropriate to raise",
    "pace of future increases", "higher for longer",
]

_EASING_GUIDANCE = [
    "rate cuts", "reduce the target range", "lower the target range",
    "will cut", "begin to reduce", "easing of monetary policy",
    "accommodation", "less restrictive", "pivot", "rate reductions",
]

_HOLD_GUIDANCE = [
    "patient", "data dependent", "hold steady", "maintain the target",
    "no change", "pause", "on hold", "wait and see",
    "incoming data", "assess further data", "carefully assess",
]

_UNCERTAINTY_HIGH = [
    "considerable uncertainty", "high uncertainty", "unusually uncertain",
    "significant uncertainty", "risks are elevated", "balance of risks",
    "unprecedented", "highly uncertain", "uncertainty remains elevated",
]

_UNCERTAINTY_LOW = [
    "clear path", "confident", "strong consensus", "decisive action",
    "strongly committed", "unambiguous", "clear signal",
]

_EMERGENCY = [
    "emergency", "unprecedented", "crisis", "covid",
    "coronavirus", "unlimited", "all available tools", "full range of tools",
    "emergency lending", "credit facilities", "quantitative easing unlimited",
    "main street lending", "paycheck protection",
]

# Rate change phrases — ordered by magnitude so first match wins
_RATE_HIKE_PATTERNS = [
    (r"\b(100|one hundred)\s*basis\s*points?\b",   100),
    (r"\b(75|seventy.five)\s*basis\s*points?\b",    75),
    (r"\b(50|fifty)\s*basis\s*points?\b",           50),
    (r"\b(25|twenty.five)\s*basis\s*points?\b",     25),
]
_RATE_CUT_KEYWORDS  = ["lower the target", "reduce the target", "cut rates",
                       "rate cut", "decreased the target"]
_RATE_HIKE_KEYWORDS = ["raise the target", "increase the target", "raised the target",
                       "hiking", "rate hike", "increased the target"]

# Pattern to find current rate level e.g. "1-1/4 to 1-1/2 percent" or "3 to 3-1/4 percent"
_RATE_NUMBER = r"(?:\d+-\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
_RATE_LEVEL_PATTERN = re.compile(
    rf"(?<![\d.])({_RATE_NUMBER})\s+to\s+({_RATE_NUMBER})\s*percent",
    re.IGNORECASE,
)


def _rate_number(value):
    if "-" in value:
        whole, fraction = value.split("-", 1)
    else:
        whole, fraction = "0", value
    if "/" in fraction:
        numerator, denominator = fraction.split("/", 1)
        return float(whole) + float(numerator) / float(denominator)
    return float(fraction)


# Policy key sentence triggers
_KEY_SENTENCE_TRIGGERS = [
    "decided to raise", "decided to lower", "decided to maintain",
    "target range", "basis points", "federal funds rate",
    "inflation", "maximum employment", "price stability",
    "ongoing increases", "strongly committed", "patient",
    "emergency", "unprecedented", "full range of tools",
]


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
class FOMCSignalExtractor:
    """
    Extracts structured numerical signals from FOMC minutes text.

    All extraction is rule-based (regex + keyword counting) — no LLM involved.
    Designed to be fast (<0.1s per document) and deterministic.
    """

    def extract(self, text: str) -> FOMCSignals:
        """
        Extract all signals from a single FOMC minutes document.

        Parameters
        ----------
        text : str
            Full plain-text content of FOMC meeting minutes.

        Returns
        -------
        FOMCSignals
            Structured signal bundle ready for prompt injection.
        """
        text_lower = text.lower()

        rate_change_bps, rate_found, rate_low, rate_high = self._extract_rate(text, text_lower)
        inflation_signal  = self._score_signal(text_lower, _HAWKISH_INFLATION, _DOVISH_INFLATION)
        growth_signal     = self._score_signal(text_lower, _HAWKISH_GROWTH,    _DOVISH_GROWTH)
        forward_guidance  = self._extract_guidance(text_lower)
        uncertainty_level = self._extract_uncertainty(text_lower)
        emergency_flag    = any(kw in text_lower for kw in _EMERGENCY)
        hawkish_score     = self._compute_hawkish_score(
            text_lower, rate_change_bps, inflation_signal,
            growth_signal, forward_guidance, emergency_flag,
        )
        key_phrases = self._extract_key_phrases(text)

        return FOMCSignals(
            rate_change_bps  = rate_change_bps,
            rate_level_low   = rate_low,
            rate_level_high  = rate_high,
            rate_found       = rate_found,
            inflation_signal = inflation_signal,
            growth_signal    = growth_signal,
            forward_guidance = forward_guidance,
            uncertainty_level = uncertainty_level,
            emergency_flag   = emergency_flag,
            hawkish_score    = hawkish_score,
            key_phrases      = key_phrases,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _extract_rate(
        self, text: str, text_lower: str
    ) -> tuple[int, bool, float, float]:
        """Extract rate change in bps and current rate level."""
        # Determine direction first
        is_cut  = any(kw in text_lower for kw in _RATE_CUT_KEYWORDS)
        is_hike = any(kw in text_lower for kw in _RATE_HIKE_KEYWORDS)

        # Magnitude
        bps = 0
        for pattern, magnitude in _RATE_HIKE_PATTERNS:
            if re.search(pattern, text_lower):
                bps = magnitude
                break

        if is_cut and not is_hike:
            bps = -abs(bps) if bps != 0 else -25   # default cut = 25bp
        elif is_hike and bps == 0:
            bps = 25   # default hike = 25bp if no magnitude found
        elif not is_hike and not is_cut:
            bps = 0

        # Rate level
        rate_low, rate_high, rate_found = 0.0, 0.0, False
        matches = _RATE_LEVEL_PATTERN.findall(text)
        for m in matches:
            try:
                # Try to parse both groups
                lo = _rate_number(m[0])
                hi = _rate_number(m[1])
                if 0.0 <= lo <= 20.0 and 0.0 <= hi <= 20.0 and hi >= lo:
                    rate_low, rate_high, rate_found = lo, hi, True
                    break
            except (ValueError, IndexError, ZeroDivisionError):
                continue

        return bps, rate_found, rate_low, rate_high

    def _score_signal(
        self, text_lower: str, hawkish_kws: list[str], dovish_kws: list[str]
    ) -> str:
        """Count keyword hits and return 'hawkish', 'dovish', or 'neutral'."""
        hawk_count = sum(1 for kw in hawkish_kws if kw in text_lower)
        dove_count = sum(1 for kw in dovish_kws  if kw in text_lower)
        if hawk_count > dove_count:
            return "hawkish"
        elif dove_count > hawk_count:
            return "dovish"
        return "neutral"

    def _extract_guidance(self, text_lower: str) -> str:
        """Classify forward guidance from keyword hits."""
        tighten = sum(1 for kw in _TIGHTENING_GUIDANCE if kw in text_lower)
        ease    = sum(1 for kw in _EASING_GUIDANCE     if kw in text_lower)
        hold    = sum(1 for kw in _HOLD_GUIDANCE        if kw in text_lower)

        if tighten > ease and tighten > hold:
            return "tightening"
        elif ease > tighten and ease > hold:
            return "easing"
        elif hold > tighten and hold > ease:
            return "hold"
        elif tighten == 0 and ease == 0:
            return "uncertain"
        return "hold"

    def _extract_uncertainty(self, text_lower: str) -> str:
        """Classify uncertainty level from keyword hits."""
        high_count = sum(1 for kw in _UNCERTAINTY_HIGH if kw in text_lower)
        low_count  = sum(1 for kw in _UNCERTAINTY_LOW  if kw in text_lower)
        if high_count >= 2 or (high_count > 0 and low_count == 0):
            return "high"
        elif low_count >= 2 or (low_count > 0 and high_count == 0):
            return "low"
        return "medium"

    def _compute_hawkish_score(
        self,
        text_lower:       str,
        rate_change_bps:  int,
        inflation_signal: str,
        growth_signal:    str,
        forward_guidance: str,
        emergency_flag:   bool,
    ) -> int:
        """
        Compute a composite hawkish score in [-10, +10].

        Positive = more hawkish (expect tightening, bearish equities overall).
        Negative = more dovish (expect easing, bullish equities overall).
        """
        score = 0

        # Rate decision (dominant signal)
        if rate_change_bps >= 75:
            score += 4
        elif rate_change_bps >= 50:
            score += 3
        elif rate_change_bps >= 25:
            score += 2
        elif rate_change_bps <= -50:
            score -= 4
        elif rate_change_bps <= -25:
            score -= 3

        # Inflation signal
        if inflation_signal == "hawkish":
            score += 2
        elif inflation_signal == "dovish":
            score -= 2

        # Growth signal (hawkish growth = less need for easing)
        if growth_signal == "hawkish":
            score += 1
        elif growth_signal == "dovish":
            score -= 2   # dovish growth = more stimulus needed

        # Forward guidance
        if forward_guidance == "tightening":
            score += 2
        elif forward_guidance == "easing":
            score -= 2
        elif forward_guidance == "hold":
            score += 0

        # Emergency flag is strongly dovish (emergency cuts)
        if emergency_flag:
            score -= 3

        return max(-10, min(10, score))

    def _extract_key_phrases(self, text: str, max_phrases: int = 3) -> list[str]:
        """
        Extract the most policy-relevant sentences from the minutes.

        Splits into sentences, scores each by trigger keyword count,
        returns the top `max_phrases` by score (truncated to 180 chars).
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        scored = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1 for trigger in _KEY_SENTENCE_TRIGGERS
                        if trigger in sent_lower)
            if score >= 2:   # only sentences with 2+ triggers
                scored.append((score, sent.strip()))

        # Sort by score descending, deduplicate, take top N
        scored.sort(key=lambda x: -x[0])
        seen, phrases = set(), []
        for _, sent in scored:
            trunc = sent[:180].strip()
            if trunc not in seen:
                seen.add(trunc)
                phrases.append(trunc)
            if len(phrases) >= max_phrases:
                break

        return phrases


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    extractor = FOMCSignalExtractor()

    test_cases = [
        {
            "label": "Sep 2022 — 75bp hike, high inflation",
            "text": """
The Committee decided to raise the target range for the federal funds rate to
3 to 3-1/4 percent. The Committee anticipates that ongoing increases in the
target range will be appropriate in order to attain a stance of monetary policy
that is sufficiently restrictive to return inflation to 2 percent over time.
Inflation remains elevated, reflecting supply and demand imbalances related to
the pandemic, higher food and energy prices, and broader price pressures.
The Committee is highly attentive to inflation risks. In determining the pace
of future increases in the target range, the Committee will take into account
the cumulative tightening of monetary policy. The Committee raised the target
range by 75 basis points.
""",
            "expected_score_sign": "+",
            "expected_bps": 75,
        },
        {
            "label": "Jan 2019 — hold, patient guidance",
            "text": """
In light of global economic and financial developments and muted inflation
pressures, the Committee will be patient as it determines what future
adjustments to the target range for the federal funds rate may be appropriate.
The Committee decided to maintain the target range for the federal funds rate
at 2-1/4 to 2-1/2 percent. Job gains have been strong and the unemployment
rate has remained low. The Committee will assess incoming data carefully.
""",
            "expected_score_sign": "0",
            "expected_bps": 0,
        },
        {
            "label": "Mar 2020 — emergency 100bp cut, COVID",
            "text": """
The coronavirus outbreak has harmed communities and disrupted economic activity.
The Committee decided to lower the target range for the federal funds rate to
0 to 1/4 percent. The Federal Reserve is prepared to use its full range of tools
to support the flow of credit. This action was taken in light of the
unprecedented circumstances created by the coronavirus pandemic. The Committee
will maintain this accommodative stance. Emergency lending facilities have been
established. The Federal Reserve stands ready to use all available tools.
""",
            "expected_score_sign": "-",
            "expected_bps": -100,
        },
    ]

    print("=" * 60)
    print("FOMCSignalExtractor self-test")
    print("=" * 60)

    all_passed = True
    for tc in test_cases:
        print(f"\n{'─'*60}")
        print(f"  {tc['label']}")
        print(f"{'─'*60}")
        signals = extractor.extract(tc["text"])
        print(signals.summary())

        # Checks
        bps_ok    = signals.rate_change_bps == tc["expected_bps"]
        score_ok  = (
            (tc["expected_score_sign"] == "+" and signals.hawkish_score > 0) or
            (tc["expected_score_sign"] == "-" and signals.hawkish_score < 0) or
            (tc["expected_score_sign"] == "0" and -2 <= signals.hawkish_score <= 2)
        )
        print(f"  PASS bps={signals.rate_change_bps}" if bps_ok
              else f"  FAIL bps: expected {tc['expected_bps']}, got {signals.rate_change_bps}")
        print(f"  PASS hawkish_score={signals.hawkish_score:+d}" if score_ok
              else f"  FAIL score sign: expected '{tc['expected_score_sign']}', got {signals.hawkish_score:+d}")

        if not (bps_ok and score_ok):
            all_passed = False

    print(f"\n{'='*60}")
    print("All checks passed." if all_passed else "SOME CHECKS FAILED — review above.")
    print("=" * 60)
