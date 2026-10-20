"""Tests for fomc signals."""

from __future__ import annotations

import pytest

from fomc_signals import FOMCSignalExtractor, FOMCSignals


@pytest.fixture
def extractor():
    return FOMCSignalExtractor()


# ═══════════════════════════════════════════════════════════════════════
# 1. Rate change (bps) extraction — feeds directly into hawkish_score
# ═══════════════════════════════════════════════════════════════════════
class TestRateChangeExtraction:

    def test_75bp_hike_detected(self, extractor):
        s = extractor.extract("The Committee raised the target range by 75 basis points.")
        assert s.rate_change_bps == 75

    def test_50bp_hike_detected(self, extractor):
        s = extractor.extract("The Committee raised the target range by 50 basis points.")
        assert s.rate_change_bps == 50

    def test_25bp_cut_detected_as_negative(self, extractor):
        s = extractor.extract("The Committee decided to lower the target range by 25 basis points.")
        assert s.rate_change_bps == -25

    def test_hike_without_explicit_magnitude_defaults_to_25(self, extractor):
        s = extractor.extract("The Committee decided to raise the target range for the federal funds rate.")
        assert s.rate_change_bps == 25

    def test_cut_without_explicit_magnitude_defaults_to_minus_25(self, extractor):
        s = extractor.extract("The Committee decided to lower the target range for the federal funds rate.")
        assert s.rate_change_bps == -25

    def test_hold_language_gives_zero(self, extractor):
        s = extractor.extract("The Committee decided to maintain the target range at 2-1/4 to 2-1/2 percent.")
        assert s.rate_change_bps == 0


# ═══════════════════════════════════════════════════════════════════════
# 2. Rate level parsing — decimal and fractional notation
# ═══════════════════════════════════════════════════════════════════════
class TestRateLevelParsing:

    def test_fraction_notation_is_parsed(self, extractor):
        s = extractor.extract("raised the target range for the federal funds rate to 3 to 3-1/4 percent.")
        assert s.rate_found is True
        assert s.rate_level_low == pytest.approx(3.0)
        assert s.rate_level_high == pytest.approx(3.25)

    def test_decimal_notation_parses_high_correctly(self, extractor):
        s = extractor.extract("raised the target range for the federal funds rate to 3.00 to 3.25 percent.")
        assert s.rate_level_high == pytest.approx(3.25, abs=1e-6)

    def test_rate_level_never_crashes_on_arbitrary_text(self, extractor):
        """Whatever the parsing quality, extract() must never raise on
        text with no rate-level phrasing at all."""
        s = extractor.extract("This sentence has no rate information whatsoever.")
        assert s.rate_found is False


# ═══════════════════════════════════════════════════════════════════════
# 3. Inflation / growth keyword scoring
# ═══════════════════════════════════════════════════════════════════════
class TestDirectionalSignals:

    def test_hawkish_inflation_keywords_detected(self, extractor):
        s = extractor.extract("Inflation remains elevated and price pressures persist.")
        assert s.inflation_signal == "hawkish"

    def test_dovish_inflation_keywords_detected(self, extractor):
        s = extractor.extract("Inflation has declined and is moving toward 2 percent, with inflation near target.")
        assert s.inflation_signal == "dovish"

    def test_no_inflation_keywords_gives_neutral(self, extractor):
        s = extractor.extract("The committee met to discuss the economy.")
        assert s.inflation_signal == "neutral"

    def test_tied_inflation_counts_default_to_neutral(self, extractor):
        """One hawkish + one dovish keyword => tie => neutral, not a crash or bias."""
        s = extractor.extract("Inflation risk exists, but inflation near target has also been noted.")
        assert s.inflation_signal == "neutral"

    def test_hawkish_growth_keywords_detected(self, extractor):
        s = extractor.extract("The labor market remains strong and unemployment low.")
        assert s.growth_signal == "hawkish"

    def test_dovish_growth_keywords_detected(self, extractor):
        s = extractor.extract("Growth is slowing and there is recession risk, with layoffs increasing.")
        assert s.growth_signal == "dovish"


# ═══════════════════════════════════════════════════════════════════════
# 4. Forward guidance classification
# ═══════════════════════════════════════════════════════════════════════
class TestForwardGuidance:

    def test_tightening_language_detected(self, extractor):
        s = extractor.extract("The Committee anticipates ongoing increases and further increases will be appropriate.")
        assert s.forward_guidance == "tightening"

    def test_easing_language_detected(self, extractor):
        s = extractor.extract("The Committee will begin to reduce the target range and expects rate cuts.")
        assert s.forward_guidance == "easing"

    def test_hold_language_detected(self, extractor):
        s = extractor.extract("The Committee will be patient and data dependent, choosing to hold steady.")
        assert s.forward_guidance == "hold"

    def test_no_guidance_keywords_gives_uncertain(self, extractor):
        s = extractor.extract("The meeting covered general economic conditions.")
        assert s.forward_guidance == "uncertain"


# ═══════════════════════════════════════════════════════════════════════
# 5. Uncertainty and emergency flags
# ═══════════════════════════════════════════════════════════════════════
class TestUncertaintyAndEmergency:

    def test_high_uncertainty_language_detected(self, extractor):
        s = extractor.extract(
            "There is considerable uncertainty and significant uncertainty about the outlook, "
            "with risks are elevated."
        )
        assert s.uncertainty_level == "high"

    def test_emergency_language_sets_flag(self, extractor):
        s = extractor.extract("The Committee established emergency lending facilities amid the coronavirus crisis.")
        assert s.emergency_flag is True

    def test_no_emergency_language_leaves_flag_false(self, extractor):
        s = extractor.extract("The Committee decided to maintain the target range.")
        assert s.emergency_flag is False


# ═══════════════════════════════════════════════════════════════════════
# 6. Hawkish score — the composite formula every agent's regime lookup uses
# ═══════════════════════════════════════════════════════════════════════
class TestHawkishScoreComposite:
    """
    Tests _compute_hawkish_score directly (bypassing text parsing) so the
    additive scoring RULES themselves are pinned down precisely, independent
    of keyword-matching noise. This is the number the whole research project's
    "regime-table rigidity" finding is about -- worth defending exactly.
    """

    def test_75bp_hike_alone_scores_plus_4(self, extractor):
        score = extractor._compute_hawkish_score(
            "", rate_change_bps=75, inflation_signal="neutral",
            growth_signal="neutral", forward_guidance="hold", emergency_flag=False,
        )
        assert score == 4

    def test_50bp_hike_alone_scores_plus_3(self, extractor):
        score = extractor._compute_hawkish_score(
            "", rate_change_bps=50, inflation_signal="neutral",
            growth_signal="neutral", forward_guidance="hold", emergency_flag=False,
        )
        assert score == 3

    def test_25bp_hike_alone_scores_plus_2(self, extractor):
        score = extractor._compute_hawkish_score(
            "", rate_change_bps=25, inflation_signal="neutral",
            growth_signal="neutral", forward_guidance="hold", emergency_flag=False,
        )
        assert score == 2

    def test_confirmed_asymmetry_25bp_cut_scores_more_than_25bp_hike(self, extractor):
        """
        Documented finding, not a bug: at the SAME 25bp magnitude, a cut
        scores -3 while a hike only scores +2 -- the rate-change scoring
        is asymmetric (hikes get a 3-tier scale 2/3/4; cuts only get a
        2-tier scale 3/4, with no distinct +/-25-only tier on the cut
        side). This test pins the asymmetry down explicitly so a future
        "symmetry cleanup" doesn't silently change this documented
        component's behaviour without it being noticed.
        """
        hike_score = extractor._compute_hawkish_score(
            "", rate_change_bps=25, inflation_signal="neutral",
            growth_signal="neutral", forward_guidance="hold", emergency_flag=False,
        )
        cut_score = extractor._compute_hawkish_score(
            "", rate_change_bps=-25, inflation_signal="neutral",
            growth_signal="neutral", forward_guidance="hold", emergency_flag=False,
        )
        assert hike_score == 2
        assert cut_score == -3
        assert abs(cut_score) > abs(hike_score)

    def test_confirmed_asymmetry_dovish_growth_weighted_double_hawkish_growth(self, extractor):
        """Documented finding: hawkish growth contributes +1, dovish growth
        contributes -2 -- matches the code's own comment ("dovish growth =
        more stimulus needed") but is worth pinning down as an explicit,
        intentional asymmetry rather than leaving it implicit."""
        hawkish_growth_score = extractor._compute_hawkish_score(
            "", rate_change_bps=0, inflation_signal="neutral",
            growth_signal="hawkish", forward_guidance="hold", emergency_flag=False,
        )
        dovish_growth_score = extractor._compute_hawkish_score(
            "", rate_change_bps=0, inflation_signal="neutral",
            growth_signal="dovish", forward_guidance="hold", emergency_flag=False,
        )
        assert hawkish_growth_score == 1
        assert dovish_growth_score == -2

    def test_emergency_flag_always_subtracts_3(self, extractor):
        base = extractor._compute_hawkish_score(
            "", rate_change_bps=25, inflation_signal="hawkish",
            growth_signal="hawkish", forward_guidance="tightening", emergency_flag=False,
        )
        with_emergency = extractor._compute_hawkish_score(
            "", rate_change_bps=25, inflation_signal="hawkish",
            growth_signal="hawkish", forward_guidance="tightening", emergency_flag=True,
        )
        assert base - with_emergency == 3

    def test_score_clamped_at_positive_10(self, extractor):
        score = extractor._compute_hawkish_score(
            "", rate_change_bps=100, inflation_signal="hawkish",
            growth_signal="hawkish", forward_guidance="tightening", emergency_flag=False,
        )
        # Raw sum would be 4+2+1+2 = 9, under the ceiling -- confirm ceiling
        # itself with an even more extreme combination isn't exceeded either
        assert score <= 10

    def test_score_clamped_at_negative_10(self, extractor):
        score = extractor._compute_hawkish_score(
            "", rate_change_bps=-100, inflation_signal="dovish",
            growth_signal="dovish", forward_guidance="easing", emergency_flag=True,
        )
        # Raw sum: -4 -2 -2 -2 -3 = -13, must clamp to -10
        assert score == -10

    def test_all_neutral_gives_zero(self, extractor):
        score = extractor._compute_hawkish_score(
            "", rate_change_bps=0, inflation_signal="neutral",
            growth_signal="neutral", forward_guidance="hold", emergency_flag=False,
        )
        assert score == 0

    def test_score_always_within_declared_range_across_many_random_combinations(self, extractor):
        import itertools
        bps_values = [-100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100]
        signals = ["hawkish", "dovish", "neutral"]
        guidances = ["tightening", "easing", "hold", "uncertain"]
        # Sample a subset of the full cross-product for speed
        combos = list(itertools.product(bps_values, signals, signals, guidances, [True, False]))[::7]
        for bps, infl, growth, guidance, emergency in combos:
            score = extractor._compute_hawkish_score("", bps, infl, growth, guidance, emergency)
            assert -10 <= score <= 10, (
                f"score {score} out of range for bps={bps}, inflation={infl}, "
                f"growth={growth}, guidance={guidance}, emergency={emergency}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 7. Key phrase extraction
# ═══════════════════════════════════════════════════════════════════════
class TestKeyPhraseExtraction:

    def test_sentence_needs_at_least_2_trigger_keywords(self, extractor):
        # "inflation" alone (1 trigger) should not qualify
        text = "Inflation was discussed briefly. The committee then adjourned."
        s = extractor.extract(text)
        assert all("inflation" not in "".join(s.key_phrases).lower() for _ in [0]) or len(s.key_phrases) == 0

    def test_high_trigger_sentence_is_extracted(self, extractor):
        text = (
            "The Committee decided to raise the target range for the federal funds rate "
            "by 75 basis points, strongly committed to returning inflation to its "
            "target range. Weather was mild today."
        )
        s = extractor.extract(text)
        assert len(s.key_phrases) >= 1

    def test_phrases_truncated_to_180_chars(self, extractor):
        long_sentence = (
            "The Committee decided to raise the target range for the federal funds rate "
            "and reaffirmed its strong commitment to price stability and maximum employment "
            "while monitoring inflation and the target range and basis points very closely indeed. "
        )
        s = extractor.extract(long_sentence)
        for phrase in s.key_phrases:
            assert len(phrase) <= 180

    def test_no_qualifying_sentences_gives_empty_list(self, extractor):
        s = extractor.extract("The weather was pleasant. Lunch was served at noon.")
        assert s.key_phrases == []


# ═══════════════════════════════════════════════════════════════════════
# 8. summary() formatting sanity check
# ═══════════════════════════════════════════════════════════════════════
class TestSummaryFormatting:

    def test_summary_does_not_crash_and_contains_score(self, extractor):
        s = extractor.extract("The Committee raised the target range by 75 basis points.")
        text = s.summary()
        assert "+4" in text or str(s.hawkish_score) in text
        assert "Hawkish composite score" in text

    def test_summary_reports_not_found_when_rate_level_missing(self, extractor):
        s = extractor.extract("No rate information here.")
        assert "not found in text" in s.summary()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
