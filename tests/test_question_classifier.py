"""
Tests for `agentxai.data.question_classifier`.

Covers the three classes the spec calls out (HIV confirmatory test → screening_or_test,
classic symptom diagnosis → diagnosis, drug side-effect → pharmacology) plus
edge cases for every other label and the `unknown` fallback.
"""

from __future__ import annotations

import pytest

from agentxai.data.question_classifier import (
    VALID_TYPES,
    classify_question,
    is_valid_type,
    matched_pattern,
)


# ---------------------------------------------------------------------------
# Required scenarios from the spec
# ---------------------------------------------------------------------------

class TestRequiredScenarios:
    def test_hiv_confirmatory_test_classified_as_screening_or_test(self):
        stem = (
            "A 34-year-old man presents to his primary care physician for "
            "routine evaluation. A reactive HIV antibody screening immunoassay "
            "is reported. Which of the following is the most appropriate "
            "confirmatory test?"
        )
        assert classify_question(stem) == "screening_or_test"

    def test_classic_symptom_diagnosis_classified_as_diagnosis(self):
        stem = (
            "A 58-year-old man presents with crushing substernal chest pain "
            "radiating to the left arm, dyspnea, and diaphoresis. ECG shows "
            "ST elevations in II, III, aVF. What is the most likely diagnosis?"
        )
        assert classify_question(stem) == "diagnosis"

    def test_drug_side_effect_classified_as_pharmacology(self):
        stem = (
            "A 65-year-old man with chronic atrial fibrillation taking warfarin "
            "develops widespread purpura. Which of the following adverse "
            "effects of warfarin is most likely responsible?"
        )
        assert classify_question(stem) == "pharmacology"


# ---------------------------------------------------------------------------
# Other supported labels
# ---------------------------------------------------------------------------

class TestOtherLabels:
    def test_treatment(self):
        stem = (
            "A 21-year-old sexually active male presents with urethral discharge. "
            "Gram stain reveals gram-negative diplococci. Which of the following "
            "is the most appropriate treatment?"
        )
        assert classify_question(stem) == "treatment"

    def test_treatment_initial_management(self):
        stem = "What is the best initial management for this patient?"
        assert classify_question(stem) == "treatment"

    def test_mechanism(self):
        stem = (
            "A patient develops hemolysis after eating fava beans. "
            "Which of the following best explains the underlying pathophysiology?"
        )
        assert classify_question(stem) == "mechanism"

    def test_risk_factor(self):
        stem = (
            "A 70-year-old smoker presents for screening. Which of the following "
            "is the strongest risk factor for abdominal aortic aneurysm?"
        )
        assert classify_question(stem) == "risk_factor"

    def test_anatomy(self):
        stem = (
            "After a humeral midshaft fracture, the patient cannot extend the "
            "wrist. Which of the following nerves is most likely damaged?"
        )
        assert classify_question(stem) == "anatomy"

    def test_prognosis(self):
        stem = (
            "A 40-year-old woman is diagnosed with stage I breast cancer. "
            "What is her 5-year survival rate?"
        )
        assert classify_question(stem) == "prognosis"


# ---------------------------------------------------------------------------
# Priority and ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_treatment_beats_diagnosis_when_both_keywords_present(self):
        # Stem mentions "diagnosis" but is asking about treatment — the
        # action-oriented label wins because it's checked first.
        stem = (
            "The patient's diagnosis is community-acquired pneumonia. "
            "What is the most appropriate treatment?"
        )
        assert classify_question(stem) == "treatment"

    def test_screening_beats_diagnosis_when_test_keyword_present(self):
        stem = (
            "The likely diagnosis is gonococcal urethritis. Which laboratory "
            "test should be ordered to confirm?"
        )
        assert classify_question(stem) == "screening_or_test"

    def test_pharmacology_beats_mechanism_for_drug_mechanism(self):
        # "mechanism of action" specifically is a pharmacology marker even
        # though "mechanism" alone would be the broader bucket.
        stem = "What is the mechanism of action of metformin?"
        assert classify_question(stem) == "pharmacology"


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------

class TestUnknownFallback:
    def test_empty_string_is_unknown(self):
        assert classify_question("") == "unknown"

    def test_none_is_unknown(self):
        assert classify_question(None) == "unknown"  # type: ignore[arg-type]

    def test_non_string_is_unknown(self):
        assert classify_question(12345) == "unknown"  # type: ignore[arg-type]
        assert classify_question({"q": "x"}) == "unknown"  # type: ignore[arg-type]

    def test_no_keywords_returns_unknown(self):
        # A rare USMLE phrasing that doesn't match any of our patterns —
        # better to flag as unknown than force-fit one of the buckets.
        stem = "The patient improves after rest. He is discharged home."
        assert classify_question(stem) == "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_valid_types_contains_all_nine_labels(self):
        expected = {
            "diagnosis", "treatment", "screening_or_test", "mechanism",
            "risk_factor", "pharmacology", "anatomy", "prognosis", "unknown",
        }
        assert set(VALID_TYPES) == expected

    def test_is_valid_type(self):
        for t in VALID_TYPES:
            assert is_valid_type(t)
        assert not is_valid_type("nonsense")
        assert not is_valid_type("")
        assert not is_valid_type(None)  # type: ignore[arg-type]

    def test_matched_pattern_returns_the_pattern_that_classified(self):
        stem = "What is the most appropriate confirmatory test?"
        pat = matched_pattern(stem)
        assert pat is not None
        assert "confirmatory" in pat

    def test_matched_pattern_returns_none_when_unknown(self):
        assert matched_pattern("The patient improves and is discharged.") is None

    def test_options_param_accepted_but_not_yet_used(self):
        # Forward-compat: the signature accepts an options dict. Currently
        # ignored; future option-aware heuristics should slot in here.
        stem = "What is the most likely diagnosis?"
        result_no_opts = classify_question(stem)
        result_with_opts = classify_question(stem, options={"A": "Acute MI"})
        assert result_no_opts == result_with_opts == "diagnosis"


# ---------------------------------------------------------------------------
# Determinism / case-insensitivity
# ---------------------------------------------------------------------------

class TestDeterminismAndCaseHandling:
    def test_classifier_is_case_insensitive(self):
        assert classify_question("WHAT IS THE MOST LIKELY DIAGNOSIS?") == "diagnosis"
        assert classify_question("which CONFIRMATORY TEST should be performed?") == "screening_or_test"

    def test_classifier_is_deterministic(self):
        stem = "What is the most appropriate confirmatory test?"
        # Running twice should always give the same answer (no random state).
        assert classify_question(stem) == classify_question(stem)
