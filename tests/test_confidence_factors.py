"""
Tests for `agentxai.xai.confidence_factors` and the dashboard panel
that renders it.

Each factor is a pure function over the live specialist-memory dicts +
the Synthesizer's final output, so the tests compose tiny synthetic
inputs to isolate one factor at a time. Plus a backward-compat test
verifying the dashboard helper silently no-ops on records without the
new field.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from agentxai.xai.confidence_factors import (
    FACTOR_KEYS,
    _contradiction_penalty,
    _evidence_coverage,
    _option_match_strength,
    _retrieval_relevance,
    _specialist_agreement,
    _word_tokens,
    compute_confidence_factors,
)


# ---------------------------------------------------------------------------
# Per-factor unit tests
# ---------------------------------------------------------------------------

class TestRetrievalRelevance:
    def test_uses_retrieval_confidence_when_present(self):
        mem_b = {"retrieval_confidence": 0.83}
        assert _retrieval_relevance(mem_b) == pytest.approx(0.83)

    def test_clamps_out_of_range_retrieval_confidence(self):
        assert _retrieval_relevance({"retrieval_confidence": 1.5}) == 1.0
        assert _retrieval_relevance({"retrieval_confidence": -0.2}) == 0.0

    def test_falls_back_to_top_evidence_mean(self):
        mem_b = {"top_evidence": [
            {"score": 0.9}, {"score": 0.6}, {"score": 0.3},
        ]}
        assert _retrieval_relevance(mem_b) == pytest.approx(0.6)

    def test_returns_zero_when_no_retrieval_data(self):
        assert _retrieval_relevance({}) == 0.0
        assert _retrieval_relevance({"top_evidence": []}) == 0.0

    def test_handles_bad_retrieval_confidence_value(self):
        mem_b = {"retrieval_confidence": "oops"}
        # Falls through to top_evidence; absent → 0.
        assert _retrieval_relevance(mem_b) == 0.0


class TestOptionMatchStrength:
    def test_correct_verdict_yields_full_strength(self):
        out = {
            "predicted_letter": "C",
            "option_analysis": [
                {"letter": "A", "verdict": "incorrect"},
                {"letter": "C", "verdict": "correct"},
            ],
        }
        assert _option_match_strength(out, {"A": "x", "C": "y"}) == 1.0

    def test_partial_verdict_lower(self):
        out = {
            "predicted_letter": "C",
            "option_analysis": [{"letter": "C", "verdict": "partial"}],
        }
        assert _option_match_strength(out, {"C": "y"}) == pytest.approx(0.6)

    def test_letter_not_in_options(self):
        out = {"predicted_letter": "Z"}
        assert _option_match_strength(out, {"A": "x", "B": "y"}) == 0.3

    def test_no_letter_at_all(self):
        out = {"predicted_letter": ""}
        assert _option_match_strength(out, {"A": "x"}) == 0.3

    def test_letter_present_no_option_analysis(self):
        out = {"predicted_letter": "A"}
        assert _option_match_strength(out, {"A": "x"}) == 0.5

    def test_letter_marked_incorrect_in_own_analysis(self):
        # Internal contradiction — picked C but option_analysis says C is wrong.
        out = {
            "predicted_letter": "C",
            "option_analysis": [{"letter": "C", "verdict": "incorrect"}],
        }
        assert _option_match_strength(out, {"C": "y"}) == 0.4

    def test_analysis_present_but_missing_predicted_letter(self):
        # option_analysis is present but doesn't include the predicted letter.
        out = {
            "predicted_letter": "C",
            "option_analysis": [{"letter": "A", "verdict": "incorrect"}],
        }
        assert _option_match_strength(out, {"A": "x", "C": "y"}) == 0.7

    def test_lowercase_letter_normalised(self):
        out = {
            "predicted_letter": "c",
            "option_analysis": [{"letter": "C", "verdict": "correct"}],
        }
        assert _option_match_strength(out, {"C": "y"}) == 1.0


class TestSpecialistAgreement:
    def test_both_specialists_agree(self):
        out = {"predicted_text": "Myocardial infarction"}
        mem_a = {"top_conditions": [["Myocardial infarction", 0.9]]}
        mem_b = {"guideline_matches": [{"match": "Myocardial infarction"}]}
        assert _specialist_agreement(out, mem_a, mem_b) == pytest.approx(1.0)

    def test_only_one_specialist_agrees(self):
        out = {"predicted_text": "Myocardial infarction"}
        mem_a = {"top_conditions": [["Pulmonary embolism", 0.8]]}
        mem_b = {"guideline_matches": [{"match": "Myocardial infarction"}]}
        assert _specialist_agreement(out, mem_a, mem_b) == pytest.approx(0.5)

    def test_neither_agrees(self):
        out = {"predicted_text": "Myocardial infarction"}
        mem_a = {"top_conditions": [["Heat exhaustion", 0.5]]}
        mem_b = {"guideline_matches": [{"match": "Sepsis"}]}
        assert _specialist_agreement(out, mem_a, mem_b) == 0.0

    def test_no_specialist_memory_returns_zero(self):
        out = {"predicted_text": "Myocardial infarction"}
        assert _specialist_agreement(out, {}, {}) == 0.0

    def test_uses_top_evidence_snippets_for_b(self):
        out = {"predicted_text": "Ceftriaxone"}
        mem_b = {"top_evidence": [
            {"snippet": "Ceftriaxone is first-line for gonorrhea."},
        ]}
        # Only B contributed → 1/1.
        assert _specialist_agreement(out, {}, mem_b) == 1.0

    def test_falls_back_to_final_diagnosis_when_predicted_text_missing(self):
        out = {"final_diagnosis": "Myocardial infarction"}
        mem_a = {"top_conditions": [["Myocardial infarction", 0.9]]}
        assert _specialist_agreement(out, mem_a, {}) == 1.0

    def test_short_tokens_dont_match(self):
        # "MI" is < _MIN_TOKEN_LEN so it doesn't anchor an agreement.
        out = {"predicted_text": "MI"}
        mem_a = {"top_conditions": [["MI", 0.9]]}
        # No tokens long enough → 0.0.
        assert _specialist_agreement(out, mem_a, {}) == 0.0


class TestEvidenceCoverage:
    def test_uses_supporting_evidence_ids_when_present(self):
        out = {"supporting_evidence_ids": ["d1", "d2", "d3"]}
        assert _evidence_coverage(out, {}) == pytest.approx(1.0)

    def test_partial_coverage(self):
        out = {"supporting_evidence_ids": ["d1"]}
        # 1/3 target → 0.333...
        assert _evidence_coverage(out, {}) == pytest.approx(1 / 3)

    def test_supporting_caps_at_one(self):
        out = {"supporting_evidence_ids": ["d1", "d2", "d3", "d4", "d5"]}
        assert _evidence_coverage(out, {}) == 1.0

    def test_falls_back_to_high_quality_top_evidence(self):
        out = {}
        mem_b = {"top_evidence": [
            {"score": 0.9}, {"score": 0.6}, {"score": 0.4}, {"score": 0.2},
        ]}
        # Two scores >= 0.5 → 2/3.
        assert _evidence_coverage(out, mem_b) == pytest.approx(2 / 3)

    def test_zero_when_no_evidence_at_all(self):
        assert _evidence_coverage({}, {}) == 0.0


class TestContradictionPenalty:
    def test_partial_options_contribute(self):
        out = {"option_analysis": [
            {"letter": "A", "verdict": "correct"},
            {"letter": "B", "verdict": "partial"},
            {"letter": "C", "verdict": "partial"},
            {"letter": "D", "verdict": "incorrect"},
        ]}
        # 2 partial / 4 total = 0.5.
        assert _contradiction_penalty(out, {}) == pytest.approx(0.5)

    def test_no_partial_means_no_contradiction(self):
        out = {"option_analysis": [
            {"letter": "A", "verdict": "correct"},
            {"letter": "B", "verdict": "incorrect"},
        ]}
        assert _contradiction_penalty(out, {}) == 0.0

    def test_falls_back_to_top_conditions_gap(self):
        out = {}
        mem_a = {"top_conditions": [["MI", 1.0], ["PE", 0.9]]}
        # second/top * 0.5 = 0.45, capped at 0.5 → 0.45
        assert _contradiction_penalty(out, mem_a) == pytest.approx(0.45)

    def test_top_conditions_clean_no_close_runner_up(self):
        out = {}
        mem_a = {"top_conditions": [["MI", 1.0], ["PE", 0.1]]}
        # 0.1 / 1.0 * 0.5 = 0.05
        assert _contradiction_penalty(out, mem_a) == pytest.approx(0.05)

    def test_zero_when_no_signals(self):
        assert _contradiction_penalty({}, {}) == 0.0
        assert _contradiction_penalty({}, {"top_conditions": []}) == 0.0
        assert _contradiction_penalty({}, {"top_conditions": [["MI", 0.5]]}) == 0.0


# ---------------------------------------------------------------------------
# Tiny utility
# ---------------------------------------------------------------------------

class TestWordTokens:
    def test_filters_short_tokens(self):
        toks = _word_tokens("MI is the diagnosis")
        # "MI", "is" are < 4 chars; "the" is 3; "diagnosis" passes.
        assert "diagnosis" in toks
        assert "the" not in toks
        assert "is" not in toks

    def test_handles_empty_input(self):
        assert _word_tokens("") == set()
        assert _word_tokens(None) == set()  # type: ignore[arg-type]

    def test_lowercases(self):
        assert "diagnosis" in _word_tokens("DIAGNOSIS confirmed")


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TestComputeConfidenceFactors:
    def test_returns_all_five_factor_keys(self):
        out = compute_confidence_factors(
            final_output={}, specialist_a_memory={}, specialist_b_memory={},
        )
        assert set(out.keys()) == set(FACTOR_KEYS)

    def test_every_factor_in_unit_interval(self):
        # Bombard with messy inputs — every value must still be in [0, 1].
        out = compute_confidence_factors(
            final_output={
                "predicted_letter": "C", "predicted_text": "Acute MI",
                "supporting_evidence_ids": ["d1", "d2", "d3", "d4", "d5"],
                "option_analysis": [
                    {"letter": "A", "verdict": "partial"},
                    {"letter": "B", "verdict": "partial"},
                    {"letter": "C", "verdict": "correct"},
                ],
            },
            specialist_a_memory={"top_conditions": [["Acute MI", 1.0],
                                                     ["PE", 0.99]]},
            specialist_b_memory={"retrieval_confidence": 0.88,
                                  "guideline_matches": [{"match": "Acute MI"}],
                                  "top_evidence": [{"score": 0.9, "snippet": "MI"}]},
            options={"A": "x", "B": "y", "C": "Acute MI"},
        )
        for k, v in out.items():
            assert 0.0 <= v <= 1.0, f"{k} out of [0, 1]: {v}"

    def test_empty_inputs_yield_all_zeros(self):
        out = compute_confidence_factors(
            final_output={}, specialist_a_memory={},
            specialist_b_memory={}, options={},
        )
        for k in FACTOR_KEYS:
            # option_match_strength returns 0.3 when no letter — so it's
            # actually not zero. Everything else IS zero.
            if k == "option_match_strength":
                assert out[k] == 0.3
            else:
                assert out[k] == 0.0

    def test_values_rounded_to_four_decimals(self):
        # The aggregator rounds each value; verify exactness.
        out = compute_confidence_factors(
            final_output={}, specialist_a_memory={},
            specialist_b_memory={"retrieval_confidence": 0.123456789},
        )
        # 0.123456789 → 0.1235.
        assert out["retrieval_relevance"] == pytest.approx(0.1235)

    def test_example_from_spec_shape(self):
        """The example in the task spec is a realistic shape — verify the
        aggregator produces a dict with that exact shape (5 keys, all
        floats)."""
        out = compute_confidence_factors(
            final_output={
                "predicted_letter": "D",
                "predicted_text": "Ceftriaxone",
                "supporting_evidence_ids": ["d1", "d2"],
                "option_analysis": [{"letter": "D", "verdict": "correct"}],
            },
            specialist_a_memory={"top_conditions": [["Gonorrhea", 0.7]]},
            specialist_b_memory={
                "retrieval_confidence": 0.57,
                "guideline_matches": [{"match": "Gonorrhea"}],
                "top_evidence": [{"score": 0.7, "snippet": "Ceftriaxone..."}],
            },
            options={"D": "Ceftriaxone"},
        )
        # Same shape as the example payload in the task.
        assert set(out.keys()) == {
            "retrieval_relevance", "option_match_strength",
            "specialist_agreement", "evidence_coverage",
            "contradiction_penalty",
        }
        assert all(isinstance(v, float) for v in out.values())


# ---------------------------------------------------------------------------
# Dashboard panel — backward compat + label mapping
# ---------------------------------------------------------------------------

class TestDashboardPanelHelper:
    def test_factor_labels_cover_all_keys(self):
        # The dashboard's label dict must include every factor key the
        # heuristic emits — otherwise rows would silently disappear.
        from agentxai.ui.dashboard import _CONFIDENCE_FACTOR_LABELS
        assert set(_CONFIDENCE_FACTOR_LABELS.keys()) == set(FACTOR_KEYS)

    def test_panel_renders_when_factors_present(self, monkeypatch):
        # Capture the markdown call so we can assert the panel rendered.
        from agentxai.ui import dashboard
        captured: list = []

        def fake_markdown(html, **kwargs):
            captured.append(html)

        monkeypatch.setattr(dashboard.st, "markdown", fake_markdown)
        dashboard._render_confidence_factors_panel(
            confidence=0.95,
            factors={
                "retrieval_relevance": 0.57,
                "option_match_strength": 0.90,
                "specialist_agreement": 0.40,
                "evidence_coverage": 0.75,
                "contradiction_penalty": 0.05,
            },
        )
        assert captured, "panel should have rendered"
        html = captured[0]
        # Every factor key + its value formatted to 2 decimals appears.
        assert "retrieval_relevance" in html and "0.57" in html
        assert "option_match_strength" in html and "0.90" in html
        assert "specialist_agreement" in html and "0.40" in html
        assert "evidence_coverage" in html and "0.75" in html
        assert "contradiction_penalty" in html and "0.05" in html
        # Heuristic disclaimer is surfaced.
        assert "not clinically calibrated" in html.lower()
        # Headline confidence is shown.
        assert "0.95" in html

    def test_panel_skips_silently_for_old_records(self, monkeypatch):
        # Backward compat: empty / missing factors → no markdown call.
        from agentxai.ui import dashboard
        captured: list = []
        monkeypatch.setattr(dashboard.st, "markdown",
                            lambda *a, **kw: captured.append(a))

        dashboard._render_confidence_factors_panel(confidence=0.9, factors={})
        dashboard._render_confidence_factors_panel(confidence=0.9, factors=None)  # type: ignore[arg-type]
        dashboard._render_confidence_factors_panel(confidence=0.9, factors="bad")  # type: ignore[arg-type]
        assert captured == [], "panel must not render for missing/bad factors"

    def test_panel_skips_factors_with_bad_values(self, monkeypatch):
        # A non-numeric value for one factor shouldn't blow up the panel
        # — that row is skipped; the others render.
        from agentxai.ui import dashboard
        captured: list = []
        monkeypatch.setattr(dashboard.st, "markdown",
                            lambda html, **kw: captured.append(html))

        dashboard._render_confidence_factors_panel(
            confidence=0.5,
            factors={
                "retrieval_relevance": "oops",
                "option_match_strength": 0.9,
            },
        )
        assert captured
        html = captured[0]
        assert "option_match_strength" in html
        # Bad value silently skipped.
        assert "0.90" in html
