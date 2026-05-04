"""
Tests for agentxai.agents._llm_utils.extract_letter_from_text — the
chain-of-thought letter-extraction fallback used by the Synthesizer when
the LLM forgets to populate `predicted_letter` and emits the answer
only in the rationale prose.
"""
from __future__ import annotations

import pytest

from agentxai.agents._llm_utils import extract_letter_from_text


VALID_AE = {"A", "B", "C", "D", "E"}


class TestBoxedAnswer:
    """LaTeX \\boxed{X} is the highest-confidence pattern (Gemini emits it)."""

    def test_dollar_boxed(self):
        assert extract_letter_from_text("The final answer is $\\boxed{B}$.", VALID_AE) == "B"

    def test_bare_boxed(self):
        assert extract_letter_from_text("Therefore \\boxed{C}", VALID_AE) == "C"

    def test_boxed_with_whitespace(self):
        assert extract_letter_from_text("\\boxed{ D }", VALID_AE) == "D"

    def test_boxed_lowercase_normalized(self):
        assert extract_letter_from_text("\\boxed{a}", VALID_AE) == "A"


class TestAnswerIsPattern:
    def test_the_final_answer_is(self):
        assert extract_letter_from_text("The final answer is B.", VALID_AE) == "B"

    def test_correct_answer_is(self):
        assert extract_letter_from_text("The correct answer is (C).", VALID_AE) == "C"

    def test_bare_the_answer_is(self):
        assert extract_letter_from_text("So the answer is D, in conclusion.", VALID_AE) == "D"

    def test_answer_colon(self):
        assert extract_letter_from_text("Answer: E", VALID_AE) == "E"


class TestOptionPattern:
    def test_option_is_appropriate(self):
        text = "Option B is the most appropriate action here."
        assert extract_letter_from_text(text, VALID_AE) == "B"

    def test_choice_seems(self):
        text = "Choice C seems closest to the diagnostic criteria."
        assert extract_letter_from_text(text, VALID_AE) == "C"

    def test_option_no_verb_does_not_match(self):
        # A bare "Option A" without a verb shouldn't trigger — that's how
        # per-option analysis sections are typically structured.
        text = "Option A. Disclose to the patient."
        assert extract_letter_from_text(text, VALID_AE) == ""


class TestPriorityAndLastWins:
    def test_boxed_beats_option_mention(self):
        text = "Option A might seem right, but \\boxed{B} is correct."
        assert extract_letter_from_text(text, VALID_AE) == "B"

    def test_last_match_wins_within_pattern(self):
        text = "The answer is A. Wait, on reflection the answer is C."
        assert extract_letter_from_text(text, VALID_AE) == "C"


class TestRealRationale:
    """Regression: the actual failing case from MedQA task 6f3299b4..."""

    def test_full_chain_of_thought_with_boxed(self):
        text = (
            "Option B is the most appropriate action. Accurate documentation "
            "in the operative report is crucial.\n\n"
            "*   **A: Disclose the error to the patient but leave it out of "
            "the operative report.** This is partially correct...\n"
            "*   **C: Tell the attending that he cannot fail to disclose...** "
            "While the resident should certainly communicate...\n\n"
            "Therefore, disclosing the error to the patient and ensuring it "
            "is accurately documented in the operative report (Option B) is "
            "the most comprehensive and ethically sound course of action.\n\n"
            "The final answer is $\\boxed{B}$."
        )
        assert extract_letter_from_text(text, VALID_AE) == "B"


class TestEdgeCases:
    def test_empty_text(self):
        assert extract_letter_from_text("", VALID_AE) == ""

    def test_no_match(self):
        assert extract_letter_from_text("Just a plain rationale with no answer markers.", VALID_AE) == ""

    def test_letter_outside_valid_set_rejected(self):
        # "the answer is F" but valid set is A-E → reject.
        assert extract_letter_from_text("The answer is F.", VALID_AE) == ""

    def test_taiwan_4_options(self):
        # Taiwan corpus uses A-D only. "answer is E" must be rejected.
        valid_ad = {"A", "B", "C", "D"}
        assert extract_letter_from_text("The answer is E.", valid_ad) == ""
        assert extract_letter_from_text("The answer is D.", valid_ad) == "D"

    def test_default_valid_letters_when_none_supplied(self):
        # Without an explicit valid set, accept any A-J.
        assert extract_letter_from_text("\\boxed{H}") == "H"

    def test_none_input(self):
        # type: ignore[arg-type]
        assert extract_letter_from_text(None) == ""  # type: ignore[arg-type]


class TestSynthesizerIntegration:
    """The fallback should fire end-to-end in Synthesizer._normalise_result."""

    def test_normalise_result_recovers_letter_from_rationale(self):
        from agentxai.agents.synthesizer import _normalise_result

        # LLM emitted only a rationale — no predicted_letter, no
        # option_analysis, no final_diagnosis. Old behaviour: empty
        # predicted_letter. New behaviour: "B" recovered from rationale.
        parsed = {
            "rationale": (
                "Option B is the most appropriate action. "
                "The final answer is $\\boxed{B}$."
            ),
        }
        out = _normalise_result(parsed, raw="{}")
        assert out["predicted_letter"] == "B"

    def test_normalise_result_does_not_overwrite_explicit_letter(self):
        from agentxai.agents.synthesizer import _normalise_result

        # Explicit predicted_letter wins; we don't try to override from
        # the rationale even if the rationale would scrape differently.
        parsed = {
            "predicted_letter": "A",
            "rationale": "On reflection the answer is B.",
        }
        out = _normalise_result(parsed, raw="{}")
        assert out["predicted_letter"] == "A"

    def test_normalise_result_legacy_rationale_unchanged(self):
        # Regression for tests/test_agents.py::test_legacy_four_key_response —
        # a short opaque rationale should NOT trigger the fallback.
        from agentxai.agents.synthesizer import _normalise_result

        parsed = {
            "final_diagnosis": "Acute MI",
            "confidence": 0.8,
            "differential": ["PE"],
            "rationale": "Old-shape rationale.",
        }
        out = _normalise_result(parsed, raw="{}")
        assert out["predicted_letter"] == ""

    def test_normalise_result_recovers_when_no_json_at_all(self):
        # Regression for the live failure mode: LLM emits chain-of-thought
        # PROSE with no JSON braces. parse_json_object returns None,
        # _normalise_result short-circuits to _empty_result — and we still
        # need to extract the answer.
        from agentxai.agents.synthesizer import _normalise_result

        raw_prose = (
            "Long discussion of the case... Option B is the most appropriate.\n"
            "The final answer is $\\boxed{B}$."
        )
        out = _normalise_result(parsed=None, raw=raw_prose)
        assert out["predicted_letter"] == "B"
        assert out["rationale"] == raw_prose.strip()
