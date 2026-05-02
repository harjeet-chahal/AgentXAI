"""
Tests for run_pipeline._match_option_letter — the function that maps a
free-text diagnosis from the synthesizer LLM onto one of the MCQ option
letters.

Includes the regression test for the bug where Gemini sometimes returned
just the bare letter (e.g. "A") and the matcher couldn't recognize it,
falsely scoring the run as incorrect.

Plus tests for `_resolve_predicted_letter`, the higher-level chooser that
prefers the Synthesizer's own `predicted_letter` over text matching when
a letter pick is supplied.
"""
from __future__ import annotations

from run_pipeline import _match_option_letter, _resolve_predicted_letter


OPTIONS = {
    "A": "Aldosterone excess",
    "B": "Cushing syndrome",
    "C": "Renal artery stenosis",
    "D": "Pheochromocytoma",
    "E": "Hypothyroidism",
}


class TestLetterPrefixShortcut:
    """Regression: a bare-letter response must resolve to its option."""

    def test_bare_letter(self):
        assert _match_option_letter("A", OPTIONS) == ("A", "Aldosterone excess")

    def test_bare_letter_lowercase(self):
        assert _match_option_letter("a", OPTIONS) == ("A", "Aldosterone excess")

    def test_bare_letter_with_dot(self):
        assert _match_option_letter("A.", OPTIONS) == ("A", "Aldosterone excess")

    def test_bare_letter_with_colon(self):
        assert _match_option_letter("A:", OPTIONS) == ("A", "Aldosterone excess")

    def test_letter_paren_text(self):
        assert _match_option_letter("A) Aldosterone excess", OPTIONS) == (
            "A", "Aldosterone excess",
        )

    def test_letter_dot_text(self):
        assert _match_option_letter("A. Aldosterone excess", OPTIONS) == (
            "A", "Aldosterone excess",
        )

    def test_wrapped_letter(self):
        assert _match_option_letter("(A)", OPTIONS) == ("A", "Aldosterone excess")

    def test_bracketed_letter(self):
        assert _match_option_letter("[A]", OPTIONS) == ("A", "Aldosterone excess")

    def test_leading_whitespace(self):
        assert _match_option_letter("   A", OPTIONS) == ("A", "Aldosterone excess")


class TestTokenOverlapFallback:
    """When no leading letter is present, token-overlap matching takes over."""

    def test_full_disease_name(self):
        letter, text = _match_option_letter("Aldosterone excess", OPTIONS)
        assert letter == "A"
        assert text == "Aldosterone excess"

    def test_partial_disease_name(self):
        letter, _ = _match_option_letter("aldosterone", OPTIONS)
        assert letter == "A"

    def test_disease_name_starting_with_letter_token_isnt_misread(self):
        # "Acute" starts with "A" but no delimiter follows — must not be
        # falsely shortcut-matched as option A.
        letter, _ = _match_option_letter("Acute appendicitis", OPTIONS_WITH_ACUTE)
        assert letter == "F"  # the actual option that contains "acute"

    def test_no_match(self):
        assert _match_option_letter("Diabetes mellitus", OPTIONS) == ("", "")

    def test_empty_diagnosis(self):
        assert _match_option_letter("", OPTIONS) == ("", "")

    def test_empty_options(self):
        assert _match_option_letter("Aldosterone excess", {}) == ("", "")


OPTIONS_WITH_ACUTE = {
    "A": "Aldosterone excess",
    "B": "Cushing syndrome",
    "C": "Renal artery stenosis",
    "D": "Pheochromocytoma",
    "E": "Hypothyroidism",
    "F": "Acute appendicitis",
}


# ---------------------------------------------------------------------------
# _resolve_predicted_letter — Synthesizer letter wins over text matching
# ---------------------------------------------------------------------------

_HIV_OPTIONS = {
    "A": "Western blot",
    "B": "Repeat HIV antibody screening test in 6 months",
    "C": "HIV-1/HIV-2 antibody differentiation immunoassay",
    "D": "p24 antigen",
    "E": "HIV RNA viral load",
}


class TestResolvePredictedLetter:
    """The Synthesizer's predicted_letter is the source of truth when present."""

    def test_synth_letter_wins_when_text_would_disagree(self):
        """
        Regression for the user-reported HIV scenario: the Synthesizer
        rationale mentions Western blot (option A) en passant, but it
        deliberately picked option C. The text-matching fallback might
        latch onto "Western blot" via Jaccard overlap; `predicted_letter`
        must override that.
        """
        synth = {
            "predicted_letter": "C",
            "predicted_text":   _HIV_OPTIONS["C"],
            "final_diagnosis":
                "Western blot is outdated — choose the antibody differentiation immunoassay.",
        }
        letter, text = _resolve_predicted_letter(synth, _HIV_OPTIONS)
        assert letter == "C"
        assert text == _HIV_OPTIONS["C"]

    def test_synth_text_used_verbatim(self):
        """
        When predicted_text is present we keep it verbatim — don't silently
        replace with the option dict's stored value (they should match
        anyway, but if the LLM phrases the option slightly differently the
        on-screen "model picked X" line should reflect the LLM's words).
        """
        synth = {"predicted_letter": "A", "predicted_text": "Western blot (legacy)"}
        letter, text = _resolve_predicted_letter(synth, _HIV_OPTIONS)
        assert letter == "A"
        assert text == "Western blot (legacy)"

    def test_synth_text_falls_back_to_option_text(self):
        synth = {"predicted_letter": "C", "predicted_text": ""}
        letter, text = _resolve_predicted_letter(synth, _HIV_OPTIONS)
        assert letter == "C"
        assert text == _HIV_OPTIONS["C"]

    def test_unknown_letter_falls_back_to_text_matching(self):
        """
        If predicted_letter isn't one of the listed options we ignore it
        and fall back to matching `final_diagnosis` against the option
        list. Catches the case where the LLM hallucinates an extra
        option letter ('F' on a 5-option question).
        """
        synth = {
            "predicted_letter": "Z",
            "predicted_text":   "irrelevant",
            "final_diagnosis":  _HIV_OPTIONS["C"],
        }
        letter, text = _resolve_predicted_letter(synth, _HIV_OPTIONS)
        assert letter == "C"
        assert text == _HIV_OPTIONS["C"]

    def test_no_letter_uses_text_matching(self):
        """Old-shape (4-key) Synthesizer output still resolves correctly."""
        synth = {"final_diagnosis": "Aldosterone excess", "rationale": "..."}
        letter, text = _resolve_predicted_letter(synth, OPTIONS)
        assert letter == "A"
        assert text == "Aldosterone excess"

    def test_no_letter_no_text_no_diagnosis_returns_empty(self):
        assert _resolve_predicted_letter({}, _HIV_OPTIONS) == ("", "")

    def test_no_options_returns_empty(self):
        synth = {"predicted_letter": "C", "predicted_text": "x"}
        assert _resolve_predicted_letter(synth, {}) == ("", "")

    def test_lowercase_letter_normalised(self):
        synth = {"predicted_letter": "c", "predicted_text": _HIV_OPTIONS["C"]}
        letter, _ = _resolve_predicted_letter(synth, _HIV_OPTIONS)
        assert letter == "C"
