"""
Tests for run_pipeline._match_option_letter — the function that maps a
free-text diagnosis from the synthesizer LLM onto one of the MCQ option
letters.

Includes the regression test for the bug where Gemini sometimes returned
just the bare letter (e.g. "A") and the matcher couldn't recognize it,
falsely scoring the run as incorrect.
"""
from __future__ import annotations

from run_pipeline import _match_option_letter


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
