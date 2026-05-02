"""
Deterministic, no-LLM heuristic classifier for MedQA question types.

Classifies a question stem (and optionally its answer options) into one of
the nine `QuestionType` values used by the responsibility-scoring priors
in ``agentxai.xai.accountability``:

    diagnosis · treatment · screening_or_test · mechanism ·
    risk_factor · pharmacology · anatomy · prognosis · unknown

The classifier is regex-based and runs in microseconds. It is deliberately
not exhaustive — USMLE phrasing varies — so the fallback is the honest
``"unknown"`` rather than a forced bucket. Routing/weighting code treats
``"unknown"`` as neutral (no prior tilt), so an under-classified question
costs nothing.

Why heuristic over LLM:
  * The classifier is in the per-task hot path; an extra LLM call would
    add latency and token spend on every task.
  * Deterministic output is testable.
  * The signal is downstream-corrective (a prior on the responsibility
    score), not load-bearing — being wrong shifts the score by ~30%, it
    doesn't choose the diagnosis.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VALID_TYPES: Tuple[str, ...] = (
    "diagnosis",
    "treatment",
    "screening_or_test",
    "mechanism",
    "risk_factor",
    "pharmacology",
    "anatomy",
    "prognosis",
    "unknown",
)

# Matchers ordered by specificity — the first matching label wins. The
# more-specific patterns (treatment, screening_or_test, pharmacology) are
# probed before the broader ones (diagnosis) so a question that mentions
# both "diagnosis" and "treatment" gets the more-action-oriented label.
_PATTERNS: List[Tuple[str, List[str]]] = [
    ("screening_or_test", [
        r"\bconfirmatory test\b",
        r"\bscreening test\b",
        r"\bdiagnostic test\b",
        r"\bbest (?:next |initial )?test\b",
        r"\bmost (?:appropriate|specific|sensitive) (?:test|study)\b",
        r"\binitial test\b",
        r"\bnext (?:best )?step in (?:the )?(?:work[- ]?up|evaluation|diagnostic work[- ]?up)\b",
        r"\bwhich (?:laboratory )?(?:test|study|panel|imaging)\b",
        r"\bgold[- ]?standard\b",
        r"\bordering which\b",
        r"\bshould (?:be )?(?:order|ordered|performed)\b",
        # HIV-style confirmatory follow-up.
        r"\bfollow[- ]?up test\b",
        r"\bnext step in (?:the )?diagnosis\b",
    ]),
    ("treatment", [
        r"\bbest (?:initial |first[- ]line )?(?:treatment|therapy|management)\b",
        r"\binitial (?:treatment|therapy|management)\b",
        r"\bmost appropriate (?:treatment|therapy|management|next step in (?:treatment|management))\b",
        r"\bdrug of choice\b",
        r"\bfirst[- ]line (?:treatment|therapy|drug|agent)\b",
        r"\bnext step in (?:the )?(?:treatment|management)\b",
        r"\bmanage(?:ment)?\b.*\bappropriate\b",
        r"\bmost appropriate (?:pharmacotherapy|antibiotic|surgery|intervention)\b",
        r"\bdefinitive (?:treatment|therapy|management)\b",
    ]),
    ("pharmacology", [
        r"\bmechanism of action\b",
        r"\b(?:side|adverse) (?:effect|reaction)s?\b",
        r"\bcontraindicat(?:ed|ion)\b",
        r"\bdrug[- ](?:induced|interaction)\b",
        r"\bpharmacokinetic(?:s)?\b",
        r"\bpharmacodynamic(?:s)?\b",
        r"\b(?:taking|on|started on)\b.*\b(?:adverse|side|toxicity)\b",
        # "Which medication / drug / agent ... cause / responsible / explain"
        r"\bwhich (?:medication|drug|agent|antibiotic)\b",
        r"\bdrug class\b",
    ]),
    ("mechanism", [
        r"\bmechanism\b",
        r"\bpathophysiolog\w*\b",
        r"\bunderlying (?:cause|mechanism|process|defect)\b",
        r"\bbest explains?\b",
        r"\bmost likely (?:explains?|underl(?:ies|ying))\b",
        r"\bresponsible for (?:this|the) (?:finding|presentation|symptom)\b",
    ]),
    ("risk_factor", [
        r"\brisk factor\b",
        r"\bpredispos(?:e|es|ed|ing|ition)\b",
        r"\bgreatest risk\b",
        r"\bmost likely to develop\b",
        r"\bincreased risk of\b",
    ]),
    ("anatomy", [
        r"\bwhich (?:structure|nerve|artery|vein|muscle|bone|ligament|gland|lobe|fissure|foramen)\b",
        r"\bmost likely (?:damaged|injured|involved|affected)\b",
        r"\binnervat(?:e|es|ed|ion)\b",
        r"\b(?:supplied by|derived from|originates from)\b",
        r"\bembryologic(?:al)?\b",
    ]),
    ("prognosis", [
        r"\bprognosis\b",
        r"\b(?:5|10)[- ]year survival\b",
        r"\blife expectancy\b",
        r"\bsurvival rate\b",
        r"\bnatural history\b",
    ]),
    # Diagnosis last — it's the broadest catch.
    ("diagnosis", [
        r"\bmost likely diagnosis\b",
        r"\bwhich (?:condition|disease|disorder|diagnosis)\b",
        r"\bbest (?:fits|explains|describes) (?:this|the) (?:presentation|patient)\b",
        r"\b(?:patient (?:most likely )?has|likely diagnosis is)\b",
    ]),
]

# Pre-compile.
_COMPILED: List[Tuple[str, List[re.Pattern]]] = [
    (label, [re.compile(p, re.IGNORECASE) for p in patterns])
    for label, patterns in _PATTERNS
]


def classify_question(
    question: str,
    options: Optional[Dict[str, str]] = None,
) -> str:
    """
    Return one of `VALID_TYPES` for this MedQA stem.

    Parameters
    ----------
    question : The full question stem (case + final question sentence).
    options  : Optional A-E option dict. Currently unused by the classifier
               but accepted so future option-aware heuristics can drop in
               without changing call sites.

    The classifier scans `_PATTERNS` in priority order and returns the
    first matching label. Returns ``"unknown"`` if nothing matches —
    callers should treat this as a neutral signal, not as evidence the
    question is degenerate.
    """
    if not question or not isinstance(question, str):
        return "unknown"
    text = question
    for label, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(text):
                return label
    return "unknown"


def matched_pattern(question: str) -> Optional[str]:
    """
    Return the literal regex pattern that classified `question`, or None.

    Useful for debugging / dashboard tooltips so the user can see *why* a
    question got a particular label.
    """
    if not question:
        return None
    for _label, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(question):
                return pat.pattern
    return None


def is_valid_type(value: str) -> bool:
    """True iff `value` is in `VALID_TYPES` — guards stored input."""
    return isinstance(value, str) and value in VALID_TYPES
