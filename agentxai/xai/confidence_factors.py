"""
Heuristic decomposition of the Synthesizer's headline ``confidence``.

The Synthesizer reports a single 0-1 confidence number with no breakdown
of *why*. This module reads the specialists' final memory states and the
Synthesizer's option-level output and produces five soft, fully
observable factor scores in [0, 1]::

    retrieval_relevance    — how relevant Specialist B's retrieved evidence
                              looked (mean FAISS similarity, basically).
    option_match_strength  — how cleanly the chosen letter matches the
                              answer options + option_analysis verdict.
    specialist_agreement   — fraction of specialists whose memory mentions
                              the predicted diagnosis.
    evidence_coverage      — count of supporting evidence ids (or
                              high-quality retrieved docs as fallback)
                              against a target of 3.
    contradiction_penalty  — fraction of options marked "partial" in
                              option_analysis (or A's two-top-conditions
                              gap), capped at 1.0.

These factors do NOT add up to the headline confidence — they are
narrative signals showing what *could* be driving it, not a calibrated
decomposition. **Confidence is heuristic and not clinically calibrated.**
The headline number should never be read as a probability of correctness;
it's the LLM's self-report, and these factors are observable cross-checks.

Each factor is independently bounded [0, 1] so the dashboard can render
them as small progress bars without normalization.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# Public name for the contract.
FACTOR_KEYS: tuple = (
    "retrieval_relevance",
    "option_match_strength",
    "specialist_agreement",
    "evidence_coverage",
    "contradiction_penalty",
)

# Target count of supporting evidence ids — anything beyond this saturates
# the evidence_coverage factor at 1.0.
_EVIDENCE_TARGET: float = 3.0

# Tokens shorter than this don't anchor a meaningful agreement match.
_MIN_TOKEN_LEN: int = 4
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_confidence_factors(
    *,
    final_output: Dict[str, Any],
    specialist_a_memory: Dict[str, Any],
    specialist_b_memory: Dict[str, Any],
    options: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Return all five factors as a single dict, each rounded to 4 decimals.

    Inputs default to empty so callers don't need to special-case missing
    specialists. Every factor returns 0.0 when its inputs are absent —
    the factor list is always complete, never sparse.
    """
    fout = final_output or {}
    mem_a = specialist_a_memory or {}
    mem_b = specialist_b_memory or {}
    opts = options or {}
    return {
        "retrieval_relevance":   round(_retrieval_relevance(mem_b), 4),
        "option_match_strength": round(_option_match_strength(fout, opts), 4),
        "specialist_agreement":  round(_specialist_agreement(fout, mem_a, mem_b), 4),
        "evidence_coverage":     round(_evidence_coverage(fout, mem_b), 4),
        "contradiction_penalty": round(_contradiction_penalty(fout, mem_a), 4),
    }


# ---------------------------------------------------------------------------
# Per-factor helpers
# ---------------------------------------------------------------------------

def _retrieval_relevance(mem_b: Dict[str, Any]) -> float:
    """
    Reuse Specialist B's `retrieval_confidence` (mean of top-3 FAISS
    scores) when present; otherwise compute from `top_evidence` directly.
    Already a [0, 1] cosine similarity, so we just clamp.
    """
    rc = mem_b.get("retrieval_confidence")
    if rc is not None:
        try:
            return max(0.0, min(1.0, float(rc)))
        except (TypeError, ValueError):
            pass
    evs = mem_b.get("top_evidence") or []
    scores: List[float] = []
    for e in evs:
        if not isinstance(e, dict):
            continue
        try:
            scores.append(float(e.get("score", 0.0)))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 0.0
    return max(0.0, min(1.0, sum(scores) / len(scores)))


def _option_match_strength(
    final_output: Dict[str, Any],
    options: Dict[str, str],
) -> float:
    """
    How cleanly the predicted letter matches the option set:
        1.0 — predicted letter named, in options, AND `option_analysis`
              marks it `correct`.
        0.7 — predicted letter named + in options, no option_analysis.
        0.6 — predicted letter named, marked `partial`.
        0.5 — predicted letter named + in options, but option_analysis
              missing (Synthesizer didn't emit per-option reasoning).
        0.4 — predicted letter named but option_analysis disagrees.
        0.3 — letter doesn't match any listed option (or no letter).
    """
    predicted_letter = (str(final_output.get("predicted_letter") or "")
                        .strip().upper())
    upper_keys = {str(k).upper() for k in options.keys()} if options else set()

    if not predicted_letter or (upper_keys and predicted_letter not in upper_keys):
        return 0.3

    analysis = final_output.get("option_analysis") or []
    if not analysis:
        return 0.5

    for entry in analysis:
        if not isinstance(entry, dict):
            continue
        if (str(entry.get("letter") or "").strip().upper()) != predicted_letter:
            continue
        verdict = str(entry.get("verdict") or "").strip().lower()
        if verdict == "correct":
            return 1.0
        if verdict == "partial":
            return 0.6
        # Marked incorrect on its own pick — internal contradiction.
        return 0.4

    # option_analysis present but doesn't include the predicted letter.
    return 0.7


def _specialist_agreement(
    final_output: Dict[str, Any],
    mem_a: Dict[str, Any],
    mem_b: Dict[str, Any],
) -> float:
    """
    Fraction of specialists whose final memory mentions the predicted
    diagnosis. With two specialists the score is 0.0 / 0.5 / 1.0.

    Match logic: word-level token overlap between the predicted text and
    the specialist's relevant memory keys (`top_conditions` for A,
    `guideline_matches` + `top_evidence` snippets for B). At least one
    shared token of length >= `_MIN_TOKEN_LEN` is enough to count as
    agreement.

    Returns 0.0 when no specialist contributed *any* relevant memory.
    """
    predicted = str(
        final_output.get("predicted_text")
        or final_output.get("final_diagnosis")
        or ""
    )
    target_tokens = _word_tokens(predicted)
    if not target_tokens:
        return 0.0

    agree = 0
    total = 0

    # --- Specialist A ---
    a_tokens: set = set()
    for c in mem_a.get("top_conditions") or []:
        if isinstance(c, (list, tuple)) and c:
            a_tokens |= _word_tokens(str(c[0]))
        else:
            a_tokens |= _word_tokens(str(c))
    if a_tokens:
        total += 1
        if a_tokens & target_tokens:
            agree += 1

    # --- Specialist B ---
    b_tokens: set = set()
    for g in mem_b.get("guideline_matches") or []:
        if isinstance(g, dict):
            b_tokens |= _word_tokens(str(g.get("match") or ""))
            b_tokens |= _word_tokens(str(g.get("queried") or ""))
    for e in mem_b.get("top_evidence") or []:
        if isinstance(e, dict):
            b_tokens |= _word_tokens(str(e.get("snippet") or ""))
    if b_tokens:
        total += 1
        if b_tokens & target_tokens:
            agree += 1

    if total == 0:
        return 0.0
    return agree / total


def _evidence_coverage(
    final_output: Dict[str, Any],
    mem_b: Dict[str, Any],
) -> float:
    """
    Prefer the Synthesizer's explicit `supporting_evidence_ids` count;
    fall back to counting Specialist B's high-quality retrieved docs
    (FAISS similarity >= 0.5). Score = count / `_EVIDENCE_TARGET`,
    capped at 1.0.
    """
    cited = final_output.get("supporting_evidence_ids") or []
    if isinstance(cited, list) and cited:
        return min(1.0, len(cited) / _EVIDENCE_TARGET)

    high_q = 0
    for e in mem_b.get("top_evidence") or []:
        if not isinstance(e, dict):
            continue
        try:
            score = float(e.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if score >= 0.5:
            high_q += 1
    if high_q == 0:
        return 0.0
    return min(1.0, high_q / _EVIDENCE_TARGET)


def _contradiction_penalty(
    final_output: Dict[str, Any],
    mem_a: Dict[str, Any],
) -> float:
    """
    Fraction of `option_analysis` entries marked `partial` (competing
    candidates that aren't outright wrong). When option_analysis is
    missing, fall back to Specialist A's top-conditions gap: if the
    second-best condition is close to the top, that's a contradiction.

    Higher = more contradiction. 0.0 = clean, 1.0 = every option is partial.
    """
    analysis = final_output.get("option_analysis") or []
    if isinstance(analysis, list) and analysis:
        partial = sum(
            1 for e in analysis
            if isinstance(e, dict)
            and str(e.get("verdict") or "").strip().lower() == "partial"
        )
        return min(1.0, partial / len(analysis))

    a_conds = mem_a.get("top_conditions") or []
    if len(a_conds) < 2:
        return 0.0
    try:
        top = float(_likelihood(a_conds[0]))
        second = float(_likelihood(a_conds[1]))
    except (TypeError, ValueError):
        return 0.0
    if top <= 0:
        return 0.0
    # second/top close to 1 → strong contradiction; close to 0 → clean.
    # Cap fallback at 0.5 since it's a weaker signal than option_analysis.
    return min(0.5, max(0.0, second / top * 0.5))


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _word_tokens(text: str) -> set:
    """Lowercased word tokens of length >= `_MIN_TOKEN_LEN`."""
    if not text:
        return set()
    return {
        tok.lower()
        for tok in _TOKEN_RE.findall(text)
        if len(tok) >= _MIN_TOKEN_LEN
    }


def _likelihood(entry: Any) -> float:
    """Extract the likelihood from a (condition, score) tuple/list entry."""
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return float(entry[1])
    return 0.0
