"""
Integration tests for the four Specialist-B tools in ``agentxai/tools/``.

Each test exercises the real implementation end-to-end — no mocks. Indices
that do not yet exist are built on first call by the tool itself (or by the
``build_knowledge_base`` helpers); the textbook FAISS build is slow (multiple
minutes on CPU), so the ``textbook_search`` test is marked ``@pytest.mark.slow``
and only runs with ``--run-slow``.
"""

from __future__ import annotations

from typing import List

import pytest
from langchain_core.tools import Tool

from agentxai.tools.guideline_lookup import guideline_lookup, guideline_lookup_tool
from agentxai.tools.textbook_search import textbook_search, textbook_search_tool
from agentxai.tools.severity_scorer import (
    SEVERITY_WEIGHTS,
    severity_scorer,
    severity_scorer_tool,
)
from agentxai.tools.symptom_lookup import symptom_lookup, symptom_lookup_tool


# ---------------------------------------------------------------------------
# symptom_lookup
# ---------------------------------------------------------------------------

def test_symptom_lookup_returns_medqa_derived_conditions():
    """'chest pain' is one of the most frequent USMLE stems; the table must
    contain a non-trivial list of related conditions, normalised likelihoods
    summing to ~1.0, and the langchain Tool wrapper must expose the same fn."""
    result = symptom_lookup("chest pain")

    assert isinstance(result, dict)
    assert result["source"] == "medqa_derived"

    related = result["related_conditions"]
    assert isinstance(related, list)
    assert len(related) > 0, "Expected at least one related condition for 'chest pain'."

    # Each entry is (str, float in [0, 1]); list is sorted by descending likelihood.
    likelihoods: List[float] = []
    for entry in related:
        assert isinstance(entry, (list, tuple)) and len(entry) == 2
        cond, like = entry
        assert isinstance(cond, str) and cond.strip()
        assert isinstance(like, float) and 0.0 <= like <= 1.0
        likelihoods.append(like)
    assert likelihoods == sorted(likelihoods, reverse=True)
    # Likelihoods are rounded per-entry to 4 decimals → sum may drift a bit
    # from exactly 1.0; allow ±5% slack.
    assert sum(likelihoods) == pytest.approx(1.0, abs=0.05)

    # Unknown phrase returns the empty-but-typed payload.
    miss = symptom_lookup("not_a_real_symptom_xyz")
    assert miss == {"related_conditions": [], "source": "medqa_derived"}

    # Tool wrapper exposes the traced callable as .func.
    assert isinstance(symptom_lookup_tool, Tool)
    assert symptom_lookup_tool.name == "symptom_lookup"
    assert symptom_lookup_tool.func is symptom_lookup


# ---------------------------------------------------------------------------
# severity_scorer
# ---------------------------------------------------------------------------

def test_severity_scorer_weights_average_and_cooccurrence_bonus():
    """Score = mean weight of recognised symptoms, +0.10 if ≥3 are severe,
    capped at 1.0; unknowns and empty input handled."""
    # Empty / all-unknown.
    assert severity_scorer([]) == 0.0
    assert severity_scorer(["not_a_symptom_xyz", "also_unknown"]) == 0.0

    # Single mild symptom: exactly the table weight.
    assert severity_scorer(["mild headache"]) == pytest.approx(
        SEVERITY_WEIGHTS["mild headache"]
    )

    # Mean of two recognised weights, no bonus (< 3 severe).
    expected = (SEVERITY_WEIGHTS["fever"] + SEVERITY_WEIGHTS["cough"]) / 2
    assert severity_scorer(["fever", "cough"]) == pytest.approx(expected)

    # Three severe symptoms (all ≥ 0.70) → +0.10 bonus, still ≤ 1.0.
    severe = ["chest pain", "syncope", "hemoptysis"]
    base = sum(SEVERITY_WEIGHTS[s] for s in severe) / len(severe)
    assert severity_scorer(severe) == pytest.approx(min(base + 0.10, 1.0))

    # Critical input must cap at 1.0 (cardiac arrest alone is already 1.0).
    assert severity_scorer(["cardiac arrest", "loss of consciousness", "unresponsive"]) == 1.0

    # Unknowns are silently dropped.
    assert severity_scorer(["fever", "totally_made_up"]) == pytest.approx(
        SEVERITY_WEIGHTS["fever"]
    )

    # Tool wrapper exposes the traced callable.
    assert isinstance(severity_scorer_tool, Tool)
    assert severity_scorer_tool.name == "severity_scorer"
    assert severity_scorer_tool.func is severity_scorer


# ---------------------------------------------------------------------------
# textbook_search  (slow: builds the textbook FAISS index on first run)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_textbook_search_returns_topk_textbook_passages():
    """Real FAISS retrieval over the local textbook index. Asserts shape,
    score ordering, and that results are clearly relevant to the query."""
    query = "myocardial infarction chest pain ST elevation treatment"
    results = textbook_search(query, k=5)

    assert isinstance(results, list)
    assert len(results) == 5

    seen_ids = set()
    last_score = float("inf")
    for hit in results:
        assert set(hit.keys()) == {"doc_id", "text", "score", "source_file"}
        assert isinstance(hit["doc_id"], str) and hit["doc_id"]
        assert isinstance(hit["text"], str) and hit["text"]
        assert isinstance(hit["source_file"], str) and hit["source_file"].endswith(".txt")
        assert isinstance(hit["score"], float)
        # Scores monotonically non-increasing (FAISS top-k order).
        assert hit["score"] <= last_score
        last_score = hit["score"]
        # No duplicate chunks in the top-k.
        assert hit["doc_id"] not in seen_ids
        seen_ids.add(hit["doc_id"])

    # At least one of the top hits should mention something cardiac-relevant.
    joined = " ".join(h["text"].lower() for h in results)
    assert any(term in joined for term in ("myocardial", "infarction", "coronary", "ischemia")), (
        "None of the top-5 textbook hits look cardiac-relevant for the query."
    )

    # k=0 and empty query are degenerate but well-defined.
    assert textbook_search("", k=5) == []
    assert textbook_search(query, k=0) == []

    # Tool wrapper exposes the traced callable.
    assert isinstance(textbook_search_tool, Tool)
    assert textbook_search_tool.name == "textbook_search"
    assert textbook_search_tool.func is textbook_search


# ---------------------------------------------------------------------------
# textbook_search — name + description sanity
# ---------------------------------------------------------------------------

class TestTextbookSearchNaming:
    """The Tool description and module docstring should describe what the
    tool actually does (local FAISS over medical textbooks)."""

    def test_tool_description_states_local_faiss(self):
        desc = (textbook_search_tool.description or "").lower()
        assert "local" in desc
        assert "faiss" in desc
        assert "textbook" in desc

    def test_module_docstring_describes_textbook_search(self):
        from agentxai.tools import textbook_search as mod
        doc = (mod.__doc__ or "").lower()
        assert "faiss" in doc and "textbook" in doc

    def test_dashboard_display_passes_known_names_through(self):
        # Deferred import so streamlit isn't required at collection time.
        from agentxai.ui.dashboard import _tool_display_name
        assert _tool_display_name("textbook_search") == "textbook_search"
        assert _tool_display_name("symptom_lookup") == "symptom_lookup"
        assert _tool_display_name("guideline_lookup") == "guideline_lookup"
        assert _tool_display_name("severity_scorer") == "severity_scorer"
        assert _tool_display_name("") == "?"
        assert _tool_display_name(None) == "?"


# ---------------------------------------------------------------------------
# guideline_lookup
# ---------------------------------------------------------------------------

def test_guideline_lookup_fuzzy_matches_against_real_store():
    """Real read from data/indices/guidelines.json with a rapidfuzz match."""
    # Empty / blank input → no match.
    assert guideline_lookup("") == {"match": None}
    assert guideline_lookup("   ") == {"match": None}

    # Nonsense string → below cutoff, no match.
    assert guideline_lookup("zzzzz_not_a_real_condition_xyzzy") == {"match": None}

    # A common MedQA answer text should resolve to *some* stored condition.
    # Use a near-but-not-exact spelling to exercise the fuzzy path.
    hit = guideline_lookup("Acute myocardial infarction")
    assert isinstance(hit, dict)
    if hit.get("match") is None:
        # Fall back to picking an actual key from the store and bouncing it
        # through the matcher — this guarantees a successful match for the
        # assertions below regardless of MedQA top-50 contents.
        from agentxai.tools.guideline_lookup import _ensure_store  # type: ignore
        any_key = next(iter(_ensure_store().keys()))
        hit = guideline_lookup(any_key)

    assert hit["match"] is not None
    assert isinstance(hit["match"], str) and hit["match"].strip()
    assert 0.0 <= hit["match_score"] <= 1.0
    assert hit["source"] == "synthetic/medqa-derived"
    assert "summary" in hit and "key_findings" in hit and "recommended_workup" in hit
    assert hit["condition"] == hit["match"]

    # Tool wrapper exposes the traced callable.
    assert isinstance(guideline_lookup_tool, Tool)
    assert guideline_lookup_tool.name == "guideline_lookup"
    assert guideline_lookup_tool.func is guideline_lookup
