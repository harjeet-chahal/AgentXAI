"""
Tests for `agentxai.xai.evidence_attribution`, the report fields it
populates, and the dashboard helper that renders it.

Required scenarios from the spec:
  * evidence IDs are preserved from Specialist B memory
  * final output includes evidence IDs
  * dashboard helper renders evidence safely
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agentxai.data.schemas import (
    AccountabilityReport,
    AgentXAIRecord,
    MemoryDiff,
    TrajectoryEvent,
)
from agentxai.store.trajectory_store import TrajectoryStore
from agentxai.xai.accountability import AccountabilityReportGenerator
from agentxai.xai.evidence_attribution import (
    infer_supporting_evidence_ids,
    latest_top_evidence_from_memory_diffs,
    rank_most_supportive_evidence,
)


# ---------------------------------------------------------------------------
# infer_supporting_evidence_ids
# ---------------------------------------------------------------------------

class TestInferSupportingEvidenceIds:
    def test_doc_id_substring_match(self):
        ev = [
            {"doc_id": "Harrison__0341", "snippet": "Some text",
             "score": 0.9, "source_file": "Harrison.txt"},
            {"doc_id": "Robbins__0099", "snippet": "Other text",
             "score": 0.7, "source_file": "Robbins.txt"},
        ]
        rationale = "As discussed in Harrison__0341, the diagnosis is MI."
        assert infer_supporting_evidence_ids(rationale, ev) == ["Harrison__0341"]

    def test_snippet_token_match(self):
        ev = [
            {"doc_id": "FA_step1_HIV", "snippet":
             "Ceftriaxone is the first-line antibiotic for gonorrhea.",
             "score": 0.85, "source_file": "First_Aid.txt"},
        ]
        rationale = "Per CDC, Ceftriaxone is the first-line treatment."
        assert infer_supporting_evidence_ids(rationale, ev) == ["FA_step1_HIV"]

    def test_dedup_preserves_order(self):
        ev = [
            {"doc_id": "doc-A", "snippet": "Ceftriaxone gonorrhea"},
            {"doc_id": "doc-B", "snippet": "Other"},
            {"doc_id": "doc-A", "snippet": "duplicate"},   # ignored
            {"doc_id": "doc-C", "snippet": "Ceftriaxone again"},
        ]
        rationale = "Ceftriaxone is mentioned"
        out = infer_supporting_evidence_ids(rationale, ev)
        assert out == ["doc-A", "doc-C"]

    def test_no_match_returns_empty(self):
        ev = [{"doc_id": "doc-A", "snippet": "Pneumonia and cough"}]
        rationale = "Findings are most consistent with MI."
        assert infer_supporting_evidence_ids(rationale, ev) == []

    def test_short_snippet_tokens_dont_match(self):
        # All snippet tokens < _MIN_TOKEN_LEN (5) → no match.
        ev = [{"doc_id": "doc-A", "snippet": "to be or not"}]
        rationale = "to be or not to be"
        # doc_id "doc-A" doesn't appear in rationale either → no match.
        assert infer_supporting_evidence_ids(rationale, ev) == []

    def test_empty_inputs(self):
        assert infer_supporting_evidence_ids("", [{"doc_id": "x"}]) == []
        assert infer_supporting_evidence_ids("text", []) == []
        assert infer_supporting_evidence_ids("", []) == []

    def test_skips_non_dict_entries(self):
        ev = ["not a dict", {"doc_id": "doc-A", "snippet": "Ceftriaxone"}]
        assert infer_supporting_evidence_ids("Ceftriaxone given", ev) == ["doc-A"]


# ---------------------------------------------------------------------------
# rank_most_supportive_evidence
# ---------------------------------------------------------------------------

class TestRankMostSupportive:
    def test_used_beats_uncited(self):
        ev = [
            {"doc_id": "high_uncited", "score": 0.95},
            {"doc_id": "low_used", "score": 0.40},
        ]
        ranked = rank_most_supportive_evidence(ev, used_ids=["low_used"])
        # low_used: 1.0 + 0.40 = 1.40; high_uncited: 0.0 + 0.95 = 0.95.
        assert ranked == ["low_used", "high_uncited"]

    def test_falls_back_to_score_when_none_used(self):
        ev = [
            {"doc_id": "low",  "score": 0.30},
            {"doc_id": "high", "score": 0.95},
            {"doc_id": "mid",  "score": 0.60},
        ]
        assert rank_most_supportive_evidence(ev, used_ids=[]) == ["high", "mid", "low"]

    def test_limit_caps_results(self):
        ev = [{"doc_id": f"d{i}", "score": float(i) / 10} for i in range(10)]
        ranked = rank_most_supportive_evidence(ev, used_ids=[], limit=3)
        assert len(ranked) == 3
        assert ranked == ["d9", "d8", "d7"]

    def test_dedup_doc_ids(self):
        ev = [
            {"doc_id": "dup", "score": 0.9},
            {"doc_id": "dup", "score": 0.8},
            {"doc_id": "other", "score": 0.5},
        ]
        ranked = rank_most_supportive_evidence(ev, used_ids=[])
        assert ranked == ["dup", "other"]

    def test_handles_missing_score(self):
        ev = [{"doc_id": "a"}, {"doc_id": "b", "score": 0.5}]
        ranked = rank_most_supportive_evidence(ev, used_ids=[])
        # b > a (0.5 + 0 vs 0 + 0).
        assert ranked == ["b", "a"]

    def test_handles_empty_input(self):
        assert rank_most_supportive_evidence([], used_ids=["x"]) == []
        assert rank_most_supportive_evidence(None, used_ids=None) == []  # type: ignore[arg-type]

    def test_skips_entries_with_no_doc_id(self):
        ev = [{"score": 0.9}, {"doc_id": "ok", "score": 0.5}]
        assert rank_most_supportive_evidence(ev, used_ids=[]) == ["ok"]


# ---------------------------------------------------------------------------
# latest_top_evidence_from_memory_diffs
# ---------------------------------------------------------------------------

class TestLatestTopEvidenceFromMemoryDiffs:
    def test_returns_latest_write(self):
        early = MemoryDiff(
            agent_id="specialist_b", operation="write", key="top_evidence",
            value_after=[{"doc_id": "old"}], timestamp=1.0,
        )
        late = MemoryDiff(
            agent_id="specialist_b", operation="write", key="top_evidence",
            value_after=[{"doc_id": "new"}], timestamp=2.0,
        )
        out = latest_top_evidence_from_memory_diffs([early, late])
        assert out == [{"doc_id": "new"}]

    def test_ignores_other_agents(self):
        diffs = [
            MemoryDiff(agent_id="specialist_a", operation="write",
                       key="top_evidence", value_after=[{"doc_id": "x"}],
                       timestamp=1.0),
        ]
        assert latest_top_evidence_from_memory_diffs(diffs) == []

    def test_ignores_other_keys(self):
        diffs = [
            MemoryDiff(agent_id="specialist_b", operation="write",
                       key="some_other_key", value_after=[{"doc_id": "x"}],
                       timestamp=1.0),
        ]
        assert latest_top_evidence_from_memory_diffs(diffs) == []

    def test_ignores_reads(self):
        diffs = [
            MemoryDiff(agent_id="specialist_b", operation="read",
                       key="top_evidence", value_after=[{"doc_id": "x"}],
                       timestamp=1.0),
        ]
        assert latest_top_evidence_from_memory_diffs(diffs) == []

    def test_handles_dicts(self):
        # The accountability scorer passes dataclasses; the dashboard
        # passes plain dicts (from the API). Both paths must work.
        diffs = [{
            "agent_id": "specialist_b", "operation": "write",
            "key": "top_evidence", "value_after": [{"doc_id": "x"}],
            "timestamp": 1.0,
        }]
        assert latest_top_evidence_from_memory_diffs(diffs) == [{"doc_id": "x"}]

    def test_returns_empty_for_no_diffs(self):
        assert latest_top_evidence_from_memory_diffs([]) == []
        assert latest_top_evidence_from_memory_diffs(None) == []   # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------

class TestSchemaRoundTrip:
    def test_evidence_fields_default_empty(self):
        report = AccountabilityReport(task_id="t")
        assert report.evidence_used_by_final_answer == []
        assert report.most_supportive_evidence_ids == []

    def test_evidence_fields_in_to_dict(self):
        report = AccountabilityReport(
            task_id="t",
            evidence_used_by_final_answer=["doc-1", "doc-2"],
            most_supportive_evidence_ids=["doc-1", "doc-3"],
        )
        d = report.to_dict()
        assert d["evidence_used_by_final_answer"] == ["doc-1", "doc-2"]
        assert d["most_supportive_evidence_ids"] == ["doc-1", "doc-3"]

    def test_round_trip_through_from_dict(self):
        report = AccountabilityReport(
            task_id="t",
            evidence_used_by_final_answer=["d1"],
            most_supportive_evidence_ids=["d2"],
        )
        rebuilt = AccountabilityReport.from_dict(report.to_dict())
        assert rebuilt.evidence_used_by_final_answer == ["d1"]
        assert rebuilt.most_supportive_evidence_ids == ["d2"]


# ---------------------------------------------------------------------------
# AccountabilityReportGenerator integration
# ---------------------------------------------------------------------------

class _NoOpLLM:
    def invoke(self, prompt):
        class _R:
            content = "explanation"
        return _R()


class _NoOpPipeline:
    def resume_from(self, snapshot, overrides):
        return {"final_diagnosis": "X", "confidence": 0.5}


def _build_store_with_evidence(
    *,
    rationale: str,
    supporting_ids: List[str],
    top_evidence: List[Dict[str, Any]],
):
    """Build a minimal task with one Specialist-B top_evidence write."""
    s = TrajectoryStore(db_url="sqlite:///:memory:")
    task_id = "EV-TASK"
    s.save_task(AgentXAIRecord(
        task_id=task_id,
        source="test",
        input={"patient_case": "x", "options": {}},
        ground_truth={},
        system_output={
            "final_diagnosis": "X",
            "confidence": 0.9,
            "correct": True,
            "rationale": rationale,
            "supporting_evidence_ids": list(supporting_ids),
        },
    ))
    for ag in ("specialist_a", "specialist_b"):
        s.save_event(task_id, TrajectoryEvent(
            agent_id=ag, event_type="agent_action",
            action="diagnose", timestamp=1.0,
        ))
    s.save_memory_diff(task_id, MemoryDiff(
        agent_id="specialist_b", operation="write", key="top_evidence",
        value_after=top_evidence, timestamp=1.0,
    ))
    return s, task_id


class TestAccountabilityReportEvidenceFields:
    def test_explicit_supporting_ids_preserved(self):
        s, task_id = _build_store_with_evidence(
            rationale="MI fits the case.",
            supporting_ids=["Harrison__0341", "FA_001"],
            top_evidence=[
                {"doc_id": "Harrison__0341", "snippet": "MI",
                 "score": 0.9, "source_file": "Harrison.txt"},
                {"doc_id": "FA_001", "snippet": "Cardiac",
                 "score": 0.7, "source_file": "FA.txt"},
                {"doc_id": "Robbins__099", "snippet": "Pathology",
                 "score": 0.6, "source_file": "Robbins.txt"},
            ],
        )
        gen = AccountabilityReportGenerator(
            store=s, pipeline=_NoOpPipeline(), llm=_NoOpLLM(),
        )
        report = gen.generate(task_id)

        # The synthesizer's own list flows verbatim into the report.
        assert report.evidence_used_by_final_answer == ["Harrison__0341", "FA_001"]
        # most_supportive ranks: cited beats uncited; tie-break by score.
        assert report.most_supportive_evidence_ids[:2] == ["Harrison__0341", "FA_001"]
        assert "Robbins__099" in report.most_supportive_evidence_ids

    def test_heuristic_inference_when_synthesizer_left_empty(self):
        # Synthesizer didn't emit supporting_evidence_ids, but the
        # rationale references content from one of the snippets.
        s, task_id = _build_store_with_evidence(
            rationale="Per Harrison, Ceftriaxone is the first-line.",
            supporting_ids=[],   # synthesizer forgot
            top_evidence=[
                {"doc_id": "Harrison__0341",
                 "snippet": "Ceftriaxone is first-line for gonorrhea.",
                 "score": 0.9, "source_file": "Harrison.txt"},
                {"doc_id": "Robbins__099",
                 "snippet": "Coronary thrombosis pathology.",
                 "score": 0.6, "source_file": "Robbins.txt"},
            ],
        )
        gen = AccountabilityReportGenerator(
            store=s, pipeline=_NoOpPipeline(), llm=_NoOpLLM(),
        )
        report = gen.generate(task_id)
        # The heuristic should pick up Harrison__0341 from "Ceftriaxone".
        assert "Harrison__0341" in report.evidence_used_by_final_answer
        # Robbins not cited → not used.
        assert "Robbins__099" not in report.evidence_used_by_final_answer

    def test_round_trip_through_store(self):
        s, task_id = _build_store_with_evidence(
            rationale="MI",
            supporting_ids=["Harrison__0341"],
            top_evidence=[
                {"doc_id": "Harrison__0341", "snippet": "MI",
                 "score": 0.9, "source_file": "Harrison.txt"},
            ],
        )
        gen = AccountabilityReportGenerator(
            store=s, pipeline=_NoOpPipeline(), llm=_NoOpLLM(),
        )
        gen.generate(task_id)

        loaded = s.get_full_record(task_id).xai_data.accountability_report
        assert loaded is not None
        assert loaded.evidence_used_by_final_answer == ["Harrison__0341"]
        assert loaded.most_supportive_evidence_ids == ["Harrison__0341"]

    def test_no_evidence_at_all_yields_empty_lists(self):
        s, task_id = _build_store_with_evidence(
            rationale="MI",
            supporting_ids=[],
            top_evidence=[],
        )
        gen = AccountabilityReportGenerator(
            store=s, pipeline=_NoOpPipeline(), llm=_NoOpLLM(),
        )
        report = gen.generate(task_id)
        assert report.evidence_used_by_final_answer == []
        assert report.most_supportive_evidence_ids == []


# ---------------------------------------------------------------------------
# Dashboard helper render-safety
# ---------------------------------------------------------------------------

class TestDashboardEvidenceRendering:
    def test_extract_top_evidence_handles_legacy_record(self):
        # Old record with no memory_diffs at all.
        from agentxai.ui.dashboard import _extract_top_evidence
        assert _extract_top_evidence({}) == []
        assert _extract_top_evidence({"memory_diffs": []}) == []

    def test_extract_top_evidence_pulls_latest_b_write(self):
        from agentxai.ui.dashboard import _extract_top_evidence
        xai = {
            "memory_diffs": [
                {"agent_id": "specialist_b", "operation": "write",
                 "key": "top_evidence",
                 "value_after": [{"doc_id": "d1"}], "timestamp": 1.0},
                {"agent_id": "specialist_b", "operation": "write",
                 "key": "top_evidence",
                 "value_after": [{"doc_id": "d2"}], "timestamp": 2.0},
            ],
        }
        assert _extract_top_evidence(xai) == [{"doc_id": "d2"}]

    def test_render_evidence_cards_handles_empty(self, monkeypatch):
        # Empty top_evidence → no rendering at all.
        from agentxai.ui import dashboard
        captured: List = []
        monkeypatch.setattr(dashboard.st, "markdown",
                            lambda *a, **kw: captured.append(("md", a)))
        monkeypatch.setattr(dashboard.st, "expander",
                            lambda *a, **kw: captured.append(("exp", a)))

        dashboard._render_evidence_cards(top_evidence=[], used_ids=[])
        assert captured == []

    def test_render_evidence_cards_marks_used(self, monkeypatch):
        from agentxai.ui import dashboard
        markdown_calls: List[str] = []
        # Streamlit returns a context manager from st.expander; emulate it.

        class _Ctx:
            def __enter__(self): return self
            def __exit__(self, *a): return False

        expander_calls: List[str] = []

        def fake_markdown(html, **kwargs):
            markdown_calls.append(html)

        def fake_expander(label, *a, **kwargs):
            expander_calls.append(label)
            return _Ctx()

        # Stub other Streamlit helpers used inside the with-block.
        monkeypatch.setattr(dashboard.st, "markdown", fake_markdown)
        monkeypatch.setattr(dashboard.st, "expander", fake_expander)
        monkeypatch.setattr(dashboard.st, "write", lambda *a, **kw: None)
        monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)
        monkeypatch.setattr(
            dashboard.st, "columns",
            lambda n: [type("C", (), {"metric": lambda *a, **kw: None})() for _ in range(n)],
        )

        dashboard._render_evidence_cards(
            top_evidence=[
                {"doc_id": "Harrison__0341",
                 "source_file": "Harrison.txt",
                 "score": 0.9,
                 "snippet": "Ceftriaxone is first-line."},
                {"doc_id": "Robbins__099",
                 "source_file": "Robbins.txt",
                 "score": 0.6,
                 "snippet": "Pathology."},
            ],
            used_ids=["Harrison__0341"],
        )
        # Header card-eyebrow rendered with cited count.
        assert any("1 cited in rationale" in m for m in markdown_calls)
        # Two expanders, one per evidence.
        assert len(expander_calls) == 2
        # Used marker on the cited doc, not the other.
        assert any("✓ used" in lbl for lbl in expander_calls)
        assert any("Harrison__0341" in lbl for lbl in expander_calls)

    def test_render_handles_missing_optional_fields(self, monkeypatch):
        # Evidence dict with no source_file / score / snippet → should
        # still render without throwing.
        from agentxai.ui import dashboard

        class _Ctx:
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr(dashboard.st, "markdown", lambda *a, **kw: None)
        monkeypatch.setattr(dashboard.st, "expander",
                            lambda *a, **kw: _Ctx())
        monkeypatch.setattr(dashboard.st, "write", lambda *a, **kw: None)
        monkeypatch.setattr(dashboard.st, "caption", lambda *a, **kw: None)
        monkeypatch.setattr(
            dashboard.st, "columns",
            lambda n: [type("C", (), {"metric": lambda *a, **kw: None})() for _ in range(n)],
        )

        # Should not raise.
        dashboard._render_evidence_cards(
            top_evidence=[
                {"doc_id": "x"},          # bare entry
                "not-a-dict",             # ignored
                {},                       # empty dict — uses fallback id
            ],
            used_ids=["x"],
        )
